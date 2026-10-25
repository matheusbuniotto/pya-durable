import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pydantic_ai_durable.extensions import (
    ExtensionDefinition,
    ExtensionRegistry,
    HookContext,
    HookDecision,
    PromptSection,
    SectionContext,
    ToolCallInfo,
    ToolWrapper,
    define_extension,
    define_section,
    global_extension_registry,
    hook,
    wrap_tool,
)
from pydantic_ai_durable.harness import DurableHarness
from pydantic_ai_durable.models import TranscriptEntryKind
from pydantic_ai_durable.replay import durable_tool
from pydantic_ai_durable.storage import AioSqliteStorage


@pytest.fixture(autouse=True)
def clean_registry():
    global_extension_registry.clear()
    yield
    global_extension_registry.clear()


def test_install_and_uninstall_extension():
    registry = ExtensionRegistry()
    ext = define_extension(name="telemetry", version="1.2.0")
    registry.install(ext)

    assert registry.list_extensions() == ["telemetry"]
    fetched = registry.get_extension("telemetry")
    assert fetched is not None
    assert fetched.version == "1.2.0"

    registry.uninstall("telemetry")
    assert registry.list_extensions() == []
    assert registry.get_extension("telemetry") is None


def test_extension_tool_overriding_by_name():
    def greet():
        return "base"

    def custom_greet():
        return "extended"
    custom_greet.__name__ = "greet"

    registry = ExtensionRegistry()
    ext = define_extension(name="override_ext", tools=[custom_greet])
    registry.install(ext)

    resolved = registry.resolve_tools([greet])
    assert len(resolved) == 1
    assert resolved[0].__name__ == "greet"
    assert resolved[0]() == "extended"


@pytest.mark.asyncio
async def test_prompt_sections_rendered_with_priorities():
    registry = ExtensionRegistry()
    sec_low = define_section("footer", lambda: "System Footer", priority=200)
    sec_high = define_section("header", lambda: "System Header", priority=10)
    sec_mid = define_section("body", lambda ctx: f"Conversation: {ctx.conversation_id}", priority=50)

    ext = define_extension("sections_ext", sections=[sec_low, sec_high, sec_mid])
    registry.install(ext)

    context = SectionContext(conversation_id="conv-123")
    rendered = await registry.render_prompt_sections(context)
    lines = rendered.split("\n\n")

    assert lines[0] == "System Header"
    assert lines[1] == "Conversation: conv-123"
    assert lines[2] == "System Footer"


def test_tool_wrapping_timing_and_metrics():
    registry = ExtensionRegistry()
    metric_calls = []

    def calc(val: int) -> int:
        return val * 2

    def calc_wrapper(fn):
        def inner(val: int) -> int:
            metric_calls.append(val)
            return fn(val) + 10
        inner.__name__ = fn.__name__
        return inner

    ext = define_extension("wrapper_ext", wraps=[wrap_tool("calc", calc_wrapper)])
    registry.install(ext)

    resolved = registry.resolve_tools([calc])
    assert len(resolved) == 1
    result = resolved[0](5)
    assert result == 20
    assert metric_calls == [5]


@pytest.mark.asyncio
async def test_before_tool_blocking_policy():
    @durable_tool
    async def restricted_action(count: int) -> str:
        return f"Executed {count}"

    def policy_hook(info: ToolCallInfo, ctx: HookContext) -> HookDecision | None:
        if info.args.get("count", 0) > 10:
            return HookDecision(block="Execution exceeds threshold limit")
        return None

    ext = define_extension("guard_ext", hooks=[hook(name="guard", before_tool=policy_hook)])
    global_extension_registry.install(ext)

    res_blocked = await restricted_action(count=15)
    assert res_blocked == {"error": "Execution exceeds threshold limit", "blocked": True}

    res_allowed = await restricted_action(count=5)
    assert res_allowed == "Executed 5"


@pytest.mark.asyncio
async def test_before_tool_argument_rewriting():
    @durable_tool
    async def process_number(n: int) -> int:
        return n * n

    def normalize_hook(info: ToolCallInfo, ctx: HookContext) -> HookDecision | None:
        val = info.args.get("n", 0)
        if val < 0:
            return HookDecision(modified_args={"n": abs(val)})
        return None

    ext = define_extension("normalizer", hooks=[hook(name="abs_rewriter", before_tool=normalize_hook)])
    global_extension_registry.install(ext)

    result = await process_number(n=-7)
    assert result == 49


@pytest.mark.asyncio
async def test_after_tool_output_redaction():
    @durable_tool
    async def read_secrets() -> str:
        return "Confidential: api_key=secret-xyz-789"

    def redact_hook(info: ToolCallInfo, result: Any, ctx: HookContext) -> Any:
        if isinstance(result, str):
            return result.replace("secret-xyz-789", "[REDACTED]")
        return result

    ext = define_extension("security", hooks=[hook(name="redactor", after_tool=redact_hook)])
    global_extension_registry.install(ext)

    output = await read_secrets()
    assert output == "Confidential: api_key=[REDACTED]"


@pytest.mark.asyncio
async def test_hook_exception_isolation():
    @durable_tool
    async def robust_calc(x: int) -> int:
        return x + 100

    def faulty_hook(info: ToolCallInfo, ctx: HookContext) -> None:
        raise RuntimeError("Telemetry failure")

    def faulty_after_hook(info: ToolCallInfo, res: Any, ctx: HookContext) -> None:
        raise ValueError("Metrics upload error")

    ext = define_extension(
        "flaky_ext",
        hooks=[hook(name="flaky", before_tool=faulty_hook, after_tool=faulty_after_hook)],
    )
    global_extension_registry.install(ext)

    result = await robust_calc(x=25)
    assert result == 125


@pytest.mark.asyncio
async def test_hot_reloading_extension_replaces_cleanly():
    registry = ExtensionRegistry()
    ext_v1 = define_extension(
        name="hot_config",
        version="1.0.0",
        sections=[define_section("cfg", lambda: "config_v1", priority=100)],
    )
    registry.install(ext_v1)
    rendered_v1 = await registry.render_prompt_sections(SectionContext())
    assert rendered_v1 == "config_v1"

    ext_v2 = define_extension(
        name="hot_config",
        version="2.0.0",
        sections=[define_section("cfg", lambda: "config_v2", priority=100)],
    )
    registry.install(ext_v2)
    assert len(registry.list_extensions()) == 1
    rendered_v2 = await registry.render_prompt_sections(SectionContext())
    assert rendered_v2 == "config_v2"


@pytest.mark.asyncio
async def test_harness_integration_with_extensions(tmp_path: Path):
    db_file = tmp_path / "test_harness_ext.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        registry = ExtensionRegistry()
        sec = define_section("policy", lambda ctx: "POLICY: Read-only transactions permitted", priority=5)
        registry.install(define_extension("enterprise_policy", sections=[sec]))

        harness = DurableHarness(agent, storage, conversation_id="conv-ext-test", registry=registry)
        turn_result = await harness.submit("Execute query")

        assert turn_result is not None
        transcript = await harness.get_transcript()
        user_entries = [e for e in transcript if e.kind == TranscriptEntryKind.USER]
        assert len(user_entries) > 0
        user_payload = json.loads(user_entries[0].message_json)
        assert "POLICY: Read-only transactions permitted\n\nExecute query" in user_payload["prompt"]
