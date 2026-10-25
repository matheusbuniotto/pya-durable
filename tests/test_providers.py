"""Tests for pluggable AI model providers, auth configuration, and Pi Durable bootstrap ergonomics."""

import os
from pathlib import Path
import tempfile
import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_durable import (
    Conversation,
    DurableConversation,
    DurableHarness,
    Harness,
    MemoryStorage,
    ModelsRegistry,
    SqliteStorage,
    anthropic_provider,
    create_models,
    gemini_provider,
    global_models_registry,
    ollama_provider,
    openai_provider,
    openrouter_provider,
    resolve_agent,
)


@pytest.mark.asyncio
async def test_create_models_registry():
    models = create_models()
    assert isinstance(models, ModelsRegistry)

    # Register openrouter provider
    models.set_provider(openrouter_provider(api_key="sk-or-test-123"))
    provider = models.get_provider("openrouter")
    assert getattr(provider, "client", None) is not None
    assert provider.client.api_key == "sk-or-test-123"

    # Resolve model with registered provider
    model = models.resolve_model("openrouter:openai/gpt-6-luna")
    assert model.model_name == "openai/gpt-6-luna"


@pytest.mark.asyncio
async def test_openai_and_anthropic_providers():
    models = create_models()
    models.set_provider(openai_provider(api_key="sk-openai-test", base_url="https://api.openai.com/v1"))
    models.set_provider(anthropic_provider(api_key="sk-ant-test"))

    oai_p = models.get_provider("openai")
    ant_p = models.get_provider("anthropic")
    assert oai_p is not None
    assert ant_p is not None

    m_oai = models.resolve_model("openai:gpt-4o")
    assert m_oai.model_name == "gpt-4o"

    m_ant = models.resolve_model("anthropic:claude-3-5-haiku-latest")
    assert m_ant.model_name == "claude-3-5-haiku-latest"


@pytest.mark.asyncio
async def test_gemini_and_ollama_providers():
    models = create_models()
    models.set_provider(gemini_provider(api_key="ai-gemini-test"))
    models.set_provider(ollama_provider(base_url="http://localhost:11434/v1"))

    gem_p = models.get_provider("gemini")
    assert gem_p is not None

    oll_p = models.get_provider("ollama")
    assert oll_p is not None


@pytest.mark.asyncio
async def test_resolve_agent_string_shortcuts():
    models = create_models()
    models.set_provider("openrouter", api_key="sk-or-test")

    # Normalize shortcut 'gpt-luna 6' -> 'openrouter:openai/gpt-6-luna'
    agent_luna = resolve_agent("gpt-luna 6", models=models)
    assert isinstance(agent_luna, Agent)
    assert agent_luna.model.model_name == "openai/gpt-6-luna"

    # Test model shortcut
    agent_test = resolve_agent("test")
    assert isinstance(agent_test.model, TestModel)


@pytest.mark.asyncio
async def test_resolve_agent_dict_spec():
    # Pi Durable style nested agent dict
    spec_pi_durable = {
        "agent": {
            "model": {
                "provider": "openrouter",
                "modelId": "openai/gpt-6-luna",
                "apiKey": "sk-or-nested-test",
            }
        }
    }
    agent = resolve_agent(spec_pi_durable)
    assert isinstance(agent, Agent)
    assert agent.model.model_name == "openai/gpt-6-luna"

    # Flat dict spec
    spec_flat = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-flat-test",
    }
    agent_flat = resolve_agent(spec_flat)
    assert agent_flat.model.model_name == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_resolve_agent_preserves_agent_and_adds_tools():
    base_agent = Agent(TestModel(), system_prompt="Base prompt")

    def dummy_tool(x: int) -> int:
        return x * 2

    # Agent instance without modifications returns same agent
    same = resolve_agent(base_agent)
    assert same is base_agent

    extended = resolve_agent(base_agent, tools=[dummy_tool])
    assert len(extended._function_toolset.tools) == 1


@pytest.mark.asyncio
async def test_harness_bootstrap_pi_durable_style():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "agent.db"

        # 1. Models configuration
        models = create_models()
        models.set_provider(openrouter_provider(api_key="sk-or-test"))

        # 2. Open Harness with SqliteStorage
        storage = SqliteStorage(db_path)
        harness = await Harness.open(storage, models=models)
        assert isinstance(harness, DurableHarness)

        # 3. Resume tasks
        recovery = await harness.resume()
        assert recovery == []

        # 4. Open root conversation with TestModel
        root = await harness.root(agent=Agent(TestModel(custom_output_text="Bootstrap verified")))
        assert isinstance(root, Conversation)
        assert isinstance(root, DurableConversation)

        # 5. Submit turn with dict input
        res = await root.submit({"content": "Test bootstrap", "requestId": "req-boot-1"})
        assert res is not None
        assert res.output == "Bootstrap verified"

        # 6. Branch root conversation
        child = await root.fork(agent=Agent(TestModel(custom_output_text="Child turn verified")))
        res_child = await child.submit("Child input")
        assert res_child.output == "Child turn verified"

        # 7. Close harness
        await harness.close()


@pytest.mark.asyncio
async def test_harness_storage_first_positional_ergonomics():
    storage = MemoryStorage()
    # User calls Harness(storage) without specifying agent
    harness = Harness(storage)
    assert harness.storage is storage
    assert harness.agent is None

    # Opening root with agent
    root = await harness.root(agent=Agent(TestModel(custom_output_text="Storage first ok")))
    res = await root.run("Hello")
    assert res.output == "Storage first ok"


@pytest.mark.asyncio
async def test_harness_conversation_by_id():
    storage = MemoryStorage()
    harness = await Harness.open(storage)

    conv_a = await harness.conversation("channel-alpha", agent=Agent(TestModel(custom_output_text="Alpha OK")))
    res_a = await conv_a.run_turn("Ping Alpha")
    assert res_a.output == "Alpha OK"

    conv_b = await harness.conversation("channel-beta", agent=Agent(TestModel(custom_output_text="Beta OK")))
    res_b = await conv_b.run_turn("Ping Beta")
    assert res_b.output == "Beta OK"
