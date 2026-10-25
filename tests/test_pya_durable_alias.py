"""Verify backward-compatibility of the pya_durable alias package."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import pydantic_ai_durable
import pya_durable


def test_top_level_alias_identity():
    assert pya_durable.DurableHarness is pydantic_ai_durable.DurableHarness
    assert pya_durable.Harness is pydantic_ai_durable.Harness
    assert pya_durable.DurableConversation is pydantic_ai_durable.DurableConversation
    assert pya_durable.Conversation is pydantic_ai_durable.Conversation
    assert pya_durable.SqliteStorage is pydantic_ai_durable.SqliteStorage
    assert pya_durable.MemoryStorage is pydantic_ai_durable.MemoryStorage
    assert pya_durable.JsonlStorage is pydantic_ai_durable.JsonlStorage
    assert pya_durable.create_models is pydantic_ai_durable.create_models
    assert pya_durable.openrouter_provider is pydantic_ai_durable.openrouter_provider
    assert pya_durable.openai_provider is pydantic_ai_durable.openai_provider
    assert pya_durable.approval_gate is pydantic_ai_durable.approval_gate
    assert pya_durable.define_saga is pydantic_ai_durable.define_saga


def test_submodule_alias_identity():
    from pya_durable.models import TaskStatus as TS1
    from pydantic_ai_durable.models import TaskStatus as TS2
    assert TS1 is TS2

    from pya_durable.harness import DurableHarness as H1
    from pydantic_ai_durable.harness import DurableHarness as H2
    assert H1 is H2

    from pya_durable.storage import SqliteStorage as S1
    from pydantic_ai_durable.storage import SqliteStorage as S2
    assert S1 is S2


@pytest.mark.asyncio
async def test_execution_via_pya_alias():
    from pya_durable import Harness, MemoryStorage

    storage = MemoryStorage()
    harness = await Harness.open(storage)
    root = await harness.root(agent=Agent(TestModel(custom_output_text="Alias execution verified")))
    result = await root.submit("Ping")
    assert result.output == "Alias execution verified"
