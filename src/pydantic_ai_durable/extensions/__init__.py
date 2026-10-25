"""Extensions, Dynamic Prompt Sections, Compaction, Durable Documents, and Subagents."""

from pydantic_ai_durable.extensions.compaction import (
    BackgroundCompactor,
    create_handoff_tool,
    create_search_history_tool,
)
from pydantic_ai_durable.extensions.docs import (
    DocDefinition,
    DocumentManager,
    define_doc,
)
from pydantic_ai_durable.extensions.memo import MemoManager
from pydantic_ai_durable.extensions.registry import (
    ExtensionDefinition,
    ExtensionRegistry,
    HookContext,
    HookDecision,
    LifecycleHook,
    ModelCallInfo,
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
from pydantic_ai_durable.extensions.subagents import SubagentManager, SubagentResult

__all__ = [
    "BackgroundCompactor",
    "DocDefinition",
    "DocumentManager",
    "ExtensionDefinition",
    "ExtensionRegistry",
    "HookContext",
    "HookDecision",
    "LifecycleHook",
    "MemoManager",
    "ModelCallInfo",
    "PromptSection",
    "SectionContext",
    "SubagentManager",
    "SubagentResult",
    "ToolCallInfo",
    "ToolWrapper",
    "create_handoff_tool",
    "create_search_history_tool",
    "define_doc",
    "define_extension",
    "define_section",
    "global_extension_registry",
    "hook",
    "wrap_tool",
]
