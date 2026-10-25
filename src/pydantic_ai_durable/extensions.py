from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger("pydantic_ai_durable.extensions")


@dataclass
class HookDecision:
    block: str | None = None
    modified_args: dict[str, Any] | None = None
    require_approval: bool = False
    approval_prompt: str | None = None


@dataclass
class ToolCallInfo:
    name: str
    args: dict[str, Any]
    conversation_id: str | None = None
    task_id: str | None = None


@dataclass
class ModelCallInfo:
    prompt: str
    conversation_id: str | None = None
    task_id: str | None = None


@dataclass
class HookContext:
    conversation_id: str | None = None
    task_id: str | None = None
    storage: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SectionContext:
    conversation_id: str | None = None
    env: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.env is None:
            self.env = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PromptSection:
    name: str
    fn: Callable[..., Any]
    priority: int = 100


def define_section(
    name: str,
    fn: Callable[..., Any],
    priority: int = 100,
) -> PromptSection:
    return PromptSection(name=name, fn=fn, priority=priority)


@dataclass
class ToolWrapper:
    tool_name: str
    wrapper_fn: Callable[..., Any]


def wrap_tool(
    tool_name: str,
    wrapper_fn: Callable[..., Any],
) -> ToolWrapper:
    return ToolWrapper(tool_name=tool_name, wrapper_fn=wrapper_fn)


@dataclass
class LifecycleHook:
    name: str = ""
    before_tool: Callable[[ToolCallInfo, HookContext], Any] | None = None
    after_tool: Callable[[ToolCallInfo, Any, HookContext], Any] | None = None
    before_model: Callable[[ModelCallInfo, HookContext], Any] | None = None
    after_model: Callable[[ModelCallInfo, Any, HookContext], Any] | None = None


def hook(
    name: str = "",
    before_tool: Callable[[ToolCallInfo, HookContext], Any] | None = None,
    after_tool: Callable[[ToolCallInfo, Any, HookContext], Any] | None = None,
    before_model: Callable[[ModelCallInfo, HookContext], Any] | None = None,
    after_model: Callable[[ModelCallInfo, Any, HookContext], Any] | None = None,
) -> LifecycleHook:
    return LifecycleHook(
        name=name,
        before_tool=before_tool,
        after_tool=after_tool,
        before_model=before_model,
        after_model=after_model,
    )


@dataclass
class ExtensionDefinition:
    name: str
    version: str = "1.0.0"
    sections: list[PromptSection] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    wraps: list[ToolWrapper] = field(default_factory=list)
    hooks: list[LifecycleHook] = field(default_factory=list)
    tasks: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sections is None:
            self.sections = []
        if self.tools is None:
            self.tools = []
        if self.wraps is None:
            self.wraps = []
        if self.hooks is None:
            self.hooks = []
        if self.tasks is None:
            self.tasks = []


def define_extension(
    name: str,
    version: str = "1.0.0",
    sections: list[PromptSection] | None = None,
    tools: list[Any] | None = None,
    wraps: list[ToolWrapper] | None = None,
    hooks: list[LifecycleHook] | None = None,
    tasks: list[Any] | None = None,
) -> ExtensionDefinition:
    return ExtensionDefinition(
        name=name,
        version=version,
        sections=sections or [],
        tools=tools or [],
        wraps=wraps or [],
        hooks=hooks or [],
        tasks=tasks or [],
    )


def _get_tool_name(tool: Any) -> str:
    if hasattr(tool, "name") and isinstance(tool.name, str):
        return tool.name
    if hasattr(tool, "__name__") and isinstance(tool.__name__, str):
        return tool.__name__
    return str(tool)


class ExtensionRegistry:
    def __init__(self) -> None:
        self._extensions: dict[str, ExtensionDefinition] = {}

    def install(self, ext: ExtensionDefinition) -> None:
        self._extensions[ext.name] = ext

    def uninstall(self, name: str) -> None:
        self._extensions.pop(name, None)

    def get_extension(self, name: str) -> ExtensionDefinition | None:
        return self._extensions.get(name)

    def list_extensions(self) -> list[str]:
        return list(self._extensions.keys())

    def clear(self) -> None:
        self._extensions.clear()

    def get_sections(self) -> list[PromptSection]:
        sections = [s for ext in self._extensions.values() for s in ext.sections]
        sections.sort(key=lambda s: s.priority)
        return sections

    async def render_prompt_sections(self, context: SectionContext) -> str:
        sections = self.get_sections()
        rendered_parts: list[str] = []
        for section in sections:
            try:
                sig = inspect.signature(section.fn)
                if len(sig.parameters) == 0:
                    res = section.fn()
                else:
                    res = section.fn(context)
                if inspect.isawaitable(res):
                    res = await res
                if res and isinstance(res, str):
                    stripped = res.strip()
                    if stripped:
                        rendered_parts.append(stripped)
            except Exception as exc:
                logger.exception("Failed to render prompt section '%s': %s", section.name, exc)
        return "\n\n".join(rendered_parts)

    def resolve_tools(self, base_tools: Sequence[Any]) -> list[Any]:
        tool_map: dict[str, Any] = {}
        for tool in base_tools:
            tool_map[_get_tool_name(tool)] = tool

        for ext in self._extensions.values():
            for tool in ext.tools:
                tool_map[_get_tool_name(tool)] = tool

        for ext in self._extensions.values():
            for wrap in ext.wraps:
                if wrap.tool_name in tool_map:
                    tool_map[wrap.tool_name] = wrap.wrapper_fn(tool_map[wrap.tool_name])

        return list(tool_map.values())

    async def run_before_tool(
        self,
        call_info: ToolCallInfo,
        context: HookContext,
    ) -> HookDecision | None:
        current_decision: HookDecision | None = None
        for ext in list(self._extensions.values()):
            for hook_item in ext.hooks:
                if hook_item.before_tool is not None:
                    try:
                        res = hook_item.before_tool(call_info, context)
                        if inspect.isawaitable(res):
                            res = await res
                        if isinstance(res, HookDecision):
                            if res.block is not None:
                                return res
                            if res.require_approval:
                                return res
                            if res.modified_args is not None:
                                call_info.args = res.modified_args
                                current_decision = res
                    except Exception as exc:
                        logger.exception(
                            "Hook '%s' failed in before_tool: %s", hook_item.name, exc
                        )
        return current_decision

    def run_before_tool_sync(
        self,
        call_info: ToolCallInfo,
        context: HookContext,
    ) -> HookDecision | None:
        current_decision: HookDecision | None = None
        for ext in list(self._extensions.values()):
            for hook_item in ext.hooks:
                if hook_item.before_tool is not None:
                    try:
                        res = hook_item.before_tool(call_info, context)
                        if isinstance(res, HookDecision):
                            if res.block is not None:
                                return res
                            if res.require_approval:
                                return res
                            if res.modified_args is not None:
                                call_info.args = res.modified_args
                                current_decision = res
                    except Exception as exc:
                        logger.exception(
                            "Hook '%s' failed in before_tool_sync: %s", hook_item.name, exc
                        )
        return current_decision

    async def run_after_tool(
        self,
        call_info: ToolCallInfo,
        result: Any,
        context: HookContext,
    ) -> Any:
        current_result = result
        for ext in list(self._extensions.values()):
            for hook_item in ext.hooks:
                if hook_item.after_tool is not None:
                    try:
                        res = hook_item.after_tool(call_info, current_result, context)
                        if inspect.isawaitable(res):
                            res = await res
                        if res is not None:
                            current_result = res
                    except Exception as exc:
                        logger.exception(
                            "Hook '%s' failed in after_tool: %s", hook_item.name, exc
                        )
        return current_result

    def run_after_tool_sync(
        self,
        call_info: ToolCallInfo,
        result: Any,
        context: HookContext,
    ) -> Any:
        current_result = result
        for ext in list(self._extensions.values()):
            for hook_item in ext.hooks:
                if hook_item.after_tool is not None:
                    try:
                        res = hook_item.after_tool(call_info, current_result, context)
                        if res is not None:
                            current_result = res
                    except Exception as exc:
                        logger.exception(
                            "Hook '%s' failed in after_tool_sync: %s", hook_item.name, exc
                        )
        return current_result

    async def run_before_model(
        self,
        call_info: ModelCallInfo,
        context: HookContext,
    ) -> None:
        for ext in list(self._extensions.values()):
            for hook_item in ext.hooks:
                if hook_item.before_model is not None:
                    try:
                        res = hook_item.before_model(call_info, context)
                        if inspect.isawaitable(res):
                            await res
                    except Exception as exc:
                        logger.exception(
                            "Hook '%s' failed in before_model: %s", hook_item.name, exc
                        )

    async def run_after_model(
        self,
        call_info: ModelCallInfo,
        result: Any,
        context: HookContext,
    ) -> Any:
        current_result = result
        for ext in list(self._extensions.values()):
            for hook_item in ext.hooks:
                if hook_item.after_model is not None:
                    try:
                        res = hook_item.after_model(call_info, current_result, context)
                        if inspect.isawaitable(res):
                            res = await res
                        if res is not None:
                            current_result = res
                    except Exception as exc:
                        logger.exception(
                            "Hook '%s' failed in after_model: %s", hook_item.name, exc
                        )
        return current_result


global_extension_registry = ExtensionRegistry()
