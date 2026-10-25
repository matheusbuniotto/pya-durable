"""Pluggable AI model providers and credential authentication registry.

Provides a unified, pluggable interface for configuring model providers
(OpenRouter, OpenAI, Anthropic, Gemini, Groq, Ollama) and resolving model
specifications into PydanticAI Agents, matching Pi Durable's architecture.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers import Provider, infer_provider, infer_provider_class

from pydantic_ai_durable.models import normalize_model_name

PROVIDER_ALIASES = {
    "openrouter": "openrouter",
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "google": "google",
    "google-gla": "google",
    "google-cloud": "google_cloud",
    "groq": "groq",
    "ollama": "ollama",
    "vllm": "vllm",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "together": "together",
}


class ModelsRegistry:
    """Pluggable registry for model providers, custom credentials, and base URLs."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider[Any]] = {}
        self._default_model: str | None = None

    def _normalize_name(self, name: str) -> str:
        return PROVIDER_ALIASES.get(name.lower().strip(), name.lower().strip())

    def set_provider(
        self,
        name_or_provider: str | Provider[Any],
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> ModelsRegistry:
        """Register or configure a model provider with explicit auth credentials."""
        if isinstance(name_or_provider, Provider):
            provider_name = getattr(name_or_provider, "name", None) or type(
                name_or_provider
            ).__name__.lower().replace("provider", "")
            self._providers[self._normalize_name(provider_name)] = name_or_provider
            return self

        provider_name = self._normalize_name(name_or_provider)
        provider_cls = infer_provider_class(provider_name)
        params: dict[str, Any] = dict(kwargs)
        if api_key is not None:
            params["api_key"] = api_key
        if base_url is not None:
            params["base_url"] = base_url
        self._providers[provider_name] = provider_cls(**params)
        return self

    register = set_provider

    def get_provider(self, name: str) -> Provider[Any] | None:
        """Get registered provider by name or alias."""
        norm = self._normalize_name(name)
        return self._providers.get(norm)

    def resolve_model(
        self,
        model_spec: Model | KnownModelName | str | dict[str, Any] | None = None,
    ) -> Model:
        """Resolve a model specification, dict spec, or shortcut into an active Model instance."""
        if isinstance(model_spec, Model):
            return model_spec

        if isinstance(model_spec, dict):
            if "agent" in model_spec and isinstance(model_spec["agent"], (dict, str, Model)):
                return self.resolve_model(model_spec["agent"])

            if "model" in model_spec and isinstance(model_spec["model"], dict):
                inner = model_spec["model"]
                merged = dict(inner)
                for k in ("api_key", "apiKey", "base_url", "baseUrl", "provider"):
                    if k in model_spec and k not in merged:
                        merged[k] = model_spec[k]
                return self.resolve_model(merged)

            provider_name = model_spec.get("provider")
            raw_model = (
                model_spec.get("model")
                or model_spec.get("model_id")
                or model_spec.get("modelId")
                or model_spec.get("name")
            )
            api_key = model_spec.get("api_key") or model_spec.get("apiKey")
            base_url = model_spec.get("base_url") or model_spec.get("baseUrl")

            if provider_name and (api_key or base_url):
                self.set_provider(provider_name, api_key=api_key, base_url=base_url)

            if provider_name and raw_model and ":" not in raw_model:
                full_id = f"{provider_name}:{raw_model}"
            else:
                full_id = raw_model

            return self.resolve_model(full_id)

        if isinstance(model_spec, str):
            normalized = normalize_model_name(model_spec)
            if not normalized:
                return self._infer_default_model()
            if normalized == "test" or normalized.startswith("test:"):
                return TestModel()

            def provider_factory(name: str) -> Provider[Any]:
                norm_name = self._normalize_name(name)
                if norm_name in self._providers:
                    return self._providers[norm_name]
                return infer_provider(name)

            return infer_model(normalized, provider_factory=provider_factory)

        return self._infer_default_model()

    def _infer_default_model(self) -> Model:
        if "openrouter" in self._providers or os.environ.get("OPENROUTER_API_KEY"):
            return self.resolve_model("openrouter:openai/gpt-6-luna")
        if "openai" in self._providers or os.environ.get("OPENAI_API_KEY"):
            return self.resolve_model("openai:gpt-4o-mini")
        if "anthropic" in self._providers or os.environ.get("ANTHROPIC_API_KEY"):
            return self.resolve_model("anthropic:claude-3-5-haiku-latest")
        if "google" in self._providers or os.environ.get("GEMINI_API_KEY"):
            return self.resolve_model("google-gla:gemini-1.5-flash")
        if "groq" in self._providers or os.environ.get("GROQ_API_KEY"):
            return self.resolve_model("groq:llama-3.3-70b-versatile")
        return TestModel()

    def create_agent(
        self,
        model_spec: Any = None,
        *,
        tools: Sequence[Any] | None = None,
        system_prompt: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Agent[Any, Any]:
        """Create a configured Agent using the registry's providers."""
        model = self.resolve_model(model_spec)
        return Agent(
            model=model,
            tools=list(tools or []),
            system_prompt=system_prompt or (),
            **kwargs,
        )


global_models_registry = ModelsRegistry()


def create_models() -> ModelsRegistry:
    """Initialize a new pluggable ModelsRegistry, matching Pi Durable's createModels()."""
    return ModelsRegistry()


def openrouter_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Provider[Any]:
    """Create an OpenRouter provider with explicit authentication."""
    cls = infer_provider_class("openrouter")
    params: dict[str, Any] = dict(kwargs)
    if api_key is not None:
        params["api_key"] = api_key
    if base_url is not None:
        params["base_url"] = base_url
    return cls(**params)


def openai_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Provider[Any]:
    """Create an OpenAI provider with explicit authentication and optional base URL."""
    cls = infer_provider_class("openai")
    params: dict[str, Any] = dict(kwargs)
    if api_key is not None:
        params["api_key"] = api_key
    if base_url is not None:
        params["base_url"] = base_url
    return cls(**params)


def anthropic_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Provider[Any]:
    """Create an Anthropic provider with explicit authentication."""
    cls = infer_provider_class("anthropic")
    params: dict[str, Any] = dict(kwargs)
    if api_key is not None:
        params["api_key"] = api_key
    if base_url is not None:
        params["base_url"] = base_url
    return cls(**params)


def gemini_provider(
    api_key: str | None = None,
    **kwargs: Any,
) -> Provider[Any]:
    """Create a Google Gemini provider with explicit authentication."""
    cls = infer_provider_class("google")
    params: dict[str, Any] = dict(kwargs)
    if api_key is not None:
        params["api_key"] = api_key
    return cls(**params)


def groq_provider(
    api_key: str | None = None,
    **kwargs: Any,
) -> Provider[Any]:
    """Create a Groq provider with explicit authentication."""
    cls = infer_provider_class("groq")
    params: dict[str, Any] = dict(kwargs)
    if api_key is not None:
        params["api_key"] = api_key
    return cls(**params)


def ollama_provider(
    base_url: str = "http://localhost:11434/v1",
    **kwargs: Any,
) -> Provider[Any]:
    """Create a local Ollama / vLLM provider."""
    cls = infer_provider_class("ollama")
    return cls(base_url=base_url, **kwargs)


def resolve_agent(
    agent_or_spec: Any = None,
    *,
    models: ModelsRegistry | None = None,
    tools: Sequence[Any] | None = None,
    system_prompt: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> Agent[Any, Any]:
    """Resolve an Agent, Model, string shortcut, or dict spec into an active Agent."""
    if isinstance(agent_or_spec, Agent):
        if tools or system_prompt:
            existing_tools: list[Any] = []
            if hasattr(agent_or_spec, "_function_toolset"):
                existing_tools = list(
                    getattr(agent_or_spec._function_toolset, "tools", {}).values()
                )
            all_tools = existing_tools + list(tools or [])
            return Agent(
                model=agent_or_spec.model,
                tools=all_tools,
                system_prompt=system_prompt or getattr(agent_or_spec, "_system_prompts", ()),
                **kwargs,
            )
        return agent_or_spec

    if (
        isinstance(agent_or_spec, dict)
        and "agent" in agent_or_spec
        and isinstance(agent_or_spec["agent"], (dict, str, Agent, Model))
    ):
        inner = agent_or_spec["agent"]
        dict_tools = agent_or_spec.get("tools")
        dict_prompt = agent_or_spec.get("system_prompt") or agent_or_spec.get("prompt")
        merged_tools = list(dict_tools or []) + list(tools or [])
        return resolve_agent(
            inner,
            models=models,
            tools=merged_tools or None,
            system_prompt=system_prompt or dict_prompt,
            **kwargs,
        )

    registry = models or global_models_registry

    if isinstance(agent_or_spec, dict):
        extra_tools = agent_or_spec.get("tools")
        extra_prompt = agent_or_spec.get("system_prompt") or agent_or_spec.get("prompt")
        merged_tools = list(extra_tools or []) + list(tools or [])
        prompt = system_prompt or extra_prompt
        return registry.create_agent(
            agent_or_spec,
            tools=merged_tools or None,
            system_prompt=prompt,
            **kwargs,
        )

    return registry.create_agent(
        agent_or_spec,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )
