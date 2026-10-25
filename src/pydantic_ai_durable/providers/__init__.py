"""Pluggable AI model providers and credential authentication registry."""

from pydantic_ai_durable.providers.registry import (
    PROVIDER_ALIASES,
    ModelsRegistry,
    anthropic_provider,
    create_models,
    gemini_provider,
    global_models_registry,
    groq_provider,
    ollama_provider,
    openai_provider,
    openrouter_provider,
    resolve_agent,
)

__all__ = [
    "PROVIDER_ALIASES",
    "ModelsRegistry",
    "anthropic_provider",
    "create_models",
    "gemini_provider",
    "global_models_registry",
    "groq_provider",
    "ollama_provider",
    "openai_provider",
    "openrouter_provider",
    "resolve_agent",
]
