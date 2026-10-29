"""LangChain chat model factory for minimal-effort OpenAI and Claude models."""

from __future__ import annotations

import os
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from beresbirag.config import Settings, get_settings

ReasoningEffort = Literal["minimal", "low", "medium", "high"]
ProviderType = Literal["openai", "anthropic"]

REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# Single source of truth for provider default models.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}

# Model-name prefixes that clearly belong to a *different* cloud provider.
FOREIGN_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("claude-",),
    "anthropic": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
}

# Provider -> (Settings attribute, environment variable) holding its API key.
API_KEY_SOURCES: dict[str, tuple[str, str]] = {
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
}


class ProviderModelMismatchError(ValueError):
    """Raised when a model name clearly belongs to a different LLM provider."""


class MissingAPIKeyError(ValueError):
    """Raised when no API key is configured for the selected LLM provider."""


def is_reasoning_model(model_name: str) -> bool:
    """Return True if model name represents an OpenAI reasoning model requiring reasoning_effort."""
    return model_name.lower().startswith(REASONING_PREFIXES)


def resolve_llm_target(
    provider: str | None, model_name: str | None, settings: Settings
) -> tuple[str, str]:
    """Return (provider, model). Explicit model wins; otherwise LLM_MODEL only when
    provider == LLM_PROVIDER; otherwise DEFAULT_MODELS[provider]."""
    active_provider = (provider or settings.llm_provider).lower()
    if active_provider not in DEFAULT_MODELS:
        expected = " or ".join(f"'{name}'" for name in DEFAULT_MODELS)
        raise ValueError(
            f"Unsupported LLM provider: {active_provider}. Expected {expected}."
        )

    settings_model = (settings.llm_model or "").strip()
    if model_name and model_name.strip():
        active_model = model_name.strip()
    elif settings_model and active_provider == settings.llm_provider:
        active_model = settings_model
    else:
        active_model = DEFAULT_MODELS[active_provider]

    if active_model.lower().startswith(FOREIGN_MODEL_PREFIXES.get(active_provider, ())):
        raise ProviderModelMismatchError(
            f"Model '{active_model}' does not belong to provider '{active_provider}'. "
            f"Choose a matching provider, or a {active_provider} model "
            f"(default: {DEFAULT_MODELS[active_provider]})."
        )
    return active_provider, active_model


def _resolve_api_key(provider: str, api_key: str | None, settings: Settings) -> str:
    """Resolve an API key from explicit arg -> settings -> environment, never inventing one."""
    settings_attr, env_var = API_KEY_SOURCES[provider]
    for candidate in (api_key, getattr(settings, settings_attr, None), os.getenv(env_var)):
        if candidate and candidate.strip():
            return candidate.strip()
    raise MissingAPIKeyError(f"{provider} API key missing: set {env_var} in .env")


def create_chat_model(
    provider: ProviderType | None = None,
    model_name: str | None = None,
    effort: ReasoningEffort | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Create a configured LangChain BaseChatModel instance optimized for latency and cost.

    Args:
        provider: 'openai' or 'anthropic'. Defaults to settings.llm_provider.
        model_name: Specific model identifier. Defaults to provider default.
        effort: Reasoning/latency effort ('minimal', 'low', 'medium', 'high').
        temperature: Sampling temperature for non-reasoning models (default 0.0).
        settings: Application Settings instance. Defaults to get_settings().
        api_key: Optional explicit API key override.

    Returns:
        BaseChatModel: Initialized ChatOpenAI or ChatAnthropic instance.

    Raises:
        ValueError: Unsupported provider.
        ProviderModelMismatchError: Model belongs to a different provider.
        MissingAPIKeyError: No API key found for the provider.
    """
    resolved_settings = settings or get_settings()
    active_provider, active_model = resolve_llm_target(
        provider, model_name, resolved_settings
    )
    active_effort = effort or resolved_settings.llm_effort
    active_temp = (
        temperature if temperature is not None else resolved_settings.llm_temperature
    )
    resolved_key = _resolve_api_key(active_provider, api_key, resolved_settings)

    if active_provider == "openai":
        if is_reasoning_model(active_model):
            return ChatOpenAI(
                model=active_model,
                reasoning_effort=active_effort,
                api_key=resolved_key,
            )

        return ChatOpenAI(
            model=active_model,
            temperature=active_temp,
            api_key=resolved_key,
        )

    if active_effort in ("minimal", "low"):
        return ChatAnthropic(
            model=active_model,
            temperature=0.0,
            max_tokens=1024,
            api_key=resolved_key,
        )

    return ChatAnthropic(
        model=active_model,
        temperature=active_temp,
        max_tokens=4096,
        api_key=resolved_key,
    )
