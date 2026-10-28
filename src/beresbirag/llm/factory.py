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


def is_reasoning_model(model_name: str) -> bool:
    """Return True if model name represents an OpenAI reasoning model requiring reasoning_effort."""
    return model_name.lower().startswith(REASONING_PREFIXES)


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
    """
    resolved_settings = settings or get_settings()
    active_provider = (provider or resolved_settings.llm_provider).lower()
    active_effort = effort or resolved_settings.llm_effort
    active_temp = (
        temperature if temperature is not None else resolved_settings.llm_temperature
    )

    if active_provider == "openai":
        active_model = model_name or (
            resolved_settings.llm_model
            if resolved_settings.llm_provider == "openai"
            else "gpt-4o-mini"
        )
        resolved_key = (
            api_key
            or (resolved_settings.openai_api_key if resolved_settings.openai_api_key else None)
            or os.getenv("OPENAI_API_KEY")
            or "dummy-openai-key"
        )

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

    if active_provider == "anthropic":
        active_model = model_name or (
            resolved_settings.llm_model
            if resolved_settings.llm_provider == "anthropic"
            else "claude-3-5-haiku-latest"
        )
        resolved_key = (
            api_key
            or (
                resolved_settings.anthropic_api_key
                if resolved_settings.anthropic_api_key
                else None
            )
            or os.getenv("ANTHROPIC_API_KEY")
            or "dummy-anthropic-key"
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

    raise ValueError(
        f"Unsupported LLM provider: {active_provider}. Expected 'openai' or 'anthropic'."
    )
