"""LLM factory, grounded prompt templates, and LCEL RAG pipeline for beresbirag."""

from beresbirag.llm.factory import (
    DEFAULT_MODELS,
    MissingAPIKeyError,
    ProviderModelMismatchError,
    create_chat_model,
    is_reasoning_model,
    resolve_llm_target,
)
from beresbirag.llm.prompts import (
    BSB_RAG_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from beresbirag.llm.rag_chain import (
    INSUFFICIENT_CONTEXT_ANSWER,
    BSBRAGPipeline,
    build_citations,
    format_context_blocks,
)

__all__ = [
    "BSBRAGPipeline",
    "BSB_RAG_PROMPT",
    "DEFAULT_MODELS",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "MissingAPIKeyError",
    "ProviderModelMismatchError",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "build_citations",
    "create_chat_model",
    "format_context_blocks",
    "is_reasoning_model",
    "resolve_llm_target",
]
