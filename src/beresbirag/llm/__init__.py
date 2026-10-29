"""LLM factory, grounded prompt templates, and LCEL RAG pipeline for beresbirag."""

from beresbirag.llm.citations import (
    extract_citations,
    parse_label,
    strip_unsupported_citations,
    validate_citations,
)
from beresbirag.llm.factory import (
    DEFAULT_MODELS,
    MissingAPIKeyError,
    ProviderModelMismatchError,
    ProviderUnavailableError,
    create_chat_model,
    is_reasoning_model,
    resolve_llm_target,
)
from beresbirag.llm.prompts import (
    BSB_RAG_PROMPT,
    BSB_RAG_PROMPT_COMPACT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from beresbirag.llm.rag_chain import (
    INSUFFICIENT_CONTEXT_ANSWER,
    BSBRAGPipeline,
    PreparedQuery,
    build_citations,
    format_context_blocks,
)

__all__ = [
    "BSBRAGPipeline",
    "BSB_RAG_PROMPT",
    "BSB_RAG_PROMPT_COMPACT",
    "DEFAULT_MODELS",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "MissingAPIKeyError",
    "ProviderModelMismatchError",
    "ProviderUnavailableError",
    "PreparedQuery",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "build_citations",
    "create_chat_model",
    "extract_citations",
    "format_context_blocks",
    "is_reasoning_model",
    "parse_label",
    "resolve_llm_target",
    "strip_unsupported_citations",
    "validate_citations",
]
