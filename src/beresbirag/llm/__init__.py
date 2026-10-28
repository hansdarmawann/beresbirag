"""LLM factory, grounded prompt templates, and LCEL RAG pipeline for beresbirag."""

from beresbirag.llm.factory import create_chat_model, is_reasoning_model
from beresbirag.llm.prompts import (
    BSB_RAG_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from beresbirag.llm.rag_chain import (
    BSBRAGPipeline,
    build_citations,
    format_context_blocks,
)

__all__ = [
    "BSBRAGPipeline",
    "BSB_RAG_PROMPT",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "build_citations",
    "create_chat_model",
    "format_context_blocks",
    "is_reasoning_model",
]
