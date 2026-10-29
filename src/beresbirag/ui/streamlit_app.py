"""Main Streamlit application for Berean Standard Bible (BSB) RAG."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import streamlit as st

from beresbirag.config import get_settings
from beresbirag.llm.factory import resolve_llm_target
from beresbirag.models.schemas import AnswerMode
from beresbirag.ui.components import (
    render_citations_table,
    render_error,
    render_intent_badge,
    render_retrieved_chunks,
    render_status_panel,
)
from beresbirag.ui.services import (
    fetch_installed_ollama_models,
    get_service_factory,
    model_options,
    provider_options,
)

if TYPE_CHECKING:
    from beresbirag.diagnostics import CheckResult


@st.cache_data(ttl=60, show_spinner=False)
def _get_cached_diagnostics(
    qdrant_url: str,
    qdrant_mode: str,
    collection_name: str,
    provider: str,
    model: str,
    resource_profile: str,
    offline_mode: bool,
) -> list[CheckResult]:
    """Run lightweight diagnostics cached for 60 seconds."""
    from beresbirag.diagnostics import run_diagnostics

    settings = get_settings()
    return run_diagnostics(settings)


def main() -> None:
    st.set_page_config(page_title="BSB RAG", layout="wide")

    settings = get_settings()
    factory = get_service_factory()

    # Determine default mode based on resource profile
    default_mode_index = 0 if settings.resource_profile == "low" else 1

    # ========================== Sidebar ==========================
    with st.sidebar:
        st.title("Settings")

        mode = st.radio(
            "Mode",
            options=["Retrieve only (no LLM)", "Answer with LLM"],
            index=default_mode_index,
        )
        is_retrieve_only = mode == "Retrieve only (no LLM)"

        # Provider selection
        avail_providers = provider_options(settings)
        selected_provider = st.selectbox(
            "Provider",
            options=avail_providers,
            index=0,
            disabled=is_retrieve_only,
        )

        # Model selection
        installed_ollama = (
            fetch_installed_ollama_models(settings)
            if selected_provider == "ollama"
            else []
        )
        avail_models = model_options(selected_provider, settings, installed_ollama)

        # Allow Custom model option
        model_choices = avail_models + ["Custom model…"]
        selected_model_choice = st.selectbox(
            "Model",
            options=model_choices,
            index=0,
            disabled=is_retrieve_only,
        )

        custom_model_error: Exception | None = None
        if selected_model_choice == "Custom model…":
            custom_model_val = st.text_input(
                "Enter model name",
                disabled=is_retrieve_only,
            )
            final_model = custom_model_val.strip()
            if final_model:
                try:
                    resolve_llm_target(selected_provider, final_model, settings)
                except Exception as exc:
                    custom_model_error = exc
                    render_error(exc)
        else:
            final_model = selected_model_choice

        # Effort select (hidden for ollama)
        effort = "minimal"
        if selected_provider != "ollama":
            effort = st.selectbox(
                "Reasoning Effort",
                options=["minimal", "low", "medium", "high"],
                index=["minimal", "low", "medium", "high"].index(
                    settings.llm_effort
                    if settings.llm_effort in ["minimal", "low", "medium", "high"]
                    else "minimal"
                ),
                disabled=is_retrieve_only,
            )

        # Top K slider
        top_k = st.slider(
            "Top K Chunks",
            min_value=1,
            max_value=10,
            value=settings.retrieval_top_k,
        )

        # System Status Expander
        with st.expander("System status"):
            st.caption(
                f"Resource Profile: **{settings.resource_profile}** | "
                f"Offline Mode: **{settings.offline_mode}**"
            )
            try:
                checks = _get_cached_diagnostics(
                    settings.qdrant_url,
                    settings.qdrant_mode,
                    settings.qdrant_collection_name,
                    settings.llm_provider,
                    settings.llm_model,
                    settings.resource_profile,
                    settings.offline_mode,
                )
                render_status_panel(checks)
            except Exception as e:
                st.warning(f"Unable to load diagnostics: {e}")

    # ========================== Main Area ==========================
    st.title("Berean Standard Bible (BSB) RAG")
    st.caption("Grounded Scripture Search & QA over the complete BSB ecosystem")

    # Example query buttons
    example_col1, example_col2, example_col3, example_col4 = st.columns(4)
    example_queries = [
        "John 3:16",
        "Genesis outline",
        'How many occurrences of the word "grace"?',
        "What does the Bible say about forgiveness?",
    ]

    if "query_text" not in st.session_state:
        st.session_state["query_text"] = ""

    if example_col1.button(example_queries[0]):
        st.session_state["query_text"] = example_queries[0]
    if example_col2.button(example_queries[1]):
        st.session_state["query_text"] = example_queries[1]
    if example_col3.button(example_queries[2]):
        st.session_state["query_text"] = example_queries[2]
    if example_col4.button(example_queries[3]):
        st.session_state["query_text"] = example_queries[3]

    user_query = st.text_area(
        "Bible Question or Search Query",
        value=st.session_state.get("query_text", ""),
        height=90,
        placeholder="e.g., What does Genesis 1:1-5 say about creation?",
    )

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        st.session_state["query_text"] = user_query

        if not user_query.strip():
            st.warning("Please enter a question or passage reference.")
            return

        if custom_model_error is not None:
            render_error(custom_model_error)
            return

        with st.spinner("Processing..."):
            start_time = time.perf_counter()

            if is_retrieve_only:
                try:
                    retriever = factory.retriever()
                    plan, results = retriever.retrieve(user_query, top_k=top_k)
                    retrieval_ms = (time.perf_counter() - start_time) * 1000

                    st.session_state["last_response"] = {
                        "mode": "retrieve",
                        "plan": plan,
                        "results": results,
                        "retrieval_ms": retrieval_ms,
                    }
                except Exception as exc:
                    render_error(exc)
                    return
            else:
                try:
                    pipeline = factory.pipeline(
                        provider=selected_provider,
                        model=final_model,
                        effort=effort,
                    )
                    prep_start = time.perf_counter()
                    prepared = pipeline.prepare(user_query, top_k=top_k)
                    retrieval_ms = (time.perf_counter() - prep_start) * 1000

                    # Render intent badge and retrieval time right away
                    render_intent_badge(prepared.plan)
                    st.caption(f"Retrieved in {retrieval_ms:.1f} ms")

                    # Handle deterministic fast path or insufficient context
                    if prepared.answer_mode != AnswerMode.LLM:
                        answer_text = prepared.fast_answer or "Not enough context found."
                        st.markdown(answer_text)
                        if prepared.answer_mode == AnswerMode.FAST_PATH:
                            st.caption("Answered deterministically without an LLM")
                        elif prepared.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT:
                            st.caption("Not enough context in the BSB sources")
                        gen_ms = 0.0
                    else:
                        gen_start = time.perf_counter()
                        stream_gen = pipeline.stream(prepared)
                        answer_text = st.write_stream(stream_gen)
                        gen_ms = (time.perf_counter() - gen_start) * 1000

                    response = pipeline.finalize(prepared, answer_text)

                    st.session_state["last_response"] = {
                        "mode": "answer",
                        "response": response,
                        "retrieval_ms": retrieval_ms,
                        "gen_ms": gen_ms,
                    }
                except Exception as exc:
                    render_error(exc)
                    return

    # Render persisted last response if available
    last_resp = st.session_state.get("last_response")
    if last_resp is not None and not ask_clicked:
        if last_resp["mode"] == "retrieve":
            render_intent_badge(last_resp["plan"])
            st.caption(f"Retrieved in {last_resp['retrieval_ms']:.1f} ms")
            render_retrieved_chunks(last_resp["results"])
        elif last_resp["mode"] == "answer":
            resp = last_resp["response"]
            render_intent_badge(resp.plan if hasattr(resp, "plan") else None)
            st.caption(
                f"Retrieved in {last_resp['retrieval_ms']:.1f} ms | "
                f"Generated in {last_resp['gen_ms']:.1f} ms"
            )
            st.markdown(resp.answer)
            if resp.answer_mode == AnswerMode.FAST_PATH:
                st.caption("Answered deterministically without an LLM")
            elif resp.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT:
                st.caption("Not enough context in the BSB sources")

            render_citations_table(resp.citations)
            render_retrieved_chunks(resp.retrieved_chunks)
    elif ask_clicked and last_resp is not None:
        if last_resp["mode"] == "retrieve":
            render_intent_badge(last_resp["plan"])
            st.caption(f"Retrieved in {last_resp['retrieval_ms']:.1f} ms")
            render_retrieved_chunks(last_resp["results"])
        elif last_resp["mode"] == "answer":
            resp = last_resp["response"]
            render_citations_table(resp.citations)
            render_retrieved_chunks(resp.retrieved_chunks)


if __name__ == "__main__":
    main()

