"""Main Streamlit application for Berean Standard Bible (BSB) RAG."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from beresbirag.config import Settings, get_settings
from beresbirag.llm.factory import resolve_llm_target
from beresbirag.models.schemas import AnswerMode
from beresbirag.ui.components import (
    render_citation_report,
    render_citations_table,
    render_error,
    render_intent_badge,
    render_retrieved_chunks,
    render_status_panel,
)
from beresbirag.ui.services import (
    append_feedback,
    fetch_installed_ollama_models,
    get_service_factory,
    model_options,
    provider_options,
)

if TYPE_CHECKING:
    from beresbirag.diagnostics import CheckResult

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", ""})

# Optional module-level override for tests (passed to append_feedback when set)
FEEDBACK_DIR: str | Path | None = None


@st.cache_data(ttl=60, show_spinner=False)
def _get_cached_diagnostics(
    qdrant_url: str,
    qdrant_mode: str,
    collection_name: str,
    provider: str,
    model: str | None,
    resource_profile: str,
    offline_mode: bool,
) -> list[CheckResult]:
    """Run lightweight diagnostics cached for 60 seconds."""
    from beresbirag.diagnostics import run_diagnostics

    settings = get_settings()
    return run_diagnostics(settings)


def _check_exposure_and_password(settings: Settings) -> bool:
    """Enforce non-loopback password requirement and UI_PASSWORD login gate."""
    raw_addr = st.get_option("server.address")
    server_addr = str(raw_addr or "127.0.0.1").strip().lower()
    configured_pw = (os.environ.get("UI_PASSWORD") or settings.ui_password or "").strip()

    if server_addr not in LOOPBACK_HOSTS and not configured_pw:
        st.error(
            f"Refusing to expose UI on non-loopback address '{server_addr}' without UI_PASSWORD set."
        )
        st.stop()
        return False

    if configured_pw:
        if st.session_state.get("ui_authenticated") is True:
            return True
        entered_pw = st.text_input("UI Password", type="password", key="ui_password_input")
        if entered_pw:
            if secrets.compare_digest(entered_pw, configured_pw):
                st.session_state["ui_authenticated"] = True
                return True
            st.error("Invalid UI password.")
            st.stop()
            return False
        st.info("Enter UI_PASSWORD to access Berean Standard Bible RAG.")
        st.stop()
        return False

    return True


def _render_feedback_controls(turn: dict[str, Any], turn_idx: int) -> None:
    """Render 👍/👎 rating buttons and optional comment box for a turn."""
    comment = st.text_input(
        "Feedback comment (optional)",
        key=f"feedback_comment_{turn_idx}",
    )
    fb_col1, fb_col2, _ = st.columns([1, 1, 6])
    if fb_col1.button("👍", key=f"feedback_up_{turn_idx}"):
        append_feedback(
            turn,
            rating="up",
            comment=comment,
            feedback_dir=FEEDBACK_DIR,
        )
        st.success("Feedback saved (👍)")
    if fb_col2.button("👎", key=f"feedback_down_{turn_idx}"):
        append_feedback(
            turn,
            rating="down",
            comment=comment,
            feedback_dir=FEEDBACK_DIR,
        )
        st.success("Feedback saved (👎)")


def _render_history_section() -> None:
    """Render previous turns collapsed with a Clear history button."""
    history: list[dict[str, Any]] = st.session_state.get("history", [])
    if not history:
        return

    st.divider()
    st.subheader(f"Session History ({len(history)})")
    st.caption("Previous turns are shown for reference and are not sent to the LLM.")

    if st.button("Clear history", key="clear_history_btn"):
        st.session_state["history"] = []
        st.session_state["last_response"] = None
        return

    for idx, turn in enumerate(history, start=1):
        q_text = turn.get("query", "")
        mode_val = turn.get("answer_mode", "")
        timings = turn.get("timings", {})
        total_ms = timings.get("total_ms", timings.get("retrieval_ms", 0.0))
        header = f"Turn #{idx}: {q_text} ({mode_val} · {total_ms:.1f} ms)"
        with st.expander(header, expanded=False):
            st.markdown(f"**Question:** {q_text}")
            st.caption(
                f"Mode: `{mode_val}` | "
                f"Retrieval: {timings.get('retrieval_ms', 0.0):.1f} ms | "
                f"Generation: {timings.get('generation_ms', 0.0):.1f} ms | "
                f"Total: {total_ms:.1f} ms"
            )
            ans = turn.get("answer", "")
            if ans:
                st.markdown(ans)


def main() -> None:
    st.set_page_config(page_title="BSB RAG", layout="wide")

    settings = get_settings()
    if not _check_exposure_and_password(settings):
        return

    factory = get_service_factory()

    if "history" not in st.session_state:
        st.session_state["history"] = []

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
            fetch_installed_ollama_models(settings) if selected_provider == "ollama" else []
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
    st.caption(
        "Grounded Scripture Search & QA over the complete BSB ecosystem — "
        "each turn runs independent retrieval; previous turns are not sent to the LLM."
    )

    # Example query buttons (buttons 0..3)
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
        key="query_text",
        height=90,
        placeholder="e.g., What does Genesis 1:1-5 say about creation?",
    )

    # Button 4 is always "Ask"
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
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

                    timings = {
                        "retrieval_ms": round(retrieval_ms, 2),
                        "generation_ms": 0.0,
                        "total_ms": round(retrieval_ms, 2),
                    }
                    turn_entry: dict[str, Any] = {
                        "mode": "retrieve",
                        "query": user_query,
                        "answer": "",
                        "answer_mode": "retrieve_only",
                        "provider": "",
                        "model": "",
                        "citation_report": None,
                        "timings": timings,
                        "plan": plan,
                        "results": results,
                        "retrieval_ms": retrieval_ms,
                    }
                    st.session_state["last_response"] = turn_entry
                    st.session_state["history"].append(turn_entry)
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
                        answer_text = str(st.write_stream(stream_gen) or "")
                        gen_ms = (time.perf_counter() - gen_start) * 1000

                    response = pipeline.finalize(prepared, answer_text)
                    timings = response.timings or {
                        "retrieval_ms": round(retrieval_ms, 2),
                        "generation_ms": round(gen_ms, 2),
                        "total_ms": round(retrieval_ms + gen_ms, 2),
                    }
                    turn_entry = {
                        "mode": "answer",
                        "query": user_query,
                        "answer": response.answer,
                        "answer_mode": (
                            response.answer_mode.value
                            if hasattr(response.answer_mode, "value")
                            else str(response.answer_mode)
                        ),
                        "provider": response.provider,
                        "model": response.model,
                        "citation_report": (
                            response.citation_report.model_dump(mode="json")
                            if response.citation_report is not None
                            else None
                        ),
                        "timings": timings,
                        "response": response,
                        "retrieval_ms": retrieval_ms,
                        "gen_ms": gen_ms,
                    }
                    st.session_state["last_response"] = turn_entry
                    st.session_state["history"].append(turn_entry)
                except Exception as exc:
                    render_error(exc)
                    return

    # Render persisted last response if available
    last_resp = st.session_state.get("last_response")
    turn_count = len(st.session_state.get("history", []))
    if last_resp is not None and not ask_clicked:
        if last_resp["mode"] == "retrieve":
            render_intent_badge(last_resp["plan"])
            st.caption(f"Retrieved in {last_resp['retrieval_ms']:.1f} ms")
            render_retrieved_chunks(last_resp["results"])
            _render_feedback_controls(last_resp, turn_count)
        elif last_resp["mode"] == "answer":
            resp = last_resp["response"]
            render_intent_badge(resp.intent if hasattr(resp, "intent") else None)
            st.caption(
                f"Retrieved in {last_resp['retrieval_ms']:.1f} ms | "
                f"Generated in {last_resp['gen_ms']:.1f} ms"
            )
            st.markdown(resp.answer)
            if resp.answer_mode == AnswerMode.FAST_PATH:
                st.caption("Answered deterministically without an LLM")
            elif resp.answer_mode == AnswerMode.INSUFFICIENT_CONTEXT:
                st.caption("Not enough context in the BSB sources")

            render_citation_report(
                resp.citation_report,
                citation_policy=settings.citation_policy,
                answer_mode=resp.answer_mode,
            )
            render_citations_table(resp.citations)
            render_retrieved_chunks(resp.retrieved_chunks)
            _render_feedback_controls(last_resp, turn_count)
    elif ask_clicked and last_resp is not None:
        if last_resp["mode"] == "retrieve":
            render_intent_badge(last_resp["plan"])
            st.caption(f"Retrieved in {last_resp['retrieval_ms']:.1f} ms")
            render_retrieved_chunks(last_resp["results"])
            _render_feedback_controls(last_resp, turn_count)
        elif last_resp["mode"] == "answer":
            resp = last_resp["response"]
            render_citation_report(
                resp.citation_report,
                citation_policy=settings.citation_policy,
                answer_mode=resp.answer_mode,
            )
            render_citations_table(resp.citations)
            render_retrieved_chunks(resp.retrieved_chunks)
            _render_feedback_controls(last_resp, turn_count)

    _render_history_section()


if __name__ == "__main__":
    main()
