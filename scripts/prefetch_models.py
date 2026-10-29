"""Prepare portable model caches on a connected build machine."""

from __future__ import annotations

import shutil
import subprocess

import typer

from beresbirag.config import get_settings

app = typer.Typer(add_completion=False)


@app.command()
def prefetch(ollama_model: str | None = typer.Option(None, "--ollama-model")) -> None:
    settings = get_settings()
    if settings.offline_mode:
        raise typer.BadParameter(
            "Set OFFLINE_MODE=false and unset HF_HUB_OFFLINE on the build machine"
        )
    from fastembed import SparseTextEmbedding, TextEmbedding
    from flashrank import Ranker

    # Release each instance before loading the next model.
    TextEmbedding(
        model_name=settings.fastembed_dense_model,
        cache_dir=settings.model_cache_dir,
        threads=settings.llm_num_thread,
    )
    SparseTextEmbedding(
        model_name=settings.fastembed_sparse_model,
        cache_dir=settings.model_cache_dir,
        threads=settings.llm_num_thread,
    )
    for model in ("ms-marco-MiniLM-L-12-v2", "ms-marco-TinyBERT-L-2-v2"):
        Ranker(model_name=model, cache_dir=settings.model_cache_dir)
    typer.echo(f"Models cached in {settings.model_cache_dir}")
    if ollama_model:
        executable = shutil.which("ollama")
        if executable:
            subprocess.run([executable, "pull", ollama_model], check=True)
        else:
            typer.echo(f"Install Ollama, then run: ollama pull {ollama_model}")


if __name__ == "__main__":
    app()
