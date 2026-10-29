"""Transfer a populated remote Qdrant collection with embedding compatibility metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import httpx
import typer

from beresbirag.config import Settings, get_settings
from beresbirag.vectorstore.qdrant_manager import QdrantManager

app = typer.Typer(add_completion=False)


def embedding_metadata(settings: Settings) -> dict:
    provider = settings.resolved_embedding_provider
    return {
        "collection": settings.qdrant_collection_name,
        "dense_dim": settings.dense_vector_size,
        "embedding_provider": provider,
        "dense_model": settings.fastembed_dense_model
        if provider == "fastembed"
        else settings.openai_embedding_model,
        "sparse_model": settings.fastembed_sparse_model,
    }


def validate_sidecar(sidecar: dict, settings: Settings) -> None:
    """Reject incompatible or incomplete metadata before contacting the server."""
    for field, expected in embedding_metadata(settings).items():
        if sidecar.get(field) != expected:
            raise ValueError(
                f"Snapshot {field} mismatch: {sidecar.get(field)!r}; expected {expected!r}"
            )


def _connection(settings: Settings) -> tuple[str, dict[str, str]]:
    if settings.qdrant_mode != "remote":
        raise ValueError("Snapshots require QDRANT_MODE=remote (native Qdrant server)")
    base = settings.qdrant_url.rstrip("/")
    collection = quote(settings.qdrant_collection_name, safe="")
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    return f"{base}/collections/{collection}/snapshots", headers


@app.command("export")
def export_snapshot(out: Annotated[Path, typer.Option("--out")]) -> None:
    settings = get_settings()
    manager = None
    try:
        url, headers = _connection(settings)
        manager = QdrantManager(settings=settings)
        if not manager.collection_exists():
            raise ValueError("Collection does not exist; ingest before exporting")
        manager.validate_collection()
        points = manager.count_points()
        snapshot = manager.client.create_snapshot(settings.qdrant_collection_name)
        if snapshot is None:
            raise ValueError("Qdrant did not return a snapshot")
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_suffix(out.suffix + ".part")
        with httpx.stream(
            "GET", f"{url}/{quote(snapshot.name, safe='')}", headers=headers, timeout=120.0
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        partial.replace(out)
        metadata = {
            **embedding_metadata(settings),
            "points": points,
            "created_at": datetime.now(UTC).isoformat(),
        }
        Path(str(out) + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Exported {points} points to {out}")
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    finally:
        if manager is not None:
            manager.client.close()


@app.command("import")
def import_snapshot(
    file: Annotated[Path, typer.Option("--file", exists=True, dir_okay=False)],
) -> None:
    settings = get_settings()
    manager = None
    try:
        sidecar = json.loads(Path(str(file) + ".json").read_text(encoding="utf-8"))
        validate_sidecar(sidecar, settings)
        url, headers = _connection(settings)
        with file.open("rb") as handle:
            response = httpx.post(
                f"{url}/upload",
                params={"priority": "snapshot"},
                files={"snapshot": (file.name, handle, "application/octet-stream")},
                headers=headers,
                timeout=600.0,
            )
        response.raise_for_status()
        manager = QdrantManager(settings=settings)
        if not manager.collection_exists():
            raise ValueError("Snapshot upload returned without creating the collection")
        manager.validate_collection()
        typer.echo(f"Imported and validated {manager.count_points()} points")
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    finally:
        if manager is not None:
            manager.client.close()


if __name__ == "__main__":
    app()
