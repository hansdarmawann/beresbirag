"""Transfer a populated remote Qdrant collection with embedding compatibility metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import httpx
import typer

from beresbirag.config import Settings, get_settings
from beresbirag.vectorstore.qdrant_manager import QdrantManager

app = typer.Typer(add_completion=False)


def embedding_metadata(settings: Settings) -> dict[str, Any]:
    provider = settings.resolved_embedding_provider
    return {
        "collection": settings.qdrant_collection_name,
        "meta_collection": f"{settings.qdrant_collection_name}__meta",
        "dense_dim": settings.dense_vector_size,
        "embedding_provider": provider,
        "dense_model": settings.fastembed_dense_model
        if provider == "fastembed"
        else settings.openai_embedding_model,
        "sparse_model": settings.fastembed_sparse_model,
    }


def validate_sidecar(sidecar: dict[str, Any], settings: Settings) -> None:
    """Reject incompatible or incomplete metadata before contacting the server."""
    expected_meta = embedding_metadata(settings)
    for field in ("collection", "dense_dim", "embedding_provider", "dense_model", "sparse_model"):
        expected = expected_meta[field]
        if sidecar.get(field) != expected:
            raise ValueError(
                f"Snapshot {field} mismatch: {sidecar.get(field)!r}; expected {expected!r}"
            )
    index_meta = sidecar.get("index_metadata")
    if isinstance(index_meta, dict):
        for field in ("dense_dim", "embedding_provider", "dense_model", "sparse_model"):
            if field in index_meta and index_meta[field] != expected_meta[field]:
                raise ValueError(
                    f"Snapshot index_metadata.{field} mismatch: {index_meta[field]!r}; "
                    f"expected {expected_meta[field]!r}"
                )


def _connection(
    settings: Settings, collection_name: str | None = None
) -> tuple[str, dict[str, str]]:
    if settings.qdrant_mode != "remote":
        raise ValueError("Snapshots require QDRANT_MODE=remote (native Qdrant server)")
    base = settings.qdrant_url.rstrip("/")
    target = collection_name or settings.qdrant_collection_name
    collection = quote(target, safe="")
    headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
    return f"{base}/collections/{collection}/snapshots", headers


def _download_snapshot(url: str, snapshot_name: str, headers: dict[str, str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream(
        "GET", f"{url}/{quote(snapshot_name, safe='')}", headers=headers, timeout=120.0
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    partial.replace(dest)


def _upload_snapshot(url: str, headers: dict[str, str], src: Path) -> None:
    with src.open("rb") as handle:
        response = httpx.post(
            f"{url}/upload",
            params={"priority": "snapshot"},
            files={"snapshot": (src.name, handle, "application/octet-stream")},
            headers=headers,
            timeout=600.0,
        )
    response.raise_for_status()


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
        _download_snapshot(url, snapshot.name, headers, out)

        raw_index_meta = manager.read_index_metadata()
        index_meta: dict[str, Any] | None = (
            raw_index_meta if isinstance(raw_index_meta, dict) else None
        )
        meta_collection = f"{settings.qdrant_collection_name}__meta"
        if index_meta is not None:
            meta_url, _ = _connection(settings, collection_name=meta_collection)
            meta_snap = manager.client.create_snapshot(meta_collection)
            if meta_snap is not None and hasattr(meta_snap, "name"):
                meta_out = Path(str(out) + ".meta.snapshot")
                _download_snapshot(meta_url, meta_snap.name, headers, meta_out)

        metadata = {
            **embedding_metadata(settings),
            "points": points,
            "index_metadata": index_meta or embedding_metadata(settings),
            "created_at": datetime.now(UTC).isoformat(),
        }
        Path(str(out) + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
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
        _upload_snapshot(url, headers, file)

        meta_file = Path(str(file) + ".meta.snapshot")
        meta_collection = f"{settings.qdrant_collection_name}__meta"
        if meta_file.is_file():
            meta_url, _ = _connection(settings, collection_name=meta_collection)
            _upload_snapshot(meta_url, headers, meta_file)

        manager = QdrantManager(settings=settings)
        if not manager.collection_exists():
            raise ValueError("Snapshot upload returned without creating the collection")
        index_meta = sidecar.get("index_metadata")
        if isinstance(index_meta, dict) and not meta_file.is_file():
            manager.write_index_metadata(index_meta)
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
