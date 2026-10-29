"""Verify or regenerate SHA-256 hashes and byte sizes for the 4 raw BSB corpus files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from beresbirag.config import Settings, get_settings, resolve_data_file

app = typer.Typer(
    name="verify-corpus",
    help="Verify or regenerate SHA-256 hashes for the 4 raw Berean Standard Bible (BSB) corpus files.",
    add_completion=False,
)

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus_manifest.json"

DEFAULT_FILES: tuple[dict[str, Any], ...] = (
    {
        "name": "bsb_usj.zip",
        "bytes": 0,
        "sha256": "",
        "source_url": "TODO: official BSB download URL",
        "notes": "Berean Standard Bible (BSB) 66-book USJ 3.1 XML archive with pericope headings, footnotes, and cross-references.",
    },
    {
        "name": "bsb_book_outlines.pdf",
        "bytes": 0,
        "sha256": "",
        "source_url": "TODO: official BSB download URL",
        "notes": "Berean Standard Bible (BSB) 66-book hierarchical book and section outlines PDF.",
    },
    {
        "name": "bsb_topical_index.xlsx",
        "bytes": 0,
        "sha256": "",
        "source_url": "TODO: official BSB download URL",
        "notes": "Berean Standard Bible (BSB) Topical Index Excel workbook (~121k rows).",
    },
    {
        "name": "bsb_concordance.xlsx",
        "bytes": 0,
        "sha256": "",
        "source_url": "TODO: official BSB download URL",
        "notes": "Berean Standard Bible (BSB) Exhaustive Concordance Excel workbook (~389k rows).",
    },
)


def compute_sha256(path: Path) -> tuple[int, str]:
    """Return ``(size_bytes, sha256_hex)`` for a file streamed in 1 MiB blocks."""
    hasher = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            total_bytes += len(chunk)
            hasher.update(chunk)
    return total_bytes, hasher.hexdigest()


def _resolve_corpus_path(
    name: str,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    if data_dir is not None:
        return data_dir / name
    cfg = settings or get_settings()
    settings_map = {
        "bsb_usj.zip": cfg.usj_zip_file,
        "bsb_book_outlines.pdf": cfg.outlines_pdf_file,
        "bsb_topical_index.xlsx": cfg.topical_xlsx_file,
        "bsb_concordance.xlsx": cfg.concordance_xlsx_file,
    }
    if name in settings_map:
        return settings_map[name]
    return resolve_data_file(name)


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load the corpus manifest JSON file."""
    path = manifest_path or DEFAULT_MANIFEST_PATH
    if not path.is_file():
        return {"files": [dict(item) for item in DEFAULT_FILES]}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hashes(manifest_path: Path | None = None) -> dict[str, str]:
    """Return ``{filename: sha256}`` from ``data/corpus_manifest.json``."""
    data = load_manifest(manifest_path)
    entries = data.get("files", []) if isinstance(data, dict) else data
    result: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") and entry.get("sha256"):
            result[str(entry["name"])] = str(entry["sha256"])
    return result


def write_manifest(
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compute file sizes and SHA-256 hashes from local files and write the manifest."""
    path = manifest_path or DEFAULT_MANIFEST_PATH
    existing = load_manifest(path)
    entries = existing.get("files", []) if isinstance(existing, dict) else existing
    if not entries:
        entries = [dict(item) for item in DEFAULT_FILES]

    updated_entries: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry["name"])
        file_path = _resolve_corpus_path(name, data_dir=data_dir, settings=settings)
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing raw corpus file for {name}: {file_path}")
        size_bytes, digest = compute_sha256(file_path)
        updated = dict(entry)
        updated["name"] = name
        updated["bytes"] = size_bytes
        updated["sha256"] = digest
        updated.setdefault("source_url", "TODO: official BSB download URL")
        updated.setdefault("notes", "")
        updated_entries.append(updated)

    payload = {"files": updated_entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_corpus(
    manifest_path: Path | None = None,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> tuple[bool, list[str], dict[str, str]]:
    """Verify local raw files against the manifest.

    Returns ``(ok, errors, hashes_by_name)``.
    """
    path = manifest_path or DEFAULT_MANIFEST_PATH
    if not path.is_file():
        return False, [f"Manifest file not found: {path}"], {}

    data = load_manifest(path)
    entries = data.get("files", []) if isinstance(data, dict) else data
    if not entries:
        return False, [f"Manifest contains no file entries: {path}"], {}

    errors: list[str] = []
    hashes: dict[str, str] = {}
    for entry in entries:
        name = str(entry.get("name", ""))
        expected_bytes = entry.get("bytes")
        expected_sha = str(entry.get("sha256", "")).lower()
        file_path = _resolve_corpus_path(name, data_dir=data_dir, settings=settings)
        if not file_path.is_file():
            errors.append(f"Missing corpus file '{name}' (expected at {file_path})")
            continue
        actual_bytes, actual_sha = compute_sha256(file_path)
        hashes[name] = actual_sha
        if (
            isinstance(expected_bytes, int)
            and expected_bytes > 0
            and actual_bytes != expected_bytes
        ):
            errors.append(
                f"Byte size mismatch for '{name}': expected {expected_bytes}, got {actual_bytes}"
            )
        if expected_sha and actual_sha.lower() != expected_sha:
            errors.append(
                f"SHA-256 mismatch for '{name}': expected {expected_sha}, got {actual_sha}"
            )

    return len(errors) == 0, errors, hashes


@app.command()
def main(
    write: Annotated[
        bool,
        typer.Option("--write", help="Regenerate bytes and SHA-256 hashes from local raw files."),
    ] = False,
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Path to corpus_manifest.json."),
    ] = DEFAULT_MANIFEST_PATH,
    data_dir: Annotated[
        Path | None,
        typer.Option("--data-dir", help="Optional directory containing raw corpus files."),
    ] = None,
) -> None:
    """Verify raw BSB corpus files against data/corpus_manifest.json, or regenerate with --write."""
    if write:
        try:
            payload = write_manifest(manifest_path=manifest, data_dir=data_dir)
            typer.echo(f"Wrote {len(payload['files'])} entries to {manifest}")
            return
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

    ok, errors, hashes = verify_corpus(manifest_path=manifest, data_dir=data_dir)
    if not ok:
        for err in errors:
            typer.echo(err, err=True)
        raise typer.Exit(code=1)

    for name, digest in hashes.items():
        typer.echo(f"OK {name} ({digest[:12]}...)")


if __name__ == "__main__":
    app()
