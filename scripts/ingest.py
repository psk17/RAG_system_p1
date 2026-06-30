#!/usr/bin/env python3
"""
scripts/ingest.py
──────────────────
CLI script for populating the vector store from local PDF / Markdown directories.

Usage
─────
    # Ingest a single file
    python scripts/ingest.py --path ./docs/handbook.pdf --collection hr_2026

    # Ingest a whole directory (recursive)
    python scripts/ingest.py --path ./documents --collection company_kb

    # Ingest with custom chunk settings
    python scripts/ingest.py --path ./docs --collection kb \\
        --chunk-size 800 --chunk-overlap 150

    # Use HuggingFace embeddings (no OpenAI key needed)
    EMBEDDING_PROVIDER=huggingface python scripts/ingest.py --path ./docs

    # Dry-run: show what would be ingested without writing to the DB
    python scripts/ingest.py --path ./docs --dry-run

Options
───────
    --path PATH             File or directory to ingest (required)
    --collection NAME       Vector DB collection name [default: default]
    --chunk-size INT        Characters per chunk [default: from settings]
    --chunk-overlap INT     Overlap between chunks [default: from settings]
    --recursive / --no-recursive  Recurse into subdirectories [default: recursive]
    --dry-run               Parse and chunk only; do not write to the DB
    --verbose               Show per-chunk details
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup (before any app imports so settings log messages appear)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ──────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Ingest documents into the RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        type=Path,
        help="File or directory to ingest.",
    )
    parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Vector DB collection name (default: 'default').",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Max characters per chunk (overrides settings).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between consecutive chunks (overrides settings).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into subdirectories when --path is a directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only — skip writing to the vector store.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-chunk details.",
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> int:
    """
    Core async logic.  Returns exit code (0 = success, 1 = errors).
    """
<<<<<<< HEAD
    from app.core.config.settings import get_settings
    try:
        from app.ingestion.chunking_service import ChunkingService
=======
    from rag_system.core.config.settings import get_settings
    try:
        from rag_system.ingestion.chunking_service import ChunkingService
>>>>>>> aa82418 (chore: add GitHub readiness docs, CI workflow, and community guides)
    except ImportError:  # pragma: no cover
        class ChunkingService:  # type: ignore[misc]
            def __init__(self, *_, **__):
                pass
            def process_file(self, *_, **__) -> list[Any]:
                return []
            def process_directory(self, *_, **__) -> list[Any]:
                return []
<<<<<<< HEAD
    from app.ingestion.ingestion_service import IngestionService
    from app.ingestion.vector_store_chroma import ChromaAdapter
=======
    from rag_system.ingestion.ingestion_service import IngestionService
    from rag_system.ingestion.vector_store_chroma import ChromaAdapter
>>>>>>> aa82418 (chore: add GitHub readiness docs, CI workflow, and community guides)

    settings = get_settings()
    target = args.path.resolve()

    if not target.exists():
        logger.error("Path does not exist: %s", target)
        return 1

    # ── Build components ──────────────────────────────────────────────────────
    chunker = ChunkingService(
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
    )

    if args.dry_run:
        logger.info("DRY RUN — no data will be written to the vector store.")
        _dry_run(target, chunker, args)
        return 0

    store = ChromaAdapter.from_settings()
    service = IngestionService(vector_store=store, chunking_service=chunker)

    # ── Ingest ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if target.is_file():
        result = await service.ingest_file(target, collection_id=args.collection)
        results = [result]
    else:
        results = await service.ingest_directory(
            target,
            collection_id=args.collection,
            recursive=args.recursive,
        )

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    total_chunks = sum(r.chunks_upserted for r in results)
    total_errors = sum(len(r.errors) for r in results)

    print("\n" + "─" * 60)
    print(f"  Collection : {args.collection}")
    print(f"  Files      : {len(results)}")
    print(f"  Chunks     : {total_chunks} upserted")
    print(f"  Errors     : {total_errors}")
    print(f"  Time       : {elapsed:.2f}s")
    print("─" * 60)

    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status}  {r.source}  ({r.chunks_upserted} chunks)")
        for err in r.errors:
            print(f"       ERROR: {err}")

    print()
    return 0 if total_errors == 0 else 1


def _dry_run(
    target: Path,
<<<<<<< HEAD
    chunker: ChunkingService,
=======
    chunker: Any,
>>>>>>> aa82418 (chore: add GitHub readiness docs, CI workflow, and community guides)
    args: argparse.Namespace,
) -> None:
    """Parse and chunk without touching the vector store."""
    if target.is_file():
        files = [target]
    else:
        supported = {".pdf", ".md", ".markdown", ".txt", ".text"}
        glob = "**/*" if args.recursive else "*"
        files = sorted(f for f in target.glob(glob) if f.suffix.lower() in supported)

    total = 0
    for f in files:
        try:
            chunks = chunker.process_file(f, collection_id=args.collection)
            total += len(chunks)
            print(f"  {f.name}: {len(chunks)} chunks")
            if args.verbose:
                for c in chunks:
                    print(
                        f"    [{c.chunk_index}] page={c.page_number} "
                        f"len={len(c.text)}  id={c.chunk_id[:8]}…"
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {f.name}: {exc}")

    print(f"\n  Total: {total} chunks across {len(files)} files (dry run, nothing written).")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
