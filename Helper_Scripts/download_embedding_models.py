#!/usr/bin/env python3
"""
Pre-download embedding model weights so the runtime can use "real" embeddings
without pausing to fetch assets on first use.

This script relies on the `huggingface_hub` package. Install it first:

    pip install huggingface_hub

Usage examples:
    # Download the default set of models into ./models/embeddings
    python Helper_Scripts/download_embedding_models.py

    # Pick a custom target directory and include an extra model
    python Helper_Scripts/download_embedding_models.py \
        --target ./cache/embeddings \
        --model sentence-transformers/all-distilroberta-v1

If you need to authenticate with Hugging Face (e.g., for gated models), set
the `HUGGINGFACE_HUB_TOKEN` environment variable before running the script.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover - handled by runtime message
    print(
        "ERROR: huggingface_hub is required. Install it with 'pip install huggingface_hub'.",
        file=sys.stderr,
    )
    sys.exit(1)

# Core models the application uses out of the box.
DEFAULT_MODEL_IDS: list[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
]


def _normalise_subdir(model_id: str) -> str:
    """Convert model IDs like 'org/name' into filesystem-friendly paths."""
    return model_id.replace("/", "__")


def download_models(
    model_ids: Iterable[str],
    target_dir: Path,
    revision: str | None = None,
    allow: list[str] | None = None,
) -> None:
    """Download the specified Hugging Face repositories into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for model_id in model_ids:
        local_dir = target_dir / _normalise_subdir(model_id)
        print(f"Downloading {model_id!r} into {local_dir}")
        try:
            kwargs = {
                "repo_id": model_id,
                "revision": revision,
                "local_dir": local_dir,
            }
            if allow:
                kwargs["allow_patterns"] = allow
            snapshot_download(**kwargs)  # nosec B615
        except HfHubHTTPError as err:
            print(f"  Failed to download {model_id}: {err}", file=sys.stderr)
        else:
            print(f"  Ready: {local_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-fetch embedding models so runtime inference does not block on downloads.",
    )
    parser.add_argument(
        "--target",
        default="models/embeddings",
        type=Path,
        help="Directory to store downloaded models (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Additional Hugging Face model ID to download. Can be specified multiple times.",
    )
    parser.add_argument(
        "--revision",
        help="Optional git revision / tag / commit hash to pin for all downloads.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Optional file glob (relative to the repo) to limit what gets downloaded. "
            "Example: --pattern '*.bin'. Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--skip-defaults",
        action="store_true",
        help="Do not include the built-in model list; only download models provided via --model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_ids: list[str] = []
    if not args.skip_defaults:
        model_ids.extend(DEFAULT_MODEL_IDS)
    if args.models:
        model_ids.extend(args.models)

    if not model_ids:
        print("No models specified; nothing to do.")
        return 0

    download_models(model_ids, args.target, revision=args.revision, allow=args.patterns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
