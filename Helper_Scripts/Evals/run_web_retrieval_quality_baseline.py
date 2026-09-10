#!/usr/bin/env python3
"""Run the deterministic offline web-retrieval quality baseline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tldw_Server_API.app.core.Evaluations.web_retrieval_quality import (  # noqa: E402
    evaluate_fixture_suite,
    load_fixture_suite,
    render_human_summary,
    serialize_report,
)

DEFAULT_FIXTURE = (
    REPOSITORY_ROOT
    / "tldw_Server_API/tests/Web_Scraping/fixtures/retrieval_quality/v1.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the offline baseline runner."""
    parser = argparse.ArgumentParser(
        description="Evaluate the checked offline web-retrieval quality fixture.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Path to the versioned retrieval-quality fixture.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for the deterministic JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the fixture, optionally write JSON, and print a stable summary."""
    args = parse_args(argv)
    suite = load_fixture_suite(args.fixture)
    report = evaluate_fixture_suite(suite)
    serialized = serialize_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized, encoding="utf-8")

    print(render_human_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
