"""Module entrypoint for running the CATS fuzzing harness with python -m."""

from __future__ import annotations

from Helper_Scripts.cats_fuzz.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
