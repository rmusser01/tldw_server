"""Compatibility runner for wizard CLI tests across supported Typer releases."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike

from click.testing import CliRunner as ClickCliRunner
from typer.testing import CliRunner as TyperCliRunner


class CliRunner(TyperCliRunner):
    """Retain Click's filesystem isolation after Typer 0.26 stopped inheriting it."""

    @contextmanager
    def isolated_filesystem(
        self,
        temp_dir: str | PathLike[str] | None = None,
    ) -> Iterator[str]:
        """Run a test in an isolated working directory."""

        with ClickCliRunner().isolated_filesystem(temp_dir=temp_dir) as directory:
            yield directory
