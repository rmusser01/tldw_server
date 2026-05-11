from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathContainmentError(ValueError):
    base: Path
    candidate: Path

    def __str__(self) -> str:
        return f"Path {self.candidate} is outside allowed base {self.base}"


def assert_path_within_base(base: Path, candidate: Path) -> Path:
    """Resolve candidate and reject paths outside the lexical base path."""
    resolved_base = base.absolute()
    resolved_candidate = candidate.resolve()
    if resolved_candidate == resolved_base or resolved_candidate.is_relative_to(resolved_base):
        return resolved_candidate
    raise PathContainmentError(base=resolved_base, candidate=resolved_candidate)
