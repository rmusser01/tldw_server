from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatsRunSummary:
    block: str
    cats_version: str
    openapi_sha256: str
    command: list[str]
    masked_command: list[str]
    exit_code: int
    failure_class: str
    stdout_path: str
    stderr_path: str
    report_dir: str
    extra: dict[str, Any] = field(default_factory=dict)


def _mask_header_arg(value: str) -> str:
    for header_name in ("X-API-KEY", "Authorization"):
        prefix = f"{header_name}="
        if value.startswith(prefix):
            return f"{prefix}${header_name}"
    return value


def mask_command(command: list[str]) -> list[str]:
    return [_mask_header_arg(value) for value in command]


def write_summary(summary: CatsRunSummary, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(summary)
    data["command"] = mask_command(summary.command)
    data["masked_command"] = mask_command(summary.masked_command)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = ["CatsRunSummary", "mask_command", "write_summary"]
