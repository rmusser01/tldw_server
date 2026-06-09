from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess  # nosec B404
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_MINERU_TIMEOUT_SEC = 120
_DEFAULT_MINERU_MAX_CONCURRENCY = 1
_MINERU_SEMAPHORE_LOCK = threading.Lock()
_MINERU_SEMAPHORE_LIMIT = _DEFAULT_MINERU_MAX_CONCURRENCY
_MINERU_SEMAPHORE = threading.BoundedSemaphore(_DEFAULT_MINERU_MAX_CONCURRENCY)
_MINERU_TEXT_MARKDOWN_FILES = ("document.md", "output.md", "result.md")


@dataclass(frozen=True)
class MinerUConfig:
    """Resolved MinerU CLI settings derived from environment variables."""
    command: list[str]
    timeout_sec: int
    max_concurrency: int
    tmp_root: Path | None
    debug_save_raw: bool


def _coerce_positive_int(raw_value: str | None, default: int) -> int:
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError, AttributeError):
        return default
    return value if value > 0 else default


def _coerce_int(raw_value: Any, default: int) -> int:
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _coerce_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _mineru_command_tokens() -> list[str]:
    raw = os.getenv("MINERU_CMD", "mineru").strip() or "mineru"
    tokens = shlex.split(raw, posix=os.name != "nt")
    if os.name == "nt":
        tokens = [
            token[1:-1]
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}
            else token
            for token in tokens
        ]
    return tokens or ["mineru"]


def load_mineru_config() -> MinerUConfig:
    """Load MinerU CLI configuration from environment variables."""
    raw_tmp_root = (os.getenv("MINERU_TMP_ROOT") or "").strip()
    tmp_root = Path(raw_tmp_root).expanduser() if raw_tmp_root else None
    return MinerUConfig(
        command=_mineru_command_tokens(),
        timeout_sec=_coerce_positive_int(os.getenv("MINERU_TIMEOUT_SEC"), _DEFAULT_MINERU_TIMEOUT_SEC),
        max_concurrency=_coerce_positive_int(
            os.getenv("MINERU_MAX_CONCURRENCY"),
            _DEFAULT_MINERU_MAX_CONCURRENCY,
        ),
        tmp_root=tmp_root,
        debug_save_raw=_coerce_bool(os.getenv("MINERU_DEBUG_SAVE_RAW"), default=False),
    )


def _build_mineru_command(
    *,
    pdf_path: Path,
    output_dir: Path,
    config: MinerUConfig | None = None,
) -> list[str]:
    """Build the argv-safe MinerU command for a single PDF run."""
    resolved_config = config or load_mineru_config()
    return [
        *resolved_config.command,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
    ]


def _mineru_available(config: MinerUConfig | None = None) -> bool:
    """Check whether the configured MinerU executable is present on PATH."""
    resolved_config = config or load_mineru_config()
    executable = resolved_config.command[0] if resolved_config.command else "mineru"
    return shutil.which(executable) is not None


def _get_mineru_semaphore(limit: int) -> threading.BoundedSemaphore:
    global _MINERU_SEMAPHORE, _MINERU_SEMAPHORE_LIMIT

    with _MINERU_SEMAPHORE_LOCK:
        if limit != _MINERU_SEMAPHORE_LIMIT:
            _MINERU_SEMAPHORE = threading.BoundedSemaphore(limit)
            _MINERU_SEMAPHORE_LIMIT = limit
        return _MINERU_SEMAPHORE


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_requested_format(output_format: str | None) -> str:
    fmt = (output_format or "").strip().lower()
    if fmt in {"text", "markdown", "json"}:
        return fmt
    return "markdown"


def _blank_page_entry(page_number: int) -> dict[str, Any]:
    return {
        "page": page_number,
        "text": "",
        "tables": [],
        "blocks": [],
        "meta": {},
    }


def _markdown_to_text(markdown_text: str) -> str:
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if set(stripped) <= {"|", "-", ":", " "}:
            continue
        stripped = stripped.lstrip("#").strip()
        if stripped.startswith(("- ", "* ", "+ ")):
            stripped = stripped[2:].strip()
        if "|" in stripped:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            stripped = " ".join(cell for cell in cells if cell)
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def _coerce_text_output(markdown_text: str, output_format: str | None) -> str:
    fmt = _normalize_requested_format(output_format)
    if fmt in {"text", "json"}:
        return _markdown_to_text(markdown_text)
    return markdown_text


def _max_page_from_tables(tables: list[dict[str, Any]]) -> int:
    max_page = 0
    for table in tables:
        if not isinstance(table, dict):
            continue
        max_page = max(max_page, _coerce_int(table.get("page"), 0))
    return max_page


def _pages_from_content_list(
    content_list: Any,
    *,
    total_pages: int | None = None,
    tables: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pages_by_index: dict[int, list[str]] = defaultdict(list)
    highest_page_index = -1

    if isinstance(content_list, list):
        for entry in content_list:
            if not isinstance(entry, dict):
                continue
            page_idx = max(_coerce_int(entry.get("page_idx"), 0), 0)
            highest_page_index = max(highest_page_index, page_idx)
            text = str(entry.get("text") or "").strip()
            if text:
                pages_by_index[page_idx].append(text)

    final_total_pages = max(
        total_pages or 0,
        highest_page_index + 1,
        _max_page_from_tables(tables or []),
    )
    if final_total_pages <= 0:
        return []

    pages = [_blank_page_entry(page_number) for page_number in range(1, final_total_pages + 1)]
    for page_idx, texts in pages_by_index.items():
        if page_idx >= len(pages):
            continue
        pages[page_idx]["text"] = "\n".join(texts).strip()
    return pages


def _tables_from_middle(middle: Any) -> list[dict[str, Any]]:
    if not isinstance(middle, dict):
        return []

    tables = middle.get("tables")
    if not isinstance(tables, list):
        return []

    out: list[dict[str, Any]] = []
    for entry in tables:
        if not isinstance(entry, dict):
            continue
        html = str(entry.get("html") or "").strip()
        if not html:
            continue
        out.append(
            {
                "page": int(entry.get("page") or 1),
                "format": "html",
                "content": html,
            }
        )
    return out


def _attach_tables_to_pages(pages: list[dict[str, Any]], tables: list[dict[str, Any]]) -> None:
    for table in tables:
        if not isinstance(table, dict):
            continue
        page_number = max(_coerce_int(table.get("page"), 1), 1)
        page_index = page_number - 1
        if page_index >= len(pages):
            continue
        page_tables = pages[page_index].setdefault("tables", [])
        if isinstance(page_tables, list):
            page_tables.append(table)


def _bounded_middle_excerpt(middle: Any) -> dict[str, Any]:
    if not isinstance(middle, dict):
        return {}

    excerpt: dict[str, Any] = {}
    tables = middle.get("tables")
    if isinstance(tables, list):
        excerpt["tables"] = tables[:5]
    return excerpt


def _include_raw_artifacts(
    artifacts: dict[str, Any],
    *,
    content_list: Any,
    middle: Any,
    debug_save_raw: bool,
) -> dict[str, Any]:
    if not debug_save_raw:
        return artifacts

    artifacts = dict(artifacts)
    artifacts["content_list_raw"] = content_list if isinstance(content_list, list) else []
    artifacts["middle_json_raw"] = middle if isinstance(middle, dict) else {}
    return artifacts


def _candidate_output_dirs(output_dir: Path) -> list[Path]:
    candidates = [output_dir]
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                candidates.append(child)
    return candidates


def _resolve_primary_output_dir(output_dir: Path) -> Path:
    best_dir = output_dir
    best_score = -1

    for candidate in _candidate_output_dirs(output_dir):
        score = 0
        if any((candidate / name).exists() for name in _MINERU_TEXT_MARKDOWN_FILES):
            score += 2
        if (candidate / "content_list.json").exists():
            score += 1
        if (candidate / "middle.json").exists():
            score += 1
        if score > best_score:
            best_dir = candidate
            best_score = score
    return best_dir


def _read_first_existing(output_dir: Path, names: tuple[str, ...]) -> str:
    for name in names:
        candidate = output_dir / name
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


def _get_pdf_page_count(source_pdf_path: Path | None) -> int:
    if source_pdf_path is None or not source_pdf_path.exists():
        return 0

    try:
        import pymupdf

        with pymupdf.open(source_pdf_path) as doc:
            return len(doc)
    except Exception:
        return 0


def _page_has_content(page: dict[str, Any]) -> bool:
    text = str(page.get("text") or "").strip()
    if text:
        return True
    tables = page.get("tables")
    if isinstance(tables, list) and tables:
        return True
    blocks = page.get("blocks")
    return isinstance(blocks, list) and bool(blocks)


def _normalize_mineru_output_dir(
    output_dir: Path,
    *,
    output_format: str | None,
    prompt_preset: str | None,
    source_pdf_path: Path | None = None,
    debug_save_raw: bool = False,
) -> dict[str, Any]:
    primary_output_dir = _resolve_primary_output_dir(output_dir)
    markdown_text = _read_first_existing(primary_output_dir, _MINERU_TEXT_MARKDOWN_FILES)
    content_list = _read_json_if_exists(primary_output_dir / "content_list.json")
    middle = _read_json_if_exists(primary_output_dir / "middle.json")
    tables = _tables_from_middle(middle)
    total_pages = _get_pdf_page_count(source_pdf_path)
    pages = _pages_from_content_list(
        content_list,
        total_pages=total_pages,
        tables=tables,
    )
    _attach_tables_to_pages(pages, tables)
    coerced_text = _coerce_text_output(markdown_text, output_format)
    artifacts = {
        "content_list_excerpt": content_list[:10] if isinstance(content_list, list) else [],
        "middle_json_excerpt": _bounded_middle_excerpt(middle),
    }
    artifacts = _include_raw_artifacts(
        artifacts,
        content_list=content_list,
        middle=middle,
        debug_save_raw=debug_save_raw,
    )

    structured = {
        "schema_version": 1,
        "text": coerced_text,
        "format": _normalize_requested_format(output_format),
        "pages": pages,
        "tables": tables,
        "artifacts": artifacts,
        "meta": {
            "backend": "mineru",
            "mode": "cli",
            "supports_per_page_metrics": bool(pages),
            "prompt_preset": prompt_preset,
            "requested_output_format": output_format,
            "output_dir_layout": str(primary_output_dir.relative_to(output_dir)) if primary_output_dir != output_dir else ".",
        },
    }
    return {
        "text": coerced_text,
        "structured": structured,
    }


def _summarize_process_output(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    message = stderr or stdout or "no process output"
    return message[:400]


def run_mineru_document_ocr(
    *,
    pdf_path: Path,
    output_format: str | None = None,
    prompt_preset: str | None = None,
    requested_lang: str | None = None,
    requested_dpi: int | None = None,
) -> dict[str, Any]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"MinerU input PDF not found: {pdf_path}")

    config = load_mineru_config()
    timeout_sec = config.timeout_sec
    max_concurrency = config.max_concurrency
    tmp_root = config.tmp_root
    if tmp_root is not None:
        tmp_root.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    temp_dir_kwargs: dict[str, Any] = {"prefix": "mineru_"}
    if tmp_root is not None:
        temp_dir_kwargs["dir"] = str(tmp_root)

    with tempfile.TemporaryDirectory(**temp_dir_kwargs) as tmp_dir:
        output_dir = Path(tmp_dir)
        command = _build_mineru_command(pdf_path=pdf_path, output_dir=output_dir, config=config)
        semaphore = _get_mineru_semaphore(max_concurrency)

        try:
            with semaphore:
                # Command tokens are argv-based, shell-free, and built from validated local paths.
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )  # nosec B603
        except subprocess.TimeoutExpired as exc:
            logger.warning("MinerU timed out for {} after {}s", pdf_path, timeout_sec)
            raise TimeoutError(f"MinerU timed out after {timeout_sec}s") from exc

        if completed.returncode != 0:
            summary = _summarize_process_output(completed)
            logger.warning(
                "MinerU command failed for {} with code {}: {}",
                pdf_path,
                completed.returncode,
                summary,
            )
            raise RuntimeError(f"MinerU exited with code {completed.returncode}: {summary}")

        normalized = _normalize_mineru_output_dir(
            output_dir,
            output_format=output_format,
            prompt_preset=prompt_preset,
            source_pdf_path=pdf_path,
            debug_save_raw=config.debug_save_raw,
        )
        structured = normalized["structured"]
        pages = structured.get("pages") if isinstance(structured, dict) else []
        total_pages = len(pages) if isinstance(pages, list) else 0
        ocr_pages = (
            sum(1 for page in pages if isinstance(page, dict) and _page_has_content(page))
            if isinstance(pages, list)
            else 0
        )
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        details = {
            "backend": "mineru",
            "mode": "cli",
            "lang": requested_lang or "eng",
            "dpi": requested_dpi or 300,
            "output_format": output_format,
            "prompt_preset": prompt_preset,
            "timeout_sec": timeout_sec,
            "max_concurrency": max_concurrency,
            "elapsed_ms": elapsed_ms,
            "total_pages": total_pages,
            "ocr_pages": ocr_pages,
            "artifacts_found": {
                "markdown": bool(structured.get("text")),
                "content_list_excerpt": bool(structured.get("artifacts", {}).get("content_list_excerpt")),
                "middle_json_excerpt": bool(structured.get("artifacts", {}).get("middle_json_excerpt")),
            },
            "supports_per_page_metrics": bool(structured.get("meta", {}).get("supports_per_page_metrics")),
        }

        return {
            "text": str(normalized.get("text") or ""),
            "structured": structured,
            "details": details,
            "warnings": [],
        }


def describe_mineru_backend() -> dict[str, Any]:
    config = load_mineru_config()
    return {
        "available": _mineru_available(config),
        "pdf_only": True,
        "document_level": True,
        "opt_in_only": True,
        "supports_per_page_metrics": True,
        "mode": "cli",
        "timeout_sec": config.timeout_sec,
        "max_concurrency": config.max_concurrency,
        "tmp_root": str(config.tmp_root) if config.tmp_root is not None else None,
        "debug_save_raw": config.debug_save_raw,
    }
