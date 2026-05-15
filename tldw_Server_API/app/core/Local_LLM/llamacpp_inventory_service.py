from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppInventoryItem,
    LlamaCppInventoryResponse,
    LlamaCppModelMetadata,
)
from tldw_Server_API.app.core.Local_LLM import handler_utils
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ModelNotFoundError, ServerError
from tldw_Server_API.app.core.Local_LLM.llamacpp_config_lock import LockAcquisitionError, llamacpp_config_write_lock
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.config import load_comprehensive_config, refresh_config_cache


_QUANT_RE = re.compile(r"(?:^|[-_.])(Q\d(?:_[A-Z0-9]+)*|F16|F32|BF16|IQ\d_[A-Z0-9_]+)(?:[-_.]|$)", re.IGNORECASE)
_PARAM_RE = re.compile(r"(?:^|[-_.])(\d+(?:\.\d+)?[bm])(?:[-_.]|$)", re.IGNORECASE)
_CTX_RE = re.compile(r"(?:^|[-_.])(?:ctx|context)[-_.]?(\d{3,6})(?:[-_.]|$)", re.IGNORECASE)


def model_id_for_path(path: Path) -> str:
    """Return the stable inventory ID for a canonical local GGUF path."""
    canonical = str(_canonical_path(path, "Model"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"gguf:{digest}"


def scan_inventory(config_state: dict[str, Any] | None = None, limit: int = 500) -> LlamaCppInventoryResponse:
    """Return a bounded GGUF inventory from configured and registered local paths."""
    saved_config = _saved_config_from_state(config_state)
    models_dir = _optional_path(saved_config.get("models_dir"))
    registered_paths = _path_list(saved_config.get("registered_model_paths"))
    warnings: list[str] = []
    items: list[LlamaCppInventoryItem] = []
    seen_ids: set[str] = set()
    scan_limited = False

    allowed_bases = _allowed_bases_for_config(saved_config)
    for path in registered_paths:
        item = _item_for_path(path, source="registered_path", allowed_bases=allowed_bases, warnings=warnings)
        if item is None:
            continue
        if item.model_id not in seen_ids:
            items.append(item)
            seen_ids.add(item.model_id)

    if models_dir is None:
        warnings.append("LlamaCpp.models_dir is not configured; only registered model paths were checked.")
    elif not models_dir.exists():
        warnings.append(f"Configured models_dir '{models_dir}' does not exist.")
    elif not models_dir.is_dir():
        warnings.append(f"Configured models_dir '{models_dir}' is not a directory.")
    else:
        model_scan_limit = max(limit - len(items), 0)
        if model_scan_limit <= 0:
            scan_limited = _has_scannable_gguf(models_dir, warnings)
        model_items_added = 0
        for path in _iter_gguf_models(models_dir, warnings, model_scan_limit):
            if model_items_added >= model_scan_limit:
                scan_limited = True
                break
            item = _item_for_path(path, source="models_dir", allowed_bases=allowed_bases, warnings=warnings)
            if item is None:
                continue
            if item.model_id not in seen_ids:
                items.append(item)
                seen_ids.add(item.model_id)
                model_items_added += 1

    items.sort(key=lambda item: (item.source != "registered_path", item.basename.lower(), item.path))
    return LlamaCppInventoryResponse(models=items, warnings=warnings, scan_limited=scan_limited)


def register_model_path(path: Path) -> LlamaCppInventoryItem:
    """Persist a local registered model path and return its inventory representation.

    Registration only persists paths under the configured models_dir or
    allowed_paths. Missing or non-GGUF paths within those roots are saved and
    reported with warnings so operators can correct them from the WebUI.
    Unresolvable paths are rejected because they cannot get a deterministic safe
    ID and should not be persisted.
    """
    canonical = _canonical_path(path, "Registered model")
    try:
        with llamacpp_config_write_lock():
            saved_config = _read_saved_config()
            allowed_bases = _allowed_bases_for_config(saved_config)
            if not allowed_bases or not handler_utils.is_path_allowed(canonical, allowed_bases):
                raise ServerError("Registered model path is outside allowed llama.cpp paths.")
            existing = _path_list(saved_config.get("registered_model_paths"))
            existing_by_id: dict[str, Path] = {}
            for item in existing:
                try:
                    existing_canonical = _canonical_path(item, "Registered model")
                    existing_by_id[model_id_for_path(existing_canonical)] = existing_canonical
                except ServerError:
                    existing_by_id[_unresolved_path_key(item)] = item.expanduser()
            existing_by_id.setdefault(model_id_for_path(canonical), canonical)

            registered_value = ", ".join(str(item) for item in existing_by_id.values())
            setup_manager.update_config({"LlamaCpp": {"registered_model_paths": registered_value}})
            refresh_config_cache()
    except Exception as exc:
        if isinstance(exc, LockAcquisitionError):
            raise ServerError("Failed to acquire the llama.cpp config write lock.") from exc
        if isinstance(exc, ServerError):
            raise
        raise ServerError("Failed to persist registered llama.cpp model path.") from exc

    saved_config["registered_model_paths"] = [registered_value]
    allowed_bases = _allowed_bases_for_config(saved_config)
    item = _item_for_path(canonical, source="registered_path", allowed_bases=allowed_bases)
    if item is None:
        raise ServerError("Registered model path could not be resolved.")
    return item


def resolve_model_id(model_id: str) -> Path:
    """Resolve a stable inventory model_id to a canonical local path."""
    wanted = str(model_id).strip()
    if not wanted:
        raise ModelNotFoundError("Model ID is required.")

    saved_config = _read_saved_config()
    allowed_bases = _allowed_bases_for_config(saved_config)
    inventory = scan_inventory(limit=500)
    for item in inventory.models:
        if item.model_id == wanted:
            path = _canonical_path(Path(item.path), "Model")
            if path.suffix.lower() != ".gguf" or not path.is_file():
                raise ModelNotFoundError(f"Model ID {wanted} does not reference an available GGUF file.")
            if not allowed_bases or not handler_utils.is_path_allowed(path, allowed_bases):
                raise ServerError("Model path is outside allowed llama.cpp paths.")
            return path
    raise ModelNotFoundError(f"Model ID {wanted} was not found in the llama.cpp inventory.")


def _iter_gguf_models(models_dir: Path, warnings: list[str], limit: int):
    if limit <= 0:
        return

    visited = 0
    max_visited = max(limit * 20, 1000)

    def _on_error(error: OSError) -> None:
        warnings.append(f"Could not scan '{error.filename}': {error.strerror or error.__class__.__name__}.")

    for root, dirs, files in os.walk(models_dir, topdown=True, onerror=_on_error):
        dirs.sort()
        files.sort()
        visited += len(dirs) + len(files)
        if visited > max_visited:
            warnings.append("Model inventory scan reached the traversal limit.")
            return
        for filename in files:
            lowered = filename.lower()
            if not lowered.endswith(".gguf"):
                continue
            if lowered.startswith("mmproj"):
                continue
            yield Path(root) / filename


def _has_scannable_gguf(models_dir: Path, warnings: list[str]) -> bool:
    def _on_error(error: OSError) -> None:
        warnings.append(f"Could not scan '{error.filename}': {error.strerror or error.__class__.__name__}.")

    for _root, _dirs, files in os.walk(models_dir, topdown=True, onerror=_on_error):
        for filename in files:
            lowered = filename.lower()
            if lowered.endswith(".gguf") and not lowered.startswith("mmproj"):
                return True
    return False


def _item_for_path(
    path: Path,
    *,
    source: str,
    allowed_bases: list[Path],
    warnings: list[str] | None = None,
) -> LlamaCppInventoryItem | None:
    try:
        canonical = _canonical_path(path, "Model")
    except ServerError:
        if warnings is not None:
            warnings.append("Could not inspect a model inventory path because it could not be resolved.")
        return None
    basename = canonical.name
    item_warnings: list[str] = []
    size_bytes: int | None = None
    modified_at: str | None = None

    if canonical.suffix.lower() != ".gguf":
        item_warnings.append("Registered path does not reference a GGUF file.")
    if not canonical.exists():
        item_warnings.append("Registered path is missing.")
    elif not canonical.is_file():
        item_warnings.append("Registered path is not a file.")
    else:
        try:
            stat_result = canonical.stat()
            size_bytes = stat_result.st_size
            modified_at = datetime.fromtimestamp(stat_result.st_mtime, UTC).isoformat()
        except PermissionError:
            item_warnings.append("Registered path is not readable.")
        except OSError:
            item_warnings.append("Could not read registered path metadata.")

    if allowed_bases and not handler_utils.is_path_allowed(canonical, allowed_bases):
        item_warnings.append("Model path is outside allowed llama.cpp paths and cannot be started until allowed_paths is updated.")

    return LlamaCppInventoryItem(
        model_id=model_id_for_path(canonical),
        display_name=_display_name(basename),
        basename=basename,
        source=source,
        path=str(canonical),
        size_bytes=size_bytes,
        modified_at=modified_at,
        metadata=_metadata_from_filename(basename),
        warnings=item_warnings,
    )


def _canonical_path(path: Path, label: str) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ServerError(f"{label} path could not be resolved.") from exc


def _unresolved_path_key(path: Path) -> str:
    digest = hashlib.sha256(str(path.expanduser()).encode("utf-8")).hexdigest()[:24]
    return f"unresolved:{digest}"


def _metadata_from_filename(filename: str) -> LlamaCppModelMetadata:
    quant_match = _QUANT_RE.search(filename)
    param_match = _PARAM_RE.search(filename)
    ctx_match = _CTX_RE.search(filename)
    return LlamaCppModelMetadata(
        quantization=quant_match.group(1).upper() if quant_match else None,
        parameter_hint=param_match.group(1).upper() if param_match else None,
        context_hint=int(ctx_match.group(1)) if ctx_match else None,
    )


def _display_name(filename: str) -> str:
    return filename[:-5] if filename.lower().endswith(".gguf") else filename


def _saved_config_from_state(config_state: dict[str, Any] | None) -> dict[str, Any]:
    if config_state and isinstance(config_state.get("saved_config"), dict):
        return dict(config_state["saved_config"])
    return _read_saved_config()


def _allowed_bases_for_config(saved_config: dict[str, Any]) -> list[Path]:
    models_dir = _optional_path(saved_config.get("models_dir"))
    allowed_paths = _path_list(saved_config.get("allowed_paths"))
    return handler_utils.build_allowed_paths(models_dir, allowed_paths) if models_dir else allowed_paths


def _read_saved_config() -> dict[str, Any]:
    parser = load_comprehensive_config()
    section = parser["LlamaCpp"] if parser and parser.has_section("LlamaCpp") else None
    if section is None:
        return {"allowed_paths": [], "registered_model_paths": []}
    return {
        "models_dir": _str_or_none(section.get("models_dir", fallback=None)),
        "allowed_paths": _split_list(section.get("allowed_paths", fallback="")),
        "registered_model_paths": _split_list(section.get("registered_model_paths", fallback="")),
    }


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _path_list(value: Any) -> list[Path]:
    if not value:
        return []
    if isinstance(value, str):
        values = _split_list(value)
    else:
        values = [str(item).strip() for item in value if str(item).strip()]
    return [Path(item).expanduser() for item in values]


def _split_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).replace(os.pathsep, ",").split(",") if part.strip()]


def _str_or_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None
