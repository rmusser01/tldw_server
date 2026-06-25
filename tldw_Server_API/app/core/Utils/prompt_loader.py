import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.config_paths import resolve_prompts_dir
from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityBlocked,
    get_global_context_integrity_resolver,
)


def _prompts_dir() -> str:
    """Resolve the Prompts directory.

    Uses the shared config root resolver to ensure consistent path behavior.
    """
    prompts_dir = resolve_prompts_dir()
    logger.debug(f"Prompt loader resolved Prompts dir: {prompts_dir}")
    return str(prompts_dir)


def _module_file_base(module: str) -> str:
    # Map module to prompts filename
    # e.g., embeddings -> embeddings.prompts.md
    sanitized = re.sub(r"[^a-z0-9_\-]", "", module.strip().lower())
    return os.path.join(_prompts_dir(), f"{sanitized}.prompts")


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", key.strip().lower())


def _prompt_env_file_key(module: str, key: str) -> str:
    module_token = re.sub(r"[^A-Za-z0-9]+", "_", module.strip().upper()).strip("_")
    key_token = re.sub(r"[^A-Za-z0-9]+", "_", key.strip().upper()).strip("_")
    return f"TLDW_PROMPT_FILE_{module_token}__{key_token}"


def _prompt_asset_id(path: str, *, source_label: str | None = None) -> str:
    filename = Path(path).name
    if source_label:
        return f"prompt_file:{source_label}:{filename}"
    return f"prompt_file:{filename}"


def _read_prompt_file_text(path: str, *, source_label: str | None = None) -> str:
    prompt_path = Path(path)
    if prompt_path.is_symlink():
        raise OSError(f"Symlinked prompt file is not allowed: {prompt_path}")

    asset_id = _prompt_asset_id(path, source_label=source_label)
    with open(prompt_path, "rb") as f:
        raw = f.read()

    metadata: dict[str, str]
    if source_label is None:
        metadata = {"path": prompt_path.name}
    else:
        metadata = {"path": str(prompt_path), "source_label": source_label}

    current_digest = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id=asset_id,
        files={prompt_path.name: raw},
        metadata=metadata,
    )
    resolver = get_global_context_integrity_resolver()
    if resolver is not None:
        resolver.require_digest_allowed(
            asset_id,
            current_digest=current_digest,
            purpose="prompt_load",
            changed_state="changed_approved_non_executable",
        )

    return raw.decode("utf-8")


def _load_env_prompt_file(module: str, key: str) -> Optional[str]:
    env_name = _prompt_env_file_key(module, key)
    raw_path = os.getenv(env_name)
    if not raw_path or not str(raw_path).strip():
        return None

    path = os.path.expanduser(str(raw_path).strip())
    try:
        return _read_prompt_file_text(path, source_label=f"env:{env_name}").strip()
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked) as exc:
        logger.warning(
            "Prompt override file read failed for env '{}' (module='{}', key='{}', error_type='{}')",
            env_name,
            module,
            key,
            exc.__class__.__name__,
        )
        return None


def _load_yaml(path: str) -> Optional[dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    try:
        raw = _read_prompt_file_text(path)
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked, TypeError, ValueError, yaml.YAMLError):
        return None


def _load_json(path: str) -> Optional[dict[str, Any]]:
    try:
        raw = _read_prompt_file_text(path)
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_prompt(module: str, key: str) -> Optional[str]:
    """Load a named prompt snippet from Prompts folder.

    Searches for a markdown heading containing the key, then returns the
    first fenced code block following that heading. If not found, returns None.
    """
    env_override = _load_env_prompt_file(module, key)
    if env_override is not None:
        return env_override

    base = _module_file_base(module)
    norm = _norm_key(key)

    # Prefer YAML
    yaml_path_1 = base + ".yaml"
    yaml_path_2 = base + ".yml"
    for ypath in (yaml_path_1, yaml_path_2):
        if os.path.exists(ypath):
            ydata = _load_yaml(ypath)
            if isinstance(ydata, dict):
                # two shapes supported: {key: str} or {templates: {name: {template:..., type:...}}}
                # Try flat map first
                if norm in {_norm_key(k): k for k in ydata}:
                    # Find original key name casing
                    for k, v in ydata.items():
                        if _norm_key(k) == norm and isinstance(v, str):
                            return v.strip()
                        if _norm_key(k) == norm and isinstance(v, dict) and isinstance(v.get("template"), str):
                            return v["template"].strip()
                # Try nested under 'templates'
                tmap = ydata.get("templates") if isinstance(ydata.get("templates"), dict) else None
                if tmap:
                    for k, v in tmap.items():
                        if _norm_key(k) == norm and isinstance(v, dict) and isinstance(v.get("template"), str):
                            return v["template"].strip()
            # If YAML present but key not found, continue to JSON/MD fallback

    # Try JSON
    json_path = base + ".json"
    if os.path.exists(json_path):
        jdata = _load_json(json_path)
        if isinstance(jdata, dict):
            for k, v in jdata.items():
                if _norm_key(k) == norm and isinstance(v, str):
                    return v.strip()
                if _norm_key(k) == norm and isinstance(v, dict) and isinstance(v.get("template"), str):
                    return v["template"].strip()

    # Find a heading that contains the key (case-insensitive)
    # Then capture the next fenced code block ```...```
    md_path = base + ".md"
    if os.path.exists(md_path):
        try:
            text = _read_prompt_file_text(md_path)
        except (OSError, UnicodeDecodeError, ContextIntegrityBlocked):
            text = ""
        if text:
            pattern = re.compile(
                rf"^\s*#{{1,6}}\s*([^\n]+?{re.escape(key)}[^\n]*)\n+```([\s\S]*?)```",
                re.IGNORECASE | re.MULTILINE,
            )
            m = pattern.search(text)
            if m:
                return m.group(2).strip()

    return None
