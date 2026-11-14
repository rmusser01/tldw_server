import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.config_paths import resolve_prompts_dir
from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityBlocked,
    ContextIntegrityResolver,
    get_global_context_integrity_resolver,
)

_USE_GLOBAL_INTEGRITY_RESOLVER = object()


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


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


class PromptAssetUnavailableError(RuntimeError):
    """Raised when a strict prompt asset cannot be used safely."""

    def __init__(self) -> None:
        super().__init__("prompt_asset_unavailable")


def _read_regular_file_bytes_no_follow(path: Path, *, max_bytes: int | None = None) -> bytes:
    expected = path.lstat()
    if not stat.S_ISREG(expected.st_mode):
        raise OSError(f"Prompt file is not a regular file: {path}")
    if max_bytes is not None and expected.st_size > max_bytes:
        raise OSError("Prompt file exceeds its byte limit")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(expected, opened):
            raise OSError(f"Prompt file changed while being opened: {path}")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise OSError("Prompt file exceeds its byte limit")
        with os.fdopen(fd, "rb", closefd=True) as file_obj:
            fd = -1
            raw = file_obj.read() if max_bytes is None else file_obj.read(max_bytes + 1)
            if max_bytes is not None and len(raw) > max_bytes:
                raise OSError("Prompt file exceeds its byte limit")
            return raw
    finally:
        if fd >= 0:
            os.close(fd)


def _read_prompt_file_text(
    path: str,
    *,
    source_label: str | None = None,
    integrity_resolver: Any = _USE_GLOBAL_INTEGRITY_RESOLVER,
    max_bytes: int | None = None,
) -> str:
    prompt_path = Path(path)
    asset_id = _prompt_asset_id(path, source_label=source_label)
    raw = _read_regular_file_bytes_no_follow(prompt_path, max_bytes=max_bytes)

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
    resolver = (
        get_global_context_integrity_resolver()
        if integrity_resolver is _USE_GLOBAL_INTEGRITY_RESOLVER
        else integrity_resolver
    )
    if resolver is not None and not isinstance(resolver, ContextIntegrityResolver):
        raise TypeError("integrity_resolver must be a ContextIntegrityResolver or None")
    if resolver is not None:
        resolver.require_digest_allowed(
            asset_id,
            current_digest=current_digest,
            purpose="prompt_load",
            changed_state="changed_approved_non_executable",
        )

    return raw.decode("utf-8")


def _load_env_prompt_file(
    module: str,
    key: str,
    *,
    integrity_resolver: Any = _USE_GLOBAL_INTEGRITY_RESOLVER,
) -> Optional[str]:
    env_name = _prompt_env_file_key(module, key)
    raw_path = os.getenv(env_name)
    if not raw_path or not str(raw_path).strip():
        return None

    path = os.path.expanduser(str(raw_path).strip())
    try:
        return _read_prompt_file_text(
            path,
            source_label=f"env:{env_name}",
            integrity_resolver=integrity_resolver,
        ).strip()
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked) as exc:
        logger.warning(
            "Prompt override file read failed for env '{}' (module='{}', key='{}', error_type='{}')",
            env_name,
            module,
            key,
            exc.__class__.__name__,
        )
        return None


def _load_yaml(
    path: str,
    *,
    integrity_resolver: Any = _USE_GLOBAL_INTEGRITY_RESOLVER,
) -> Optional[dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    try:
        raw = _read_prompt_file_text(path, integrity_resolver=integrity_resolver)
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked, TypeError, ValueError, yaml.YAMLError):
        return None


def _load_json(
    path: str,
    *,
    integrity_resolver: Any = _USE_GLOBAL_INTEGRITY_RESOLVER,
) -> Optional[dict[str, Any]]:
    try:
        raw = _read_prompt_file_text(path, integrity_resolver=integrity_resolver)
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return None
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_prompt(
    module: str,
    key: str,
    *,
    integrity_resolver: Any = _USE_GLOBAL_INTEGRITY_RESOLVER,
) -> Optional[str]:
    """Load a named prompt snippet from Prompts folder.

    Searches for a markdown heading containing the key, then returns the
    first fenced code block following that heading. If not found, returns None.
    """
    env_override = _load_env_prompt_file(
        module,
        key,
        integrity_resolver=integrity_resolver,
    )
    if env_override is not None:
        return env_override

    base = _module_file_base(module)
    norm = _norm_key(key)

    # Prefer YAML
    yaml_path_1 = base + ".yaml"
    yaml_path_2 = base + ".yml"
    for ypath in (yaml_path_1, yaml_path_2):
        if os.path.exists(ypath):
            ydata = _load_yaml(ypath, integrity_resolver=integrity_resolver)
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
        jdata = _load_json(json_path, integrity_resolver=integrity_resolver)
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
            text = _read_prompt_file_text(
                md_path,
                integrity_resolver=integrity_resolver,
            )
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


def _strict_prompt_text(value: object, *, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise PromptAssetUnavailableError()
    prompt = value.strip()
    if not prompt or "\x00" in prompt or any(0xD800 <= ord(char) <= 0xDFFF for char in prompt):
        raise PromptAssetUnavailableError()
    try:
        encoded = prompt.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PromptAssetUnavailableError() from exc
    if len(encoded) > max_bytes:
        raise PromptAssetUnavailableError()
    return prompt


def _strict_mapping_prompt(data: object, *, key: str, max_bytes: int) -> str:
    if not isinstance(data, dict):
        raise PromptAssetUnavailableError()
    norm = _norm_key(key)
    matches: list[object] = []
    for candidate_key, value in data.items():
        if _norm_key(str(candidate_key)) != norm:
            continue
        matches.append(value.get("template") if isinstance(value, dict) else value)
    templates = data.get("templates")
    if isinstance(templates, dict):
        for candidate_key, value in templates.items():
            if _norm_key(str(candidate_key)) != norm:
                continue
            matches.append(value.get("template") if isinstance(value, dict) else value)
    if len(matches) != 1:
        raise PromptAssetUnavailableError()
    return _strict_prompt_text(matches[0], max_bytes=max_bytes)


def _strict_prompt_from_asset(path: str, *, key: str, max_bytes: int) -> str:
    raw_text = _read_prompt_file_text(path, max_bytes=max_bytes)
    suffix = Path(path).suffix.casefold()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise PromptAssetUnavailableError() from exc

        class UniqueKeyLoader(yaml.SafeLoader):
            pass

        def construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
            mapping: dict[Any, Any] = {}
            for key_node, value_node in node.value:
                item_key = loader.construct_object(key_node, deep=deep)
                if item_key in mapping:
                    raise PromptAssetUnavailableError()
                mapping[item_key] = loader.construct_object(value_node, deep=deep)
            return mapping

        UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_unique_mapping,
        )
        try:
            loader = UniqueKeyLoader(raw_text)
            try:
                data = loader.get_single_data()
            finally:
                loader.dispose()
            return _strict_mapping_prompt(
                data,
                key=key,
                max_bytes=max_bytes,
            )
        except (RecursionError, TypeError, ValueError, yaml.YAMLError):
            raise PromptAssetUnavailableError() from None
    if suffix == ".json":

        def reject_constant(_value: str) -> None:
            raise PromptAssetUnavailableError()

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            mapping: dict[str, Any] = {}
            for item_key, value in pairs:
                if item_key in mapping:
                    raise PromptAssetUnavailableError()
                mapping[item_key] = value
            return mapping

        try:
            data = json.loads(
                raw_text,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
            return _strict_mapping_prompt(data, key=key, max_bytes=max_bytes)
        except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
            raise PromptAssetUnavailableError() from None

    pattern = re.compile(
        rf"^\s*#{{1,6}}\s*([^\n]*{re.escape(key)}[^\n]*)\n+```(?:[^\n]*)\n([\s\S]*?)```",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(raw_text))
    if len(matches) != 1:
        raise PromptAssetUnavailableError()
    return _strict_prompt_text(matches[0].group(2), max_bytes=max_bytes)


def load_prompt_strict(module: str, key: str, max_bytes: int) -> str:
    """Load one required prompt without weakening a configured override.

    Unlike :func:`load_prompt`, any configured override failure is terminal and
    never falls back to the packaged prompt.
    """

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise PromptAssetUnavailableError() from None

    env_name = _prompt_env_file_key(module, key)
    if env_name in os.environ:
        raw_path = str(os.environ[env_name]).strip()
        if not raw_path:
            raise PromptAssetUnavailableError() from None
        try:
            value = _read_prompt_file_text(
                os.path.expanduser(raw_path),
                source_label=f"env:{env_name}",
                max_bytes=max_bytes,
            )
            return _strict_prompt_text(value, max_bytes=max_bytes)
        except (OSError, UnicodeDecodeError, ContextIntegrityBlocked, PromptAssetUnavailableError, RecursionError):
            raise PromptAssetUnavailableError() from None

    base = _module_file_base(module)
    candidates = (base + ".yaml", base + ".yml", base + ".json", base + ".md")
    selected = next((path for path in candidates if os.path.lexists(path)), None)
    if selected is None:
        raise PromptAssetUnavailableError() from None
    try:
        return _strict_prompt_from_asset(selected, key=key, max_bytes=max_bytes)
    except (OSError, UnicodeDecodeError, ContextIntegrityBlocked, PromptAssetUnavailableError, RecursionError):
        raise PromptAssetUnavailableError() from None
