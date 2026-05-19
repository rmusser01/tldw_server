from __future__ import annotations

import os
from typing import Any

from loguru import logger


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default) or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _gmail_connector_enabled_from_env() -> bool:
    value = str(os.getenv("EMAIL_GMAIL_CONNECTOR_ENABLED", "false") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_default_policy_from_env(org_id: int) -> dict[str, Any]:
    """Return a default org policy derived from environment variables."""
    default_providers = (
        "drive,notion,gmail,zotero"
        if _gmail_connector_enabled_from_env()
        else "drive,notion,zotero"
    )
    return {
        "org_id": org_id,
        "enabled_providers": _csv_env("ORG_CONNECTORS_ENABLED_PROVIDERS", default_providers),
        "allowed_export_formats": _csv_env("ORG_CONNECTORS_ALLOWED_EXPORT_FORMATS", "md,txt,pdf"),
        "allowed_file_types": _csv_env("ORG_CONNECTORS_ALLOWED_FILE_TYPES", ""),
        "max_file_size_mb": int(os.getenv("ORG_CONNECTORS_MAX_FILE_SIZE_MB", "500") or 500),
        "account_linking_role": os.getenv("ORG_CONNECTORS_ACCOUNT_LINKING_ROLE", "admin"),
        "allowed_account_domains": _csv_env("ORG_CONNECTORS_ALLOWED_ACCOUNT_DOMAINS", ""),
        "allowed_remote_paths": _csv_env("ORG_CONNECTORS_ALLOWED_REMOTE_PATHS", ""),
        "denied_remote_paths": _csv_env("ORG_CONNECTORS_DENIED_REMOTE_PATHS", ""),
        "allowed_notion_workspaces": _csv_env("ORG_CONNECTORS_ALLOWED_NOTION_WORKSPACES", ""),
        "denied_notion_workspaces": _csv_env("ORG_CONNECTORS_DENIED_NOTION_WORKSPACES", ""),
        "quotas_per_role": {},
    }


def _path_allowed(path: str | None, allow: list[str], deny: list[str]) -> bool:
    from fnmatch import fnmatch
    if not path:
        return True
    for pat in deny:
        if fnmatch(path, pat):
            return False
    if not allow:
        return True
    return any(fnmatch(path, pat) for pat in allow)


def evaluate_policy_constraints(
    policy: dict[str, Any],
    *,
    provider: str,
    remote_path: str | None = None,
    notion_workspace_id: str | None = None,
    account_email: str | None = None,
) -> tuple[bool, str | None]:
    """Check org policy for provider enablement and path/workspace/domain constraints."""
    try:
        enabled = [str(p).lower() for p in (policy.get("enabled_providers") or [])]
        if provider.lower() not in enabled:
            return False, f"Provider '{provider}' disabled by org policy"
        # account domain
        if account_email:
            allowed_domains = policy.get("allowed_account_domains") or []
            if allowed_domains:
                domain = account_email.split("@")[-1].lower() if "@" in account_email else ""
                if not any(domain == d.lower() for d in allowed_domains):
                    return False, f"Account domain not permitted by org policy: {domain}"
        # remote paths
        allow = policy.get("allowed_remote_paths") or []
        deny = policy.get("denied_remote_paths") or []
        if not _path_allowed(remote_path, allow, deny):
            return False, "Remote path denied by org policy"
        # notion workspace
        if notion_workspace_id:
            allow_ws = policy.get("allowed_notion_workspaces") or []
            deny_ws = policy.get("denied_notion_workspaces") or []
            if deny_ws and notion_workspace_id in deny_ws:
                return False, "Notion workspace denied by org policy"
            if allow_ws and notion_workspace_id not in allow_ws:
                return False, "Notion workspace not in allowed set"
        return True, None
    except Exception:
        logger.warning("Policy evaluation error")
        return False, "Policy evaluation failed"


def is_file_type_allowed(*, name: str | None, mime: str | None, allowed: list[str] | None) -> bool:
    """Return True if a file name/mime is allowed by 'allowed' list.

    - allowed may contain extensions (with/without dot), full mimes, or mime prefixes (e.g., 'text/', 'application/pdf').
    - Empty or None 'allowed' means allow all.
    """
    allowed = [a.strip().lower() for a in (allowed or []) if a and a.strip()]
    if not allowed:
        return True
    nm = (name or "").lower()
    mm = (mime or "").lower()
    ext = ""
    if "." in nm:
        ext = nm.split(".")[-1]
    candidates = set()
    if ext:
        candidates.add(ext)
        candidates.add("." + ext)
    if mm:
        candidates.add(mm)
        # mime prefix like 'text/'
        if "/" in mm:
            candidates.add(mm.split("/")[0] + "/")
    # Match if any allowed token equals any candidate
    return any(a in candidates for a in allowed)
