from __future__ import annotations

from pathlib import PurePosixPath


def _normalize_member_name(member_name: str) -> str:
    return member_name.replace("\\", "/").strip().strip("/")


def _single_common_root(member_names: list[str]) -> str | None:
    roots: set[str] = set()
    for member_name in member_names:
        normalized = _normalize_member_name(member_name)
        if not normalized or member_name.endswith("/"):
            continue
        parts = PurePosixPath(normalized).parts
        if len(parts) < 2:
            return None
        roots.add(parts[0])
        if len(roots) > 1:
            return None
    return next(iter(roots), None)


def _strip_common_root(member_name: str, common_root: str | None) -> str:
    normalized = _normalize_member_name(member_name)
    if not common_root:
        return normalized
    parts = PurePosixPath(normalized).parts
    if parts and parts[0] == common_root:
        return str(PurePosixPath(*parts[1:]))
    return normalized


def normalized_archive_member_paths(member_names: list[str]) -> dict[str, str]:
    common_root = _single_common_root(member_names)
    paths_by_member: dict[str, str] = {}
    seen_relative_paths: set[str] = set()
    for member_name in member_names:
        normalized_name = _normalize_member_name(member_name)
        if not normalized_name or member_name.endswith("/"):
            continue
        relative_path = _strip_common_root(member_name, common_root)
        if relative_path in seen_relative_paths:
            raise ValueError(f"Duplicate normalized archive member path: {relative_path}")
        seen_relative_paths.add(relative_path)
        paths_by_member[member_name] = relative_path
    return paths_by_member


def normalize_archive_members(
    member_names: list[str],
    hashes: dict[str, str],
) -> dict[str, dict[str, str | None]]:
    items: dict[str, dict[str, str | None]] = {}
    for member_name, relative_path in normalized_archive_member_paths(member_names).items():
        normalized_name = _normalize_member_name(member_name)
        items[relative_path] = {
            "relative_path": relative_path,
            "content_hash": hashes.get(member_name, hashes.get(normalized_name)),
        }
    return items


def diff_snapshots(
    *,
    previous: dict[str, dict[str, str | None]],
    current: dict[str, dict[str, str | None]],
) -> dict[str, list[dict[str, str | None]]]:
    previous_keys = set(previous)
    current_keys = set(current)

    created = [current[key] for key in sorted(current_keys - previous_keys)]
    deleted = [previous[key] for key in sorted(previous_keys - current_keys)]
    changed: list[dict[str, str | None]] = []
    unchanged: list[dict[str, str | None]] = []

    for key in sorted(previous_keys & current_keys):
        previous_hash = previous[key].get("content_hash")
        current_hash = current[key].get("content_hash")
        if previous_hash != current_hash:
            changed.append(current[key])
        else:
            unchanged.append(current[key])

    return {
        "created": created,
        "changed": changed,
        "unchanged": unchanged,
        "deleted": deleted,
    }
