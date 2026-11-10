"""Prompt Studio request-bound resource ownership checks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastapi import HTTPException, status


def authoritative_prompt_project(
    db: Any,
    prompt_id: int,
    *,
    compatibility_project_id: int | None = None,
) -> tuple[dict[str, Any], int]:
    """Load a live prompt and return its authoritative project identifier."""

    get_with_project = getattr(db, "get_prompt_with_project", None)
    if callable(get_with_project):
        prompt = get_with_project(prompt_id, include_deleted=False)
    else:
        prompt = db.get_prompt(prompt_id)

    try:
        project_id = int(prompt["project_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt {prompt_id} not found",
        ) from None

    if prompt.get("deleted") or (
        compatibility_project_id is not None
        and int(compatibility_project_id) != project_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt {prompt_id} not found",
        )
    return prompt, project_id


def require_test_cases_in_project(
    db: Any,
    test_case_ids: Iterable[int],
    project_id: int,
) -> list[int]:
    """Require every requested live test case to belong to ``project_id``."""

    identifiers = list(dict.fromkeys(int(item) for item in test_case_ids))
    if not identifiers:
        return []

    get_many = getattr(db, "get_test_cases_by_ids", None)
    rows = get_many(identifiers) if callable(get_many) else []
    rows_by_id: dict[int, dict[str, Any]] = {}
    for row in rows or []:
        try:
            rows_by_id[int(row["id"])] = row
        except (KeyError, TypeError, ValueError):
            continue

    get_one = getattr(db, "get_test_case", None)
    for identifier in identifiers:
        row = rows_by_id.get(identifier)
        if (row is None or row.get("project_id") is None) and callable(get_one):
            row = get_one(identifier)
        try:
            row_project_id = int(row["project_id"])
        except (KeyError, TypeError, ValueError):
            row_project_id = -1
        if not row or row.get("deleted") or row_project_id != int(project_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more test cases were not found in the prompt project",
            )
    return identifiers
