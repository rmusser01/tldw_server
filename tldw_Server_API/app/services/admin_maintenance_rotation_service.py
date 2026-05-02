from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from tldw_Server_API.app.api.v1.utils.pagination import (
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    MaintenanceRotationRunItem,
    MaintenanceRotationRunListResponse,
)


_ALLOWED_ROTATION_FIELDS = {"payload", "result"}
_ENV_ROTATION_KEY_SOURCE = "env:jobs_crypto_rotate"


@dataclass
class AdminMaintenanceRotationService:
    """Service for validating and recording authoritative maintenance rotation runs."""

    repo: Any

    @staticmethod
    def _is_duplicate_active_execute_error(exc: Exception) -> bool:
        """Return True when an insert failure matches the active execute uniqueness guard."""
        message = str(exc).strip().lower()
        return any(
            token in message
            for token in (
                "idx_maintenance_rotation_runs_active_execute",
                "duplicate key value violates unique constraint",
                "unique constraint failed",
            )
        )

    @staticmethod
    def _normalize_optional_scope_text(value: str | None) -> str | None:
        """Trim optional scope text and collapse blanks to None."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _normalize_fields(fields: list[str] | None) -> list[str]:
        """Return a deduplicated, validated set of rotation fields."""
        if fields is None:
            return ["payload", "result"]
        normalized: list[str] = []
        seen: set[str] = set()
        for entry in fields:
            item = str(entry).strip().lower()
            if not item or item in seen:
                continue
            if item not in _ALLOWED_ROTATION_FIELDS:
                raise HTTPException(status_code=400, detail="invalid_rotation_fields")
            seen.add(item)
            normalized.append(item)
        if not normalized:
            raise HTTPException(status_code=400, detail="invalid_rotation_fields")
        return normalized

    @staticmethod
    def resolve_key_source() -> str | None:
        """Return the configured server-side rotation key source, or None if unavailable."""
        old_key = os.getenv("JOBS_CRYPTO_ROTATE_OLD_KEY", "").strip()
        new_key = os.getenv("JOBS_CRYPTO_ROTATE_NEW_KEY", "").strip()
        if not old_key or not new_key:
            return None
        return _ENV_ROTATION_KEY_SOURCE

    @staticmethod
    def build_scope_summary(
        *,
        domain: str | None,
        queue: str | None,
        job_type: str | None,
        fields: list[str],
        limit: int | None,
    ) -> str:
        """Build a stable human-readable scope summary for operator history."""
        domain_text = domain or "*"
        queue_text = queue or "*"
        job_type_text = job_type or "*"
        fields_text = ",".join(fields)
        limit_text = "*" if limit is None else str(int(limit))
        return (
            f"domain={domain_text}, queue={queue_text}, "
            f"job_type={job_type_text}, fields={fields_text}, limit={limit_text}"
        )

    async def create_run(
        self,
        *,
        mode: str,
        domain: str | None,
        queue: str | None,
        job_type: str | None,
        fields: list[str] | None,
        limit: int,
        confirmed: bool,
        requested_by_user_id: int | None,
        requested_by_label: str | None,
    ) -> dict[str, Any]:
        """Validate a maintenance rotation request and persist the queued run."""
        if mode not in {"dry_run", "execute"}:
            raise HTTPException(status_code=400, detail="invalid_rotation_mode")
        if mode == "execute" and not confirmed:
            raise HTTPException(status_code=400, detail="confirmation_required")

        key_source = self.resolve_key_source()
        if key_source is None:
            raise HTTPException(status_code=503, detail="rotation_key_source_unavailable")

        normalized_domain = self._normalize_optional_scope_text(domain)
        normalized_queue = self._normalize_optional_scope_text(queue)
        normalized_job_type = self._normalize_optional_scope_text(job_type)
        normalized_fields = self._normalize_fields(fields)
        normalized_limit = int(limit)
        if normalized_limit < 1:
            raise HTTPException(status_code=400, detail="invalid_rotation_limit")

        if mode == "execute" and await self.repo.has_active_execute_run():
            raise HTTPException(status_code=409, detail="active_execute_run_exists")

        try:
            created = await self.repo.create_run(
                mode=mode,
                domain=normalized_domain,
                queue=normalized_queue,
                job_type=normalized_job_type,
                fields_json=json.dumps(normalized_fields, separators=(",", ":")),
                limit=normalized_limit,
                requested_by_user_id=requested_by_user_id,
                requested_by_label=requested_by_label.strip() if requested_by_label else None,
                confirmation_recorded=bool(confirmed),
                scope_summary=self.build_scope_summary(
                    domain=normalized_domain,
                    queue=normalized_queue,
                    job_type=normalized_job_type,
                    fields=normalized_fields,
                    limit=normalized_limit,
                ),
                key_source=key_source,
            )
        except HTTPException:
            raise
        except Exception as exc:
            if mode == "execute" and self._is_duplicate_active_execute_error(exc):
                raise HTTPException(status_code=409, detail="active_execute_run_exists") from exc
            raise
        return MaintenanceRotationRunItem.model_validate(created).model_dump(mode="json")

    async def list_runs(
        self,
        *,
        limit: int,
        offset: int,
        allowed_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return paginated maintenance rotation run history."""
        rows, total = await self.repo.list_runs(
            limit=limit,
            offset=offset,
            allowed_domains=allowed_domains,
        )
        response = MaintenanceRotationRunListResponse(
            items=[MaintenanceRotationRunItem.model_validate(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(rows),
            ),
        )
        return response.model_dump(mode="json")

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a single maintenance rotation run by id."""
        row = await self.repo.get_run(run_id)
        if row is None:
            return None
        return MaintenanceRotationRunItem.model_validate(row).model_dump(mode="json")
