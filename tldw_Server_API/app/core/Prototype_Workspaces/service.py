"""Orchestration service for prototype workspace runtime and promotion flows."""
from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
    PrototypeWorkspacesRepo,
)

from .models import (
    PrototypePromotionResult,
    PrototypeRuntimeStatus,
)
from .preview_broker import PrototypePreviewBroker


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _normalize_promoter_ids(raw: Any) -> set[int]:
    if not isinstance(raw, list):
        return set()
    out: set[int] = set()
    for item in raw:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


class PrototypeWorkspaceService:
    """Coordinate branch sessions, snapshots, previews, and promotions."""

    def __init__(
        self,
        *,
        repo: PrototypeWorkspacesRepo,
        preview_broker: PrototypePreviewBroker | None = None,
        publish_validator: Any | None = None,
    ) -> None:
        self._repo = repo
        self._preview_broker = preview_broker or PrototypePreviewBroker(repo=repo)
        self._publish_validator = publish_validator

    async def create_workspace(
        self,
        *,
        owner_user_id: int,
        title: str,
        creation_source: str,
        description: str | None = None,
        prompt: str | None = None,
        preview_policy: dict[str, Any] | None = None,
        share_policy: dict[str, Any] | None = None,
        runtime_policy: dict[str, Any] | None = None,
        designated_promoter_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        async with self._repo.transaction() as repo:
            workspace = await repo.create_workspace(
                owner_user_id=int(owner_user_id),
                title=title,
                description=description,
                creation_source=creation_source,
                preview_policy=preview_policy,
                share_policy=share_policy,
                runtime_policy=runtime_policy,
                designated_promoter_ids=designated_promoter_ids,
            )
            seed_snapshot = await repo.create_snapshot(
                prototype_workspace_id=workspace["id"],
                snapshot_id=f"psnap_{uuid.uuid4().hex}",
                created_by_user_id=int(owner_user_id),
                storage_ref="prototype://seed",
                prompt_summary=prompt,
                diff_summary={"creation_source": creation_source},
            )
            updated = await repo.update_workspace_state(
                workspace["id"],
                canonical_snapshot_id=seed_snapshot["snapshot_id"],
                last_known_good_snapshot_id=seed_snapshot["snapshot_id"],
                canonical_preview_status="uninitialized",
                publish_validation_status="pending",
            )
            if not updated or updated.get("canonical_snapshot_id") != seed_snapshot["snapshot_id"]:
                raise RuntimeError("failed to persist prototype workspace seed snapshot")
            return updated

    async def create_or_reuse_branch_session(
        self,
        *,
        prototype_workspace_id: str,
        actor_type: str,
        actor_user_id: int | None = None,
        actor_shared_actor_id: str | None = None,
        request_nonce: str | None = None,
        share_link_id: int | None = None,
        expires_at: str | None = None,
        base_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise ValueError("prototype workspace not found")
        if workspace.get("is_archived"):
            raise RuntimeError("archived workspaces cannot create branch sessions")

        resolved_base_snapshot_id = str(
            base_snapshot_id
            or workspace.get("canonical_snapshot_id")
            or workspace.get("last_known_good_snapshot_id")
            or ""
        ).strip()
        if not resolved_base_snapshot_id:
            raise ValueError("prototype workspace does not have a canonical snapshot")
        await self._assert_branch_actor_active(
            actor_type=actor_type,
            actor_shared_actor_id=actor_shared_actor_id,
        )

        existing = await self._repo.find_active_session(
            prototype_workspace_id=prototype_workspace_id,
            base_snapshot_id=resolved_base_snapshot_id,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            actor_shared_actor_id=actor_shared_actor_id,
            share_link_id=share_link_id,
        )
        if existing:
            return {
                "created": False,
                "request_nonce": request_nonce,
                "session": existing,
            }

        session = await self._repo.create_session(
            prototype_workspace_id=prototype_workspace_id,
            base_snapshot_id=resolved_base_snapshot_id,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            actor_shared_actor_id=actor_shared_actor_id,
            share_link_id=share_link_id,
            expires_at=expires_at,
        )
        updated = await self._repo.update_session_state(
            session["id"],
            runtime_status=PrototypeRuntimeStatus.QUEUED.value,
            last_activity_at=_utc_now(),
        )
        return {
            "created": True,
            "request_nonce": request_nonce,
            "session": updated or session,
        }

    async def boot_preview(
        self,
        *,
        prototype_workspace_id: str,
        snapshot_id: str,
        runtime_target_url: str,
        prototype_session_id: str | None = None,
        runtime_policy_profile: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._preview_broker.issue_preview_grant(
            prototype_workspace_id=prototype_workspace_id,
            prototype_session_id=prototype_session_id,
            snapshot_id=snapshot_id,
            runtime_target_url=runtime_target_url,
            runtime_policy_profile=runtime_policy_profile,
            metadata=metadata,
        )

    async def renew_preview_grant(self, *, preview_handle: str) -> dict[str, Any]:
        return await self._preview_broker.renew_preview_grant(preview_handle)

    async def save_session_snapshot(
        self,
        *,
        prototype_session_id: str,
        snapshot_id: str | None = None,
        storage_ref: str | None = None,
        diff_summary: dict[str, Any] | None = None,
        prompt_summary: str | None = None,
        preview_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._repo.transaction() as repo:
            session = await repo.get_session(prototype_session_id)
            if not session:
                raise ValueError("prototype session not found")
            workspace = await repo.get_workspace(str(session["prototype_workspace_id"]))
            if not workspace:
                raise ValueError("prototype workspace not found")
            if workspace.get("is_archived"):
                raise RuntimeError("archived workspaces cannot save snapshots")
            await self._assert_session_is_active(session, repo=repo)

            new_snapshot_id = str(snapshot_id or f"psnap_{uuid.uuid4().hex}")
            snapshot = await repo.create_snapshot(
                prototype_workspace_id=str(session["prototype_workspace_id"]),
                snapshot_id=new_snapshot_id,
                created_by_user_id=session.get("actor_user_id"),
                created_by_shared_actor_id=session.get("actor_shared_actor_id"),
                parent_snapshot_id=str(
                    session.get("last_saved_snapshot_id")
                    or session.get("base_snapshot_id")
                    or ""
                ).strip()
                or None,
                created_from_session_id=prototype_session_id,
                storage_ref=storage_ref,
                diff_summary=diff_summary,
                prompt_summary=prompt_summary,
                preview_health=preview_health,
            )
            updated_session = await repo.update_session_state(
                prototype_session_id,
                last_saved_snapshot_id=snapshot["snapshot_id"],
                last_activity_at=_utc_now(),
            )
            if not updated_session or updated_session.get("last_saved_snapshot_id") != snapshot["snapshot_id"]:
                raise RuntimeError("failed to persist session snapshot state")
            return snapshot

    async def promote_candidate(
        self,
        *,
        prototype_workspace_id: str,
        candidate_snapshot_id: str,
        reviewer_user_id: int,
        review_baseline_snapshot_id: str | None = None,
        promotion_request_id: str | None = None,
        review_notes: str | None = None,
    ) -> dict[str, Any]:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise ValueError("prototype workspace not found")

        reviewer_id = int(reviewer_user_id)
        if not self._is_promoter(workspace, reviewer_id):
            raise PermissionError("reviewer does not have prototype.promote permission")

        candidate = await self._repo.get_snapshot(candidate_snapshot_id)
        if not candidate or candidate.get("prototype_workspace_id") != prototype_workspace_id:
            raise ValueError("candidate snapshot not found in prototype workspace")

        promotion_request = None
        if promotion_request_id:
            promotion_request = await self._repo.get_promotion_request(promotion_request_id)
            if not promotion_request:
                raise ValueError("promotion request not found")
            if promotion_request.get("prototype_workspace_id") != prototype_workspace_id:
                raise ValueError("promotion request does not belong to prototype workspace")
            if promotion_request.get("candidate_snapshot_id") != candidate_snapshot_id:
                raise ValueError("promotion request candidate does not match requested candidate")

        canonical_snapshot_id = str(workspace.get("canonical_snapshot_id") or "").strip()
        resolved_review_baseline_snapshot_id = str(
            review_baseline_snapshot_id
            or workspace.get("last_known_good_snapshot_id")
            or canonical_snapshot_id
            or ""
        ).strip()

        stale_result = await self._detect_stale_candidate(
            candidate=candidate,
            review_baseline_snapshot_id=resolved_review_baseline_snapshot_id,
        )
        if stale_result:
            if promotion_request:
                await self._repo.update_promotion_request(
                    promotion_request["id"],
                    status="stale",
                    reviewed_by_user_id=reviewer_id,
                    review_notes=review_notes or "Candidate is stale against the canonical snapshot",
                )
            return PrototypePromotionResult(
                status="stale",
                failure_code="stale_candidate",
                prototype_workspace_id=prototype_workspace_id,
                candidate_snapshot_id=candidate_snapshot_id,
                canonical_snapshot_id=canonical_snapshot_id or None,
            ).to_dict()

        validation = await self._run_publish_validator(
            workspace=workspace,
            candidate=candidate,
            reviewer_user_id=reviewer_id,
            canonical_snapshot_id=canonical_snapshot_id or None,
            review_baseline_snapshot_id=resolved_review_baseline_snapshot_id or None,
            promotion_request=promotion_request,
        )
        validation_ok = bool(validation.get("ok"))
        if not validation_ok:
            failure_reason = str(
                validation.get("reason")
                or validation.get("failure_code")
                or "publish validation failed"
            )
            await self._repo.update_workspace_state(
                prototype_workspace_id,
                publish_validation_status="failed",
            )
            if promotion_request:
                await self._repo.update_promotion_request(
                    promotion_request["id"],
                    status="rejected",
                    reviewed_by_user_id=reviewer_id,
                    review_notes=review_notes or failure_reason,
                )
            return PrototypePromotionResult(
                status="failed",
                failure_code="publish_validation_failed",
                prototype_workspace_id=prototype_workspace_id,
                candidate_snapshot_id=candidate_snapshot_id,
                canonical_snapshot_id=canonical_snapshot_id or None,
                details={"reason": failure_reason},
            ).to_dict()

        previous_workspace_state = {
            "canonical_snapshot_id": workspace.get("canonical_snapshot_id"),
            "last_known_good_snapshot_id": workspace.get("last_known_good_snapshot_id"),
            "canonical_preview_status": workspace.get("canonical_preview_status"),
            "publish_validation_status": workspace.get("publish_validation_status"),
        }
        preview_grant = await self._preview_broker.issue_preview_grant(
            prototype_workspace_id=prototype_workspace_id,
            snapshot_id=candidate_snapshot_id,
            runtime_target_url=str(
                validation.get("runtime_target_url")
                or f"runtime://canonical/{prototype_workspace_id}/{candidate_snapshot_id}"
            ),
            metadata={
                "candidate_snapshot_id": candidate_snapshot_id,
                "validation_mode": "publish",
            },
        )
        try:
            updated_workspace = await self._repo.update_workspace_state(
                prototype_workspace_id,
                canonical_snapshot_id=candidate_snapshot_id,
                last_known_good_snapshot_id=candidate_snapshot_id,
                canonical_preview_status="ready",
                publish_validation_status="validated",
            )
            if not updated_workspace or updated_workspace.get("canonical_snapshot_id") != candidate_snapshot_id:
                raise RuntimeError("failed to persist canonical workspace update")
            if promotion_request:
                updated_request = await self._repo.update_promotion_request(
                    promotion_request["id"],
                    status="promoted",
                    reviewed_by_user_id=reviewer_id,
                    review_notes=review_notes,
                )
                if not updated_request or updated_request.get("status") != "promoted":
                    raise RuntimeError("failed to persist promotion request update")
        except Exception:
            await self._preview_broker.revoke_preview_handle(preview_grant["preview_handle"])
            await self._repo.update_workspace_state(
                prototype_workspace_id,
                canonical_snapshot_id=previous_workspace_state["canonical_snapshot_id"],
                last_known_good_snapshot_id=previous_workspace_state["last_known_good_snapshot_id"],
                canonical_preview_status=previous_workspace_state["canonical_preview_status"],
                publish_validation_status=previous_workspace_state["publish_validation_status"],
            )
            raise

        return PrototypePromotionResult(
            status="promoted",
            prototype_workspace_id=prototype_workspace_id,
            candidate_snapshot_id=candidate_snapshot_id,
            canonical_snapshot_id=candidate_snapshot_id,
            preview_handle=preview_grant["preview_handle"],
            details={"preview_url": preview_grant["preview_url"]},
        ).to_dict()

    async def _detect_stale_candidate(
        self,
        *,
        candidate: dict[str, Any],
        review_baseline_snapshot_id: str,
    ) -> bool:
        if not review_baseline_snapshot_id:
            return False

        session_id = str(candidate.get("created_from_session_id") or "").strip()
        if session_id:
            session = await self._repo.get_session(session_id)
            if not session:
                return True
            return str(session.get("base_snapshot_id") or "").strip() != review_baseline_snapshot_id

        parent_snapshot_id = str(candidate.get("parent_snapshot_id") or "").strip()
        if parent_snapshot_id:
            return parent_snapshot_id != review_baseline_snapshot_id
        return False

    async def _run_publish_validator(self, **kwargs: Any) -> dict[str, Any]:
        validator = self._publish_validator
        if validator is None:
            return {"ok": True}

        candidate = None
        if hasattr(validator, "validate_publish_candidate"):
            candidate = validator.validate_publish_candidate(**kwargs)
        elif hasattr(validator, "validate"):
            candidate = validator.validate(**kwargs)
        elif callable(validator):
            candidate = validator(**kwargs)
        else:
            return {"ok": False, "reason": "publish validator is not callable"}

        if inspect.isawaitable(candidate):
            candidate = await candidate

        if isinstance(candidate, bool):
            return {"ok": bool(candidate)}
        if isinstance(candidate, dict):
            return dict(candidate)
        return {"ok": bool(candidate)}

    @staticmethod
    def _is_promoter(workspace: dict[str, Any], reviewer_user_id: int) -> bool:
        owner_user_id = int(workspace.get("owner_user_id"))
        if reviewer_user_id == owner_user_id:
            return True
        return reviewer_user_id in _normalize_promoter_ids(workspace.get("designated_promoter_ids"))

    async def _assert_branch_actor_active(
        self,
        *,
        actor_type: str,
        actor_shared_actor_id: str | None,
    ) -> None:
        if str(actor_type or "").strip().lower() != "external_collaborator":
            return
        actor = await self._repo.get_shared_actor(str(actor_shared_actor_id or ""))
        if not actor or actor.get("is_revoked") or actor.get("revoked_at"):
            raise RuntimeError("revoked shared actor cannot create or reuse branch sessions")
        expires_at = _normalize_datetime(actor.get("expires_at"))
        if expires_at and expires_at <= datetime.now(timezone.utc):
            raise RuntimeError("expired shared actor cannot create or reuse branch sessions")

    async def _assert_session_is_active(
        self,
        session: dict[str, Any],
        *,
        repo: PrototypeWorkspacesRepo | None = None,
    ) -> None:
        if session.get("is_revoked") or session.get("revoked_at"):
            raise RuntimeError("revoked session cannot save snapshots")
        expires_at = _normalize_datetime(session.get("expires_at"))
        if expires_at and expires_at <= datetime.now(timezone.utc):
            raise RuntimeError("expired session cannot save snapshots")
        actor_type = str(session.get("actor_type") or "").strip().lower()
        if actor_type != "external_collaborator":
            return
        actor_repo = repo or self._repo
        actor = await actor_repo.get_shared_actor(str(session.get("actor_shared_actor_id") or ""))
        if not actor or actor.get("is_revoked") or actor.get("revoked_at"):
            raise RuntimeError("revoked shared actor cannot save snapshots")
        actor_expires_at = _normalize_datetime(actor.get("expires_at"))
        if actor_expires_at and actor_expires_at <= datetime.now(timezone.utc):
            raise RuntimeError("expired shared actor cannot save snapshots")


class PrototypePromotionService(PrototypeWorkspaceService):
    """Compatibility alias for tests and future focused promotion entrypoints."""
