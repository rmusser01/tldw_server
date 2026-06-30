"""
Guardian Controls API

Endpoints for managing guardian-child relationships and supervised
content moderation policies.

Routes:
- POST   /relationships          Create a guardian link
- GET    /relationships          List relationships (as guardian)
- GET    /relationships/{id}     Get relationship details
- POST   /relationships/{id}/accept   Dependent accepts
- POST   /relationships/{id}/dissolve  Dissolve relationship
- POST   /relationships/{id}/suspend   Suspend relationship
- POST   /relationships/{id}/reactivate  Reactivate relationship
- POST   /policies               Create supervised policy
- GET    /policies               List policies for a relationship
- GET    /policies/{id}          Get policy details
- PATCH  /policies/{id}          Update policy
- DELETE /policies/{id}          Delete policy
- GET    /audit/{relationship_id}  Get audit log
- GET    /dependent/status       Dependent sees their supervision status
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.guardian_deps import (
    get_guardian_db_for_user,
    get_guardian_db_for_user_id,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.guardian_schemas import (
    DetailResponse,
    DissolveRequest,
    GuardianRelationshipCreate,
    GuardianRelationshipList,
    GuardianRelationshipResponse,
    SupervisedPolicyCreate,
    SupervisedPolicyList,
    SupervisedPolicyResponse,
    SupervisedPolicyUpdate,
    SupervisionAuditList,
    SupervisionAuditResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.Moderation.family_wizard_materializer import (
    materialize_pending_plans_for_relationship,
)

router = APIRouter()


# ── Helper ───────────────────────────────────────────────────

def _user_id(user: User) -> str:
    return str(user.id)


# ── Relationships ────────────────────────────────────────────

@router.post(
    "/relationships",
    response_model=GuardianRelationshipResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_relationship(
    body: GuardianRelationshipCreate,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Create a guardian relationship (current user becomes guardian)."""
    try:
        rel = db.create_relationship(
            guardian_user_id=_user_id(user),
            dependent_user_id=body.dependent_user_id,
            relationship_type=body.relationship_type,
            dependent_visible=body.dependent_visible,
        )
        db.log_action(
            relationship_id=rel.id,
            actor_user_id=_user_id(user),
            action="relationship_created",
            target_user_id=body.dependent_user_id,
            detail=f"type={body.relationship_type}",
        )
        return GuardianRelationshipResponse(
            id=rel.id,
            guardian_user_id=rel.guardian_user_id,
            dependent_user_id=rel.dependent_user_id,
            relationship_type=rel.relationship_type,
            status=rel.status,
            consent_given_by_dependent=rel.consent_given_by_dependent,
            consent_given_at=rel.consent_given_at,
            dependent_visible=rel.dependent_visible,
            dissolution_reason=rel.dissolution_reason,
            dissolved_at=rel.dissolved_at,
            created_at=rel.created_at,
            updated_at=rel.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@router.get("/relationships", response_model=GuardianRelationshipList)
def list_relationships(
    role: str = Query("guardian", description="'guardian' or 'dependent'"),
    filter_status: str | None = Query(None, alias="status"),
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """List guardian relationships for the current user."""
    uid = _user_id(user)
    if role == "dependent":
        rels = db.get_relationships_for_dependent(uid, status=filter_status)
    else:
        rels = db.get_relationships_for_guardian(uid, status=filter_status)
    items = [
        GuardianRelationshipResponse(
            id=r.id,
            guardian_user_id=r.guardian_user_id,
            dependent_user_id=r.dependent_user_id,
            relationship_type=r.relationship_type,
            status=r.status,
            consent_given_by_dependent=r.consent_given_by_dependent,
            consent_given_at=r.consent_given_at,
            dependent_visible=r.dependent_visible,
            dissolution_reason=r.dissolution_reason,
            dissolved_at=r.dissolved_at,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rels
    ]
    return GuardianRelationshipList(items=items, total=len(items))


@router.get("/relationships/{relationship_id}", response_model=GuardianRelationshipResponse)
def get_relationship(
    relationship_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Get relationship details (must be guardian or dependent)."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    uid = _user_id(user)
    if rel.guardian_user_id != uid and rel.dependent_user_id != uid:
        raise HTTPException(status_code=403, detail="Not authorized for this relationship")
    return GuardianRelationshipResponse(
        id=rel.id,
        guardian_user_id=rel.guardian_user_id,
        dependent_user_id=rel.dependent_user_id,
        relationship_type=rel.relationship_type,
        status=rel.status,
        consent_given_by_dependent=rel.consent_given_by_dependent,
        consent_given_at=rel.consent_given_at,
        dependent_visible=rel.dependent_visible,
        dissolution_reason=rel.dissolution_reason,
        dissolved_at=rel.dissolved_at,
        created_at=rel.created_at,
        updated_at=rel.updated_at,
    )


@router.post("/relationships/{relationship_id}/accept", response_model=DetailResponse)
def accept_relationship(
    relationship_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Dependent accepts the guardian relationship (gives consent)."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    if rel.dependent_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only the dependent can accept")
    ok = db.accept_relationship(relationship_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot accept (may already be active or dissolved)")
    materialization_db = db
    if rel.guardian_user_id != _user_id(user):
        materialization_db = get_guardian_db_for_user_id(rel.guardian_user_id)
    materialization_result = materialize_pending_plans_for_relationship(
        db=materialization_db,
        relationship_id=relationship_id,
        actor_user_id=_user_id(user),
    )
    db.log_action(
        relationship_id=relationship_id,
        actor_user_id=_user_id(user),
        action="relationship_accepted",
        detail=(
            "consent_given "
            f"materialized={materialization_result['materialized_count']} "
            f"failed={materialization_result['failed_count']}"
        ),
    )
    return DetailResponse(detail="Relationship accepted")


@router.post("/relationships/{relationship_id}/dissolve", response_model=DetailResponse)
def dissolve_relationship(
    relationship_id: str,
    body: DissolveRequest | None = None,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Dissolve a guardian relationship (either party can dissolve)."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    uid = _user_id(user)
    if rel.guardian_user_id != uid and rel.dependent_user_id != uid:
        raise HTTPException(status_code=403, detail="Not authorized")
    reason = (body.reason if body else None) or "manual"
    ok = db.dissolve_relationship(relationship_id, reason=reason)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot dissolve (may already be dissolved)")
    db.log_action(
        relationship_id=relationship_id,
        actor_user_id=uid,
        action="relationship_dissolved",
        detail=f"reason={reason}",
    )
    return DetailResponse(detail="Relationship dissolved")


@router.post("/relationships/{relationship_id}/suspend", response_model=DetailResponse)
def suspend_relationship(
    relationship_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Suspend a guardian relationship (guardian only)."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    if rel.guardian_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only guardian can suspend")
    ok = db.suspend_relationship(relationship_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot suspend (must be active)")
    db.log_action(
        relationship_id=relationship_id,
        actor_user_id=_user_id(user),
        action="relationship_suspended",
    )
    return DetailResponse(detail="Relationship suspended")


@router.post("/relationships/{relationship_id}/reactivate", response_model=DetailResponse)
def reactivate_relationship(
    relationship_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Reactivate a suspended relationship (guardian only)."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    if rel.guardian_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only guardian can reactivate")
    ok = db.reactivate_relationship(relationship_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot reactivate (must be suspended)")
    db.log_action(
        relationship_id=relationship_id,
        actor_user_id=_user_id(user),
        action="relationship_reactivated",
    )
    return DetailResponse(detail="Relationship reactivated")


# ── Supervised Policies ──────────────────────────────────────

@router.post(
    "/policies",
    response_model=SupervisedPolicyResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_policy(
    body: SupervisedPolicyCreate,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Create a supervised policy (guardian only)."""
    rel = db.get_relationship(body.relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    if rel.guardian_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only guardian can create policies")
    if rel.status != "active":
        raise HTTPException(status_code=400, detail="Relationship must be active")
    try:
        pol = db.create_policy(
            relationship_id=body.relationship_id,
            policy_type=body.policy_type,
            category=body.category,
            pattern=body.pattern,
            pattern_type=body.pattern_type,
            action=body.action,
            phase=body.phase,
            severity=body.severity,
            notify_guardian=body.notify_guardian,
            notify_context=body.notify_context,
            message_to_dependent=body.message_to_dependent,
            enabled=body.enabled,
            governance_policy_id=body.governance_policy_id,
        )
        db.log_action(
            relationship_id=body.relationship_id,
            actor_user_id=_user_id(user),
            action="policy_created",
            policy_id=pol.id,
            detail=f"category={body.category} action={body.action}",
        )
        return _policy_response(pol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/policies", response_model=SupervisedPolicyList)
def list_policies(
    relationship_id: str = Query(...),
    enabled_only: bool = Query(False),
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """List supervised policies for a relationship."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    uid = _user_id(user)
    if rel.guardian_user_id != uid and rel.dependent_user_id != uid:
        raise HTTPException(status_code=403, detail="Not authorized")
    policies = db.list_policies_for_relationship(relationship_id, enabled_only=enabled_only)
    items = [_policy_response(p) for p in policies]
    return SupervisedPolicyList(items=items, total=len(items))


@router.get("/policies/{policy_id}", response_model=SupervisedPolicyResponse)
def get_policy(
    policy_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Get policy details."""
    pol = db.get_policy(policy_id)
    if not pol:
        raise HTTPException(status_code=404, detail="Policy not found")
    rel = db.get_relationship(pol.relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    uid = _user_id(user)
    if rel.guardian_user_id != uid and rel.dependent_user_id != uid:
        raise HTTPException(status_code=403, detail="Not authorized")
    return _policy_response(pol)


@router.patch("/policies/{policy_id}", response_model=SupervisedPolicyResponse)
def update_policy(
    policy_id: str,
    body: SupervisedPolicyUpdate,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Update a supervised policy (guardian only)."""
    pol = db.get_policy(policy_id)
    if not pol:
        raise HTTPException(status_code=404, detail="Policy not found")
    rel = db.get_relationship(pol.relationship_id)
    if not rel or rel.guardian_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only guardian can update policies")
    try:
        updates = body.model_dump(exclude_unset=True)
        if updates:
            db.update_policy(policy_id, **updates)
            db.log_action(
                relationship_id=pol.relationship_id,
                actor_user_id=_user_id(user),
                action="policy_updated",
                policy_id=policy_id,
                detail=f"fields={list(updates.keys())}",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    updated = db.get_policy(policy_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Policy not found after update")
    return _policy_response(updated)


@router.delete("/policies/{policy_id}", response_model=DetailResponse)
def delete_policy(
    policy_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Delete a supervised policy (guardian only)."""
    pol = db.get_policy(policy_id)
    if not pol:
        raise HTTPException(status_code=404, detail="Policy not found")
    rel = db.get_relationship(pol.relationship_id)
    if not rel or rel.guardian_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only guardian can delete policies")
    db.delete_policy(policy_id)
    db.log_action(
        relationship_id=pol.relationship_id,
        actor_user_id=_user_id(user),
        action="policy_deleted",
        policy_id=policy_id,
    )
    return DetailResponse(detail="Policy deleted")


# ── Audit Log ────────────────────────────────────────────────

@router.get("/audit/{relationship_id}", response_model=SupervisionAuditList)
def get_audit_log(
    relationship_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Get audit log for a relationship (guardian only)."""
    rel = db.get_relationship(relationship_id)
    if not rel:
        raise HTTPException(status_code=404, detail="Relationship not found")
    if rel.guardian_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Only guardian can view audit log")
    entries = db.get_audit_log(relationship_id, limit=limit, offset=offset)
    total = db.count_audit_entries(relationship_id)
    items = [
        SupervisionAuditResponse(
            id=e.id,
            relationship_id=e.relationship_id,
            actor_user_id=e.actor_user_id,
            action=e.action,
            target_user_id=e.target_user_id,
            policy_id=e.policy_id,
            detail=e.detail,
            created_at=e.created_at,
        )
        for e in entries
    ]
    pagination = build_offset_pagination_meta(total=total, limit=limit, offset=offset, count=len(items))
    return SupervisionAuditList(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        pagination=pagination,
    )


# ── Dependent Status ─────────────────────────────────────────

@router.get("/dependent/status")
def dependent_supervision_status(
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Dependent user checks if they have active guardians and what's visible."""
    uid = _user_id(user)
    rels = db.get_relationships_for_dependent(uid, status="active")
    if not rels:
        return {"supervised": False, "guardians": []}
    guardians = []
    for rel in rels:
        info = {"relationship_id": rel.id, "relationship_type": rel.relationship_type}
        if rel.dependent_visible:
            info["monitoring_active"] = True
            policies = db.list_policies_for_relationship(rel.id, enabled_only=True)
            info["policy_count"] = len(policies)
            info["categories"] = list({p.category for p in policies if p.category})
        guardians.append(info)
    return {"supervised": True, "guardians": guardians}


# ── Helpers ──────────────────────────────────────────────────

def _policy_response(pol) -> SupervisedPolicyResponse:
    return SupervisedPolicyResponse(
        id=pol.id,
        relationship_id=pol.relationship_id,
        governance_policy_id=pol.governance_policy_id,
        policy_type=pol.policy_type,
        category=pol.category,
        pattern=pol.pattern,
        pattern_type=pol.pattern_type,
        action=pol.action,
        phase=pol.phase,
        severity=pol.severity,
        notify_guardian=pol.notify_guardian,
        notify_context=pol.notify_context,
        message_to_dependent=pol.message_to_dependent,
        enabled=pol.enabled,
        created_at=pol.created_at,
        updated_at=pol.updated_at,
    )
