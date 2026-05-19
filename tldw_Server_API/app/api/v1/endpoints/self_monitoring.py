"""
Self-Monitoring API

Endpoints for managing self-awareness monitoring rules, alerts,
governance policies, and crisis resources.

Routes:
- POST   /rules                  Create a monitoring rule
- GET    /rules                  List rules
- GET    /rules/{id}             Get rule details
- PATCH  /rules/{id}             Update rule
- DELETE /rules/{id}             Delete rule
- POST   /rules/{id}/deactivate              Request deactivation (with cooldown)
- POST   /rules/{id}/confirm-deactivation    Confirm deactivation (token)
- POST   /rules/{id}/approve-deactivation    Approve deactivation (partner)
- GET    /alerts                              List alerts
- POST   /alerts/mark-read       Mark alerts as read
- GET    /alerts/unread-count     Count unread alerts
- POST   /governance-policies     Create governance policy
- GET    /governance-policies     List governance policies
- DELETE /governance-policies/{id}  Delete governance policy
- GET    /crisis-resources        Get crisis resources
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from tldw_Server_API.app.api.v1.API_Deps.guardian_deps import get_guardian_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.guardian_schemas import (
    CrisisResource,
    CrisisResourceList,
    DeactivationApproveRequest,
    DeactivationConfirmRequest,
    DetailResponse,
    GovernancePolicyCreate,
    GovernancePolicyList,
    GovernancePolicyResponse,
    MarkAlertsReadRequest,
    SelfMonitoringAlertList,
    SelfMonitoringAlertResponse,
    SelfMonitoringRuleCreate,
    SelfMonitoringRuleList,
    SelfMonitoringRuleResponse,
    SelfMonitoringRuleUpdate,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.Monitoring.self_monitoring_service import (
    CRISIS_DISCLAIMER,
    CRISIS_RESOURCES,
    SelfMonitoringService,
    get_self_monitoring_service,
)

router = APIRouter()


# ── Helper ───────────────────────────────────────────────────

def _user_id(user: User) -> str:
    return str(user.id)


def _get_service(db: GuardianDB) -> SelfMonitoringService:
    return get_self_monitoring_service(db)


# ── Self-Monitoring Rules ────────────────────────────────────

@router.post(
    "/rules",
    response_model=SelfMonitoringRuleResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_rule(
    body: SelfMonitoringRuleCreate,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Create a self-monitoring rule."""
    try:
        rule = db.create_self_monitoring_rule(
            user_id=_user_id(user),
            name=body.name,
            category=body.category,
            patterns=body.patterns,
            pattern_type=body.pattern_type,
            except_patterns=body.except_patterns,
            rule_type=body.rule_type,
            action=body.action,
            phase=body.phase,
            severity=body.severity,
            display_mode=body.display_mode,
            block_message=body.block_message,
            context_note=body.context_note,
            notification_frequency=body.notification_frequency,
            notification_channels=body.notification_channels,
            webhook_url=body.webhook_url,
            trusted_contact_email=body.trusted_contact_email,
            crisis_resources_enabled=body.crisis_resources_enabled,
            cooldown_minutes=body.cooldown_minutes,
            bypass_protection=body.bypass_protection,
            bypass_partner_user_id=body.bypass_partner_user_id,
            escalation_session_threshold=body.escalation_session_threshold,
            escalation_session_action=body.escalation_session_action,
            escalation_window_days=body.escalation_window_days,
            escalation_window_threshold=body.escalation_window_threshold,
            escalation_window_action=body.escalation_window_action,
            min_context_length=body.min_context_length,
            governance_policy_id=body.governance_policy_id,
            enabled=body.enabled,
        )
        svc = _get_service(db)
        svc.invalidate_cache(_user_id(user))
        return _rule_response(rule, svc)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/rules", response_model=SelfMonitoringRuleList)
def list_rules(
    enabled_only: bool = Query(False),
    category: str | None = Query(None),
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """List self-monitoring rules for the current user."""
    rules = db.list_self_monitoring_rules(
        _user_id(user),
        enabled_only=enabled_only,
        category=category,
    )
    svc = _get_service(db)
    items = [_rule_response(r, svc) for r in rules]
    return SelfMonitoringRuleList(items=items, total=len(items))


@router.get("/rules/{rule_id}", response_model=SelfMonitoringRuleResponse)
def get_rule(
    rule_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Get self-monitoring rule details."""
    rule = db.get_self_monitoring_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Not your rule")
    svc = _get_service(db)
    return _rule_response(rule, svc)


@router.patch("/rules/{rule_id}", response_model=SelfMonitoringRuleResponse)
def update_rule(
    rule_id: str,
    body: SelfMonitoringRuleUpdate,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Update a self-monitoring rule."""
    rule = db.get_self_monitoring_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Not your rule")
    try:
        updates = body.model_dump(exclude_unset=True)
        if updates:
            db.update_self_monitoring_rule(rule_id, **updates)
            svc = _get_service(db)
            svc.invalidate_cache(_user_id(user))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    updated = db.get_self_monitoring_rule(rule_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Rule not found after update")
    svc = _get_service(db)
    return _rule_response(updated, svc)


@router.delete("/rules/{rule_id}", response_model=DetailResponse)
def delete_rule(
    rule_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Delete a self-monitoring rule and its alerts."""
    rule = db.get_self_monitoring_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    if rule.user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Not your rule")
    db.delete_self_monitoring_rule(rule_id)
    svc = _get_service(db)
    svc.invalidate_cache(_user_id(user))
    return DetailResponse(detail="Rule deleted")


@router.post("/rules/{rule_id}/deactivate")
def request_deactivation(
    rule_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Request deactivation of a rule (respects bypass protection)."""
    svc = _get_service(db)
    result = svc.request_deactivation(rule_id, _user_id(user))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed"))
    return result


@router.post("/rules/{rule_id}/confirm-deactivation")
def confirm_deactivation(
    rule_id: str,
    body: DeactivationConfirmRequest,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Confirm deactivation of a rule (confirmation bypass mode)."""
    svc = _get_service(db)
    result = svc.confirm_deactivation(rule_id, _user_id(user), body.token)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed"))
    return result


@router.post("/rules/{rule_id}/approve-deactivation")
def approve_deactivation(
    rule_id: str,
    body: DeactivationApproveRequest,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Approve deactivation of a rule (partner_approval bypass mode)."""
    svc = _get_service(db)
    result = svc.approve_deactivation(rule_id, _user_id(user), body.token)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed"))
    return result


# ── Alerts ───────────────────────────────────────────────────

@router.get("/alerts", response_model=SelfMonitoringAlertList)
def list_alerts(
    rule_id: str | None = Query(None),
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """List self-monitoring alerts."""
    alerts = db.list_self_monitoring_alerts(
        _user_id(user),
        rule_id=rule_id,
        unread_only=unread_only,
        limit=limit,
        offset=offset,
    )
    items = [_alert_response(a) for a in alerts]
    total = db.count_self_monitoring_alerts(_user_id(user), rule_id=rule_id, unread_only=unread_only)
    pagination = build_offset_pagination_meta(total=total, limit=limit, offset=offset, count=len(items))
    return SelfMonitoringAlertList(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        pagination=pagination,
    )


@router.post("/alerts/mark-read", response_model=DetailResponse)
def mark_alerts_read(
    body: MarkAlertsReadRequest,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Mark alerts as read."""
    count = db.mark_alerts_read(_user_id(user), body.alert_ids)
    return DetailResponse(detail=f"Marked {count} alerts as read")


@router.get("/alerts/unread-count")
def unread_alert_count(
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Get count of unread alerts."""
    count = db.count_unread_alerts(_user_id(user))
    return {"unread_count": count}


# ── Governance Policies ──────────────────────────────────────

@router.post(
    "/governance-policies",
    response_model=GovernancePolicyResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_governance_policy(
    body: GovernancePolicyCreate,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Create a governance policy (named policy group)."""
    try:
        gp = db.create_governance_policy(
            owner_user_id=_user_id(user),
            name=body.name,
            description=body.description,
            policy_mode=body.policy_mode,
            scope_chat_types=body.scope_chat_types,
            enabled=body.enabled,
            schedule_start=body.schedule_start,
            schedule_end=body.schedule_end,
            schedule_days=body.schedule_days,
            schedule_timezone=body.schedule_timezone,
            transparent=body.transparent,
        )
        return _governance_policy_response(gp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/governance-policies", response_model=GovernancePolicyList)
def list_governance_policies(
    policy_mode: str | None = Query(None),
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """List governance policies for the current user."""
    policies = db.list_governance_policies(_user_id(user), policy_mode=policy_mode)
    items = [_governance_policy_response(gp) for gp in policies]
    return GovernancePolicyList(items=items, total=len(items))


@router.delete("/governance-policies/{policy_id}", response_model=DetailResponse)
def delete_governance_policy(
    policy_id: str,
    user: User = Depends(get_request_user),
    db: GuardianDB = Depends(get_guardian_db_for_user),
):
    """Delete a governance policy."""
    gp = db.get_governance_policy(policy_id)
    if not gp:
        raise HTTPException(status_code=404, detail="Governance policy not found")
    if gp.owner_user_id != _user_id(user):
        raise HTTPException(status_code=403, detail="Not your policy")
    db.delete_governance_policy(policy_id)
    return DetailResponse(detail="Governance policy deleted")


# ── Crisis Resources ─────────────────────────────────────────

@router.get("/crisis-resources", response_model=CrisisResourceList)
def get_crisis_resources():
    """Get crisis helpline resources and disclaimer."""
    resources = [
        CrisisResource(
            name=r["name"],
            description=r["description"],
            contact=r["contact"],
            url=r.get("url"),
            available_24_7=r.get("available_24_7", True),
        )
        for r in CRISIS_RESOURCES
    ]
    return CrisisResourceList(resources=resources, disclaimer=CRISIS_DISCLAIMER)


# ── Response Builders ────────────────────────────────────────

def _rule_response(rule, svc: SelfMonitoringService) -> SelfMonitoringRuleResponse:
    return SelfMonitoringRuleResponse(
        id=rule.id,
        user_id=rule.user_id,
        governance_policy_id=rule.governance_policy_id,
        name=rule.name,
        category=rule.category,
        patterns=rule.patterns,
        pattern_type=rule.pattern_type,
        except_patterns=rule.except_patterns,
        rule_type=rule.rule_type,
        action=rule.action,
        phase=rule.phase,
        severity=rule.severity,
        display_mode=rule.display_mode,
        block_message=rule.block_message,
        context_note=rule.context_note,
        notification_frequency=rule.notification_frequency,
        notification_channels=rule.notification_channels,
        webhook_url=rule.webhook_url,
        trusted_contact_email=rule.trusted_contact_email,
        crisis_resources_enabled=rule.crisis_resources_enabled,
        cooldown_minutes=rule.cooldown_minutes,
        bypass_protection=rule.bypass_protection,
        bypass_partner_user_id=rule.bypass_partner_user_id,
        escalation_session_threshold=rule.escalation_session_threshold,
        escalation_session_action=rule.escalation_session_action,
        escalation_window_days=rule.escalation_window_days,
        escalation_window_threshold=rule.escalation_window_threshold,
        escalation_window_action=rule.escalation_window_action,
        min_context_length=rule.min_context_length,
        enabled=rule.enabled,
        can_disable=svc.can_disable_rule(rule),
        pending_deactivation_at=rule.pending_deactivation_at,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


def _alert_response(alert) -> SelfMonitoringAlertResponse:
    return SelfMonitoringAlertResponse(
        id=alert.id,
        rule_id=alert.rule_id,
        rule_name=alert.rule_name,
        category=alert.category,
        severity=alert.severity,
        matched_pattern=alert.matched_pattern,
        context_snippet=alert.context_snippet,
        snippet_mode=alert.snippet_mode,
        conversation_id=alert.conversation_id,
        session_id=alert.session_id,
        chat_type=alert.chat_type,
        phase=alert.phase,
        action_taken=alert.action_taken,
        notification_sent=alert.notification_sent,
        notification_channels_used=alert.notification_channels_used,
        crisis_resources_shown=alert.crisis_resources_shown,
        display_mode=alert.display_mode,
        escalation_info=alert.escalation_info,
        is_read=alert.is_read,
        created_at=alert.created_at,
    )


def _governance_policy_response(gp) -> GovernancePolicyResponse:
    return GovernancePolicyResponse(
        id=gp.id,
        owner_user_id=gp.owner_user_id,
        name=gp.name,
        description=gp.description,
        policy_mode=gp.policy_mode,
        scope_chat_types=gp.scope_chat_types,
        enabled=gp.enabled,
        schedule_start=gp.schedule_start,
        schedule_end=gp.schedule_end,
        schedule_days=gp.schedule_days,
        schedule_timezone=gp.schedule_timezone,
        transparent=gp.transparent,
        created_at=gp.created_at,
        updated_at=gp.updated_at,
    )
