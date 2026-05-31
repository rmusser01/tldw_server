from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import AdminPrincipal, get_auth_principal, get_request_user, RequirePermission, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.claims_schemas import (
    ClaimNotificationResponse,
    ClaimNotificationsAckRequest,
    ClaimNotificationsDigestResponse,
    ClaimReviewBulkRequest,
    ClaimReviewRequest,
    ClaimReviewRuleCreate,
    ClaimReviewRuleUpdate,
    ClaimsAlertConfigCreate,
    ClaimsAlertConfigResponse,
    ClaimsAlertConfigUpdate,
    ClaimsAnalyticsDashboardResponse,
    ClaimsAnalyticsExportListResponse,
    ClaimsAnalyticsExportRequest,
    ClaimsAnalyticsExportResponse,
    ClaimsClusterLinkCreate,
    ClaimsClusterLinkResponse,
    ClaimsExtractorCatalogResponse,
    ClaimsMonitoringSettingsResponse,
    ClaimsMonitoringSettingsUpdate,
    ClaimsReviewExtractorMetricsResponse,
    ClaimsSearchResponse,
    ClaimsSettingsResponse,
    ClaimsSettingsUpdate,
    ClaimUpdateRequest,
    FVASettingsResponse,
    FVAVerifyRequest,
    FVAVerifyResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import get_claims_rebuild_service

router = APIRouter(prefix="/claims", tags=["claims"])


@router.get("/status")
def claims_rebuild_status(_principal: AdminPrincipal) -> dict[str, Any]:
    """Return statistics about the claims rebuild worker. Admin only."""
    return claims_service.claims_rebuild_status(
        rebuild_service=get_claims_rebuild_service(),
    )


@router.get("")
def list_all_claims(
    media_id: Optional[int] = None,
    review_status: Optional[str] = None,
    reviewer_id: Optional[int] = None,
    review_group: Optional[str] = None,
    claim_cluster_id: Optional[int] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    include_deleted: bool = Query(False),
    envelope: bool = Query(False),
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """List claims across accessible media for the current user."""
    return claims_service.list_all_claims(
        media_id=media_id,
        review_status=review_status,
        reviewer_id=reviewer_id,
        review_group=review_group,
        claim_cluster_id=claim_cluster_id,
        limit=limit,
        offset=offset,
        include_deleted=include_deleted,
        envelope=envelope,
        user_id=user_id,
        current_user=current_user,
        db=db,
    )


@router.get("/notifications", response_model=list[ClaimNotificationResponse])
def list_claim_notifications(
    kind: Optional[str] = None,
    target_user_id: Optional[str] = None,
    target_review_group: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    delivered: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> list[dict[str, Any]]:
    """List claim notifications visible to the caller."""
    return claims_service.list_claim_notifications(
        kind=kind,
        target_user_id=target_user_id,
        target_review_group=target_review_group,
        resource_type=resource_type,
        resource_id=resource_id,
        delivered=delivered,
        limit=limit,
        offset=offset,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/notifications/digest", response_model=ClaimNotificationsDigestResponse)
def claim_notifications_digest(
    kind: Optional[str] = None,
    target_user_id: Optional[str] = None,
    target_review_group: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    delivered: Optional[bool] = None,
    include_items: bool = Query(False),
    ack: bool = Query(False),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0, le=100000),
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return aggregated counts and optional items for claim notifications."""
    return claims_service.claim_notifications_digest(
        kind=kind,
        target_user_id=target_user_id,
        target_review_group=target_review_group,
        resource_type=resource_type,
        resource_id=resource_id,
        delivered=delivered,
        limit=limit,
        offset=offset,
        include_items=include_items,
        ack=ack,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/notifications/ack")
def claim_notifications_ack(
    payload: ClaimNotificationsAckRequest,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Mark claim notifications as delivered."""
    return claims_service.mark_claim_notifications_delivered(
        ids=payload.ids,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/notifications/watchlists/evaluate")
def evaluate_watchlist_notifications(
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Evaluate watchlist cluster subscriptions and emit notifications."""
    return claims_service.evaluate_watchlist_cluster_notifications(
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/settings", response_model=ClaimsSettingsResponse)
def get_claims_settings(
    _principal: AdminPrincipal,
) -> ClaimsSettingsResponse:
    """Return current claims settings."""
    return claims_service.get_claims_settings(_principal)


@router.put("/settings", response_model=ClaimsSettingsResponse)
def update_claims_settings(
    payload: ClaimsSettingsUpdate,
    principal: AdminPrincipal,
    _perm: AuthPrincipal = Depends(RequirePermission(SYSTEM_CONFIGURE)),  # noqa: B008
) -> ClaimsSettingsResponse:
    """Update claims settings (optionally persisted)."""
    return claims_service.update_claims_settings(
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
    )


@router.get("/monitoring/config", response_model=ClaimsMonitoringSettingsResponse)
def get_claims_monitoring_config(
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> ClaimsMonitoringSettingsResponse:
    """Return claims monitoring configuration (admin only)."""
    return claims_service.get_claims_monitoring_config(
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.patch("/monitoring/config", response_model=ClaimsMonitoringSettingsResponse)
def update_claims_monitoring_config(
    payload: ClaimsMonitoringSettingsUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> ClaimsMonitoringSettingsResponse:
    """Update claims monitoring configuration (optionally persisted)."""
    return claims_service.update_claims_monitoring_config(
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/alerts", response_model=list[ClaimsAlertConfigResponse])
def list_claims_alerts(
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> list[ClaimsAlertConfigResponse]:
    """List claims alert configs for the current user (admin can override user_id)."""
    return claims_service.list_claims_alerts(
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/alerts", response_model=ClaimsAlertConfigResponse)
def create_claims_alert(
    payload: ClaimsAlertConfigCreate,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> ClaimsAlertConfigResponse:
    """Create a claims alert config."""
    return claims_service.create_claims_alert(
        payload=payload.model_dump(exclude_unset=True),
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.patch("/alerts/{config_id}", response_model=ClaimsAlertConfigResponse)
def update_claims_alert(
    config_id: int,
    payload: ClaimsAlertConfigUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> ClaimsAlertConfigResponse:
    """Update a claims alert config."""
    return claims_service.update_claims_alert(
        config_id=config_id,
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.delete("/alerts/{config_id}")
def delete_claims_alert(
    config_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Delete a claims alert config."""
    return claims_service.delete_claims_alert(
        config_id=config_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/alerts/evaluate")
def evaluate_claims_alerts(
    window_sec: int = 3600,
    baseline_sec: int = 86400,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Evaluate claims alert ratios and optionally dispatch notifications."""
    return claims_service.evaluate_claims_alerts(
        window_sec=window_sec,
        baseline_sec=baseline_sec,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/rebuild/health")
def claims_rebuild_health(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Return health for the claims rebuild worker."""
    return claims_service.claims_rebuild_health(principal)


@router.get("/review-queue")
def get_review_queue(
    status_filter: Optional[str] = None,
    reviewer_id: Optional[int] = None,
    review_group: Optional[str] = None,
    media_id: Optional[int] = None,
    extractor: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    include_deleted: bool = Query(False),
    envelope: bool = Query(False),
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """Return claims queued for review."""
    return claims_service.get_review_queue(
        status_filter=status_filter,
        reviewer_id=reviewer_id,
        review_group=review_group,
        media_id=media_id,
        extractor=extractor,
        limit=limit,
        offset=offset,
        include_deleted=include_deleted,
        envelope=envelope,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.patch("/{claim_id}/review")
async def review_claim(
    claim_id: int,
    payload: ClaimReviewRequest,
    request: Request,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Update claim review status and notes."""
    return await claims_service.review_claim(
        claim_id=claim_id,
        payload=payload.model_dump(),
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
        request=request,
    )


@router.get("/{claim_id}/history")
def get_claim_review_history(
    claim_id: int,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> list[dict[str, Any]]:
    """Return review history for a claim."""
    return claims_service.get_claim_review_history(
        claim_id=claim_id,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/review/bulk")
def bulk_review_claims(
    payload: ClaimReviewBulkRequest,
    request: Request,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Bulk review update (admin only)."""
    return claims_service.bulk_review_claims(
        payload=payload.model_dump(),
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
        request=request,
    )


@router.get("/review/rules")
def list_review_rules(
    user_id: Optional[int] = None,
    active_only: bool = Query(False),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> list[dict[str, Any]]:
    """List claim review rules."""
    return claims_service.list_review_rules(
        user_id=user_id,
        active_only=active_only,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/review/rules")
def create_review_rule(
    payload: ClaimReviewRuleCreate,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Create a claim review rule."""
    return claims_service.create_review_rule(
        payload=payload.model_dump(exclude_unset=True),
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.patch("/review/rules/{rule_id}")
def update_review_rule(
    rule_id: int,
    payload: ClaimReviewRuleUpdate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Update a claim review rule."""
    return claims_service.update_review_rule(
        rule_id=rule_id,
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.delete("/review/rules/{rule_id}")
def delete_review_rule(
    rule_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Delete a claim review rule."""
    return claims_service.delete_review_rule(
        rule_id=rule_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/review/analytics")
def review_analytics(
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return summary review analytics."""
    return claims_service.review_analytics(principal, db)


@router.get("/extractors", response_model=ClaimsExtractorCatalogResponse)
def list_claims_extractors(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Return the available claims extractor catalog."""
    return claims_service.list_claims_extractors(principal)


@router.get("/review/metrics", response_model=ClaimsReviewExtractorMetricsResponse)
def list_review_metrics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    extractor: Optional[str] = None,
    extractor_version: Optional[str] = None,
    user_id: Optional[int] = None,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0, le=100000),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return daily review metrics grouped by extractor."""
    return claims_service.list_claims_review_metrics(
        start_date=start_date,
        end_date=end_date,
        extractor=extractor,
        extractor_version=extractor_version,
        user_id=user_id,
        limit=limit,
        offset=offset,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/analytics/dashboard", response_model=ClaimsAnalyticsDashboardResponse)
def claims_dashboard_analytics(
    window_days: int = Query(7, ge=1, le=365),
    window_sec: int = Query(3600, ge=60, le=604800),
    baseline_sec: int = Query(86400, ge=60, le=2592000),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return dashboard-ready claims analytics."""
    return claims_service.claims_dashboard_analytics(
        window_days=window_days,
        window_sec=window_sec,
        baseline_sec=baseline_sec,
        principal=principal,
        db=db,
    )


@router.post("/analytics/export", response_model=ClaimsAnalyticsExportResponse)
def export_claims_analytics(
    payload: ClaimsAnalyticsExportRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """Export claims analytics in JSON or CSV."""
    return claims_service.export_claims_analytics(
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/analytics/exports", response_model=ClaimsAnalyticsExportListResponse)
def list_claims_analytics_exports(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    status: Optional[str] = None,
    format_filter: Optional[str] = Query(None, alias="format"),
    workspace_id: Optional[str] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """List available claims analytics exports."""
    return claims_service.list_claims_analytics_exports(
        limit=limit,
        offset=offset,
        status_filter=status,
        format_filter=format_filter,
        workspace_id=workspace_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get(
    "/analytics/export/{export_id}",
    responses={
        200: {
            "description": "Prepared claims analytics export as JSON or CSV.",
            "content": {
                "application/json": {},
                "text/csv": {},
            },
        },
    },
)
def download_claims_analytics_export(
    export_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """Download a prepared claims analytics export."""
    result = claims_service.get_claims_analytics_export(
        export_id=export_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )
    if result.get("format") == "csv":
        return Response(content=str(result.get("payload") or ""), media_type="text/csv")
    return result.get("payload") or {}


@router.get("/clusters")
def list_claim_clusters(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    updated_since: Optional[str] = Query(None, alias="since"),
    keyword: Optional[str] = None,
    min_size: Optional[int] = Query(None, ge=1),
    watchlisted: Optional[bool] = None,
    envelope: bool = Query(False),
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """List cluster summaries, optionally filtered by timeframe or keyword."""
    return claims_service.list_claim_clusters(
        limit=limit,
        offset=offset,
        updated_since=updated_since,
        keyword=keyword,
        min_size=min_size,
        watchlisted=watchlisted,
        envelope=envelope,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/clusters/rebuild")
def rebuild_claim_clusters(
    min_size: int = Query(2, ge=1, le=1000),
    method: Optional[str] = Query(None, description="embeddings or exact"),
    similarity_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Rebuild claim clusters using stored embeddings."""
    return claims_service.rebuild_claim_clusters(
        min_size=min_size,
        method=method,
        similarity_threshold=similarity_threshold,
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/clusters/{cluster_id}")
def get_claim_cluster(
    cluster_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return a cluster summary."""
    return claims_service.get_claim_cluster(
        cluster_id=cluster_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/clusters/{cluster_id}/links", response_model=list[ClaimsClusterLinkResponse])
def list_claim_cluster_links(
    cluster_id: int,
    direction: str = Query("both", pattern="^(both|inbound|outbound|parent|child)$"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> list[dict[str, Any]]:
    """List cluster relationships."""
    return claims_service.list_claim_cluster_links(
        cluster_id=cluster_id,
        direction=direction,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/clusters/{cluster_id}/links", response_model=ClaimsClusterLinkResponse)
def create_claim_cluster_link(
    cluster_id: int,
    payload: ClaimsClusterLinkCreate,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Create a cluster relationship."""
    return claims_service.create_claim_cluster_link(
        cluster_id=cluster_id,
        payload=payload.model_dump(exclude_unset=True),
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.delete("/clusters/{cluster_id}/links/{child_cluster_id}")
def delete_claim_cluster_link(
    cluster_id: int,
    child_cluster_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Delete a cluster relationship."""
    return claims_service.delete_claim_cluster_link(
        cluster_id=cluster_id,
        child_cluster_id=child_cluster_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/clusters/{cluster_id}/members")
def list_claim_cluster_members(
    cluster_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    envelope: bool = Query(False),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """Return cluster members."""
    return claims_service.list_claim_cluster_members(
        cluster_id=cluster_id,
        limit=limit,
        offset=offset,
        envelope=envelope,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/clusters/{cluster_id}/timeline")
def claim_cluster_timeline(
    cluster_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    envelope: bool = Query(False),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return aggregated cluster timeline."""
    return claims_service.claim_cluster_timeline(
        cluster_id=cluster_id,
        limit=limit,
        offset=offset,
        envelope=envelope,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/clusters/{cluster_id}/evidence")
def claim_cluster_evidence(
    cluster_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    envelope: bool = Query(False),
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Return aggregated evidence for a cluster."""
    return claims_service.claim_cluster_evidence(
        cluster_id=cluster_id,
        limit=limit,
        offset=offset,
        envelope=envelope,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.get("/search", response_model=ClaimsSearchResponse)
def search_claims(
    q: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    group_by_cluster: bool = Query(False),
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Search claims using the FTS index."""
    return claims_service.search_claims(
        query=q,
        limit=limit,
        offset=offset,
        group_by_cluster=group_by_cluster,
        user_id=user_id,
        current_user=current_user,
        db=db,
    )


@router.get("/{media_id}")
def list_claims(
    media_id: int,
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0, le=100000),
    envelope: bool = Query(False),
    absolute_links: bool = Query(False),
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> Any:
    """List claims for a media item."""
    return claims_service.list_claims_by_media(
        media_id=media_id,
        limit=limit,
        offset=offset,
        envelope=envelope,
        absolute_links=absolute_links,
        user_id=user_id,
        current_user=current_user,
        db=db,
        request=request,
    )


@router.get("/items/{claim_id}")
def get_claim_item(
    claim_id: int,
    include_deleted: bool = Query(False),
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Fetch a single claim by id."""
    return claims_service.get_claim_item(
        claim_id=claim_id,
        include_deleted=include_deleted,
        user_id=user_id,
        current_user=current_user,
        db=db,
    )


@router.patch("/items/{claim_id}")
async def update_claim_item(
    claim_id: int,
    payload: ClaimUpdateRequest,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Update a claim entry."""
    return await claims_service.update_claim_item(
        claim_id=claim_id,
        payload=payload.model_dump(exclude_unset=True),
        user_id=user_id,
        principal=principal,
        current_user=current_user,
        db=db,
    )


@router.post("/{media_id}/rebuild")
def rebuild_claims(
    media_id: int,
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Enqueue a claims rebuild for a media item."""
    return claims_service.rebuild_claims(
        media_id=media_id,
        user_id=user_id,
        current_user=current_user,
        db=db,
        rebuild_service=get_claims_rebuild_service(),
    )


@router.post("/rebuild/all")
def rebuild_all_media(
    policy: str = "missing",
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Enqueue rebuild tasks for all media based on policy."""
    return claims_service.rebuild_all_media(
        policy=policy,
        user_id=user_id,
        current_user=current_user,
        db=db,
        rebuild_service=get_claims_rebuild_service(),
    )


@router.post("/rebuild_fts")
def rebuild_claims_fts(
    user_id: Optional[int] = None,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """Rebuild the claims FTS table."""
    return claims_service.rebuild_claims_fts(
        user_id=user_id,
        current_user=current_user,
        db=db,
    )


# =============================================================================
# FVA (Falsification-Verification Alignment) Endpoints
# =============================================================================


@router.post("/verify/fva", response_model=FVAVerifyResponse)
async def verify_claims_fva(
    payload: FVAVerifyRequest,
    user_id: Optional[int] = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
) -> dict[str, Any]:
    """
    Verify claims using the FVA (Falsification-Verification Alignment) pipeline.

    The FVA pipeline extends standard claim verification with active counter-evidence
    retrieval. For each claim, it:

    1. Performs standard verification against retrieved documents
    2. Determines if falsification checking is needed based on confidence/claim type
    3. If triggered, retrieves anti-context (counter-evidence) documents
    4. Adjudicates between supporting and contradicting evidence
    5. Returns final verification status which may include CONTESTED for conflicting evidence

    This implements the approach from the FVA-RAG paper (arXiv:2512.07015).
    """
    # Convert claims to dict format
    claims_data = [c.model_dump() for c in payload.claims]

    # Get FVA config
    fva_config = payload.fva_config.model_dump() if payload.fva_config else None

    # Resolve user_id string
    resolved_user_id: Optional[str] = None
    if user_id is not None:
        resolved_user_id = str(user_id)
    elif current_user is not None:
        resolved_user_id = str(getattr(current_user, "username", None) or getattr(current_user, "id", None))

    return await claims_service.verify_claims_with_fva(
        claims=claims_data,
        query=payload.query,
        sources=payload.sources,
        top_k=payload.top_k,
        fva_config=fva_config,
        user_id=resolved_user_id,
        current_user=current_user,
        db=db,
    )


@router.get("/verify/fva/settings", response_model=FVASettingsResponse)
def get_fva_settings(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Return current FVA pipeline settings."""
    return claims_service.get_fva_settings()
