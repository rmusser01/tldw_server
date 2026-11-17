"""Existing-only, owner-safe best effort after cluster publication.

The worker calls this only for newly published output. A failure or a crash may
omit notifications; evaluation never invalidates publication or retries a rebuild.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from tldw_Server_API.app.core.claims_analytics_export_contract import is_routable_claims_owner_id_text
from tldw_Server_API.app.core.Claims_Extraction.claims_notifications import record_watchlist_cluster_notifications
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import MEDIA_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def evaluate_published_cluster_watchlists(db: MediaDatabase, *, owner_user_id: str) -> None:
    """Evaluate owned subscriptions only against explicitly owner-scoped clusters."""
    try:
        if not is_routable_claims_owner_id_text(owner_user_id):
            raise ValueError("owner_user_id must be a routable Claims owner")
        watch_db = WatchlistsDatabase.open_existing(user_id=owner_user_id, backend=db.backend)
        if watch_db is None:
            return
        subscriptions: dict[int, list[int]] = {}
        for row in watch_db.list_watchlist_cluster_subscriptions():
            subscriptions.setdefault(int(row["cluster_id"]), []).append(int(row["job_id"]))
        if not subscriptions:
            return
        clusters = db.get_claim_clusters_by_ids(list(subscriptions), owner_user_id=owner_user_id)
        cluster_map = {int(cluster["id"]): cluster for cluster in clusters}
        eligible = {cluster_id: jobs for cluster_id, jobs in subscriptions.items() if cluster_id in cluster_map}
        if not eligible:
            return
        cluster_ids = list(eligible)
        member_counts = db.get_claim_cluster_member_counts(cluster_ids, owner_user_id=owner_user_id)
        counts = watch_db.list_watchlist_cluster_counts(cluster_ids=cluster_ids)
        db.update_claim_clusters_watchlist_counts(counts, owner_user_id=owner_user_id)
        record_watchlist_cluster_notifications(
            db=db,
            owner_user_id=owner_user_id,
            clusters=cluster_map,
            member_counts=member_counts,
            subscriptions=eligible,
        )
    except MEDIA_NONCRITICAL_EXCEPTIONS:
        logger.warning("Published cluster Watchlists evaluation failed")
