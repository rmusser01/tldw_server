"""Closed opt-in registration for Jobs resource-required types."""

import re

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import CLAIMS_REBUILD_CLUSTERS_JOB_TYPE

_CLUSTER_ALIAS = re.compile(re.escape(CLAIMS_REBUILD_CLUSTERS_JOB_TYPE) + r"\.v[1-9][0-9]*\Z")


def resource_requirement_for(domain: str, job_type: str) -> str | None:
    """Resolve the registered resource or fail closed on reserved version aliases."""
    if domain == "claims":
        if job_type == CLAIMS_REBUILD_CLUSTERS_JOB_TYPE:
            return "cluster-rebuild"
        if _CLUSTER_ALIAS.fullmatch(job_type):
            raise ValueError("unsupported cluster rebuild job type alias")
    return None
