/** Dispatched by features after a localStorage write so quota state refreshes. */
export const STORAGE_QUOTA_REFRESH_EVENT = "tldw:storage-quota-refresh"

/** Dispatched by the quota hook when the level transitions (e.g., ok → warning). */
export const STORAGE_QUOTA_WARNING_EVENT = "tldw:storage-quota-warning"
