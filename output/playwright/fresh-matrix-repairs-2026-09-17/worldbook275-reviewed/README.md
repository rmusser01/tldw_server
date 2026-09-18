# Reviewed parent World Book summary refresh (UAT275)

Successful entry mutations invalidate the existing parent list as well as the per-book entries. Successful destination copies also refresh parent counts before a potentially failing source deletion. Failed ordinary add preserves the existing count and refetch behavior.

Three maintained real-QueryClient controls pass independently; exact baseline overlay produces two expected failures and a passing negative control. Adjacent author suite passes 17 with one pre-existing retired-drawer skip. Partial bulk-add is source-reviewed only. Scoped lint has no new errors; Bandit cannot parse TypeScript and no clean whole-UI compiler result is claimed.

Native original PostgreSQL count and persistence acceptance remains pending. Exact evidence is retained with lossless gzip where applicable, credential scanning and source hashes. No full-matrix claim.
