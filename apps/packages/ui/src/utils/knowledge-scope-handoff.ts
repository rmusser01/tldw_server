/** An explicit empty or malformed transfer must never become a library search. */
export function buildKnowledgeMediaScopePath(mediaIds: number[]): string {
  const valid =
    mediaIds.length > 0 &&
    mediaIds.every((id) => Number.isSafeInteger(id) && id > 0)
  const params = new URLSearchParams({
    media_ids: valid ? [...new Set(mediaIds)].join(",") : "",
  })
  return `/knowledge?${params}`
}

export function parseKnowledgeMediaScope(
  search: string,
): { mediaIds: number[]; invalid: boolean } | null {
  const values = new URLSearchParams(search).getAll("media_ids")
  if (values.length === 0) return null
  const tokens = values[0].split(",")
  const mediaIds = tokens.map(Number)
  if (
    values.length !== 1 ||
    tokens.some((id) => !/^\d+$/.test(id)) ||
    mediaIds.some((id) => !Number.isSafeInteger(id) || id <= 0)
  ) {
    return { mediaIds: [], invalid: true }
  }
  return { mediaIds: [...new Set(mediaIds)], invalid: false }
}
