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

const CANONICAL_NOTE_ID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function buildKnowledgeNoteScopePath(noteIds: string[]): string {
  const valid =
    noteIds.length > 0 && noteIds.every((id) => CANONICAL_NOTE_ID.test(id))
  return `/knowledge?${new URLSearchParams({ note_ids: valid ? [...new Set(noteIds)].join(",") : "" })}`
}

/** Every explicitly supplied category must be valid before any search can run. */
export function parseKnowledgeScope(search: string): {
  mediaIds: number[]
  noteIds: string[]
  invalid: boolean
} | null {
  const media = parseKnowledgeMediaScope(search)
  const values = new URLSearchParams(search).getAll("note_ids")
  if (!media && values.length === 0) return null
  const noteIds = values.length ? values[0].split(",") : []
  if (
    media?.invalid ||
    values.length > 1 ||
    noteIds.some((id) => !CANONICAL_NOTE_ID.test(id))
  ) {
    return { mediaIds: [], noteIds: [], invalid: true }
  }
  return {
    mediaIds: media?.mediaIds ?? [],
    noteIds: [...new Set(noteIds)],
    invalid: false
  }
}
