export type SourceAnnotation = {
  id: string
  quote: string
  note: string
  createdAt: number
  updatedAt: number
}

export type SourceAnnotations = Record<string, SourceAnnotation[]>

const storageKey = (workspaceId: string | null | undefined) =>
  `tldw:research-workspace:source-annotations:v1:${workspaceId || "local"}`

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isAnnotation = (value: unknown): value is SourceAnnotation =>
  isRecord(value) &&
  typeof value.id === "string" &&
  value.id.length > 0 &&
  typeof value.quote === "string" &&
  typeof value.note === "string" &&
  typeof value.createdAt === "number" &&
  Number.isFinite(value.createdAt) &&
  value.createdAt >= 0 &&
  typeof value.updatedAt === "number" &&
  Number.isFinite(value.updatedAt) &&
  value.updatedAt >= 0

const isLocalSourceAnnotations = (value: unknown): value is SourceAnnotations =>
  isRecord(value) && Object.entries(value).every(([sourceId, annotations]) =>
    sourceId.length > 0 && Array.isArray(annotations) && annotations.every(isAnnotation)
  )

/** Import references identify bundle sources; they never grant access to Media. */
export const validateSourceAnnotations = (
  value: unknown,
  allowedSourceIds?: Iterable<string>
): SourceAnnotations | null => {
  if (!isRecord(value)) return null
  const allowed = allowedSourceIds ? new Set(allowedSourceIds) : null
  const entries = Object.entries(value)
  if (entries.length > 1000) return null
  let count = 0
  const result: Array<[string, SourceAnnotation[]]> = []
  for (const [sourceId, annotations] of entries) {
    if (
      !sourceId ||
      sourceId.length > 512 ||
      (allowed && !allowed.has(sourceId)) ||
      !Array.isArray(annotations) ||
      !annotations.every(annotation =>
        isAnnotation(annotation) && annotation.id.length <= 512 &&
        annotation.quote.length <= 100_000 && annotation.note.length <= 100_000
      )
    )
      return null
    count += annotations.length
    if (count > 5000) return null
    result.push([
      sourceId,
      annotations.map(({ id, quote, note, createdAt, updatedAt }) => ({
        id,
        quote,
        note,
        createdAt,
        updatedAt
      }))
    ])
  }
  const normalized = Object.fromEntries(result)
  return JSON.stringify(normalized).length <= 4_000_000 ? normalized : null
}

export const readPersistedSourceAnnotations = (
  workspaceId: string | null | undefined,
  allowedSourceIds?: Iterable<string>
): SourceAnnotations => {
  if (typeof window === "undefined") return {}
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(storageKey(workspaceId)) || "{}"
    )
    if (!isRecord(parsed)) return {}
    const allowed = allowedSourceIds ? new Set(allowedSourceIds) : null
    const entries = Object.entries(parsed)
      .filter(
        ([id, values]) => (!allowed || allowed.has(id)) && Array.isArray(values)
      )
      .map(([id, values]) => [id, (values as unknown[]).filter(isAnnotation)])
    return Object.fromEntries(entries)
  } catch {
    return {}
  }
}

/** Refuse incomplete exports instead of treating invalid saved data as no annotations. */
export const readSourceAnnotationsForExport = (
  workspaceId: string | null | undefined,
  allowedSourceIds: Iterable<string>
): SourceAnnotations | null => {
  if (typeof window === "undefined") return {}
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(storageKey(workspaceId)) || "{}"
    )
    if (!isRecord(parsed)) return null
    const allowed = new Set(allowedSourceIds)
    return validateSourceAnnotations(Object.fromEntries(
      Object.entries(parsed).filter(([sourceId]) => allowed.has(sourceId))
    ))
  } catch {
    return null
  }
}

export const persistSourceAnnotations = (
  workspaceId: string | null | undefined,
  annotations: SourceAnnotations
): boolean => {
  if (typeof window === "undefined") return false
  if (!isLocalSourceAnnotations(annotations)) return false
  try {
    const key = storageKey(workspaceId)
    const existing = window.localStorage.getItem(key)
    // A salvaged read must never overwrite entries it could not recover.
    if (existing && !isLocalSourceAnnotations(JSON.parse(existing))) return false
    window.localStorage.setItem(key, JSON.stringify(annotations))
    return true
  } catch {
    return false
  }
}
