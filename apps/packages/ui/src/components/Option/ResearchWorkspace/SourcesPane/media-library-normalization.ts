type MediaLibraryRecord = Record<string, unknown>

export type NormalizedMediaLibraryResponse = {
  items: unknown[]
  totalCount: number
}

const isRecord = (value: unknown): value is MediaLibraryRecord =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const normalizeId = (value: unknown): string | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  return null
}

export const getMediaLibraryItemKey = (item: unknown): string | null => {
  if (!isRecord(item)) return null

  return normalizeId(item.media_id) ?? normalizeId(item.id)
}

const firstArray = (...values: unknown[]): unknown[] | null => {
  for (const value of values) {
    if (Array.isArray(value)) return value
  }
  return null
}

const normalizeTotal = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return Math.trunc(value)
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed >= 0) {
      return Math.trunc(parsed)
    }
  }
  return null
}

const firstTotal = (...values: unknown[]): number | null => {
  for (const value of values) {
    const total = normalizeTotal(value)
    if (total != null) return total
  }
  return null
}

export const normalizeMediaLibraryResponse = (
  response: unknown,
  fallbackTotal?: number
): NormalizedMediaLibraryResponse => {
  if (Array.isArray(response)) {
    const fallback = normalizeTotal(fallbackTotal)
    return {
      items: response,
      totalCount: Math.max(response.length, fallback ?? response.length)
    }
  }

  if (!isRecord(response)) {
    const fallback = normalizeTotal(fallbackTotal)
    return {
      items: [],
      totalCount: fallback ?? 0
    }
  }

  const nestedData = isRecord(response.data) ? response.data : null
  const items =
    firstArray(response.media, response.results, response.items, response.data) ??
    (nestedData
      ? firstArray(nestedData.items, nestedData.media, nestedData.results)
      : null) ??
    []

  const rootPagination = isRecord(response.pagination) ? response.pagination : null
  const nestedPagination =
    nestedData && isRecord(nestedData.pagination) ? nestedData.pagination : null
  const total =
    firstTotal(
      response.total_count,
      response.total,
      response.count,
      response.results_count,
      rootPagination?.total,
      rootPagination?.total_items,
      nestedData?.total_count,
      nestedData?.total,
      nestedData?.count,
      nestedData?.results_count,
      nestedPagination?.total,
      nestedPagination?.total_items
    ) ?? normalizeTotal(fallbackTotal)

  return {
    items,
    totalCount: Math.max(items.length, total ?? items.length)
  }
}
