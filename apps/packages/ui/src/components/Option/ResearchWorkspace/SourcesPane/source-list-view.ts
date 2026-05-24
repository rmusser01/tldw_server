import type {
  WorkspaceSource,
  WorkspaceSourceStatus,
  WorkspaceSourceType
} from "@/types/workspace"

export type SourceListSortOption =
  | "manual"
  | "name_asc"
  | "name_desc"
  | "added_desc"
  | "added_asc"
  | "source_created_desc"
  | "source_created_asc"
  | "file_size_desc"
  | "file_size_asc"
  | "duration_desc"
  | "duration_asc"
  | "page_count_desc"
  | "page_count_asc"

export interface SourceListViewState {
  expanded: boolean
  typeFilters: WorkspaceSourceType[]
  statusFilters: WorkspaceSourceStatus[]
  dateField: "addedAt" | "sourceCreatedAt"
  dateFrom: string | null
  dateTo: string | null
  requireUrl: boolean
  requireFileSize: boolean
  requireDuration: boolean
  requirePageCount: boolean
  fileSizeMin: number | null
  fileSizeMax: number | null
  durationMin: number | null
  durationMax: number | null
  pageCountMin: number | null
  pageCountMax: number | null
  sort: SourceListSortOption
}

export const DEFAULT_SOURCE_LIST_VIEW_STATE: SourceListViewState = {
  expanded: false,
  typeFilters: [],
  statusFilters: [],
  dateField: "addedAt",
  dateFrom: null,
  dateTo: null,
  requireUrl: false,
  requireFileSize: false,
  requireDuration: false,
  requirePageCount: false,
  fileSizeMin: null,
  fileSizeMax: null,
  durationMin: null,
  durationMax: null,
  pageCountMin: null,
  pageCountMax: null,
  sort: "manual"
}

const parseStartOfDayUtc = (value: string | null): number | null => {
  if (!value) return null
  const parsed = Date.parse(`${value}T00:00:00.000Z`)
  return Number.isFinite(parsed) ? parsed : null
}

const parseEndOfDayUtc = (value: string | null): number | null => {
  if (!value) return null
  const parsed = Date.parse(`${value}T23:59:59.999Z`)
  return Number.isFinite(parsed) ? parsed : null
}

const isFiniteNumber = (value: number | undefined): value is number =>
  Number.isFinite(value)

const matchesNumericRange = (
  value: number | undefined,
  min: number | null,
  max: number | null
): boolean => {
  if (min == null && max == null) return true
  if (!isFiniteNumber(value)) return false
  if (min != null && value < min) return false
  if (max != null && value > max) return false
  return true
}

const matchesDateRange = (
  source: WorkspaceSource,
  dateField: SourceListViewState["dateField"],
  dateFrom: string | null,
  dateTo: string | null
): boolean => {
  const start = parseStartOfDayUtc(dateFrom)
  const end = parseEndOfDayUtc(dateTo)
  if (start == null && end == null) return true
  const sourceDate = source[dateField]
  if (!(sourceDate instanceof Date) || Number.isNaN(sourceDate.getTime())) return false
  const timestamp = sourceDate.getTime()
  if (start != null && timestamp < start) return false
  if (end != null && timestamp > end) return false
  return true
}

export const hasActiveSourceFilters = (viewState: SourceListViewState): boolean =>
  viewState.typeFilters.length > 0 ||
  viewState.statusFilters.length > 0 ||
  viewState.dateFrom !== null ||
  viewState.dateTo !== null ||
  viewState.requireUrl ||
  viewState.requireFileSize ||
  viewState.requireDuration ||
  viewState.requirePageCount ||
  viewState.fileSizeMin !== null ||
  viewState.fileSizeMax !== null ||
  viewState.durationMin !== null ||
  viewState.durationMax !== null ||
  viewState.pageCountMin !== null ||
  viewState.pageCountMax !== null

export const filterSources = (
  sources: WorkspaceSource[],
  viewState: SourceListViewState
): WorkspaceSource[] =>
  sources.filter((source) => {
    if (
      viewState.typeFilters.length > 0 &&
      !viewState.typeFilters.includes(source.type)
    ) {
      return false
    }

    const status = source.status ?? "ready"
    if (
      viewState.statusFilters.length > 0 &&
      !viewState.statusFilters.includes(status)
    ) {
      return false
    }

    if (!matchesDateRange(source, viewState.dateField, viewState.dateFrom, viewState.dateTo)) {
      return false
    }

    if (viewState.requireUrl && !source.url) {
      return false
    }

    if (viewState.requireFileSize && !isFiniteNumber(source.fileSize)) {
      return false
    }

    if (viewState.requireDuration && !isFiniteNumber(source.duration)) {
      return false
    }

    if (viewState.requirePageCount && !isFiniteNumber(source.pageCount)) {
      return false
    }

    if (!matchesNumericRange(source.fileSize, viewState.fileSizeMin, viewState.fileSizeMax)) {
      return false
    }

    if (!matchesNumericRange(source.duration, viewState.durationMin, viewState.durationMax)) {
      return false
    }

    if (!matchesNumericRange(source.pageCount, viewState.pageCountMin, viewState.pageCountMax)) {
      return false
    }

    return true
  })

const compareText = (left: string, right: string, direction: "asc" | "desc"): number => {
  const result = left.localeCompare(right, undefined, { sensitivity: "base" })
  return direction === "asc" ? result : -result
}

const compareOptionalNumber = (
  left: number | undefined,
  right: number | undefined,
  direction: "asc" | "desc"
): number => {
  const leftMissing = !isFiniteNumber(left)
  const rightMissing = !isFiniteNumber(right)
  if (leftMissing && rightMissing) return 0
  if (leftMissing) return 1
  if (rightMissing) return -1
  return direction === "asc" ? left - right : right - left
}

const compareOptionalDate = (
  left: Date | undefined,
  right: Date | undefined,
  direction: "asc" | "desc"
): number => {
  const leftTime = left instanceof Date ? left.getTime() : Number.NaN
  const rightTime = right instanceof Date ? right.getTime() : Number.NaN
  const leftMissing = !Number.isFinite(leftTime)
  const rightMissing = !Number.isFinite(rightTime)
  if (leftMissing && rightMissing) return 0
  if (leftMissing) return 1
  if (rightMissing) return -1
  return direction === "asc" ? leftTime - rightTime : rightTime - leftTime
}

export const sortSources = (
  sources: WorkspaceSource[],
  sort: SourceListSortOption
): WorkspaceSource[] => {
  if (sort === "manual") {
    return sources
  }

  return sources
    .map((source, index) => ({ source, index }))
    .sort((left, right) => {
      let result = 0

      switch (sort) {
        case "name_asc":
          result = compareText(left.source.title, right.source.title, "asc")
          break
        case "name_desc":
          result = compareText(left.source.title, right.source.title, "desc")
          break
        case "added_asc":
          result = compareOptionalDate(left.source.addedAt, right.source.addedAt, "asc")
          break
        case "added_desc":
          result = compareOptionalDate(left.source.addedAt, right.source.addedAt, "desc")
          break
        case "source_created_asc":
          result = compareOptionalDate(
            left.source.sourceCreatedAt,
            right.source.sourceCreatedAt,
            "asc"
          )
          break
        case "source_created_desc":
          result = compareOptionalDate(
            left.source.sourceCreatedAt,
            right.source.sourceCreatedAt,
            "desc"
          )
          break
        case "file_size_asc":
          result = compareOptionalNumber(left.source.fileSize, right.source.fileSize, "asc")
          break
        case "file_size_desc":
          result = compareOptionalNumber(left.source.fileSize, right.source.fileSize, "desc")
          break
        case "duration_asc":
          result = compareOptionalNumber(left.source.duration, right.source.duration, "asc")
          break
        case "duration_desc":
          result = compareOptionalNumber(left.source.duration, right.source.duration, "desc")
          break
        case "page_count_asc":
          result = compareOptionalNumber(left.source.pageCount, right.source.pageCount, "asc")
          break
        case "page_count_desc":
          result = compareOptionalNumber(left.source.pageCount, right.source.pageCount, "desc")
          break
        default:
          result = 0
      }

      return result !== 0 ? result : left.index - right.index
    })
    .map(({ source }) => source)
}

const capitalize = (value: string): string =>
  value.length === 0 ? value : value[0].toUpperCase() + value.slice(1)

export const SOURCE_LIST_SORT_LABELS: Record<
  Exclude<SourceListSortOption, "manual">,
  string
> = {
  name_asc: "Name (A-Z)",
  name_desc: "Name (Z-A)",
  added_desc: "Added date (newest)",
  added_asc: "Added date (oldest)",
  source_created_desc: "Source date (newest)",
  source_created_asc: "Source date (oldest)",
  file_size_desc: "File size (largest)",
  file_size_asc: "File size (smallest)",
  duration_desc: "Duration (longest)",
  duration_asc: "Duration (shortest)",
  page_count_desc: "Page count (highest)",
  page_count_asc: "Page count (lowest)"
}

export const buildSourceFilterSummary = (viewState: SourceListViewState): string => {
  const parts: string[] = []

  if (viewState.typeFilters.length > 0) {
    parts.push(`Type=${viewState.typeFilters.map((value) => value.toUpperCase()).join(", ")}`)
  }

  if (viewState.statusFilters.length > 0) {
    parts.push(
      `Status=${viewState.statusFilters.map((value) => capitalize(value)).join(", ")}`
    )
  }

  if (viewState.sort !== "manual") {
    parts.push(`Sort: ${SOURCE_LIST_SORT_LABELS[viewState.sort]}`)
  }

  return parts.join(" · ")
}
