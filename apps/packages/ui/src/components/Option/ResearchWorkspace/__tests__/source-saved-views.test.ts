import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import {
  DEFAULT_SOURCE_LIST_VIEW_STATE,
  type SourceListViewState
} from "../SourcesPane/source-list-view"
import {
  LARGE_SOURCE_FILE_BYTES,
  SOURCE_SAVED_VIEW_SCHEMA_VERSION,
  SOURCE_VIEW_PRESETS,
  applySavedSourceViewState,
  areSourceViewStatesEqual,
  deserializeSourceViewState,
  getSourceListViewStateSignature,
  getSourceViewStateSignature,
  isSourceListViewStateModified,
  serializeSourceListViewState
} from "../SourcesPane/source-saved-views"
import { useSourceListViewState } from "../use-source-list-view-state"
import {
  WORKSPACE_SOURCE_SAVED_VIEW_INVALID_REASONS,
  WORKSPACE_SOURCE_SAVED_VIEW_SORTS,
  type WorkspaceSourceSavedViewStateV1
} from "@/types/workspace-source-saved-view"

const defaultV1: WorkspaceSourceSavedViewStateV1 = {
  type_filters: [],
  status_filters: [],
  review_state_filters: [],
  lifecycle_state_filters: [],
  date_field: "added_at",
  date_from: null,
  date_to: null,
  require_url: false,
  require_file_size: false,
  require_duration: false,
  require_page_count: false,
  file_size_min: null,
  file_size_max: null,
  duration_min: null,
  duration_max: null,
  page_count_min: null,
  page_count_max: null,
  sort: "manual"
}

describe("source saved view V1 contract", () => {
  it("pins schema version, invalid reasons, and the exact sort enum", () => {
    expect(SOURCE_SAVED_VIEW_SCHEMA_VERSION).toBe(1)
    expect(WORKSPACE_SOURCE_SAVED_VIEW_INVALID_REASONS).toEqual([
      "invalid_json",
      "invalid_state",
      "unsupported_schema_version"
    ])
    expect(WORKSPACE_SOURCE_SAVED_VIEW_SORTS).toEqual([
      "manual",
      "name_asc",
      "name_desc",
      "added_desc",
      "added_asc",
      "source_created_desc",
      "source_created_asc",
      "file_size_desc",
      "file_size_asc",
      "duration_desc",
      "duration_asc",
      "page_count_desc",
      "page_count_asc"
    ])
  })

  it("defaults every omitted V1 field", () => {
    expect(deserializeSourceViewState({})).toEqual(defaultV1)
  })

  it("canonicalizes enum arrays by declaration order and removes duplicates", () => {
    expect(
      deserializeSourceViewState({
        type_filters: ["text", "pdf", "website", "pdf"],
        status_filters: ["error", "processing", "ready", "error"],
        review_state_filters: ["reviewed", "unset", "needs_review", "unset"],
        lifecycle_state_filters: [
          "unknown",
          "partially_queryable",
          "queued",
          "failed",
          "queued"
        ]
      })
    ).toEqual({
      ...defaultV1,
      type_filters: ["pdf", "website", "text"],
      status_filters: ["processing", "ready", "error"],
      review_state_filters: ["unset", "needs_review", "reviewed"],
      lifecycle_state_filters: [
        "queued",
        "partially_queryable",
        "failed",
        "unknown"
      ]
    })
  })

  it("accepts the exact declared lifecycle and sort values", () => {
    const lifecycleStates = [
      "queued",
      "ingesting",
      "extracting",
      "chunking",
      "indexing",
      "queryable",
      "partially_queryable",
      "failed",
      "retrying",
      "missing_media",
      "blocked_by_permissions",
      "unknown"
    ] as const

    expect(
      deserializeSourceViewState({ lifecycle_state_filters: lifecycleStates })
        ?.lifecycle_state_filters
    ).toEqual(lifecycleStates)
    for (const sort of WORKSPACE_SOURCE_SAVED_VIEW_SORTS) {
      expect(deserializeSourceViewState({ sort })?.sort).toBe(sort)
    }
  })

  it.each([
    ["type_filters", ["spreadsheet"]],
    ["status_filters", ["complete"]],
    ["review_state_filters", ["pending"]],
    ["lifecycle_state_filters", ["partial"]],
    ["date_field", "created_at"],
    ["sort", "relevance"]
  ])("rejects an unknown value for %s", (field, value) => {
    expect(deserializeSourceViewState({ [field]: value })).toBeNull()
  })

  it("rejects unknown fields", () => {
    expect(
      deserializeSourceViewState({ ...defaultV1, expanded: true })
    ).toBeNull()
  })

  it.each([
    ["2024-02-29", true],
    ["2023-02-29", false],
    ["2026-04-31", false],
    ["2026-1-01", false],
    ["not-a-date", false]
  ])("validates %s as a real YYYY-MM-DD date", (date, valid) => {
    const value = deserializeSourceViewState({ date_from: date })
    expect(value !== null).toBe(valid)
  })

  it("rejects inverted date ranges", () => {
    expect(
      deserializeSourceViewState({
        date_from: "2026-03-02",
        date_to: "2026-03-01"
      })
    ).toBeNull()
  })

  it.each([
    ["negative", -1],
    ["NaN", Number.NaN],
    ["positive infinity", Number.POSITIVE_INFINITY],
    ["boolean", true]
  ])("rejects %s numeric values", (_name, value) => {
    expect(deserializeSourceViewState({ file_size_min: value })).toBeNull()
  })

  it.each([
    ["file_size_min", "file_size_max"],
    ["duration_min", "duration_max"],
    ["page_count_min", "page_count_max"]
  ])("rejects an inverted %s/%s range", (minimum, maximum) => {
    expect(
      deserializeSourceViewState({ [minimum]: 2, [maximum]: 1 })
    ).toBeNull()
  })

  it("round trips every supported field and excludes expanded", () => {
    const local: SourceListViewState = {
      expanded: true,
      typeFilters: ["website", "pdf", "pdf"],
      statusFilters: ["error"],
      reviewStateFilters: ["needs_review"],
      lifecycleStateFilters: ["partially_queryable"],
      dateField: "sourceCreatedAt",
      dateFrom: "2024-02-29",
      dateTo: "2026-03-01",
      requireUrl: true,
      requireFileSize: true,
      requireDuration: true,
      requirePageCount: true,
      fileSizeMin: 0,
      fileSizeMax: 100,
      durationMin: 1,
      durationMax: 200,
      pageCountMin: 2,
      pageCountMax: 300,
      sort: "source_created_desc"
    }

    const serialized = serializeSourceListViewState(local)

    expect(serialized.ok).toBe(true)
    if (!serialized.ok) return
    expect(serialized.state).not.toHaveProperty("expanded")
    expect(deserializeSourceViewState(serialized.state)).toEqual(
      serialized.state
    )
    expect(serialized.state.type_filters).toEqual(["pdf", "website"])
  })

  it("returns field-specific issues instead of payloads for invalid local state", () => {
    const serialized = serializeSourceListViewState({
      ...DEFAULT_SOURCE_LIST_VIEW_STATE,
      dateFrom: "2026-02-30",
      fileSizeMin: -1,
      durationMin: 20,
      durationMax: 10
    })

    expect(serialized.ok).toBe(false)
    if (serialized.ok) return
    expect(serialized.issues.map((issue) => issue.field)).toEqual(
      expect.arrayContaining(["dateFrom", "fileSizeMin", "durationMax"])
    )
    expect(serialized).not.toHaveProperty("state")
  })

  it.each([null, [], "{}", { unknown: true }, { type_filters: false }])(
    "returns null for malformed runtime state %#",
    (payload) => {
      expect(deserializeSourceViewState(payload)).toBeNull()
    }
  )

  it("fully applies saved fields while preserving only expanded", () => {
    const current: SourceListViewState = {
      ...DEFAULT_SOURCE_LIST_VIEW_STATE,
      expanded: true,
      typeFilters: ["audio"],
      dateFrom: "2020-01-01",
      fileSizeMin: 999,
      sort: "name_desc"
    }
    const saved: WorkspaceSourceSavedViewStateV1 = {
      ...defaultV1,
      lifecycle_state_filters: ["partially_queryable"],
      require_url: true
    }

    expect(applySavedSourceViewState(current, saved)).toEqual({
      ...DEFAULT_SOURCE_LIST_VIEW_STATE,
      expanded: true,
      lifecycleStateFilters: ["partially_queryable"],
      requireUrl: true
    })
  })

  it("builds deterministic signatures and treats invalid local state as Modified", () => {
    const left = deserializeSourceViewState({
      type_filters: ["website", "pdf", "website"]
    })
    const right = deserializeSourceViewState({
      type_filters: ["pdf", "website"]
    })
    expect(left).not.toBeNull()
    expect(right).not.toBeNull()
    if (!left || !right) return

    const signature = getSourceViewStateSignature(left)
    expect(getSourceViewStateSignature(right)).toBe(signature)
    expect(areSourceViewStatesEqual(left, right)).toBe(true)

    const local = applySavedSourceViewState(
      DEFAULT_SOURCE_LIST_VIEW_STATE,
      left
    )
    expect(getSourceListViewStateSignature(local)).toBe(signature)
    expect(isSourceListViewStateModified(local, signature)).toBe(false)
    expect(
      isSourceListViewStateModified(
        { ...local, dateFrom: "not-a-date" },
        signature
      )
    ).toBe(true)
    expect(
      getSourceListViewStateSignature({ ...local, dateFrom: "not-a-date" })
    ).toBeNull()
  })
})

describe("built-in source view presets", () => {
  it("defines all seven immutable presets in fixed order", () => {
    expect(Object.keys(SOURCE_VIEW_PRESETS)).toEqual([
      "needsReview",
      "unreviewed",
      "failedIngest",
      "partiallyIndexed",
      "pdfs",
      "webCaptures",
      "largeFiles"
    ])
    expect(
      Object.values(SOURCE_VIEW_PRESETS).map((preset) => preset.label)
    ).toEqual([
      "Needs review",
      "Unreviewed",
      "Failed ingest",
      "Partially indexed",
      "PDFs",
      "Web captures",
      "Large files"
    ])
    expect(Object.isFrozen(SOURCE_VIEW_PRESETS)).toBe(true)
    expect(
      Object.values(SOURCE_VIEW_PRESETS).every(
        (preset) => Object.isFrozen(preset) && Object.isFrozen(preset.state)
      )
    ).toBe(true)
  })

  it("maps every preset to a complete replacement state", () => {
    const expected = {
      needsReview: { reviewStateFilters: ["needs_review"] },
      unreviewed: { reviewStateFilters: ["unset"] },
      failedIngest: { statusFilters: ["error"] },
      partiallyIndexed: { lifecycleStateFilters: ["partially_queryable"] },
      pdfs: { typeFilters: ["pdf"] },
      webCaptures: { typeFilters: ["website"] },
      largeFiles: { fileSizeMin: LARGE_SOURCE_FILE_BYTES }
    } as const

    for (const [key, fields] of Object.entries(expected)) {
      expect(
        SOURCE_VIEW_PRESETS[key as keyof typeof SOURCE_VIEW_PRESETS].state
      ).toEqual({
        ...DEFAULT_SOURCE_LIST_VIEW_STATE,
        ...fields
      })
    }
    expect(
      SOURCE_VIEW_PRESETS.partiallyIndexed.state.lifecycleStateFilters
    ).toEqual(["partially_queryable"])
    expect(SOURCE_VIEW_PRESETS.largeFiles.state.fileSizeMin).toBe(
      50 * 1024 * 1024
    )
  })
})

describe("source list state hook", () => {
  it("replaces the complete state supplied to applySourceListViewState", () => {
    const { result } = renderHook(() => useSourceListViewState())
    const next: SourceListViewState = {
      ...DEFAULT_SOURCE_LIST_VIEW_STATE,
      expanded: true,
      lifecycleStateFilters: ["failed"],
      sort: "added_desc"
    }

    act(() => {
      result.current.patchSourceListViewState({ typeFilters: ["pdf"] })
      result.current.applySourceListViewState(next)
    })

    expect(result.current.sourceListViewState).toEqual(next)
  })
})
