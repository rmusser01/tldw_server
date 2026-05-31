import React from "react"
import {
  SOURCE_LIST_SORT_LABELS,
  hasActiveSourceFilters,
  type SourceListSortOption,
  type SourceListViewState
} from "./source-list-view"

interface SourceAdvancedControlsProps {
  viewState: SourceListViewState
  summary: string
  hasFileSizeSources: boolean
  hasDurationSources: boolean
  hasPageCountSources: boolean
  onPatchViewState: (patch: Partial<SourceListViewState>) => void
  onResetAdvancedFilters: () => void
}

const toggleListValue = <T extends string>(values: T[], value: T): T[] =>
  values.includes(value)
    ? values.filter((entry) => entry !== value)
    : [...values, value]

const parseOptionalNumber = (value: string): number | null => {
  const trimmed = value.trim()
  if (!trimmed) return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

const SORT_OPTIONS: SourceListSortOption[] = [
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
]

export const SourceAdvancedControls: React.FC<SourceAdvancedControlsProps> = ({
  viewState,
  summary,
  hasFileSizeSources,
  hasDurationSources,
  hasPageCountSources,
  onPatchViewState,
  onResetAdvancedFilters
}) => {
  const showClearFilters = hasActiveSourceFilters(viewState) || viewState.sort !== "manual"

  return (
    <div className="mt-2 rounded-lg border border-border/70 bg-surface/50 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          aria-expanded={viewState.expanded}
          onClick={() => onPatchViewState({ expanded: !viewState.expanded })}
          className="rounded border border-border bg-surface px-2 py-1 text-[11px] font-medium text-text transition hover:bg-surface2"
        >
          Advanced
        </button>
        {!viewState.expanded && summary ? (
          <span className="min-w-0 flex-1 truncate text-[11px] text-text-muted">
            {summary}
          </span>
        ) : null}
        {showClearFilters ? (
          <button
            type="button"
            onClick={onResetAdvancedFilters}
            className="rounded border border-border px-2 py-1 text-[11px] text-text-muted transition hover:bg-surface2 hover:text-text"
          >
            Clear filters
          </button>
        ) : null}
      </div>

      {viewState.expanded ? (
        <div className="mt-3 space-y-3">
          <div className="space-y-1">
            <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
              Status
            </p>
            <div className="flex flex-wrap gap-3 text-sm text-text">
              {(["ready", "processing", "error"] as const).map((status) => (
                <label key={status} className="inline-flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={viewState.statusFilters.includes(status)}
                    onChange={() =>
                      onPatchViewState({
                        statusFilters: toggleListValue(viewState.statusFilters, status)
                      })
                    }
                  />
                  <span>{`Status ${status[0].toUpperCase()}${status.slice(1)}`}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-1">
            <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
              Type
            </p>
            <div className="flex flex-wrap gap-3 text-sm text-text">
              {(["pdf", "video", "audio", "website", "document", "text"] as const).map(
                (type) => (
                  <label key={type} className="inline-flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={viewState.typeFilters.includes(type)}
                      onChange={() =>
                        onPatchViewState({
                          typeFilters: toggleListValue(viewState.typeFilters, type)
                        })
                      }
                    />
                    <span>{`Type ${type.toUpperCase()}`}</span>
                  </label>
                )
              )}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-sm text-text">
              <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                Sort by
              </span>
              <select
                aria-label="Sort by"
                value={viewState.sort}
                onChange={(event) =>
                  onPatchViewState({
                    sort: event.target.value as SourceListSortOption
                  })
                }
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
              >
                {SORT_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option === "manual" ? "Manual order" : SOURCE_LIST_SORT_LABELS[option]}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1 text-sm text-text">
              <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                Date field
              </span>
              <select
                aria-label="Date field"
                value={viewState.dateField}
                onChange={(event) =>
                  onPatchViewState({
                    dateField: event.target.value as SourceListViewState["dateField"]
                  })
                }
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
              >
                <option value="addedAt">Added date</option>
                <option value="sourceCreatedAt">Source date</option>
              </select>
            </label>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-sm text-text">
              <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                Date from
              </span>
              <input
                aria-label="Date from"
                type="date"
                value={viewState.dateFrom ?? ""}
                onChange={(event) =>
                  onPatchViewState({ dateFrom: event.target.value || null })
                }
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
              />
            </label>

            <label className="space-y-1 text-sm text-text">
              <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                Date to
              </span>
              <input
                aria-label="Date to"
                type="date"
                value={viewState.dateTo ?? ""}
                onChange={(event) =>
                  onPatchViewState({ dateTo: event.target.value || null })
                }
                className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
              />
            </label>
          </div>

          <div className="flex flex-wrap gap-3 text-sm text-text">
            <label className="inline-flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={viewState.requireUrl}
                onChange={() =>
                  onPatchViewState({ requireUrl: !viewState.requireUrl })
                }
              />
              <span>Has URL</span>
            </label>

            {hasFileSizeSources ? (
              <label className="inline-flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={viewState.requireFileSize}
                  onChange={() =>
                    onPatchViewState({ requireFileSize: !viewState.requireFileSize })
                  }
                />
                <span>Has file size</span>
              </label>
            ) : null}

            {hasDurationSources ? (
              <label className="inline-flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={viewState.requireDuration}
                  onChange={() =>
                    onPatchViewState({ requireDuration: !viewState.requireDuration })
                  }
                />
                <span>Has duration</span>
              </label>
            ) : null}

            {hasPageCountSources ? (
              <label className="inline-flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={viewState.requirePageCount}
                  onChange={() =>
                    onPatchViewState({ requirePageCount: !viewState.requirePageCount })
                  }
                />
                <span>Has page count</span>
              </label>
            ) : null}
          </div>

          {(hasFileSizeSources || hasDurationSources || hasPageCountSources) && (
            <div className="grid gap-3 md:grid-cols-2">
              {hasFileSizeSources ? (
                <>
                  <label className="space-y-1 text-sm text-text">
                    <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                      File size min
                    </span>
                    <input
                      aria-label="File size min"
                      type="number"
                      value={viewState.fileSizeMin ?? ""}
                      onChange={(event) =>
                        onPatchViewState({
                          fileSizeMin: parseOptionalNumber(event.target.value)
                        })
                      }
                      className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
                    />
                  </label>
                  <label className="space-y-1 text-sm text-text">
                    <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                      File size max
                    </span>
                    <input
                      aria-label="File size max"
                      type="number"
                      value={viewState.fileSizeMax ?? ""}
                      onChange={(event) =>
                        onPatchViewState({
                          fileSizeMax: parseOptionalNumber(event.target.value)
                        })
                      }
                      className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
                    />
                  </label>
                </>
              ) : null}

              {hasDurationSources ? (
                <>
                  <label className="space-y-1 text-sm text-text">
                    <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                      Duration min
                    </span>
                    <input
                      aria-label="Duration min"
                      type="number"
                      value={viewState.durationMin ?? ""}
                      onChange={(event) =>
                        onPatchViewState({
                          durationMin: parseOptionalNumber(event.target.value)
                        })
                      }
                      className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
                    />
                  </label>
                  <label className="space-y-1 text-sm text-text">
                    <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                      Duration max
                    </span>
                    <input
                      aria-label="Duration max"
                      type="number"
                      value={viewState.durationMax ?? ""}
                      onChange={(event) =>
                        onPatchViewState({
                          durationMax: parseOptionalNumber(event.target.value)
                        })
                      }
                      className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
                    />
                  </label>
                </>
              ) : null}

              {hasPageCountSources ? (
                <>
                  <label className="space-y-1 text-sm text-text">
                    <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                      Page count min
                    </span>
                    <input
                      aria-label="Page count min"
                      type="number"
                      value={viewState.pageCountMin ?? ""}
                      onChange={(event) =>
                        onPatchViewState({
                          pageCountMin: parseOptionalNumber(event.target.value)
                        })
                      }
                      className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
                    />
                  </label>
                  <label className="space-y-1 text-sm text-text">
                    <span className="block text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                      Page count max
                    </span>
                    <input
                      aria-label="Page count max"
                      type="number"
                      value={viewState.pageCountMax ?? ""}
                      onChange={(event) =>
                        onPatchViewState({
                          pageCountMax: parseOptionalNumber(event.target.value)
                        })
                      }
                      className="w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-text"
                    />
                  </label>
                </>
              ) : null}
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}
