import React from "react"
import { ChevronLeft, ChevronRight, FileText, Search } from "lucide-react"
import { Tooltip } from "antd"
import { useTranslation } from "react-i18next"
import type { SharedSource, SharedSourceQuery } from "@/types/shared-workspace"
import type { SharedResearchWorkspaceController } from "./useSharedResearchWorkspace"
import { formatSharedActionReason } from "./shared-action-reason"

type SharedWorkspaceSourcesPaneProps = {
  controller: SharedResearchWorkspaceController
  onPreview: (
    sourceId: string,
    chunkIndex: number | undefined,
    trigger: HTMLElement
  ) => void
}

export const SharedWorkspaceSourcesPane: React.FC<
  SharedWorkspaceSourcesPaneProps
> = ({ controller, onPreview }) => {
  const { t } = useTranslation("playground")
  const { state } = controller
  const page = state.sources
  const summary = state.sourceSummary
  const inspectAction = state.allowedActions.inspect_sources
  const canInspect = inspectAction.allowed
  const selectionBusy = state.selectionMaterializing
  const inspectReason = formatSharedActionReason(inspectAction.reason_code)

  const runQuery = (patch: Partial<SharedSourceQuery>) => {
    if (!canInspect || selectionBusy) return
    const query = { ...state.sourceQuery, ...patch }
    controller.setSourceQuery(query)
    void controller.refreshSources(query)
  }

  const selectedCount =
    state.sourceScopeMode === "all"
      ? summary?.queryable ?? 0
      : state.selectedSourceIds.length
  const queryableCount = summary?.queryable ?? 0
  const allScopeOverLimit = state.sourceScopeMode === "all" && queryableCount > 500
  const reasonLabel = (source: SharedSource): string => {
    if (source.reason_code === "transcription_pending") {
      return t("sharedWorkspace.transcriptionPending", "Transcription pending")
    }
    if (source.state === "processing") {
      return t("sharedWorkspace.stillProcessing", "Still processing")
    }
    if (source.state === "failed") {
      return t("sharedWorkspace.processingFailed", "Processing failed")
    }
    return source.reason_code?.replaceAll("_", " ") ||
      t("sharedWorkspace.notQueryable", "Not queryable")
  }

  return (
    <section
      data-testid="shared-workspace-sources-pane"
      aria-labelledby="shared-workspace-sources-heading"
      className="flex min-h-0 min-w-0 flex-col overflow-hidden border-r border-border bg-surface"
    >
      <div className="shrink-0 space-y-2 border-b border-border px-3 py-3">
        <div className="flex min-w-0 items-center justify-between gap-2">
          <div className="min-w-0">
            <h2
              id="shared-workspace-sources-heading"
              className="text-sm font-semibold"
            >
              {t("sharedWorkspace.sources", "Sources")}
            </h2>
            <p className="text-xs text-text-muted">
              {t(
                "sharedWorkspace.selectedCount",
                "{{selected}} of {{total}} queryable sources selected",
                { selected: selectedCount, total: queryableCount }
              )}
            </p>
          </div>
          <span className="inline-flex h-7 shrink-0 items-center rounded-full bg-surface2 px-2 text-xs text-text-muted">
            {summary?.processing
              ? t("sharedWorkspace.processingBadge", "{{count}} processing", {
                  count: summary.processing
                })
              : t("sharedWorkspace.readyBadge", "Ready")}
          </span>
        </div>

        {summary?.processing ? (
          <p className="text-xs text-text-muted">
            {t(
              "sharedWorkspace.processing",
              "Shared sources are still processing. You can inspect available items while you wait."
            )}
          </p>
        ) : null}

        {!canInspect && inspectReason ? (
          <p role="status" className="text-xs text-warn">
            {inspectReason}
          </p>
        ) : null}

        <label className="relative block">
          <span className="sr-only">
            {t("sharedWorkspace.search", "Search shared sources")}
          </span>
          <Search
            className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-text-subtle"
            aria-hidden="true"
          />
          <input
            type="search"
            aria-label={t("sharedWorkspace.search", "Search shared sources")}
            value={state.sourceQuery.q ?? ""}
            disabled={!canInspect || selectionBusy}
            onChange={(event) => runQuery({ q: event.target.value, offset: 0 })}
            className="h-9 w-full rounded-md border border-border bg-surface2 pl-8 pr-3 text-sm outline-none focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-focus"
          />
        </label>

        <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-2">
          <label className="min-w-0">
            <span className="sr-only">
              {t(
                "sharedWorkspace.stateFilter",
                "Filter shared sources by state"
              )}
            </span>
            <select
              aria-label={t(
                "sharedWorkspace.stateFilter",
                "Filter shared sources by state"
              )}
              value={state.sourceQuery.state ?? ""}
              disabled={!canInspect || selectionBusy}
              onChange={(event) =>
                runQuery({ state: event.target.value || undefined, offset: 0 })
              }
              className="h-9 w-full min-w-0 rounded-md border border-border bg-surface2 px-2 text-sm outline-none focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-focus"
            >
              <option value="">{t("sharedWorkspace.allStates", "All states")}</option>
              <option value="ready">{t("sharedWorkspace.ready", "Ready")}</option>
              <option value="processing">
                {t("sharedWorkspace.processingState", "Processing")}
              </option>
              <option value="failed">{t("sharedWorkspace.failed", "Failed")}</option>
            </select>
          </label>
          <button
            type="button"
            disabled={!canInspect || selectionBusy}
            onClick={controller.selectAllSources}
            className="h-9 whitespace-nowrap rounded-md px-2 text-xs font-medium text-primary outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus"
            aria-label={t(
              "sharedWorkspace.selectAll",
              "Select all queryable sources"
            )}
          >
            {t("sharedWorkspace.all", "All")}
          </button>
          <button
            type="button"
            disabled={!canInspect || selectionBusy}
            onClick={controller.clearSelectedSources}
            className="h-9 whitespace-nowrap rounded-md px-2 text-xs font-medium text-text-muted outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus"
            aria-label={t(
              "sharedWorkspace.clearSelected",
              "Clear selected sources"
            )}
          >
            {t("sharedWorkspace.clear", "Clear")}
          </button>
        </div>

        {allScopeOverLimit ? (
          <p className="text-xs text-warn">
            {t(
              "sharedWorkspace.scopeLimit",
              "Clear the selection, then choose up to 500 sources."
            )}
          </p>
        ) : null}
        {selectionBusy ? (
          <p role="status" className="text-xs text-text-muted">
            {t(
              "sharedWorkspace.selectionPreparing",
              "Preparing complete source selection..."
            )}
          </p>
        ) : null}
        {state.errors.selection ? (
          <p role="alert" className="text-xs text-danger">
            {t("sharedWorkspace.selectionUnavailable", state.errors.selection.message)}
          </p>
        ) : null}
        {state.errors.sources ? (
          <p className="text-xs text-danger">{state.errors.sources.message}</p>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {page?.items.length ? (
          <ul
            className="divide-y divide-border"
            aria-label={t("sharedWorkspace.sourceList", "Shared sources")}
          >
            {page.items.map((source) => {
              const checked =
                source.retrieval_ready &&
                (state.sourceScopeMode === "all" ||
                  state.selectedSourceIds.includes(source.source_id))
              return (
                <li
                  key={source.source_id}
                  className="grid min-w-0 grid-cols-[2rem_minmax(0,1fr)] gap-1 px-2 py-2"
                >
                  <label className="flex h-10 w-8 items-center justify-center">
                    <input
                      type="checkbox"
                      aria-label={t(
                        "sharedWorkspace.selectSource",
                        "Select {{title}}",
                        { title: source.title }
                      )}
                      checked={checked}
                      disabled={
                        !canInspect ||
                        !source.retrieval_ready ||
                        selectionBusy ||
                        allScopeOverLimit
                      }
                      onChange={(event) =>
                        void controller.toggleSource(
                          source,
                          event.target.checked
                        )
                      }
                      className="h-4 w-4 accent-primary focus-visible:ring-2 focus-visible:ring-focus"
                    />
                  </label>
                  <button
                    type="button"
                    disabled={!canInspect}
                    aria-label={t(
                      "sharedWorkspace.previewSource",
                      "Preview {{title}}",
                      { title: source.title }
                    )}
                    onClick={(event) =>
                      onPreview(source.source_id, undefined, event.currentTarget)
                    }
                    className="flex min-h-10 min-w-0 items-start gap-2 rounded-md px-1.5 py-1 text-left outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <FileText
                      className="mt-0.5 h-4 w-4 shrink-0 text-text-subtle"
                      aria-hidden="true"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium">
                        {source.title ||
                          t("sharedWorkspace.untitled", "Untitled source")}
                      </span>
                      <span className="flex min-w-0 items-center gap-1.5 text-xs text-text-muted">
                        <span className="truncate">{source.source_type}</span>
                        {!source.retrieval_ready ? (
                          <span className="shrink-0 text-warn">
                            {reasonLabel(source)}
                          </span>
                        ) : null}
                      </span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="p-4 text-sm text-text-muted">
            {t(
              "sharedWorkspace.empty",
              "This workspace has no shared sources yet."
            )}
          </p>
        )}
      </div>

      <div className="flex h-11 shrink-0 items-center justify-between border-t border-border px-3">
        <span className="text-xs text-text-muted">
          {page
            ? t("sharedWorkspace.sourceRange", "{{start}}-{{end}} of {{total}}", {
                start: page.pagination.total ? page.pagination.offset + 1 : 0,
                end: Math.min(
                  page.pagination.offset + page.items.length,
                  page.pagination.total
                ),
                total: page.pagination.total
              })
            : ""}
        </span>
        <div className="flex items-center gap-1">
          <Tooltip
            title={t("sharedWorkspace.previousPage", "Previous source page")}
          >
            <button
              type="button"
              aria-label={t(
                "sharedWorkspace.previousPage",
                "Previous source page"
              )}
              disabled={
                !canInspect ||
                selectionBusy ||
                !page ||
                page.pagination.offset === 0
              }
              onClick={() =>
                runQuery({
                  offset: Math.max(
                    0,
                    (page?.pagination.offset ?? 0) -
                      (page?.pagination.limit ?? state.sourceQuery.limit)
                  )
                })
              }
              className="inline-flex h-9 w-9 items-center justify-center rounded-md outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-40"
            >
              <ChevronLeft className="h-4 w-4" aria-hidden="true" />
            </button>
          </Tooltip>
          <Tooltip title={t("sharedWorkspace.nextPage", "Next source page")}>
            <button
              type="button"
              aria-label={t(
                "sharedWorkspace.nextPage",
                "Next source page"
              )}
              disabled={
                !canInspect || selectionBusy || !page?.pagination.has_more
              }
              onClick={() =>
                runQuery({
                  offset:
                    (page?.pagination.offset ?? 0) +
                    (page?.pagination.limit ?? state.sourceQuery.limit)
                })
              }
              className="inline-flex h-9 w-9 items-center justify-center rounded-md outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-40"
            >
              <ChevronRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </Tooltip>
        </div>
      </div>
    </section>
  )
}
