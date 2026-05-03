import { useMemo, useState } from "react"
import type { WorkspaceSource } from "@/types/workspace"

export type WorkspaceRailProps = {
  workspaceName: string
  sources: WorkspaceSource[]
  browsedSourceId: string | null
  stagedSourceIds: string[]
  onBrowseSource: (sourceId: string) => void
  onStageSources: (sourceIds: string[]) => void
}

const panelClass = "rounded-md border border-border bg-surface px-3 py-2"
const headingClass = "text-[11px] font-semibold text-text-muted"
const buttonClass =
  "inline-flex min-h-[28px] min-w-0 items-center justify-center break-words rounded-md border border-border px-2.5 py-1 text-xs font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

const getSourceStatus = (source: WorkspaceSource) => source.status || "ready"

export const WorkspaceRail = ({
  workspaceName,
  sources,
  browsedSourceId,
  stagedSourceIds,
  onBrowseSource,
  onStageSources
}: WorkspaceRailProps) => {
  const [filter, setFilter] = useState("")
  const stagedSourceIdSet = useMemo(
    () => new Set(stagedSourceIds),
    [stagedSourceIds]
  )
  const normalizedFilter = filter.trim().toLowerCase()
  const visibleSources = useMemo(
    () =>
      normalizedFilter
        ? sources.filter((source) =>
            source.title.toLowerCase().includes(normalizedFilter)
          )
        : sources,
    [normalizedFilter, sources]
  )
  const titleCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const source of visibleSources) {
      counts.set(source.title, (counts.get(source.title) ?? 0) + 1)
    }
    return counts
  }, [visibleSources])

  return (
    <aside
      aria-label="Chat workspace sources"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section className={panelClass}>
        <p className={headingClass}>Workspace</p>
        <h2 className="mt-1 min-w-0 break-words text-sm font-semibold text-text">
          {workspaceName}
        </h2>
      </section>

      <section className={panelClass}>
        <label
          htmlFor="chat-workspace-source-filter"
          className={headingClass}
        >
          Filter sources
        </label>
        <input
          id="chat-workspace-source-filter"
          type="search"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          className="mt-2 min-h-[32px] w-full rounded-md border border-border bg-surface2 px-2 py-1 text-sm text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        />
      </section>

      <section className={panelClass}>
        <h3 className={headingClass}>Library</h3>
        <div className="mt-2 flex flex-wrap gap-2">
          <button type="button" className={buttonClass} disabled>
            Add source unavailable
          </button>
          <button type="button" className={buttonClass} disabled>
            Open library unavailable
          </button>
        </div>
      </section>

      <section className={panelClass}>
        <h3 className={headingClass}>Sources</h3>
        {visibleSources.length > 0 ? (
          <ul className="mt-2 space-y-2">
            {visibleSources.map((source) => {
              const status = getSourceStatus(source)
              const isReady = status === "ready"
              const isStaged = stagedSourceIdSet.has(source.id)
              const isBrowsed = browsedSourceId === source.id
              const hasDuplicateTitle = (titleCounts.get(source.title) ?? 0) > 1
              const actionNameSuffix = hasDuplicateTitle ? ` ${source.id}` : ""

              return (
                <li
                  key={source.id}
                  className="min-w-0 rounded-md border border-border bg-surface2/50 px-2 py-1.5"
                >
                  <div className="flex min-w-0 flex-col gap-1">
                    <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="min-w-0 break-words text-sm font-medium text-text">
                        {source.title}
                      </span>
                      <span className="text-xs text-text-muted">{source.type}</span>
                      {!isReady ? (
                        <span className="text-xs text-text-muted">{status}</span>
                      ) : null}
                      {isStaged ? (
                        <span className="text-xs font-medium text-text">
                          Context staged
                        </span>
                      ) : null}
                      {isBrowsed ? (
                        <span className="text-xs text-text-muted">Browsing</span>
                      ) : null}
                    </div>
                    {source.statusMessage ? (
                      <p className="min-w-0 break-words text-xs text-text-muted">
                        {source.statusMessage}
                      </p>
                    ) : null}
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        className={buttonClass}
                        aria-label={`Browse ${source.title}${actionNameSuffix}`}
                        onClick={() => onBrowseSource(source.id)}
                      >
                        Browse {source.title}
                      </button>
                      <button
                        type="button"
                        className={buttonClass}
                        aria-label={`Stage ${source.title}${actionNameSuffix} for chat`}
                        disabled={!isReady}
                        onClick={() => {
                          if (isReady) {
                            onStageSources([source.id])
                          }
                        }}
                      >
                        Stage {source.title} for chat
                      </button>
                    </div>
                  </div>
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="mt-2 rounded-md border border-dashed border-border bg-surface2/40 px-2 py-1.5 text-xs text-text-muted">
            No sources match the filter
          </p>
        )}
      </section>

      <section className={panelClass}>
        <h3 className={headingClass}>Study</h3>
        <p className="mt-1 text-xs text-text-muted">No generated study set</p>
      </section>
    </aside>
  )
}
