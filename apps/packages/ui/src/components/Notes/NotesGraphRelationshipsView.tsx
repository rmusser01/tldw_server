import type {
  NotesGraphEdgeType,
  NotesGraphResponse,
  NotesGraphSuggestion
} from "@/services/note-graph-suggestions"
import React from "react"
import { useTranslation } from "react-i18next"

import type { ProvisionalNotesGraphOverlay } from "./hooks/useNotesGraphSuggestions"

const PAGE_SIZE = 100
const GROUP_ORDER = ["outgoing", "incoming", "connected", "suggested"] as const
const EDGE_ORDER: Record<
  NotesGraphEdgeType | "provisional_suggestion",
  number
> = {
  manual: 0,
  wikilink: 1,
  backlink: 2,
  tag_membership: 3,
  source_membership: 4,
  provisional_suggestion: 5
}

type RelationshipGroupId = (typeof GROUP_ORDER)[number]

export type NotesGraphRelationshipRow = {
  id: string
  group: RelationshipGroupId
  edgeType: NotesGraphEdgeType | "provisional_suggestion"
  counterpart: { id: string; label: string }
  suggestion: NotesGraphSuggestion | null
}

export type NotesGraphRelationshipGroup = {
  id: RelationshipGroupId
  rows: NotesGraphRelationshipRow[]
}

type BuildGroupsInput = {
  graph: NotesGraphResponse
  selectedNodeId: string | null
  provisionalOverlays: ProvisionalNotesGraphOverlay[]
  suggestions: NotesGraphSuggestion[]
}

const normalizedLabel = (value: string): string =>
  value.normalize("NFKC").trim().toLocaleLowerCase()

export const buildNotesGraphRelationshipGroups = ({
  graph,
  selectedNodeId,
  provisionalOverlays,
  suggestions
}: BuildGroupsInput): NotesGraphRelationshipGroup[] => {
  if (!selectedNodeId) return []
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
  const grouped = new Map<RelationshipGroupId, NotesGraphRelationshipRow[]>()
  const add = (group: RelationshipGroupId, row: NotesGraphRelationshipRow) => {
    grouped.set(group, [...(grouped.get(group) ?? []), row])
  }

  graph.edges.forEach((edge) => {
    if (edge.source !== selectedNodeId && edge.target !== selectedNodeId) return
    const counterpartId =
      edge.source === selectedNodeId ? edge.target : edge.source
    const group = !edge.directed
      ? "connected"
      : edge.source === selectedNodeId
        ? "outgoing"
        : "incoming"
    add(group, {
      id: edge.id,
      group,
      edgeType: edge.type,
      counterpart: {
        id: counterpartId,
        label: nodes.get(counterpartId)?.label ?? counterpartId
      },
      suggestion: null
    })
  })

  const suggestionsById = new Map(suggestions.map((item) => [item.id, item]))
  provisionalOverlays.forEach((overlay) => {
    if (
      overlay.edge.source !== selectedNodeId &&
      overlay.edge.target !== selectedNodeId
    )
      return
    const item = suggestionsById.get(overlay.edge.suggestionId)
    if (!item || item.kind !== "related_note") return
    const counterpartId =
      overlay.edge.source === selectedNodeId
        ? overlay.edge.target
        : overlay.edge.source
    add("suggested", {
      id: overlay.edge.id,
      group: "suggested",
      edgeType: "provisional_suggestion",
      counterpart: {
        id: counterpartId,
        label:
          item.target_title ??
          nodes.get(counterpartId)?.label ??
          overlay.node?.label ??
          counterpartId
      },
      suggestion: item
    })
  })

  const compare = (
    left: NotesGraphRelationshipRow,
    right: NotesGraphRelationshipRow
  ) =>
    EDGE_ORDER[left.edgeType] - EDGE_ORDER[right.edgeType] ||
    normalizedLabel(left.counterpart.label).localeCompare(
      normalizedLabel(right.counterpart.label)
    ) ||
    left.id.localeCompare(right.id)

  return GROUP_ORDER.flatMap((id) => {
    const rows = grouped.get(id)
    return rows?.length ? [{ id, rows: [...rows].sort(compare) }] : []
  })
}

type ReviewRowProps = {
  item: NotesGraphSuggestion
  title: string
  isOnline: boolean
  canAccept: boolean
  canReject: boolean
  onSelect?: () => void
  onAccept?: (suggestionId: string) => void
  onReject?: (suggestionId: string) => void
}

export const NotesGraphSuggestionReviewRow: React.FC<ReviewRowProps> = ({
  item,
  title,
  isOnline,
  canAccept,
  canReject,
  onSelect,
  onAccept,
  onReject
}) => {
  const { t } = useTranslation("option")
  const sourceEvidence = item.evidence.filter(
    (entry) => entry.side === "source"
  )
  const targetEvidence = item.evidence.filter(
    (entry) => entry.side === "target"
  )
  return (
    <div
      className="border-b border-border px-3 py-3 text-sm text-text"
      data-suggestion-review-row={item.id}>
      {onSelect ? (
        <button
          type="button"
          data-testid="notes-graph-relationship-row"
          className="min-h-11 w-full break-words text-left font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          onClick={onSelect}>
          {title}
        </button>
      ) : (
        <h3 className="break-words text-sm font-semibold">{title}</h3>
      )}
      <p className="mt-1 text-xs font-medium text-text-muted">
        {item.match_strength === "strong"
          ? t("notesSearch.graphStrongMatch")
          : t("notesSearch.graphPossibleMatch")}
      </p>
      {item.rationale ? (
        <p className="mt-2 break-words">{item.rationale}</p>
      ) : null}
      {sourceEvidence.length ? (
        <div className="mt-2">
          <h4 className="text-xs font-semibold text-text-muted">
            {t("notesSearch.graphSourceEvidence")}
          </h4>
          {sourceEvidence.map((entry, index) => (
            <p
              key={`${entry.side}:${entry.note_id}:${entry.field}:${index}`}
              className="mt-1 break-words text-xs">
              {entry.text}
            </p>
          ))}
        </div>
      ) : null}
      {targetEvidence.length ? (
        <div className="mt-2">
          <h4 className="text-xs font-semibold text-text-muted">
            {t("notesSearch.graphTargetEvidence")}
          </h4>
          {targetEvidence.map((entry, index) => (
            <p
              key={`${entry.side}:${entry.note_id}:${entry.field}:${index}`}
              className="mt-1 break-words text-xs">
              {entry.text}
            </p>
          ))}
        </div>
      ) : null}
      {onAccept || onReject ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {onAccept ? (
            <button
              type="button"
              className="min-h-11 border border-border bg-primary px-3 text-sm text-primary-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:opacity-50"
              disabled={!isOnline || !canAccept}
              onClick={() => onAccept(item.id)}>
              {t("notesSearch.graphAcceptSuggestion", { title })}
            </button>
          ) : null}
          {onReject ? (
            <button
              type="button"
              className="min-h-11 border border-border bg-surface px-3 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:opacity-50"
              disabled={!isOnline || !canReject}
              onClick={() => onReject(item.id)}>
              {t("notesSearch.graphRejectSuggestion", { title })}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

export type NotesGraphSuggestionDecisionHandler = (
  action: "accept" | "reject",
  suggestionId: string
) => Promise<boolean>

type NotesGraphRelationshipsViewProps = BuildGroupsInput & {
  suggestionsAuthorized: boolean
  isOnline: boolean
  canAccept?: boolean
  canReject?: boolean
  onSelectNode: (nodeId: string) => void
  onDecideSuggestion?: NotesGraphSuggestionDecisionHandler
}

const NotesGraphRelationshipsView: React.FC<
  NotesGraphRelationshipsViewProps
> = ({
  graph,
  selectedNodeId,
  provisionalOverlays,
  suggestions,
  suggestionsAuthorized,
  isOnline,
  canAccept = false,
  canReject = false,
  onSelectNode,
  onDecideSuggestion
}) => {
  const { t } = useTranslation("option")
  const groups = React.useMemo(
    () =>
      buildNotesGraphRelationshipGroups({
        graph,
        selectedNodeId,
        provisionalOverlays: suggestionsAuthorized ? provisionalOverlays : [],
        suggestions: suggestionsAuthorized ? suggestions : []
      }),
    [
      graph,
      provisionalOverlays,
      selectedNodeId,
      suggestions,
      suggestionsAuthorized
    ]
  )
  const rows = React.useMemo(
    () => groups.flatMap((group) => group.rows),
    [groups]
  )
  const [page, setPage] = React.useState(0)
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const safePage = Math.min(page, pageCount - 1)
  const pageRows = rows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE)
  const rowSetMetadata = React.useMemo(() => {
    const metadata = new Map<string, { position: number; setSize: number }>()
    groups.forEach((group) => {
      group.rows.forEach((row, index) => {
        metadata.set(row.id, {
          position: index + 1,
          setSize: group.rows.length
        })
      })
    })
    return metadata
  }, [groups])
  const rootRef = React.useRef<HTMLElement | null>(null)
  const firstRowRef = React.useRef<HTMLDivElement | null>(null)
  const previousPageRef = React.useRef(safePage)

  const decide = async (
    action: "accept" | "reject",
    suggestionId: string,
    origin: HTMLElement | null
  ) => {
    const reviewRows = Array.from(
      rootRef.current?.querySelectorAll<HTMLElement>(
        "[data-suggestion-review-row]"
      ) ?? []
    )
    const currentIndex = reviewRows.findIndex(
      (row) => row.dataset.suggestionReviewRow === suggestionId
    )
    const nextSuggestionId =
      currentIndex >= 0
        ? reviewRows[currentIndex + 1]?.dataset.suggestionReviewRow ?? null
        : null
    let succeeded = false
    try {
      succeeded = (await onDecideSuggestion?.(action, suggestionId)) ?? false
    } catch {
      succeeded = false
    }
    requestAnimationFrame(() => {
      if (!succeeded) {
        origin?.focus()
        return
      }
      const next = Array.from(
        rootRef.current?.querySelectorAll<HTMLElement>(
          "[data-suggestion-review-row]"
        ) ?? []
      ).find((row) => row.dataset.suggestionReviewRow === nextSuggestionId)
      ;(
        next?.querySelector<HTMLButtonElement>("button") ?? rootRef.current
      )?.focus()
    })
  }

  React.useEffect(() => setPage(0), [selectedNodeId])
  React.useEffect(() => {
    const changed = previousPageRef.current !== safePage
    previousPageRef.current = safePage
    if (!changed) return
    firstRowRef.current
      ?.querySelector<HTMLButtonElement>(
        'button[data-testid="notes-graph-relationship-row"]'
      )
      ?.focus()
  }, [safePage])

  return (
    <section
      ref={rootRef}
      tabIndex={-1}
      className="h-full min-h-0 overflow-y-auto bg-bg"
      data-testid="notes-graph-relationships-view"
      aria-label={t("notesSearch.graphRelationships")}>
      {pageRows.length ? (
        <div>
          {GROUP_ORDER.map((groupId) => {
            const groupRows = pageRows.filter((row) => row.group === groupId)
            if (!groupRows.length) return null
            const headingId = `notes-graph-relationship-group-${groupId}`
            return (
              <div key={groupId}>
                <h2
                  id={headingId}
                  className="sticky top-0 border-b border-border bg-surface px-3 py-2 text-xs font-semibold uppercase text-text-muted">
                  {t(`notesSearch.graphRelationshipGroup.${groupId}`)}
                </h2>
                <div role="list" aria-labelledby={headingId}>
                  {groupRows.map((row) => {
                    const setMetadata = rowSetMetadata.get(row.id) ?? {
                      position: 1,
                      setSize: groupRows.length
                    }
                    const firstOnPage = pageRows[0] === row
                    if (row.suggestion) {
                      return (
                        <div
                          ref={firstOnPage ? firstRowRef : undefined}
                          key={row.id}
                          role="listitem"
                          aria-posinset={setMetadata.position}
                          aria-setsize={setMetadata.setSize}>
                          <NotesGraphSuggestionReviewRow
                            item={row.suggestion}
                            title={row.counterpart.label}
                            isOnline={isOnline}
                            canAccept={canAccept}
                            canReject={canReject}
                            onSelect={
                              row.counterpart.id.startsWith("suggestion-node:")
                                ? undefined
                                : () => onSelectNode(row.counterpart.id)
                            }
                            onAccept={
                              onDecideSuggestion
                                ? (suggestionId) => {
                                    const origin =
                                      document.activeElement as HTMLElement | null
                                    void decide("accept", suggestionId, origin)
                                  }
                                : undefined
                            }
                            onReject={
                              onDecideSuggestion
                                ? (suggestionId) => {
                                    const origin =
                                      document.activeElement as HTMLElement | null
                                    void decide("reject", suggestionId, origin)
                                  }
                                : undefined
                            }
                          />
                        </div>
                      )
                    }
                    return (
                      <div
                        ref={firstOnPage ? firstRowRef : undefined}
                        key={row.id}
                        role="listitem"
                        aria-posinset={setMetadata.position}
                        aria-setsize={setMetadata.setSize}
                        className="border-b border-border px-3 py-1">
                        <button
                          type="button"
                          data-testid="notes-graph-relationship-row"
                          className="min-h-11 w-full break-words text-left text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                          onClick={() => onSelectNode(row.counterpart.id)}>
                          {row.counterpart.label}
                        </button>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <p className="p-4 text-sm text-text-muted">
          {t("notesSearch.graphNoRelationships")}
        </p>
      )}
      {pageCount > 1 ? (
        <nav
          className="flex items-center justify-between border-t border-border px-3 py-2"
          aria-label={t("notesSearch.graphRelationshipPages")}>
          <button
            type="button"
            className="min-h-11 px-3 text-sm disabled:opacity-50"
            disabled={safePage === 0}
            onClick={() => setPage((current) => Math.max(0, current - 1))}>
            {t("notesSearch.graphPreviousPage")}
          </button>
          <span className="text-xs text-text-muted">
            {safePage + 1} / {pageCount}
          </span>
          <button
            type="button"
            className="min-h-11 px-3 text-sm disabled:opacity-50"
            disabled={safePage >= pageCount - 1}
            onClick={() =>
              setPage((current) => Math.min(pageCount - 1, current + 1))
            }>
            {t("notesSearch.graphNextPage")}
          </button>
        </nav>
      ) : null}
    </section>
  )
}

export default NotesGraphRelationshipsView
