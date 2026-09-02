import type {
  NotesGraphEdge,
  NotesGraphEdgeType,
  NotesGraphResponse,
  NotesGraphSuggestion
} from "@/services/note-graph-suggestions"
import React from "react"
import { useTranslation } from "react-i18next"

import type { ProvisionalNotesGraphOverlay } from "./hooks/useNotesGraphSuggestions"
import {
  getNotesGraphEdgeLabel,
  groupNotesGraphEdgesByPair
} from "./notes-manager-utils"

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
  semantic: 5,
  provisional_suggestion: 6
}

type RelationshipGroupId = (typeof GROUP_ORDER)[number]

export type NotesGraphRelationshipRow = {
  id: string
  group: RelationshipGroupId
  edgeType: NotesGraphEdgeType | "provisional_suggestion"
  edgeIds: string[]
  edgeTypes: NotesGraphEdgeType[]
  edges: NotesGraphEdge[]
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
  visibleEdgeTypes?: ReadonlySet<NotesGraphEdgeType>
}

const normalizedLabel = (value: string): string =>
  value.normalize("NFKC").trim().toLocaleLowerCase()

export const buildNotesGraphRelationshipGroups = ({
  graph,
  selectedNodeId,
  provisionalOverlays,
  suggestions,
  visibleEdgeTypes
}: BuildGroupsInput): NotesGraphRelationshipGroup[] => {
  if (!selectedNodeId) return []
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
  const grouped = new Map<RelationshipGroupId, NotesGraphRelationshipRow[]>()
  const add = (group: RelationshipGroupId, row: NotesGraphRelationshipRow) => {
    grouped.set(group, [...(grouped.get(group) ?? []), row])
  }

  const visibleEdges = graph.edges.filter(
    (edge) =>
      (!visibleEdgeTypes || visibleEdgeTypes.has(edge.type)) &&
      (edge.source === selectedNodeId || edge.target === selectedNodeId)
  )
  groupNotesGraphEdgesByPair(visibleEdges).forEach((edgeGroup) => {
    const representative =
      edgeGroup.edges.find((edge) => edge.type === "manual") ??
      edgeGroup.edges.find(
        (edge) => edge.type === "wikilink" || edge.type === "backlink"
      ) ??
      edgeGroup.edges[0]
    const counterpartId =
      representative.source === selectedNodeId
        ? representative.target
        : representative.source
    const group = !representative.directed
      ? "connected"
      : representative.source === selectedNodeId
        ? "outgoing"
        : "incoming"
    add(group, {
      id: edgeGroup.id,
      group,
      edgeType: representative.type,
      edgeIds: edgeGroup.edges.map((edge) => edge.id),
      edgeTypes: edgeGroup.edges.map((edge) => edge.type),
      edges: edgeGroup.edges,
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
      edgeIds: [],
      edgeTypes: [],
      edges: [],
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

export type NotesGraphManualLinkHandler = (
  edge: NotesGraphEdge,
  origin: HTMLElement | null
) => Promise<boolean>

type SemanticRelationshipDetailsProps = {
  edge: NotesGraphEdge
  manualLinkAuthorized: boolean
  isOnline: boolean
  hasManualRelationship: boolean
  manualLinkPending?: boolean
  onCreateManualLink?: NotesGraphManualLinkHandler
  showHeading?: boolean
}

export const NotesSemanticRelationshipDetails: React.FC<
  SemanticRelationshipDetailsProps
> = ({
  edge,
  manualLinkAuthorized,
  isOnline,
  hasManualRelationship,
  manualLinkPending = false,
  onCreateManualLink,
  showHeading = true
}) => {
  const { t } = useTranslation("option")
  const evidence = edge.evidence
  const similarity = evidence?.similarity ?? edge.weight

  return (
    <section
      className={
        showHeading
          ? "mt-2 min-w-0 border-t border-border pt-2 text-xs text-text"
          : "min-w-0 pb-3 text-xs text-text"
      }
      aria-label={t("notesSearch.graphSimilarContent")}>
      {showHeading ? (
        <h3 className="font-semibold">
          {t("notesSearch.graphSimilarContent")}
        </h3>
      ) : null}
      {similarity !== null ? (
        <p className="mt-1 break-words">
          {t("notesSearch.graphPassageSimilarity", {
            value: String(similarity)
          })}
        </p>
      ) : null}
      {evidence ? (
        <>
          <p className="mt-1 font-medium text-text-muted">
            {t(`notesSearch.graphSimilarityBand.${evidence.qualitative_band}`)}
          </p>
          <dl className="mt-2 grid min-w-0 grid-cols-[auto,minmax(0,1fr)] gap-x-2 gap-y-1 text-text-muted">
            <dt>{t("notesSearch.graphSemanticProvider")}</dt>
            <dd className="break-words text-text">
              {t("notesSearch.graphSemanticProviderModel", {
                provider: evidence.provider_label,
                model: evidence.model_label
              })}
            </dd>
            <dt>{t("notesSearch.graphSemanticFreshness")}</dt>
            <dd className="break-words text-text">
              {t("notesSearch.graphSemanticVersions", {
                source: evidence.source_content_version,
                target: evidence.target_content_version
              })}
            </dd>
            <dt>{t("notesSearch.graphSemanticGenerationLabel")}</dt>
            <dd className="break-all text-text">
              {t("notesSearch.graphSemanticGeneration", {
                generation: evidence.generation_id
              })}
            </dd>
            <dt>{t("notesSearch.graphSemanticIndexVersion")}</dt>
            <dd className="break-words text-text">
              {t("notesSearch.graphSemanticIndexVersionValue", {
                index: evidence.semantic_index_revision,
                configuration: evidence.configuration_revision
              })}
            </dd>
            <dt>{t("notesSearch.graphSemanticModelRevision")}</dt>
            <dd className="break-words text-text">
              {evidence.model_revision ??
                t("notesSearch.graphSemanticVersionUnavailable")}
            </dd>
            <dt>{t("notesSearch.graphSemanticNormalizationVersion")}</dt>
            <dd className="break-words text-text">
              {evidence.normalization_version}
            </dd>
            <dt>{t("notesSearch.graphSemanticChunkerVersion")}</dt>
            <dd className="break-words text-text">
              {evidence.chunker_version}
            </dd>
          </dl>
          {evidence.excerpt_pairs.slice(0, 3).map((pair, index) => (
            <div
              key={`${edge.id}:evidence:${index}`}
              className="mt-3 grid min-w-0 grid-cols-1 gap-2 xl:grid-cols-2">
              <blockquote className="min-w-0 border-l-2 border-border pl-2">
                <span className="block font-semibold text-text-muted">
                  {t("notesSearch.graphSourceEvidence")}
                </span>
                <span className="mt-1 block break-words">
                  {pair.source.text}
                </span>
              </blockquote>
              <blockquote className="min-w-0 border-l-2 border-border pl-2">
                <span className="block font-semibold text-text-muted">
                  {t("notesSearch.graphTargetEvidence")}
                </span>
                <span className="mt-1 block break-words">
                  {pair.target.text}
                </span>
              </blockquote>
            </div>
          ))}
        </>
      ) : edge.evidence_omitted ? (
        <p className="mt-2 break-words text-text-muted">
          {t("notesSearch.graphEvidenceOmitted")}
        </p>
      ) : null}
      {manualLinkAuthorized &&
      (evidence || edge.evidence_omitted === "response_byte_cap") &&
      !hasManualRelationship &&
      onCreateManualLink ? (
        <button
          type="button"
          className="mt-3 min-h-11 border border-border bg-primary px-3 text-sm text-primary-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:opacity-50"
          disabled={!isOnline || manualLinkPending}
          onClick={(event) => {
            const origin = event.currentTarget
            void onCreateManualLink(edge, origin).then((succeeded) => {
              requestAnimationFrame(() => {
                const row = origin.closest<HTMLElement>(
                  "[data-notes-graph-relationship-group]"
                )
                const nextFocus = succeeded
                  ? row?.querySelector<HTMLButtonElement>(
                      '[data-testid="notes-graph-relationship-row"]'
                    ) ?? document.getElementById("notes-graph-details-tab")
                  : origin
                nextFocus?.focus()
              })
            })
          }}>
          {t("notesSearch.graphCreateManualLink")}
        </button>
      ) : null}
    </section>
  )
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
  manualLinkAuthorized?: boolean
  isOnline: boolean
  canAccept?: boolean
  canReject?: boolean
  onSelectNode: (nodeId: string) => void
  onSelectEdge?: (edgeId: string) => void
  onCreateManualLink?: NotesGraphManualLinkHandler
  onDecideSuggestion?: NotesGraphSuggestionDecisionHandler
  manualLinkPendingEdgeIds?: ReadonlySet<string>
  queryIdentity?: string
}

const NotesGraphRelationshipsView: React.FC<
  NotesGraphRelationshipsViewProps
> = ({
  graph,
  selectedNodeId,
  provisionalOverlays,
  suggestions,
  visibleEdgeTypes,
  suggestionsAuthorized,
  manualLinkAuthorized = false,
  isOnline,
  canAccept = false,
  canReject = false,
  onSelectNode,
  onSelectEdge,
  onCreateManualLink,
  onDecideSuggestion,
  manualLinkPendingEdgeIds,
  queryIdentity
}) => {
  const { t } = useTranslation("option")
  const groups = React.useMemo(
    () =>
      buildNotesGraphRelationshipGroups({
        graph,
        selectedNodeId,
        provisionalOverlays: suggestionsAuthorized ? provisionalOverlays : [],
        suggestions: suggestionsAuthorized ? suggestions : [],
        visibleEdgeTypes
      }),
    [
      graph,
      provisionalOverlays,
      selectedNodeId,
      suggestions,
      suggestionsAuthorized,
      visibleEdgeTypes
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
  const edgeFilterIdentity = React.useMemo(
    () =>
      Array.from(visibleEdgeTypes ?? [])
        .sort()
        .join(","),
    [visibleEdgeTypes]
  )

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

  React.useEffect(
    () => setPage(0),
    [edgeFilterIdentity, queryIdentity, selectedNodeId]
  )
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
                        data-notes-graph-relationship-group={row.id}
                        role="listitem"
                        aria-posinset={setMetadata.position}
                        aria-setsize={setMetadata.setSize}
                        className="border-b border-border px-3 py-1">
                        <button
                          type="button"
                          data-testid="notes-graph-relationship-row"
                          aria-describedby={`${row.id}-types`}
                          className="min-h-11 w-full break-words text-left text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                          onClick={() => onSelectNode(row.counterpart.id)}>
                          {row.counterpart.label}
                        </button>
                        <p id={`${row.id}-types`} className="sr-only">
                          {row.edgeTypes
                            .map((edgeType) =>
                              t(`notesSearch.graphEdgeType.${edgeType}`, {
                                defaultValue: getNotesGraphEdgeLabel(edgeType)
                              })
                            )
                            .join(", ")}
                        </p>
                        <div className="flex min-w-0 flex-wrap gap-1 pb-2">
                          {row.edges.map((edge) => (
                            <button
                              key={edge.id}
                              type="button"
                              className="min-h-9 border border-border bg-surface px-2 text-xs text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                              onClick={() => onSelectEdge?.(edge.id)}>
                              {t(`notesSearch.graphEdgeType.${edge.type}`, {
                                defaultValue: getNotesGraphEdgeLabel(edge.type)
                              })}
                            </button>
                          ))}
                        </div>
                        {row.edges
                          .filter((edge) => edge.type === "semantic")
                          .map((edge) => {
                            const evidence = edge.evidence
                            const similarity =
                              evidence?.similarity ?? edge.weight
                            return (
                              <details
                                key={edge.id}
                                className="border-t border-border text-xs text-text">
                                <summary
                                  data-testid="notes-graph-semantic-evidence-toggle"
                                  className="min-h-11 cursor-pointer py-2 pl-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus">
                                  <span
                                    data-testid="notes-graph-semantic-treatment-label"
                                    className="font-semibold">
                                    {t("notesSearch.graphSimilarContent")}
                                  </span>
                                  <span className="ml-2 text-text-muted">
                                    {evidence
                                      ? t(
                                          `notesSearch.graphSimilarityBand.${evidence.qualitative_band}`
                                        )
                                      : null}
                                    {evidence && similarity !== null
                                      ? " / "
                                      : null}
                                    {similarity !== null
                                      ? t(
                                          "notesSearch.graphPassageSimilarity",
                                          {
                                            value: String(similarity)
                                          }
                                        )
                                      : null}
                                  </span>
                                </summary>
                                <div className="pl-5">
                                  <NotesSemanticRelationshipDetails
                                    edge={edge}
                                    manualLinkAuthorized={manualLinkAuthorized}
                                    isOnline={isOnline}
                                    hasManualRelationship={row.edgeTypes.includes(
                                      "manual"
                                    )}
                                    manualLinkPending={manualLinkPendingEdgeIds?.has(
                                      edge.id
                                    )}
                                    onCreateManualLink={onCreateManualLink}
                                    showHeading={false}
                                  />
                                </div>
                              </details>
                            )
                          })}
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
      {graph.semantic_status?.truncated_by.length ? (
        <p
          className="border-t border-border p-3 text-xs text-warn"
          role="status">
          {t("notesSearch.graphSemanticTruncated")}
        </p>
      ) : null}
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
