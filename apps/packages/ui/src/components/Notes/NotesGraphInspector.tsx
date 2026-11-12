import IconButton from "@/components/Common/IconButton"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import type {
  NotesGraphEdge,
  NotesGraphResponse,
  NotesGraphSuggestion
} from "@/services/note-graph-suggestions"
import {
  NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES,
  type NotesSemanticCapabilities,
  NotesSemanticClientError,
  type NotesSemanticIndexStatus,
  type NotesSemanticRun
} from "@/services/note-semantic-index"
import { Ellipsis } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import {
  type NotesGraphManualLinkHandler,
  type NotesGraphSuggestionDecisionHandler,
  NotesGraphSuggestionReviewRow,
  NotesSemanticRelationshipDetails,
  buildNotesGraphRelationshipGroups
} from "./NotesGraphRelationshipsView"
import type { NotesGraphSuggestionsController } from "./hooks/useNotesGraphSuggestions"
import type { NotesSemanticIndexController } from "./hooks/useNotesSemanticIndex"
import { getNotesGraphEdgeLabel } from "./notes-manager-utils"

type NotesGraphInspectorProps = {
  graph: NotesGraphResponse
  selectedNodeId: string | null
  selectedEdgeId?: string | null
  suggestionsAuthorized: boolean
  manualLinkAuthorized?: boolean
  isOnline: boolean
  controller: NotesGraphSuggestionsController
  semanticController?: NotesSemanticIndexController
  semanticEnabled?: boolean
  onSemanticEnabledChange?: (enabled: boolean) => void
  onSelectNode: (nodeId: string) => void
  onSelectEdge?: (edgeId: string) => void
  onCreateManualLink?: NotesGraphManualLinkHandler
  manualLinkPendingEdgeIds?: ReadonlySet<string>
  onAnnounce: (message: string) => void
  onDecideSuggestion: NotesGraphSuggestionDecisionHandler
}

const pending = (mutation: unknown): boolean =>
  Boolean((mutation as { isPending?: boolean } | undefined)?.isPending)

const CANCELLABLE_SEMANTIC_RUN_MODES = new Set([
  "build",
  "rebuild",
  "retry_failed"
])
const SEMANTIC_DETAIL_REASONS = new Set([
  "building",
  "degraded",
  "stale_configuration",
  "consent_required",
  "cleanup_pending",
  "cleanup_stalled",
  "unavailable",
  "rebuild_required"
])
type SemanticAction =
  | "enable"
  | "renew"
  | "rebuild"
  | "retry"
  | "cancel"
  | "deleteIndex"

const semanticDetailKey = (reason: string): string =>
  SEMANTIC_DETAIL_REASONS.has(reason)
    ? `notesSearch.semanticDetail.${reason}`
    : "notesSearch.semanticDetail.generic"

const hasCompleteConsentDisclosure = (
  capability: NotesSemanticCapabilities | null
): boolean => {
  if (!capability) return false
  const outbound = new Set(capability.outbound_data_categories)
  const hasIdentity = [
    capability.provider_label,
    capability.model,
    capability.storage_label
  ].every((value) => value.trim().length > 0)
  const dimensionsAreCoherent =
    (capability.resolved_dimensions === null) ===
    capability.dimension_probe_required
  return Boolean(
    capability.indexing_available &&
      hasIdentity &&
      capability.endpoint_display !== null &&
      capability.endpoint_display.trim().length > 0 &&
      capability.storage_boundary !== "unavailable" &&
      capability.unavailable_reason === null &&
      dimensionsAreCoherent &&
      capability.outbound_data_categories.length ===
        NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES.length &&
      NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES.every((category) =>
        outbound.has(category)
      )
  )
}

const semanticManagementActions = ({
  capability,
  status,
  activeRun
}: {
  capability: NotesSemanticCapabilities | null
  status: NotesSemanticIndexStatus | null
  activeRun: NotesSemanticRun | null
}): SemanticAction[] => {
  if (!capability?.manage_authorized || !status) return []
  const cleanupBlocked = Boolean(
    status.cleanup_pending ||
      status.detail_reason === "cleanup_pending" ||
      status.detail_reason === "cleanup_stalled"
  )
  if (cleanupBlocked) return []
  if (activeRun) {
    return status.desired_state === "enabled" &&
      CANCELLABLE_SEMANTIC_RUN_MODES.has(activeRun.mode)
      ? ["cancel"]
      : []
  }
  const capabilityUsable = hasCompleteConsentDisclosure(capability)
  if (status.desired_state === "disabled") {
    return status.state === "off" &&
      status.detail_reason === null &&
      capabilityUsable &&
      !capability.renewal_requires_delete
      ? ["enable"]
      : []
  }
  if (
    status.detail_reason === "stale_configuration" ||
    status.detail_reason === "consent_required"
  ) {
    return capabilityUsable && !capability.renewal_requires_delete
      ? ["renew", "deleteIndex"]
      : ["deleteIndex"]
  }
  if (!capabilityUsable) {
    return ["deleteIndex"]
  }
  if (!status.active_generation_usable) return ["rebuild", "deleteIndex"]
  if (status.state === "ready") return ["rebuild", "deleteIndex"]
  if (
    status.state === "needs_attention" &&
    status.detail_reason === "degraded"
  ) {
    return [
      ...(status.failed_notes > 0 ? (["retry"] as const) : []),
      "rebuild",
      "deleteIndex"
    ]
  }
  return ["deleteIndex"]
}

const semanticMutationErrorKey = (error: unknown): string => {
  if (!(error instanceof NotesSemanticClientError)) {
    return "notesSearch.semanticActionFailed"
  }
  switch (error.code) {
    case "notes_semantic_permission_denied":
      return "notesSearch.semanticError.permission"
    case "notes_semantic_capability_revision_conflict":
    case "notes_semantic_configuration_revision_conflict":
    case "notes_semantic_run_revision_conflict":
    case "notes_semantic_active_generation_required":
      return "notesSearch.semanticError.refresh"
    case "notes_semantic_backend_change_requires_delete":
      return "notesSearch.semanticError.backendChangeRequiresDelete"
    case "notes_semantic_writer_conflict":
      return "notesSearch.semanticError.writerConflict"
    case "notes_semantic_quota_exceeded":
      return "notesSearch.semanticError.quota"
    case "notes_semantic_dataset_authority_unavailable":
    case "notes_semantic_jobs_unavailable":
    case "notes_semantic_provider_unavailable":
      return "notesSearch.semanticError.unavailable"
    default:
      return "notesSearch.semanticActionFailed"
  }
}

const NotesGraphInspector: React.FC<NotesGraphInspectorProps> = ({
  graph,
  selectedNodeId,
  selectedEdgeId = null,
  suggestionsAuthorized,
  manualLinkAuthorized = false,
  isOnline,
  controller,
  semanticController,
  semanticEnabled = false,
  onSemanticEnabledChange,
  onSelectNode,
  onSelectEdge,
  onCreateManualLink,
  manualLinkPendingEdgeIds,
  onAnnounce,
  onDecideSuggestion
}) => {
  const { t } = useTranslation("option")
  const confirmDanger = useConfirmDanger()
  const [tab, setTab] = React.useState<"details" | "semantic" | "suggestions">(
    "details"
  )
  const [menuOpen, setMenuOpen] = React.useState(false)
  const [semanticActionError, setSemanticActionError] = React.useState<
    string | null
  >(null)
  const inspectorRef = React.useRef<HTMLElement | null>(null)
  const detailsTabRef = React.useRef<HTMLButtonElement | null>(null)
  const suggestionsTabRef = React.useRef<HTMLButtonElement | null>(null)
  const semanticTabRef = React.useRef<HTMLButtonElement | null>(null)
  const semanticHeadingRef = React.useRef<HTMLHeadingElement | null>(null)
  const semanticActionRegionRef = React.useRef<HTMLDivElement | null>(null)
  const semanticActionWasFocused = React.useRef(false)
  const previousSemanticRunId = React.useRef<string | null>(null)
  const suggestionHeadingRef = React.useRef<HTMLHeadingElement | null>(null)
  const menuTriggerRef = React.useRef<HTMLButtonElement | null>(null)
  const menuItemRef = React.useRef<HTMLButtonElement | null>(null)
  const selectedNode =
    graph.nodes.find((node) => node.id === selectedNodeId) ?? null
  const selectedEdge =
    graph.edges.find((edge) => edge.id === selectedEdgeId) ?? null
  const selectedSemanticEdge =
    selectedEdge?.type === "semantic" ? selectedEdge : null
  const selectedEdgeNodes = selectedSemanticEdge
    ? [
        graph.nodes.find((node) => node.id === selectedSemanticEdge.source),
        graph.nodes.find((node) => node.id === selectedSemanticEdge.target)
      ]
    : []
  const selectedPairHasManual = selectedSemanticEdge
    ? graph.edges.some(
        (edge) =>
          edge.type === "manual" &&
          ((edge.source === selectedSemanticEdge.source &&
            edge.target === selectedSemanticEdge.target) ||
            (edge.source === selectedSemanticEdge.target &&
              edge.target === selectedSemanticEdge.source))
      )
    : false
  const suggestions = controller.suggestions ?? []
  const capabilities = controller.capabilities ?? null
  const activeRun = controller.activeRun ?? null
  const visibleRun = activeRun ?? controller.lastTerminalRun ?? null
  const allowedActions = new Set(capabilities?.allowed_actions ?? [])
  const isDecisionPending =
    pending(controller.mutations?.acceptance) ||
    pending(controller.mutations?.rejection)

  React.useEffect(() => {
    if (!suggestionsAuthorized && tab === "suggestions") setTab("details")
  }, [suggestionsAuthorized, tab])

  React.useEffect(() => {
    if (!semanticController && tab === "semantic") setTab("details")
  }, [semanticController, tab])

  React.useEffect(() => {
    if (selectedEdgeId) setTab("details")
  }, [selectedEdgeId])

  const semanticStatus = semanticController?.status ?? null
  const semanticCapabilities = semanticController?.capabilities ?? null
  const semanticActiveRun = semanticController?.activeRun ?? null
  const activeIndexingRun =
    semanticActiveRun &&
    CANCELLABLE_SEMANTIC_RUN_MODES.has(semanticActiveRun.mode)
      ? semanticActiveRun
      : null
  const semanticDisplayState =
    activeIndexingRun && semanticStatus
      ? semanticStatus.active_generation_id
        ? "updating"
        : "preparing"
      : semanticStatus?.state
  const semanticProgress = activeIndexingRun ?? semanticStatus
  const semanticDetail = semanticCapabilities?.renewal_requires_delete
    ? t("notesSearch.semanticBackendChangeRequiresDelete")
    : semanticStatus?.detail_reason
      ? t(semanticDetailKey(semanticStatus.detail_reason))
      : ""
  const semanticActions = new Set(
    semanticManagementActions({
      capability: semanticCapabilities,
      status: semanticStatus,
      activeRun: semanticActiveRun
    })
  )
  const semanticEdgesUsable = Boolean(semanticStatus?.active_generation_usable)
  const semanticAnnouncement = semanticStatus
    ? t("notesSearch.semanticStatusAnnouncement", {
        state: t(`notesSearch.semanticState.${semanticDisplayState}`),
        detail: semanticDetail
      }).trim()
    : ""
  const previousSemanticAnnouncement = React.useRef("")
  React.useEffect(() => {
    if (
      !semanticAnnouncement ||
      previousSemanticAnnouncement.current === semanticAnnouncement
    )
      return
    previousSemanticAnnouncement.current = semanticAnnouncement
    onAnnounce(semanticAnnouncement)
  }, [onAnnounce, semanticAnnouncement])

  const focusSemanticStatus = React.useCallback(() => {
    requestAnimationFrame(() => semanticHeadingRef.current?.focus())
  }, [])

  React.useEffect(() => {
    const previous = previousSemanticRunId.current
    previousSemanticRunId.current = semanticActiveRun?.run_id ?? null
    if (previous && !semanticActiveRun && semanticActionWasFocused.current) {
      semanticActionWasFocused.current = false
      focusSemanticStatus()
    }
  }, [focusSemanticStatus, semanticActiveRun])

  React.useEffect(() => {
    if (!menuOpen) return
    const frame = requestAnimationFrame(() => menuItemRef.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [menuOpen])

  const handleTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    const tabs = [
      "details",
      ...(suggestionsAuthorized ? (["suggestions"] as const) : []),
      ...(semanticController ? (["semantic"] as const) : [])
    ] as const
    let next: (typeof tabs)[number] | null = null
    if (event.key === "Home") next = tabs[0]
    if (event.key === "End") next = tabs[tabs.length - 1]
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      const direction = event.key === "ArrowRight" ? 1 : -1
      const current = Math.max(0, tabs.indexOf(tab))
      next = tabs[(current + direction + tabs.length) % tabs.length]
    }
    if (!next) return
    event.preventDefault()
    setTab(next)
    const refs = {
      details: detailsTabRef,
      semantic: semanticTabRef,
      suggestions: suggestionsTabRef
    }
    refs[next].current?.focus()
  }

  const detailGroups = React.useMemo(
    () =>
      buildNotesGraphRelationshipGroups({
        graph,
        selectedNodeId,
        provisionalOverlays: [],
        suggestions: []
      }),
    [graph, selectedNodeId]
  )
  const connectedNodes = React.useMemo(() => {
    if (!selectedNodeId) return []
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
    return graph.edges.flatMap((edge) => {
      if (edge.source !== selectedNodeId && edge.target !== selectedNodeId)
        return []
      const id = edge.source === selectedNodeId ? edge.target : edge.source
      const node = nodes.get(id)
      return node ? [{ edge, node }] : []
    })
  }, [graph, selectedNodeId])
  const tags = connectedNodes
    .filter(
      ({ edge, node }) => edge.type === "tag_membership" && node.type === "tag"
    )
    .map(({ node }) => node.label)
  const source = connectedNodes.find(
    ({ edge, node }) =>
      edge.type === "source_membership" && node.type === "source"
  )?.node

  const focusAfterRemoval = (nextSuggestionId: string | null) => {
    requestAnimationFrame(() => {
      const rows = Array.from(
        inspectorRef.current?.querySelectorAll<HTMLElement>(
          "[data-suggestion-review-row]"
        ) ?? []
      )
      const next = rows.find(
        (row) => row.dataset.suggestionReviewRow === nextSuggestionId
      )
      const button = next?.querySelector<HTMLButtonElement>("button")
      ;(button ?? suggestionHeadingRef.current)?.focus()
    })
  }

  const decide = async (
    action: "accept" | "reject",
    item: NotesGraphSuggestion,
    origin: HTMLElement | null
  ) => {
    const index = suggestions.findIndex((entry) => entry.id === item.id)
    const nextSuggestionId = suggestions[index + 1]?.id ?? null
    const succeeded = await onDecideSuggestion(action, item.id)
    if (succeeded) {
      focusAfterRemoval(nextSuggestionId)
    } else {
      requestAnimationFrame(() => origin?.focus())
    }
  }

  const runGenerate = async () => {
    try {
      await controller.generate()
      onAnnounce(t("notesSearch.graphGenerationStarted"))
    } catch {
      onAnnounce(t("notesSearch.graphGenerationFailed"))
    }
  }

  const runCancel = async () => {
    try {
      await controller.cancel()
      onAnnounce(t("notesSearch.graphCancellationRequested"))
    } catch {
      onAnnounce(t("notesSearch.graphCancellationFailed"))
    }
  }

  const resetRejections = async () => {
    setMenuOpen(false)
    const confirmed = await confirmDanger({
      title: t("notesSearch.graphResetDismissedTitle"),
      content: t("notesSearch.graphResetDismissedBody"),
      okText: t("notesSearch.graphResetDismissedConfirm"),
      cancelText: t("notesSearch.graphCancel")
    })
    if (!confirmed) return
    try {
      await controller.resetRejections()
      onAnnounce(t("notesSearch.graphDismissedReset"))
    } catch {
      onAnnounce(t("notesSearch.graphDismissedResetFailed"))
    }
  }

  const runSemanticAction = async (action: SemanticAction) => {
    if (!semanticController) return
    const confirmed = await confirmDanger({
      title: t(`notesSearch.semanticConfirm.${action}.title`),
      content: t(`notesSearch.semanticConfirm.${action}.body`),
      okText: t(`notesSearch.semanticConfirm.${action}.confirm`),
      cancelText: t("notesSearch.graphCancel")
    })
    if (!confirmed) return
    setSemanticActionError(null)
    const commands = {
      enable: semanticController.enable,
      renew: semanticController.enable,
      rebuild: semanticController.rebuild,
      retry: semanticController.retryFailed,
      cancel: semanticController.cancel,
      deleteIndex: semanticController.deleteIndex
    }
    const successKeys = {
      enable: "notesSearch.semanticEnableStarted",
      renew: "notesSearch.semanticRenewStarted",
      rebuild: "notesSearch.semanticRebuildStarted",
      retry: "notesSearch.semanticRetryStarted",
      cancel: "notesSearch.semanticCancelRequested",
      deleteIndex: "notesSearch.semanticDeleteStarted"
    }
    try {
      await commands[action]()
      onAnnounce(t(successKeys[action]))
      focusSemanticStatus()
    } catch (error) {
      const errorKey = semanticMutationErrorKey(error)
      setSemanticActionError(errorKey)
      onAnnounce(t(errorKey))
    }
  }

  return (
    <aside
      ref={inspectorRef}
      className="min-w-0 bg-bg text-text"
      aria-label={t("notesSearch.graphInspector")}>
      <div
        className="border-b border-border px-3 pt-3"
        role="tablist"
        aria-label={t("notesSearch.graphInspectorTabs")}>
        <button
          ref={detailsTabRef}
          id="notes-graph-details-tab"
          type="button"
          role="tab"
          tabIndex={tab === "details" ? 0 : -1}
          aria-selected={tab === "details"}
          aria-controls="notes-graph-details-panel"
          className="min-h-11 border-b-2 border-transparent px-3 text-sm aria-selected:border-primary aria-selected:font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
          onKeyDown={handleTabKeyDown}
          onClick={() => setTab("details")}>
          {t("notesSearch.graphDetails")}
        </button>
        {suggestionsAuthorized ? (
          <button
            ref={suggestionsTabRef}
            id="notes-graph-suggestions-tab"
            type="button"
            role="tab"
            tabIndex={tab === "suggestions" ? 0 : -1}
            aria-selected={tab === "suggestions"}
            aria-controls="notes-graph-suggestions-panel"
            className="min-h-11 border-b-2 border-transparent px-3 text-sm aria-selected:border-primary aria-selected:font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
            onKeyDown={handleTabKeyDown}
            onClick={() => setTab("suggestions")}>
            {t("notesSearch.graphSuggestions")}
          </button>
        ) : null}
        {semanticController ? (
          <button
            ref={semanticTabRef}
            id="notes-graph-semantic-tab"
            type="button"
            role="tab"
            tabIndex={tab === "semantic" ? 0 : -1}
            aria-selected={tab === "semantic"}
            aria-controls="notes-graph-semantic-panel"
            className="min-h-11 border-b-2 border-transparent px-3 text-sm aria-selected:border-primary aria-selected:font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
            onKeyDown={handleTabKeyDown}
            onClick={() => setTab("semantic")}>
            {t("notesSearch.graphSimilarContent")}
          </button>
        ) : null}
      </div>

      {tab === "details" ? (
        <div
          id="notes-graph-details-panel"
          role="tabpanel"
          aria-labelledby="notes-graph-details-tab"
          className="p-3">
          {selectedSemanticEdge ? (
            <div className="min-w-0">
              <h2 className="break-words text-base font-semibold">
                {t("notesSearch.graphSimilarContent")}
              </h2>
              <p className="mt-1 break-words text-sm text-text-muted">
                {selectedEdgeNodes
                  .map((node) => node?.label ?? "")
                  .filter(Boolean)
                  .join(" / ")}
              </p>
              <NotesSemanticRelationshipDetails
                edge={selectedSemanticEdge}
                manualLinkAuthorized={manualLinkAuthorized}
                isOnline={isOnline}
                hasManualRelationship={selectedPairHasManual}
                manualLinkPending={manualLinkPendingEdgeIds?.has(
                  selectedSemanticEdge.id
                )}
                onCreateManualLink={onCreateManualLink}
                showHeading={false}
              />
              {graph.semantic_status?.truncated_by.length ? (
                <p className="mt-3 break-words text-xs text-warn" role="status">
                  {t("notesSearch.graphSemanticTruncated")}
                </p>
              ) : null}
            </div>
          ) : selectedNode ? (
            <>
              <h2 className="break-words text-base font-semibold">
                {selectedNode.label}
              </h2>
              <p className="mt-1 text-xs font-medium uppercase text-text-muted">
                {t(`notesSearch.graphNodeType.${selectedNode.type}`)}
              </p>
              {tags.length ? (
                <div className="mt-4">
                  <h3 className="text-xs font-semibold text-text-muted">
                    {t("notesSearch.graphTags")}
                  </h3>
                  <p className="mt-1 break-words text-sm">{tags.join(", ")}</p>
                </div>
              ) : null}
              {source ? (
                <div className="mt-4">
                  <h3 className="text-xs font-semibold text-text-muted">
                    {t("notesSearch.graphSource")}
                  </h3>
                  <button
                    type="button"
                    className="mt-1 min-h-11 break-words text-left text-sm underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                    onClick={() => onSelectNode(source.id)}>
                    {source.label}
                  </button>
                </div>
              ) : null}
              <div className="mt-4">
                <h3 className="text-xs font-semibold text-text-muted">
                  {t("notesSearch.graphRelationships")}
                </h3>
                {detailGroups.length ? (
                  detailGroups.map((group) => (
                    <div
                      key={group.id}
                      className="mt-3 border-t border-border pt-2">
                      <h4 className="text-xs font-semibold text-text-muted">
                        {t(`notesSearch.graphRelationshipGroup.${group.id}`)}
                      </h4>
                      {group.rows.map((row) => (
                        <div key={row.id} className="border-b border-border/60">
                          <button
                            type="button"
                            className="block min-h-11 w-full break-words text-left text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                            onClick={() => onSelectNode(row.counterpart.id)}>
                            {row.counterpart.label}
                          </button>
                          <div className="flex min-w-0 flex-wrap gap-1 pb-2">
                            {row.edges.map((edge: NotesGraphEdge) => (
                              <button
                                key={edge.id}
                                type="button"
                                className="min-h-9 border border-border bg-surface px-2 text-xs focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                                onClick={() => onSelectEdge?.(edge.id)}>
                                {t(`notesSearch.graphEdgeType.${edge.type}`, {
                                  defaultValue: getNotesGraphEdgeLabel(
                                    edge.type
                                  )
                                })}
                              </button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ))
                ) : (
                  <p className="mt-2 text-sm text-text-muted">
                    {t("notesSearch.graphNoRelationships")}
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="text-sm text-text-muted">
              {t("notesSearch.graphSelectNode")}
            </p>
          )}
        </div>
      ) : tab === "semantic" && semanticController ? (
        <div
          id="notes-graph-semantic-panel"
          role="tabpanel"
          aria-labelledby="notes-graph-semantic-tab"
          className="p-3">
          <div className="flex items-start justify-between gap-3">
            <h2
              ref={semanticHeadingRef}
              tabIndex={-1}
              className="text-base font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus">
              {t("notesSearch.semanticIndex")}
            </h2>
            {semanticStatus ? (
              <p className="text-sm font-semibold" role="status">
                {t(`notesSearch.semanticState.${semanticDisplayState}`)}
              </p>
            ) : null}
          </div>

          {semanticController.capabilitiesQuery.isLoading ||
          semanticController.statusQuery.isLoading ? (
            <p className="mt-3 text-sm text-text-muted">
              {t("notesSearch.semanticLoading")}
            </p>
          ) : semanticController.capabilitiesQuery.error ||
            semanticController.statusQuery.error ? (
            <p className="mt-3 text-sm text-error" role="alert">
              {t("notesSearch.semanticLoadFailed")}
            </p>
          ) : null}

          {semanticCapabilities ? (
            <>
              <p className="mt-3 text-sm">
                {t("notesSearch.semanticActiveNotes", {
                  count: semanticCapabilities.active_note_count
                })}
              </p>
              <p className="mt-1 text-sm text-text-muted">
                {t("notesSearch.semanticEstimate", {
                  chunks: semanticCapabilities.estimated_chunk_count,
                  runs: semanticCapabilities.estimated_run_count
                })}
              </p>
              <dl className="mt-4 grid grid-cols-[auto,minmax(0,1fr)] gap-x-3 gap-y-2 text-sm">
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.semanticProvider")}
                </dt>
                <dd className="break-words">
                  {semanticCapabilities.provider_label}
                </dd>
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.semanticModel")}
                </dt>
                <dd className="break-words">{semanticCapabilities.model}</dd>
                {semanticCapabilities.endpoint_display ? (
                  <>
                    <dt className="font-medium text-text-muted">
                      {t("notesSearch.semanticEndpoint")}
                    </dt>
                    <dd className="break-words">
                      {semanticCapabilities.endpoint_display}
                    </dd>
                  </>
                ) : null}
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.semanticExecutionBoundary")}
                </dt>
                <dd>
                  {t(
                    `notesSearch.semanticBoundary.${semanticCapabilities.execution_boundary}`
                  )}
                </dd>
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.semanticStorageBoundary")}
                </dt>
                <dd className="break-words">
                  {semanticCapabilities.storage_label} (
                  {t(
                    `notesSearch.semanticBoundary.${semanticCapabilities.storage_boundary}`
                  )}
                  )
                </dd>
              </dl>
              {semanticCapabilities.dimension_probe_required ? (
                <p className="mt-4 text-sm text-text-muted">
                  {t("notesSearch.semanticDimensionProbeDisclosure")}
                </p>
              ) : null}
              {semanticCapabilities.outbound_data_categories.length ? (
                <div className="mt-4">
                  <h3 className="text-xs font-semibold text-text-muted">
                    {t("notesSearch.semanticOutboundData")}
                  </h3>
                  <ul className="mt-1 list-disc pl-5 text-sm">
                    {semanticCapabilities.outbound_data_categories
                      .filter((category) =>
                        NOTES_SEMANTIC_OUTBOUND_DATA_CATEGORIES.includes(
                          category
                        )
                      )
                      .map((category) => (
                        <li key={category}>
                          {t(`notesSearch.semanticOutbound.${category}`)}
                        </li>
                      ))}
                  </ul>
                </div>
              ) : null}
              <details className="mt-4 border-t border-border pt-3 text-sm">
                <summary className="min-h-11 cursor-pointer py-2 font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus">
                  {t("notesSearch.semanticTechnicalDetails")}
                </summary>
                <dl className="grid grid-cols-[auto,minmax(0,1fr)] gap-x-3 gap-y-2 pb-1 pt-2 text-sm">
                  <dt className="font-medium text-text-muted">
                    {t("notesSearch.semanticVectorBackend")}
                  </dt>
                  <dd className="break-words">
                    {semanticCapabilities.storage_label}
                  </dd>
                  <dt className="font-medium text-text-muted">
                    {t("notesSearch.semanticDimensions")}
                  </dt>
                  <dd>
                    {semanticCapabilities.resolved_dimensions ??
                      t("notesSearch.semanticDimensionsPending")}
                  </dd>
                  <dt className="font-medium text-text-muted">
                    {t("notesSearch.semanticMetric")}
                  </dt>
                  <dd>
                    {t(
                      `notesSearch.semanticMetricValue.${semanticCapabilities.metric}`
                    )}
                  </dd>
                  <dt className="font-medium text-text-muted">
                    {t("notesSearch.semanticConfigurationRevision")}
                  </dt>
                  <dd>{semanticStatus?.configuration_revision ?? 0}</dd>
                  <dt className="font-medium text-text-muted">
                    {t("notesSearch.semanticIndexRevision")}
                  </dt>
                  <dd>{semanticStatus?.semantic_index_revision ?? 0}</dd>
                  <dt className="font-medium text-text-muted">
                    {t("notesSearch.semanticPublishedChunks")}
                  </dt>
                  <dd>{semanticStatus?.published_chunks ?? 0}</dd>
                </dl>
              </details>
            </>
          ) : null}

          {semanticStatus ? (
            <>
              <p className="mt-4 text-sm">
                {t("notesSearch.semanticProgress", {
                  indexed: semanticProgress?.indexed_notes ?? 0,
                  total:
                    semanticCapabilities?.active_note_count ??
                    (semanticProgress?.indexed_notes ?? 0) +
                      (semanticProgress?.excluded_notes ?? 0) +
                      (semanticProgress?.failed_notes ?? 0) +
                      (semanticProgress?.pending_notes ?? 0)
                })}
              </p>
              {semanticStatus.detail_reason ? (
                <p className="mt-2 text-sm text-text-muted">{semanticDetail}</p>
              ) : null}
            </>
          ) : null}

          {semanticCapabilities && !semanticCapabilities.indexing_available ? (
            <p className="mt-3 text-sm text-error">
              {t("notesSearch.semanticUnavailable")}
            </p>
          ) : null}
          {semanticCapabilities && !semanticCapabilities.manage_authorized ? (
            <p className="mt-3 text-sm text-text-muted">
              {t("notesSearch.semanticPermissionReadOnly")}
            </p>
          ) : null}
          {!isOnline ? (
            <p className="mt-3 text-sm text-text-muted">
              {t("notesSearch.semanticOffline")}
            </p>
          ) : null}
          {semanticActionError ? (
            <p className="mt-3 text-sm text-error" role="alert">
              {t(semanticActionError)}
            </p>
          ) : null}

          {semanticEdgesUsable ? (
            <label className="mt-4 flex min-h-11 items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={semanticEnabled}
                disabled={!isOnline}
                className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                onChange={(event) =>
                  onSemanticEnabledChange?.(event.target.checked)
                }
              />
              {t("notesSearch.semanticShowEdges")}
            </label>
          ) : null}

          {semanticActions.size ? (
            <div
              ref={semanticActionRegionRef}
              className="mt-4 flex flex-wrap gap-2"
              onFocusCapture={() => {
                semanticActionWasFocused.current = true
              }}
              onBlurCapture={(event) => {
                const next = event.relatedTarget
                if (
                  !next ||
                  !semanticActionRegionRef.current?.contains(next as Node)
                ) {
                  semanticActionWasFocused.current = false
                }
              }}>
              {semanticActions.has("enable") ? (
                <button
                  type="button"
                  className="min-h-11 border border-border bg-primary px-3 text-sm text-primary-foreground disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                  disabled={
                    !isOnline || pending(semanticController.mutations.enable)
                  }
                  onClick={() => void runSemanticAction("enable")}>
                  {t("notesSearch.semanticEnable")}
                </button>
              ) : null}
              {semanticActions.has("renew") ? (
                <button
                  type="button"
                  className="min-h-11 border border-border bg-primary px-3 text-sm text-primary-foreground disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                  disabled={
                    !isOnline || pending(semanticController.mutations.enable)
                  }
                  onClick={() => void runSemanticAction("renew")}>
                  {t("notesSearch.semanticRenewConsent")}
                </button>
              ) : null}
              {semanticActions.has("cancel") ? (
                <button
                  type="button"
                  className="min-h-11 border border-border bg-surface px-3 text-sm disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                  disabled={
                    !isOnline || pending(semanticController.mutations.cancel)
                  }
                  onClick={() => void runSemanticAction("cancel")}>
                  {t("notesSearch.semanticCancel")}
                </button>
              ) : null}
              {semanticActions.has("retry") ? (
                <button
                  type="button"
                  className="min-h-11 border border-border bg-primary px-3 text-sm text-primary-foreground disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                  disabled={
                    !isOnline || pending(semanticController.mutations.retry)
                  }
                  onClick={() => void runSemanticAction("retry")}>
                  {t("notesSearch.semanticRetry")}
                </button>
              ) : null}
              {semanticActions.has("rebuild") ? (
                <button
                  type="button"
                  className="min-h-11 border border-border bg-surface px-3 text-sm disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                  disabled={
                    !isOnline || pending(semanticController.mutations.rebuild)
                  }
                  onClick={() => void runSemanticAction("rebuild")}>
                  {t("notesSearch.semanticRebuild")}
                </button>
              ) : null}
              {semanticActions.has("deleteIndex") ? (
                <button
                  type="button"
                  className="min-h-11 border border-error bg-surface px-3 text-sm text-error disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                  disabled={
                    !isOnline ||
                    pending(semanticController.mutations.deleteIndex)
                  }
                  onClick={() => void runSemanticAction("deleteIndex")}>
                  {t("notesSearch.semanticDelete")}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : suggestionsAuthorized ? (
        <div
          id="notes-graph-suggestions-panel"
          role="tabpanel"
          aria-labelledby="notes-graph-suggestions-tab">
          <div className="border-b border-border p-3">
            <div className="flex items-start justify-between gap-2">
              <h2
                ref={suggestionHeadingRef}
                tabIndex={-1}
                className="text-base font-semibold">
                {t("notesSearch.graphGroundedSuggestions")}
              </h2>
              {allowedActions.has("reset_rejections") ? (
                <div className="relative">
                  <IconButton
                    ref={menuTriggerRef}
                    ariaLabel={t("notesSearch.graphSuggestionMenu")}
                    ariaExpanded={menuOpen}
                    ariaControls="notes-graph-suggestion-menu"
                    hasPopup="menu"
                    onClick={() => setMenuOpen((open) => !open)}>
                    <Ellipsis size={18} aria-hidden="true" />
                  </IconButton>
                  {menuOpen ? (
                    <div
                      id="notes-graph-suggestion-menu"
                      role="menu"
                      className="absolute right-0 z-10 min-w-[220px] border border-border bg-elevated p-1 shadow-lg">
                      <button
                        ref={menuItemRef}
                        type="button"
                        role="menuitem"
                        className="min-h-11 w-full px-3 text-left text-sm text-error focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                        disabled={
                          !isOnline || pending(controller.mutations?.reset)
                        }
                        onKeyDown={(event) => {
                          if (event.key !== "Escape") return
                          event.preventDefault()
                          setMenuOpen(false)
                          requestAnimationFrame(() =>
                            menuTriggerRef.current?.focus()
                          )
                        }}
                        onClick={() => void resetRejections()}>
                        {t("notesSearch.graphResetDismissed")}
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
            {controller.capabilitiesQuery?.isLoading ? (
              <p className="mt-3 text-sm text-text-muted">
                {t("notesSearch.graphLoadingCapabilities")}
              </p>
            ) : controller.capabilitiesQuery?.error ? (
              <p className="mt-3 text-sm text-error">
                {t("notesSearch.graphCapabilitiesFailed")}
              </p>
            ) : capabilities ? (
              <dl className="mt-3 grid grid-cols-[auto,minmax(0,1fr)] gap-x-3 gap-y-2 text-sm">
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.graphProvider")}
                </dt>
                <dd className="break-words">{capabilities.provider}</dd>
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.graphModel")}
                </dt>
                <dd className="break-words">{capabilities.model}</dd>
                <dt className="font-medium text-text-muted">
                  {t("notesSearch.graphDataBoundary")}
                </dt>
                <dd>
                  {t(
                    `notesSearch.graphBoundary.${capabilities.data_boundary === "local" ? "local" : "external"}`
                  )}
                </dd>
              </dl>
            ) : null}
            {capabilities?.outbound_data_categories.length ? (
              <div className="mt-3">
                <h3 className="text-xs font-semibold text-text-muted">
                  {t("notesSearch.graphOutboundData")}
                </h3>
                <ul className="mt-1 list-disc pl-5 text-sm">
                  {capabilities.outbound_data_categories.map((category) => (
                    <li key={category}>
                      {t(`notesSearch.graphOutbound.${category}`)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
            {capabilities?.unavailable_reason ? (
              <p className="mt-3 text-sm text-error">
                {t(
                  `notesSearch.graphUnavailable.${capabilities.unavailable_reason}`
                )}
              </p>
            ) : null}
            {!isOnline ? (
              <p className="mt-3 text-sm font-medium text-text-muted">
                {t("notesSearch.graphSuggestionOffline")}
              </p>
            ) : null}
            {visibleRun ? (
              <p
                className="mt-3 text-sm font-medium"
                data-testid="notes-graph-suggestion-run-status">
                {t(`notesSearch.graphRunState.${visibleRun.state}`)}
              </p>
            ) : null}
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                className="min-h-11 border border-border bg-primary px-3 text-sm text-primary-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:opacity-50"
                disabled={
                  !isOnline ||
                  !capabilities?.generation_available ||
                  !allowedActions.has("generate") ||
                  pending(controller.mutations?.generation) ||
                  Boolean(activeRun)
                }
                onClick={() => void runGenerate()}>
                {suggestions.length || controller.lastTerminalRun
                  ? t("notesSearch.graphRegenerate")
                  : t("notesSearch.graphGenerate")}
              </button>
              {activeRun ? (
                <button
                  type="button"
                  className="min-h-11 border border-border bg-surface px-3 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:opacity-50"
                  disabled={
                    !isOnline ||
                    !allowedActions.has("cancel") ||
                    !activeRun.cancellation_available ||
                    pending(controller.mutations?.cancellation)
                  }
                  onClick={() => void runCancel()}>
                  {t("notesSearch.graphCancelRun")}
                </button>
              ) : null}
            </div>
          </div>
          {controller.suggestionsQuery?.isLoading ? (
            <p className="p-3 text-sm text-text-muted">
              {t("notesSearch.graphLoadingSuggestions")}
            </p>
          ) : null}
          {controller.suggestionsQuery?.error ? (
            <p className="p-3 text-sm text-error">
              {t("notesSearch.graphSuggestionsFailed")}
            </p>
          ) : null}
          {suggestions.map((item) => {
            const title =
              item.kind === "related_note"
                ? item.target_title ?? t("notesSearch.graphSuggestedNote")
                : item.display_tag ??
                  item.normalized_tag ??
                  t("notesSearch.graphSuggestedTag")
            const tagPrefix =
              item.kind === "tag"
                ? t(
                    item.existing_tag
                      ? "notesSearch.graphExistingTag"
                      : "notesSearch.graphNewTag"
                  )
                : null
            return (
              <div key={item.id}>
                {tagPrefix ? (
                  <p className="px-3 pt-3 text-xs font-semibold text-text-muted">
                    {tagPrefix}
                  </p>
                ) : null}
                <NotesGraphSuggestionReviewRow
                  item={item}
                  title={title}
                  isOnline={isOnline}
                  canAccept={allowedActions.has("accept") && !isDecisionPending}
                  canReject={allowedActions.has("reject") && !isDecisionPending}
                  onAccept={(id) => {
                    const origin = document.activeElement as HTMLElement | null
                    const selected = suggestions.find(
                      (entry) => entry.id === id
                    )
                    if (selected) void decide("accept", selected, origin)
                  }}
                  onReject={(id) => {
                    const origin = document.activeElement as HTMLElement | null
                    const selected = suggestions.find(
                      (entry) => entry.id === id
                    )
                    if (selected) void decide("reject", selected, origin)
                  }}
                />
              </div>
            )
          })}
          {!suggestions.length && !controller.suggestionsQuery?.isLoading ? (
            <p className="p-3 text-sm text-text-muted">
              {t("notesSearch.graphNoSuggestions")}
            </p>
          ) : null}
        </div>
      ) : null}
    </aside>
  )
}

export default NotesGraphInspector
