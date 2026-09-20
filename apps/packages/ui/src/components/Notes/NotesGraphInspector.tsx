import IconButton from "@/components/Common/IconButton"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import type {
  NotesGraphResponse,
  NotesGraphSuggestion
} from "@/services/note-graph-suggestions"
import { Ellipsis } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import {
  type NotesGraphSuggestionDecisionHandler,
  NotesGraphSuggestionReviewRow,
  buildNotesGraphRelationshipGroups
} from "./NotesGraphRelationshipsView"
import type { NotesGraphSuggestionsController } from "./hooks/useNotesGraphSuggestions"

type NotesGraphInspectorProps = {
  graph: NotesGraphResponse
  selectedNodeId: string | null
  suggestionsAuthorized: boolean
  isOnline: boolean
  controller: NotesGraphSuggestionsController
  onSelectNode: (nodeId: string) => void
  onAnnounce: (message: string) => void
  onDecideSuggestion: NotesGraphSuggestionDecisionHandler
}

const pending = (mutation: unknown): boolean =>
  Boolean((mutation as { isPending?: boolean } | undefined)?.isPending)

const NotesGraphInspector: React.FC<NotesGraphInspectorProps> = ({
  graph,
  selectedNodeId,
  suggestionsAuthorized,
  isOnline,
  controller,
  onSelectNode,
  onAnnounce,
  onDecideSuggestion
}) => {
  const { t } = useTranslation("option")
  const confirmDanger = useConfirmDanger()
  const [tab, setTab] = React.useState<"details" | "suggestions">("details")
  const [menuOpen, setMenuOpen] = React.useState(false)
  const inspectorRef = React.useRef<HTMLElement | null>(null)
  const detailsTabRef = React.useRef<HTMLButtonElement | null>(null)
  const suggestionsTabRef = React.useRef<HTMLButtonElement | null>(null)
  const suggestionHeadingRef = React.useRef<HTMLHeadingElement | null>(null)
  const menuTriggerRef = React.useRef<HTMLButtonElement | null>(null)
  const menuItemRef = React.useRef<HTMLButtonElement | null>(null)
  const selectedNode =
    graph.nodes.find((node) => node.id === selectedNodeId) ?? null
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
    if (!menuOpen) return
    const frame = requestAnimationFrame(() => menuItemRef.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [menuOpen])

  const handleTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (!suggestionsAuthorized) return
    let next: "details" | "suggestions" | null = null
    if (event.key === "Home") next = "details"
    if (event.key === "End") next = "suggestions"
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      next = tab === "details" ? "suggestions" : "details"
    }
    if (!next) return
    event.preventDefault()
    setTab(next)
    ;(next === "details" ? detailsTabRef : suggestionsTabRef).current?.focus()
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
      </div>

      {tab === "details" ? (
        <div
          id="notes-graph-details-panel"
          role="tabpanel"
          aria-labelledby="notes-graph-details-tab"
          className="p-3">
          {selectedNode ? (
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
                        <button
                          key={row.id}
                          type="button"
                          className="block min-h-11 w-full break-words text-left text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
                          onClick={() => onSelectNode(row.counterpart.id)}>
                          {row.counterpart.label}
                        </button>
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
