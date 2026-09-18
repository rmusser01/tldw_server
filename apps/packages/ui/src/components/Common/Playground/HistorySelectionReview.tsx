import { Button } from "@/components/Common/Button"
import { RecoveryCallout, StatePanel } from "@/components/ui/state"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { HistoryCursorV1 } from "@/types/history-selection"
import { useVirtualizer } from "@tanstack/react-virtual"
import React from "react"
import { useTranslation } from "react-i18next"

const forkCopy = {
  prepared: [
    "Fork prepared",
    "This fork has not been sent. No copy will be retried automatically."
  ],
  dispatching: [
    "Fork in progress",
    "The copy outcome is not confirmed yet. No copy will be retried automatically."
  ],
  unknown: [
    "Fork status unknown",
    "A copy may have been created, but its outcome could not be confirmed. No copy will be retried automatically."
  ],
  partial: [
    "Fork incomplete",
    "A copy was created. Some content may be missing because copying could not be confirmed as complete. No copy will be retried automatically."
  ],
  completed: [
    "Fork saved",
    "The selected supported content was copied. You can inspect the saved copy."
  ],
  rejected: [
    "Fork not started",
    "No copy was created by this operation. Review the details before starting a new fork."
  ]
} as const

/** Complete-manifest review; virtualization affects presentation, never consent. */
export function HistorySelectionReview({
  selection,
  onExpand,
  onBind
}: {
  selection: HistorySelectionController
  onExpand?: () => void
  onBind?: () => void
}) {
  const { t } = useTranslation("playground")
  const text = (key: string, fallback: string) =>
    t(`historySelection.${key}`, fallback)
  const [expanded, setExpanded] = React.useState(false)
  const [included, setIncluded] = React.useState<string[]>([])
  const [cursor, setCursor] = React.useState<HistoryCursorV1>({ kind: "empty" })
  const [focusedId, setFocusedId] = React.useState<string | null>(null)
  const trigger = React.useRef<HTMLButtonElement>(null)
  const scroller = React.useRef<HTMLDivElement>(null)
  const errorRegion = React.useRef<HTMLDivElement>(null)
  const identity = React.useId()
  const nodes = selection.capture?.snapshot.nodes || []
  const rowId = (id: string) => `${identity}-${encodeURIComponent(id)}`
  const virtualizer = useVirtualizer({
    count: nodes.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => 132,
    getItemKey: (index) => nodes[index].id,
    overscan: 3,
    initialRect: { width: 320, height: 480 }
  })
  const pending =
    selection.status === "pending" || selection.status === "pending_unknown"
  const stale =
    selection.status === "stale_selection" || selection.status === "error"
  React.useEffect(() => {
    if (stale && expanded) errorRegion.current?.focus()
  }, [stale, expanded])
  const focusRow = (index: number) => {
    const node = nodes[Math.max(0, Math.min(index, nodes.length - 1))]
    if (!node) return
    setFocusedId(node.id)
    virtualizer.scrollToIndex(nodes.indexOf(node), { align: "auto" })
    requestAnimationFrame(() =>
      document.getElementById(rowId(node.id))?.focus()
    )
  }
  const beginReview = () => {
    // Start from the complete owner manifest, including offscreen alternatives.
    if (!expanded && included.length === 0) {
      const ids = nodes.map((node) => node.id)
      setIncluded(ids)
      setCursor(
        ids.length
          ? { kind: "after_message", message_id: ids[ids.length - 1] }
          : { kind: "empty" }
      )
    }
    setExpanded(true)
    setFocusedId(nodes[0]?.id ?? null)
  }
  const changeIncluded = (id: string) => {
    setIncluded((previous) =>
      previous.includes(id)
        ? previous.filter((item) => item !== id)
        : [...previous, id]
    )
    if (
      cursor.kind !== "empty" &&
      cursor.message_id === id &&
      included.includes(id)
    )
      setCursor({ kind: "empty" })
  }
  const move = (id: string, direction: -1 | 1) => {
    setIncluded((previous) => {
      const index = previous.indexOf(id)
      const destination = index + direction
      if (index < 0 || destination < 0 || destination >= previous.length)
        return previous
      const next = [...previous]
      ;[next[index], next[destination]] = [next[destination], next[index]]
      return next
    })
  }
  if (selection.status === "idle") return null
  return (
    <section
      className="w-full min-w-0 px-3 py-2"
      aria-label={text("region", "Conversation history selection")}>
      {selection.forkOperationsError && (
        <p role="alert">
          {text("forkLoadError", "Could not load retained fork outcomes.")}
        </p>
      )}
      {selection.forkOperations?.map((entry) => (
        <div
          key={entry.operation_id}
          role="status"
          className="mb-3 rounded border p-3">
          <p className="font-medium">
            {text(`forkStates.${entry.state}.title`, forkCopy[entry.state][0])}
          </p>
          <p>
            {text(
              `forkStates.${entry.state}.message`,
              forkCopy[entry.state][1]
            )}
          </p>
          <details className="text-sm text-text-muted">
            <summary>{text("forkDetails", "Operation details")}</summary>
            <p className="break-words">{entry.operation_id}</p>
            {entry.result && "code" in entry.result && (
              <p>{entry.result.code}</p>
            )}
          </details>
          {entry.candidate_child_id && (
            <Button onClick={() => void selection.inspectForkOperation(entry)}>
              {text("forkInspect", "Inspect copy")}: {entry.candidate_child_id}
            </Button>
          )}
          {entry.active_intent &&
            (entry.state === "prepared" ||
              entry.state === "unknown" ||
              entry.state === "partial") && (
              <Button onClick={() => void selection.allowNewFork(entry)}>
                {text("forkAllowNew", "Allow a new fork action")}
              </Button>
            )}
          {(entry.state === "prepared" ||
            entry.state === "unknown" ||
            entry.state === "partial") && (
            <p>
              {text(
                "forkNewExplanation",
                "Allowing a new action permits a separate copy and keeps this result. It does not retry this operation."
              )}
            </p>
          )}
        </div>
      ))}
      {selection.recoveryError && (
        <p role="alert">
          {text(
            "recoveryLoadError",
            "Could not load retained send outcomes. Refresh history to try again."
          )}
        </p>
      )}
      {selection.recoveries?.map((entry) => (
        <div
          key={entry.turn.operation_id}
          className="mb-3 rounded border p-3"
          role="status">
          <p className="font-medium">
            {text("recoveryTitle", "Turn needs review")}
          </p>
          <p>
            {text(
              "recoveryMessage",
              "The original send outcome is retained separately from history. It will not be sent again automatically."
            )}
          </p>
          <p>
            {entry.turn.persistence === "server" && entry.turn.admission
              ? text(
                  "recoveryState.accepted_response_unknown",
                  "User input accepted; response outcome unknown"
                )
              : text(`recoveryState.${entry.turn.state}`, entry.turn.state)}
          </p>
          <details>
            <summary>
              {text("inspectRecovery", "Inspect original input and result")}
            </summary>
            <pre className="whitespace-pre-wrap break-words">
              {entry.turn.input_text}
            </pre>
            <pre className="whitespace-pre-wrap break-words">
              {entry.turn.result_text}
            </pre>
          </details>
          <Button
            onClick={() =>
              void navigator.clipboard.writeText(
                entry.turn.result_text || entry.turn.input_text
              )
            }>
            {text("copyRecovery", "Copy recovered text")}
          </Button>
          <Button onClick={() => void selection.dismissRecovery(entry)}>
            {text("dismissRecovery", "Dismiss recovery")}
          </Button>
        </div>
      ))}
      {selection.status === "loading" && (
        <StatePanel
          state="loading"
          title={text("loading", "Loading selected history")}
          aria-live="polite"
        />
      )}
      {pending && (
        <StatePanel
          state="loading"
          title={text("pending", "History confirmation pending")}
          message={text(
            "pendingMessage",
            "The original choices are retained. Check the same confirmation before creating another interpretation."
          )}
          primaryAction={
            selection.status === "pending_unknown"
              ? {
                  label: text(
                    "checkConfirmation",
                    "Check original confirmation"
                  ),
                  onClick: () => void selection.confirm([], { kind: "empty" })
                }
              : undefined
          }
        />
      )}
      {stale && (
        <div ref={errorRegion} tabIndex={-1}>
          <RecoveryCallout
            state="blocked"
            title={text("changed", "Conversation history changed")}
            role="alert"
            message={text(
              "changedMessage",
              "Your choices are retained. Refresh the source, then review missing messages or choose a new boundary."
            )}
            diagnostics={
              selection.error
                ? [{ label: text("reason", "Reason"), value: selection.error }]
                : undefined
            }
            primaryAction={{
              label: text("refresh", "Refresh history"),
              onClick: () => void selection.refresh()
            }}
          />
        </div>
      )}
      {selection.status === "unsupported_history_capability" && (
        <RecoveryCallout
          state="unavailable"
          title={text("unavailable", "Selected history unavailable")}
          message={text(
            "unavailableMessage",
            "You can still read this conversation. A supported, verified history owner is required to continue from a selection."
          )}
          diagnostics={
            selection.error
              ? [{ label: text("reason", "Reason"), value: selection.error }]
              : undefined
          }
          primaryAction={
            selection.error === "unbound_server_mirror" && onBind
              ? {
                  label: text("bind", "Connect this conversation"),
                  onClick: onBind
                }
              : undefined
          }
        />
      )}
      <div className="flex flex-wrap items-center gap-2">
        {nodes.length > 0 && (
          <Button
            ref={trigger}
            variant="text"
            onClick={beginReview}
            disabled={pending}>
            {text("review", "Review conversation history")}
          </Button>
        )}
        {selection.status === "ready" && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void selection.choose({ kind: "empty" })}>
            {text("empty", "Use no prior messages")}
          </Button>
        )}
        {onExpand && (
          <Button variant="ghost" size="sm" onClick={onExpand}>
            {text("expand", "Expand in full page")}
          </Button>
        )}
      </div>
      {expanded && (
        <div className="mt-2 border-t border-border pt-3">
          <h3 className="text-base font-semibold">
            {text("review", "Review conversation history")}
          </h3>
          <p className="text-sm text-text-muted">
            {text("counts", "Complete source / included path")}: {nodes.length}{" "}
            / {included.length}
          </p>
          <p className="my-2 text-sm text-text-muted">
            {text(
              "instructions",
              "Include messages in the order to use. Omitted messages remain alternatives. Arrow keys browse rows; Alt + arrow keys reorder included messages."
            )}
          </p>
          <div
            ref={scroller}
            className="max-h-[30rem] overflow-auto"
            style={{ height: Math.min(480, nodes.length * 132) }}
            role="list"
            aria-label={text("source", "Complete source messages")}>
            <div
              style={{
                height: virtualizer.getTotalSize(),
                position: "relative"
              }}>
              {virtualizer.getVirtualItems().map((item) => {
                const node = nodes[item.index]
                const position = included.indexOf(node.id)
                const preview =
                  node.preview ||
                  (node.assets?.length
                    ? text("imageOnly", "Image or attachment only")
                    : text("emptyMessage", "Empty message"))
                return (
                  <div
                    key={node.id}
                    id={rowId(node.id)}
                    data-history-message-id={node.id}
                    role="listitem"
                    aria-setsize={nodes.length}
                    aria-posinset={item.index + 1}
                    tabIndex={focusedId === node.id ? 0 : -1}
                    onFocus={() => setFocusedId(node.id)}
                    className="absolute left-0 top-0 w-full border-b border-border p-2 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-focus"
                    style={{
                      transform: `translateY(${item.start}px)`,
                      minHeight: item.size
                    }}
                    onKeyDown={(event) => {
                      if (event.key !== "ArrowDown" && event.key !== "ArrowUp")
                        return
                      event.preventDefault()
                      const direction = event.key === "ArrowDown" ? 1 : -1
                      if (event.altKey) move(node.id, direction)
                      else focusRow(item.index + direction)
                    }}>
                    <label className="flex items-start gap-2">
                      <input
                        type="checkbox"
                        checked={position >= 0}
                        disabled={pending}
                        onChange={() => changeIncluded(node.id)}
                        aria-label={`${text("include", "Include")} ${node.role}: ${preview}`}
                      />
                      <span>
                        <strong>{node.role}</strong> · {preview}
                      </span>
                    </label>
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <span className="text-text-muted">
                        {position < 0
                          ? text("omitted", "Omitted alternative")
                          : `${text("position", "Path position")} ${position + 1}`}
                      </span>
                      {position >= 0 && (
                        <>
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={pending || position === 0}
                            onClick={() => move(node.id, -1)}
                            ariaLabel={`${text("up", "Move earlier")}: ${preview}`}>
                            {text("up", "Move earlier")}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={
                              pending || position === included.length - 1
                            }
                            onClick={() => move(node.id, 1)}
                            ariaLabel={`${text("down", "Move later")}: ${preview}`}>
                            {text("down", "Move later")}
                          </Button>
                        </>
                      )}
                      <Button
                        size="sm"
                        variant="text"
                        disabled={pending}
                        onClick={() =>
                          void selection.choose({
                            kind: "after_message",
                            message_id: node.id
                          })
                        }>
                        {text("chooseBoundary", "Continue after this message")}
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
          <div className="my-3 flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              disabled={pending || !included.length}
              onClick={() =>
                setCursor({ kind: "before_message", message_id: included[0] })
              }>
              {text("beforeFirst", "Before first included message")}
            </Button>
            <Button
              variant="secondary"
              disabled={pending}
              onClick={() => setCursor({ kind: "empty" })}>
              {text("empty", "Use no prior messages")}
            </Button>
            <Button
              variant="secondary"
              disabled={pending || !included.length}
              onClick={() =>
                setCursor({
                  kind: "after_message",
                  message_id: included[included.length - 1]
                })
              }>
              {text("throughLast", "Through last included message")}
            </Button>
          </div>
          <p className="my-2 text-sm text-text-muted">
            {text(
              "confirmEffect",
              "Confirm creates a saved interpretation of this complete source. It does not delete omitted alternatives."
            )}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={pending || selection.status === "loading"}
              onClick={() => void selection.confirm(included, cursor)}>
              {text("confirm", "Confirm selected history")}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setExpanded(false)
                trigger.current?.focus()
              }}>
              {text("cancel", "Cancel review")}
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
