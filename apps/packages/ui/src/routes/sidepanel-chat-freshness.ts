/**
 * Keep a side-panel tab bound to a server chat in step with that chat (XP-08,
 * #3105).
 *
 * A tab restores its saved transcript and history-selection bookmark, so after
 * the chat is continued elsewhere (the full page, another window) the tab still
 * ends at its old message. Sending from it would reply to that old message and
 * silently fork the chat. The panel therefore checks the server's message
 * manifest when it shows a tab, when it regains focus and before a send. When
 * the server has newer turns, it moves the tab's history selection to the
 * latest message (which reloads the transcript and re-parents the next send),
 * or shows an "updated elsewhere" notice when moving would discard messages
 * that exist only in this tab.
 */
import React from "react"

import type { FreshnessNotice } from "@/components/Sidepanel/Chat/ServerChatFreshnessNotice"
import type { SidepanelSendGateResult } from "@/hooks/chat/sidepanel-send-gate"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { captureHistorySnapshot, type HistoryOwnerV1 } from "@/services/chat-history-selection"
import { useStoreMessageOption, type Message } from "@/store/option"
import type {
  HistoryCaptureResultV1,
  HistoryNodeV1,
  HistorySelectionCaptureV1,
  HistoryViewSelectionV1
} from "@/types/history-selection"
import { latestHistoryTipId } from "@/utils/history-selection"

/** Focus and visibility events arrive in bursts; check once they settle. */
export const FOCUS_CHECK_DEBOUNCE_MS = 400
/** At most one focus-triggered check per tab in this window. */
export const FOCUS_CHECK_MIN_INTERVAL_MS = 5_000

export type ServerTurnsComparison =
  | { status: "current"; latestId: string | null }
  | { status: "behind"; latestId: string; newCount: number }

/** Every server message id a tab shows, including assistant variants. */
export const shownServerMessageIds = (
  messages: readonly Message[]
): Set<string> => {
  const ids = new Set<string>()
  for (const message of messages) {
    if (message.id) ids.add(message.id)
    if (message.serverMessageId) ids.add(message.serverMessageId)
    for (const variant of message.variants ?? []) {
      if (variant.id) ids.add(variant.id)
      if (variant.serverMessageId) ids.add(variant.serverMessageId)
    }
  }
  return ids
}

/**
 * Compare a tab with the server's message graph.
 *
 * The tab is current when the server's latest message is the one its view ends
 * at, one it shows (an assistant variant), or the latest message it saw when it
 * last matched the server (so a position chosen on purpose is kept). Otherwise
 * the chat gained turns elsewhere; `newCount` counts the latest message's
 * ancestors the tab doesn't show.
 */
export const compareWithServerTurns = ({
  nodes,
  cursorId,
  shownIds,
  latestSeenId
}: {
  nodes: readonly Pick<HistoryNodeV1, "id" | "parent_id">[]
  cursorId: string | null
  shownIds: ReadonlySet<string>
  latestSeenId: string | null
}): ServerTurnsComparison => {
  const latestId = latestHistoryTipId(nodes)
  if (
    !latestId ||
    latestId === cursorId ||
    shownIds.has(latestId) ||
    latestId === latestSeenId
  ) {
    return { status: "current", latestId }
  }
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const counted = new Set<string>()
  let node = byId.get(latestId)
  while (node && !shownIds.has(node.id) && !counted.has(node.id)) {
    counted.add(node.id)
    node = node.parent_id ? byId.get(node.parent_id) : undefined
  }
  return { status: "behind", latestId, newCount: Math.max(1, counted.size) }
}

/** Whether the tab shows messages the server doesn't have (a refresh would drop them). */
export const hasUnsentLocalMessages = (
  messages: readonly Message[],
  nodes: readonly Pick<HistoryNodeV1, "id">[]
): boolean => {
  const serverIds = new Set(nodes.map((node) => node.id))
  return messages.some(
    (message) =>
      !(message.id && serverIds.has(message.id)) &&
      !(message.serverMessageId && serverIds.has(message.serverMessageId))
  )
}

/**
 * Read the chat's message manifest only: the same capture the view uses, with
 * an empty cursor so the server returns no message content.
 */
export const captureServerChatManifest = (
  owner: HistoryOwnerV1,
  view: HistoryViewSelectionV1,
  signal?: AbortSignal
): Promise<HistoryCaptureResultV1> =>
  captureHistorySnapshot(owner, { ...view, cursor: { kind: "empty" } }, "send", signal)

type CheckMode = "show" | "focus" | "send" | "manual"

const PROCEED: SidepanelSendGateResult = { proceed: true, refreshed: false }
const HOLD: SidepanelSendGateResult = { proceed: false, refreshed: false }

type Options = {
  historySelection: HistorySelectionController
  /**
   * The active tab's id when the panel may check it: the tab is bound to a
   * server chat, nothing is restoring, opening or streaming. Otherwise null.
   */
  checkableTabId: () => string | null
}

export const useSidepanelChatFreshness = ({
  historySelection,
  checkableTabId
}: Options) => {
  /** The active tab's last-seen latest server message (persisted in its snapshot). */
  const latestSeenRef = React.useRef<string | null>(null)
  const [notice, setNotice] = React.useState<FreshnessNotice | null>(null)
  const selectionRef = React.useRef(historySelection)
  selectionRef.current = historySelection
  const checkableRef = React.useRef(checkableTabId)
  checkableRef.current = checkableTabId
  const queueRef = React.useRef<Promise<unknown>>(Promise.resolve())
  const lastCheckAtRef = React.useRef(new Map<string, number>())
  const focusTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = React.useRef(true)

  React.useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const publish = React.useCallback((next: FreshnessNotice | null) => {
    if (mountedRef.current) setNotice(next)
  }, [])

  const check = React.useCallback(
    async (mode: CheckMode): Promise<SidepanelSendGateResult> => {
      const tabId = checkableRef.current()
      if (!tabId) return PROCEED
      const selection = selectionRef.current
      const current = selection.getCurrent()
      const { owner, view, capture } = current
      if (
        current.status !== "ready" ||
        owner?.kind !== "native" ||
        !view ||
        capture?.status !== "captured"
      ) {
        return PROCEED
      }
      // A send whose tab changed while it was checked is held: it would
      // otherwise go to the tab the user switched to.
      const changed = mode === "send" ? HOLD : PROCEED
      const sameView = selection.fence()
      const onTab = () => mountedRef.current && checkableRef.current() === tabId
      lastCheckAtRef.current.set(tabId, Date.now())

      let nodes: readonly HistoryNodeV1[]
      if (mode === "show") {
        // The view was just loaded from the server; its manifest is current.
        nodes = capture.snapshot.nodes
      } else {
        try {
          const manifest = await captureServerChatManifest(owner, view, selection.getSignal())
          nodes = manifest.snapshot.nodes
        } catch (error) {
          // A failed check never blocks: the send path reports server errors.
          console.warn("[sidepanel] Could not check the chat for newer messages", error)
          return onTab() ? PROCEED : changed
        }
        if (!onTab()) return changed
        if (!sameView()) return mode === "send" ? check(mode) : PROCEED
      }

      const messages = useStoreMessageOption.getState().messages
      const cursorId =
        view.cursor.kind === "after_message" ? view.cursor.message_id : null
      const comparison = compareWithServerTurns({
        nodes,
        cursorId,
        shownIds: shownServerMessageIds(messages),
        latestSeenId: latestSeenRef.current
      })
      if (comparison.status === "current") {
        latestSeenRef.current = comparison.latestId ?? latestSeenRef.current
        if (mountedRef.current) {
          setNotice((previous) =>
            previous?.tabId === tabId && previous.kind === "stale" ? null : previous
          )
        }
        return PROCEED
      }

      const { latestId, newCount } = comparison
      if ((mode === "show" || mode === "focus") && hasUnsentLocalMessages(messages, nodes)) {
        publish({ tabId, kind: "stale", newCount })
        return PROCEED
      }
      let moved = false
      try {
        moved = await selection.choose({ kind: "after_message", message_id: latestId })
      } catch (error) {
        console.warn("[sidepanel] Could not load the chat's newer messages", error)
      }
      if (!onTab()) return changed
      const now = selection.getCurrent()
      if (
        moved &&
        now.status === "ready" &&
        now.view?.cursor.kind === "after_message" &&
        now.view.cursor.message_id === latestId
      ) {
        latestSeenRef.current = latestId
        publish(
          mode === "manual"
            ? null
            : { tabId, kind: mode === "send" ? "refreshedBeforeSend" : "refreshed", newCount }
        )
        return { proceed: true, refreshed: true }
      }
      publish({ tabId, kind: mode === "send" ? "sendBlocked" : "stale", newCount })
      return mode === "send" ? HOLD : PROCEED
    },
    [publish]
  )

  /** Run checks one at a time, so a focus check and a send never race. */
  const enqueue = React.useCallback(
    (mode: CheckMode): Promise<SidepanelSendGateResult> => {
      const run = queueRef.current.then(
        () => check(mode),
        () => check(mode)
      )
      queueRef.current = run.catch(() => undefined)
      return run
    },
    [check]
  )

  /** After a tab is shown (panel opened, tab switched), compare it with what it just loaded. */
  const checkShownTab = React.useCallback(() => {
    void enqueue("show")
  }, [enqueue])

  /** The Refresh action on a notice: load the newer messages, discarding unsent ones. */
  const refreshNow = React.useCallback(() => {
    void enqueue("manual")
  }, [enqueue])

  const dismissNotice = React.useCallback(() => setNotice(null), [])

  /** Forget the previous tab's notice and seen message when another tab is shown. */
  const showTab = React.useCallback((latestSeenId: string | null | undefined) => {
    latestSeenRef.current = latestSeenId ?? null
    setNotice(null)
  }, [])

  /**
   * Note a capture the history selection installed for the active tab. A view
   * that ends at its chat's latest message matches the server.
   */
  const noteCapture = React.useCallback(
    (installed: Partial<Pick<HistorySelectionCaptureV1, "rows" | "snapshot">>) => {
      // Only a hint for later checks: never let it break the load that installed it.
      const nodes = installed.snapshot?.nodes ?? []
      const rows = installed.rows ?? []
      const latestId = latestHistoryTipId(nodes)
      const leafId = rows.length ? rows[rows.length - 1].id : null
      if (latestId && leafId === latestId) latestSeenRef.current = latestId
    },
    []
  )

  /** The pre-send check `useMessage` runs through `SidepanelSendGateContext`. */
  const beforeSend = React.useCallback(
    async ({ message }: { message: string }): Promise<SidepanelSendGateResult> => {
      if (!checkableRef.current()) return PROCEED
      const result = await enqueue("send")
      if (!result.proceed) {
        // The composer cleared itself when the send started; put the message back.
        window.dispatchEvent(
          new CustomEvent("tldw:set-composer-message", {
            detail: { message, ifEmptyOnly: true }
          })
        )
      }
      return result
    },
    [enqueue]
  )

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const schedule = () => {
      if (focusTimerRef.current) clearTimeout(focusTimerRef.current)
      focusTimerRef.current = setTimeout(() => {
        focusTimerRef.current = null
        const tabId = checkableRef.current()
        if (!tabId) return
        const last = lastCheckAtRef.current.get(tabId)
        if (last !== undefined && Date.now() - last < FOCUS_CHECK_MIN_INTERVAL_MS) return
        void enqueue("focus")
      }, FOCUS_CHECK_DEBOUNCE_MS)
    }
    const onVisibility = () => {
      if (document.visibilityState !== "hidden") schedule()
    }
    window.addEventListener("focus", schedule)
    document.addEventListener("visibilitychange", onVisibility)
    return () => {
      window.removeEventListener("focus", schedule)
      document.removeEventListener("visibilitychange", onVisibility)
      if (focusTimerRef.current) clearTimeout(focusTimerRef.current)
      focusTimerRef.current = null
    }
  }, [enqueue])

  return {
    latestSeenRef,
    notice,
    dismissNotice,
    refreshNow,
    checkShownTab,
    showTab,
    noteCapture,
    beforeSend
  }
}
