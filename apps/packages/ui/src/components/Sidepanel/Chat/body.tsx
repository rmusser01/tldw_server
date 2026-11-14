import { useExplicitStopFocus } from "@/hooks/useExplicitStopFocus"
import { resolveCharacterMessageType } from "@/utils/character-greetings"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import React from "react"
import { useTranslation } from "react-i18next"
import { PlaygroundMessage } from "~/components/Common/Playground/Message"
import { useMessage } from "~/hooks/useMessage"
import { EmptySidePanel } from "../Chat/empty"
import { useWebUI } from "@/store/webui"
import { useUiModeStore } from "@/store/ui-mode"
import { useVirtualizer } from "@tanstack/react-virtual"
import { useStorage } from "@plasmohq/storage/hook"
import { applyVariantToMessage } from "@/utils/message-variants"
import { generateID } from "@/db/dexie/helpers"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { useStoreMessageOption } from "@/store/option"
import type { Character } from "@/types/character"
import { characterToAssistantSelection, getAssistantSelectionMode, isProtectedTrackedPersonaUnavailable } from "@/types/assistant-selection"
import {
  EDIT_MESSAGE_EVENT,
  type TimelineActionDetail
} from "@/utils/timeline-actions"

type Props = {
  scrollParentRef?: React.RefObject<HTMLDivElement>
  searchQuery?: string
  timelineAction?: TimelineActionDetail | null
  onTimelineActionHandled?: () => void
  inputRef?: React.RefObject<HTMLTextAreaElement>
}

const EMPTY_IMAGES: string[] = []

/**
 * Per-row handler interning: rows receive the SAME callback instance across
 * renders (PlaygroundMessage's strict memo comparator requires this), while
 * the interned closures only close over stable refs, so they still read the
 * latest store state at invocation time.
 */
const getStableRowHandler = <T extends (...args: never[]) => unknown>(
  cache: React.MutableRefObject<Map<string, unknown>>,
  key: string,
  factory: () => T
): T => {
  const existing = cache.current.get(key) as T | undefined
  if (existing) return existing
  const handler = factory()
  cache.current.set(key, handler)
  return handler
}

export const SidePanelBody = ({
  scrollParentRef,
  searchQuery,
  inputRef,
  timelineAction,
  onTimelineActionHandled
}: Props) => {
  const historySelection = useHistorySelectionContext()
  const {
    messages,
    setMessages,
    streaming,
    isProcessing,
    regenerateLastMessage,
    effectiveAssistantState,
    editMessage,
    deleteMessage,
    isSearchingInternet,
    createChatBranch,
    historyId,
    temporaryChat,
    stopStreamingRequest,
    serverChatId,
    isEmbedding
  } = useMessage()
  const { t } = useTranslation(["option"])
  const regenerateDisabledReason = isProtectedTrackedPersonaUnavailable(
    Boolean(historySelection), effectiveAssistantState?.kind,
    effectiveAssistantState?.mode === "overlay" ? "overlay" : "tracked"
  ) ? t("assistant.trackedPersonaUnavailable", "Tracked persona chat is unavailable here. Use a persona prompt overlay in a regular chat, or choose a character for tracked chat.") : null
  const stopStreamingFromUi = useExplicitStopFocus(stopStreamingRequest)
  const { ttsEnabled } = useWebUI()
  const [openReasoning] = useStorage("openReasoning", false)
  const [selectedCharacter] = useSelectedCharacter<Character | null>(null)
  const serverChatCharacterId = useStoreMessageOption(
    (state) => state.serverChatCharacterId
  )
  const uiMode = useUiModeStore((state) => state.mode)
  const scrollAnchorRef = React.useRef<number | null>(null)
  const topPaddingClass = historySelection && historySelection.status !== "idle" ? "pt-4" : "pt-12"
  const stableHistoryId =
    temporaryChat || historyId === "temp" ? null : historyId
  const [conversationInstanceId, setConversationInstanceId] = React.useState(
    () => generateID()
  )
  const previousMessageCount = React.useRef(messages.length)

  React.useEffect(() => {
    const hasStableId = Boolean(serverChatId || stableHistoryId)
    if (
      !hasStableId &&
      messages.length === 0 &&
      previousMessageCount.current > 0
    ) {
      setConversationInstanceId(generateID())
    }
    previousMessageCount.current = messages.length
  }, [messages.length, serverChatId, stableHistoryId])

  const handleVariantSwipe = React.useCallback(
    (messageId: string | undefined, direction: "prev" | "next") => {
      if (!messageId) return
      if (historySelection) {
        const message = messages.find(row => row.id === messageId)
        const variants = message?.variants || []
        const currentIndex = message?.activeVariantIndex ?? variants.length - 1
        const candidate = variants[currentIndex + (direction === "prev" ? -1 : 1)]
        if (candidate?.id) void historySelection.choose({ kind: "after_message", message_id: candidate.id })
        return
      }
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== messageId) return msg
          const variants = msg.variants ?? []
          if (variants.length < 2) return msg
          const currentIndex =
            typeof msg.activeVariantIndex === "number"
              ? msg.activeVariantIndex
              : variants.length - 1
          const nextIndex =
            direction === "prev" ? currentIndex - 1 : currentIndex + 1
          if (nextIndex < 0 || nextIndex >= variants.length) return msg
          return applyVariantToMessage(msg, variants[nextIndex], nextIndex)
        })
      )
    },
    [historySelection, messages, setMessages]
  )

  // Stable callbacks for PlaygroundMessage rows. The hook's action
  // callbacks are re-created every render (their closures carry the
  // render-time `messages` snapshot dev's refusal guards rely on), so rows
  // receive interned props whose identity never depends on that churn:
  // edit submits capture only stable row coordinates (index + role) and
  // dispatch through the ref below at submit time, while Delete sources are
  // ref-like objects resolved by PlaygroundMessage at confirmation-open.
  // This keeps PlaygroundMessage's strict memo comparator effective:
  // without it, inline per-row closures would re-render every visible row
  // on each streaming flush.
  const latestRowHandlers = React.useRef({
    editMessage,
    deleteMessage,
    createChatBranch,
    handleVariantSwipe,
    regenerateLastMessage,
    stopStreaming: stopStreamingFromUi
  })
  latestRowHandlers.current = {
    editMessage,
    deleteMessage,
    createChatBranch,
    handleVariantSwipe,
    regenerateLastMessage,
    stopStreaming: stopStreamingFromUi
  }
  const handleRegenerateRow = React.useCallback(
    (...args: Parameters<typeof regenerateLastMessage>) =>
      latestRowHandlers.current.regenerateLastMessage(...args),
    []
  )
  const handleStopStreamingRow = React.useCallback(
    (...args: Parameters<typeof stopStreamingFromUi>) => {
      latestRowHandlers.current.stopStreaming(...args)
    },
    []
  )
  // Per-row handler cache keyed by message id; cleared when the conversation
  // instance changes so it cannot grow unbounded across conversations. The
  // initial mount is skipped: clearing right after mount would recreate every
  // row handler on the first post-mount update (e.g. stream start) and
  // re-render all rows, defeating the memo boundary (WP-03).
  const rowHandlerCache = React.useRef(new Map<string, unknown>())
  const rowHandlerCacheConversationRef = React.useRef<string | null>(null)
  React.useEffect(() => {
    if (rowHandlerCacheConversationRef.current === null) {
      rowHandlerCacheConversationRef.current = conversationInstanceId
      return
    }
    if (rowHandlerCacheConversationRef.current === conversationInstanceId) {
      return
    }
    rowHandlerCacheConversationRef.current = conversationInstanceId
    rowHandlerCache.current.clear()
  }, [conversationInstanceId])
  // O(1) row index lookup by message id, rebuilt once per `messages`
  // identity (a streaming flush rebuilds the map once, not per row).
  const rowIndexById = React.useMemo(() => {
    const map = new Map<string, number>()
    messages.forEach((row, index) => {
      if (row.id != null) map.set(row.id, index)
    })
    return map
  }, [messages])
  // Interned edit handlers capture the row INDEX and role at intern time
  // and dispatch through the hook's LATEST editMessage instance at submit
  // time: the hook re-creates editMessage every render (its closure carries
  // the current `messages` snapshot that the hook's own message_changed
  // guards compare against), so submits must reach the freshest instance.
  // Capturing only the stable row coordinates keeps the interned identity —
  // and the row props — stable across streaming flushes and hook identity
  // churn. When the captured row disappears, the entry is kept and keeps
  // calling editMessage with the captured index (dev's semantics: the
  // hook's target guards decide, no silent no-op).
  const getRowEditSubmitHandler = React.useCallback(
    (messageId: string | undefined) => {
      const key = `edit:${messageId ?? ""}`
      const interned = rowHandlerCache.current.get(key) as
        | {
            handler: (
              value: string,
              isSend: boolean,
              editIsCurrent?: () => boolean,
              editOriginIsCurrent?: () => boolean
            ) => ReturnType<typeof editMessage>
            index: number
            isBot: boolean
          }
        | undefined
      const index =
        messageId == null
          ? messages.findIndex((row) => row.id == null)
          : rowIndexById.get(messageId)
      const isBot =
        index !== undefined && index >= 0
          ? Boolean(messages[index]?.isBot)
          : interned?.isBot ?? false
      if (
        interned &&
        (index === undefined ||
          (interned.index === index && interned.isBot === isBot))
      ) {
        return interned.handler
      }
      const targetIndex = index ?? -1
      const targetIsBot = isBot
      const handler = (
        value: string,
        isSend: boolean,
        editIsCurrent?: () => boolean,
        editOriginIsCurrent?: () => boolean
      ) => {
        if (targetIndex < 0) return
        return latestRowHandlers.current.editMessage(
          targetIndex,
          value,
          !targetIsBot,
          isSend,
          editIsCurrent,
          editOriginIsCurrent
        )
      }
      rowHandlerCache.current.set(
        key,
        { handler, index: targetIndex, isBot: targetIsBot }
      )
      return handler
    },
    [rowIndexById, messages]
  )
  // Interned delete sources are callable functions carrying a `current`
  // getter. Direct invocation (dev's pending-outcome contract) dispatches
  // through the hook's latest deleteMessage instance. PlaygroundMessage
  // instead resolves `.current` once, when the Delete confirmation OPENS,
  // so the pending onOk keeps the exact deleteMessage instance and row
  // index that were live at confirmation-open (dev's retained-refusal
  // semantics) even if later renders replace the source or the target.
  // The interned function's identity never changes — not across streaming
  // flushes and not across the hook's per-render deleteMessage identities
  // (useMessage re-creates deleteMessage whenever `messages` changes; that
  // churn is what delivers each instance's render-time snapshot to the
  // confirmation) — so row props stay memo-stable (WP-03) while every
  // confirmation still captures a fresh source at open.
  const getRowDeleteSource = React.useCallback(
    (messageId: string | undefined) => {
      const key = `delete-source:${messageId ?? ""}`
      const existing = rowHandlerCache.current.get(key) as
        | (() => boolean | Promise<boolean>) & {
            readonly current: () => () => boolean | Promise<boolean>
          }
        | undefined
      if (existing) return existing
      const resolveCapture = (): (() => boolean | Promise<boolean>) => {
        const deleteMessageNow = latestRowHandlers.current.deleteMessage
        const index = messagesRef.current.findIndex(
          (row) => row.id === messageId
        )
        // Never hand the hook a missing target: refuse rather than delete
        // whatever now sits at a stale position.
        if (index < 0) return async () => false
        // Rows await the pending delete outcome; pass it through.
        return () => deleteMessageNow(index)
      }
      const source = (() => resolveCapture()()) as (() => boolean | Promise<boolean>) & {
        readonly current: () => () => boolean | Promise<boolean>
      }
      Object.defineProperty(source, "current", {
        get: resolveCapture
      })
      rowHandlerCache.current.set(key, source)
      return source
    },
    []
  )
  const getRowNewBranchHandler = React.useCallback(
    (messageId: string | undefined) =>
      getStableRowHandler(
        rowHandlerCache,
        `branch:${messageId ?? ""}`,
        () => () => {
          void latestRowHandlers.current.createChatBranch(messageId ?? "")
        }
      ),
    []
  )
  const getRowSwipeHandler = React.useCallback(
    (messageId: string | undefined, direction: "prev" | "next") =>
      getStableRowHandler(
        rowHandlerCache,
        `swipe:${direction}:${messageId ?? ""}`,
        () => () => {
          latestRowHandlers.current.handleVariantSwipe(messageId, direction)
        }
      ),
    []
  )
  const rowSwipeDisabled = Boolean(
    historySelection &&
      (!historySelection.view || historySelection.status === "loading")
  )

  const previousUserMessageByIndex = React.useMemo(() => {
    const map = new Map<number, (typeof messages)[number] | null>()
    let previous: (typeof messages)[number] | null = null
    for (let i = 0; i < messages.length; i++) {
      map.set(i, previous)
      const candidate = messages[i]
      if (candidate && !candidate.isBot) {
        previous = candidate
      }
    }
    return map
  }, [messages])

  const parentEl = scrollParentRef?.current || null
  const rowVirtualizer = useVirtualizer({
    count: messages.length,
    getItemKey: index => messages[index].id || index,
    getScrollElement: () => parentEl,
    estimateSize: () => 120,
    // Reduced from 6 to 3 for better performance on large conversations
    overscan: 3,
    measureElement: (el) => el?.getBoundingClientRect().height || 120
  })

  const messagesRef = React.useRef(messages)
  React.useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  React.useEffect(() => {
    if (!timelineAction) return
    if (timelineAction.historyId && timelineAction.historyId !== historyId) {
      return
    }

    const findIndex = (messageId: string) =>
      messagesRef.current.findIndex(
        (message) =>
          message.id === messageId || message.serverMessageId === messageId
      )

    if (timelineAction.action === "branch") {
      if (!timelineAction.messageId) {
        onTimelineActionHandled?.()
        return
      }
      if (messagesRef.current.length === 0) {
        return
      }
      const index = findIndex(timelineAction.messageId)
      if (index >= 0) {
        void createChatBranch(messagesRef.current[index]?.id ?? "")
      }
      onTimelineActionHandled?.()
      return
    }

    if (!timelineAction.messageId) {
      onTimelineActionHandled?.()
      return
    }

    if (messagesRef.current.length === 0) {
      return
    }
    const index = findIndex(timelineAction.messageId)
    if (index < 0) {
      onTimelineActionHandled?.()
      return
    }

    rowVirtualizer.scrollToIndex(index, { align: "center" })
    if (timelineAction.action === "edit" && typeof window !== "undefined") {
      window.dispatchEvent(
        new CustomEvent(EDIT_MESSAGE_EVENT, {
          detail: { messageId: timelineAction.messageId }
        })
      )
    }
    onTimelineActionHandled?.()
  }, [
    createChatBranch,
    historyId,
    messages.length,
    onTimelineActionHandled,
    rowVirtualizer,
    timelineAction
  ])

  // Lock scroll position during streaming to prevent virtualizer jumps
  React.useEffect(() => {
    if (!parentEl) return

    if (streaming) {
      // Save current scroll position when streaming starts
      scrollAnchorRef.current = parentEl.scrollTop
    } else {
      // Clear anchor when streaming ends
      scrollAnchorRef.current = null
    }
  }, [streaming, parentEl])

  // Prevent virtualizer scroll jumps during streaming
  React.useEffect(() => {
    if (!parentEl || !streaming || scrollAnchorRef.current === null) return

    const handleScroll = () => {
      // If user manually scrolls, update the anchor
      if (Math.abs(parentEl.scrollTop - scrollAnchorRef.current!) > 10) {
        scrollAnchorRef.current = parentEl.scrollTop
      }
    }

    parentEl.addEventListener('scroll', handleScroll, { passive: true })
    return () => parentEl.removeEventListener('scroll', handleScroll)
  }, [streaming, parentEl])

  const characterIdentityEnabled = React.useMemo(() => {
    if (!selectedCharacter?.id) return false
    if (serverChatId) {
      if (serverChatCharacterId == null) return false
      return String(serverChatCharacterId) === String(selectedCharacter.id)
    }
    return true
  }, [selectedCharacter?.id, serverChatCharacterId, serverChatId])

  return (
    <>
      <div
        role="log"
        aria-live="polite"
        className={`relative flex w-full flex-col items-center ${topPaddingClass} pb-4`}
      >
        {messages.length === 0 && !(selectedCharacter && getAssistantSelectionMode(characterToAssistantSelection(selectedCharacter)) === "tracked" && !historyId && !serverChatId) && <EmptySidePanel inputRef={inputRef} />}
        <div style={{ height: rowVirtualizer.getTotalSize(), width: '100%', position: 'relative' }}>
          {rowVirtualizer.getVirtualItems().map((vr) => {
            const index = vr.index
            const message = messages[index]
            const previousUserMessage = previousUserMessageByIndex.get(index) ?? null
            return (
              <div key={vr.key} ref={rowVirtualizer.measureElement} data-index={index} style={{ position: 'absolute', top: 0, left: 0, transform: `translateY(${vr.start}px)`, width: '100%' }}>
                <PlaygroundMessage
                  isBot={message.isBot}
                  message={message.message}
                  name={message.name}
                  role={message.role}
                  images={message.images || EMPTY_IMAGES}
                  currentMessageIndex={index}
                  totalMessages={messages.length}
                  onRegenerate={handleRegenerateRow}
                  regenerateDisabledReason={regenerateDisabledReason}
                  message_type={resolveCharacterMessageType(message, index, Boolean(serverChatId && characterIdentityEnabled))}
                  isProcessing={isProcessing}
                  isSearchingInternet={isSearchingInternet}
                  sources={message.sources}
                  onEditFormSubmit={getRowEditSubmitHandler(message.id)}
                  onDeleteMessage={getRowDeleteSource(message.id)}
                  onNewBranch={getRowNewBranchHandler(message.id)}
                  isTTSEnabled={ttsEnabled}
                  generationInfo={message?.generationInfo}
                  toolCalls={message?.toolCalls}
                  toolResults={message?.toolResults}
                  isStreaming={streaming && index === messages.length - 1}
                  reasoningTimeTaken={message?.reasoning_time_taken}
                  openReasoning={openReasoning}
                  modelImage={message?.modelImage}
                  modelName={message?.modelName}
                  createdAt={message?.createdAt}
                  temporaryChat={temporaryChat}
                  onStopStreaming={handleStopStreamingRow}
                  serverChatId={serverChatId}
                  serverMessageId={message.serverMessageId}
                  messageId={message.id}
                  discoSkillComment={message.discoSkillComment}
                  historyId={stableHistoryId ?? undefined}
                  conversationInstanceId={conversationInstanceId}
                  feedbackQuery={previousUserMessage?.message ?? null}
                  searchQuery={searchQuery}
                  isEmbedding={isEmbedding}
                  characterIdentity={selectedCharacter}
                  characterIdentityEnabled={characterIdentityEnabled}
                  speakerCharacterId={message.speakerCharacterId ?? null}
                  speakerCharacterName={message.speakerCharacterName}
                  moodLabel={message.moodLabel ?? null}
                  moodConfidence={message.moodConfidence ?? null}
                  moodTopic={message.moodTopic ?? null}
                  variants={message.variants}
                  activeVariantIndex={message.activeVariantIndex}
                  metadataExtra={message.metadataExtra}
                  dynamicUISurface="extension-sidepanel"
                  onSwipePrev={rowSwipeDisabled ? undefined : getRowSwipeHandler(message.id, "prev")}
                  onSwipeNext={rowSwipeDisabled ? undefined : getRowSwipeHandler(message.id, "next")}
                />
              </div>
            )
          })}
        </div>
      </div>

    </>
  )
}
