import React from "react"
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
import { ChatGreetingPicker } from "@/components/Common/ChatGreetingPicker"
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

export const SidePanelBody = ({
  scrollParentRef,
  searchQuery,
  inputRef,
  timelineAction,
  onTimelineActionHandled
}: Props) => {
  const {
    messages,
    setMessages,
    streaming,
    isProcessing,
    regenerateLastMessage,
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
  const { ttsEnabled } = useWebUI()
  const [openReasoning] = useStorage("openReasoning", false)
  const [selectedCharacter] = useSelectedCharacter<Character | null>(null)
  const serverChatCharacterId = useStoreMessageOption(
    (state) => state.serverChatCharacterId
  )
  const uiMode = useUiModeStore((state) => state.mode)
  const scrollAnchorRef = React.useRef<number | null>(null)
  const topPaddingClass = "pt-12"
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
    [setMessages]
  )

  // Stable callbacks for PlaygroundMessage
  const handleEditMessage = React.useCallback(
    (index: number, value: string, isUser: boolean, isSend: boolean) =>
      editMessage(index, value, isUser, isSend),
    [editMessage]
  )
  const handleDeleteMessage = React.useCallback(
    (index: number) => deleteMessage(index),
    [deleteMessage]
  )
  const handleNewBranch = React.useCallback(
    (index: number) => createChatBranch(index),
    [createChatBranch]
  )
  const handleSwipePrev = React.useCallback(
    (id: string) => handleVariantSwipe(id, "prev"),
    [handleVariantSwipe]
  )
  const handleSwipeNext = React.useCallback(
    (id: string) => handleVariantSwipe(id, "next"),
    [handleVariantSwipe]
  )

  const getPreviousUserMessage = (index: number) => {
    for (let i = index - 1; i >= 0; i--) {
      const candidate = messages[i]
      if (!candidate?.isBot) {
        return candidate
      }
    }
    return null
  }

  const parentEl = scrollParentRef?.current || null
  const rowVirtualizer = useVirtualizer({
    count: messages.length,
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
        void createChatBranch(index)
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
        {messages.length === 0 && <EmptySidePanel inputRef={inputRef} />}
        <ChatGreetingPicker
          selectedCharacter={selectedCharacter}
          messages={messages}
          historyId={historyId}
          serverChatId={serverChatId}
          className="mb-4"
        />
        <div style={{ height: rowVirtualizer.getTotalSize(), width: '100%', position: 'relative' }}>
          {rowVirtualizer.getVirtualItems().map((vr) => {
            const index = vr.index
            const message = messages[index]
            const previousUserMessage = getPreviousUserMessage(index)
            return (
              <div key={vr.key} ref={rowVirtualizer.measureElement} data-index={index} style={{ position: 'absolute', top: 0, left: 0, transform: `translateY(${vr.start}px)`, width: '100%' }}>
                <PlaygroundMessage
                  isBot={message.isBot}
                  message={message.message}
                  name={message.name}
                  role={message.role}
                  images={message.images || []}
                  currentMessageIndex={index}
                  totalMessages={messages.length}
                  onRegenerate={regenerateLastMessage}
                  message_type={message.messageType}
                  isProcessing={isProcessing}
                  isSearchingInternet={isSearchingInternet}
                  sources={message.sources}
                  onEditFormSubmit={(value, isSend) => {
                    void handleEditMessage(index, value, !message.isBot, isSend)
                  }}
                  onDeleteMessage={() => {
                    void handleDeleteMessage(index)
                  }}
                  onNewBranch={() => {
                    void handleNewBranch(index)
                  }}
                  isTTSEnabled={ttsEnabled}
                  generationInfo={message?.generationInfo}
                  toolCalls={message?.toolCalls}
                  toolResults={message?.toolResults}
                  isStreaming={streaming}
                  reasoningTimeTaken={message?.reasoning_time_taken}
                  openReasoning={openReasoning}
                  modelImage={message?.modelImage}
                  modelName={message?.modelName}
                  createdAt={message?.createdAt}
                  temporaryChat={temporaryChat}
                  onStopStreaming={stopStreamingRequest}
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
                  onSwipePrev={() => handleSwipePrev(message.id)}
                  onSwipeNext={() => handleSwipeNext(message.id)}
                />
              </div>
            )
          })}
        </div>
      </div>

    </>
  )
}
