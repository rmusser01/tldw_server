import { useCallback, useEffect, useMemo, useRef } from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import {
  useQuickChatStore,
  QuickChatMessage,
  type QuickChatAssistantMode
} from "@/store/quick-chat"
import { TldwChatService, TldwChatOptions } from "@/services/tldw/TldwChat"
import { ChatMessage, tldwClient } from "@/services/tldw/TldwApiClient"
import { buildQuickChatRagReply } from "@/components/Common/QuickChatHelper/rag-response"
import {
  buildQuickChatDocsRagProfile,
  normalizeQuickChatDocsMediaIds,
  QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE
} from "@/components/Common/QuickChatHelper/docs-rag-profile"
import {
  QUICK_CHAT_WORKFLOW_GUIDES,
  QUICK_CHAT_WORKFLOW_GUIDES_STORAGE_KEY,
  resolveQuickChatWorkflowGuides
} from "@/components/Common/QuickChatHelper/workflow-guides"

// Create a dedicated chat service instance for quick chat
const quickChatService = new TldwChatService()

export type QuickChatSendOptions = {
  mode?: QuickChatAssistantMode
  currentRoute?: string | null
}

export const useQuickChat = () => {
  const { selectedModel } = useSelectedModel()
  const [quickChatStrictDocsOnly] = useStorage<boolean>(
    "quickChatStrictDocsOnly",
    true
  )
  const [quickChatDocsNamespace] = useStorage<string>(
    "quickChatDocsIndexNamespace",
    QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE
  )
  const [quickChatDocsMediaIdsRaw] = useStorage<unknown>(
    "quickChatDocsProjectMediaIds",
    []
  )
  const [quickChatWorkflowGuidesRaw] = useStorage<unknown>(
    QUICK_CHAT_WORKFLOW_GUIDES_STORAGE_KEY,
    QUICK_CHAT_WORKFLOW_GUIDES
  )
  const ragControllerRef = useRef<AbortController | null>(null)
  const quickChatWorkflowGuides = useMemo(
    () => resolveQuickChatWorkflowGuides(quickChatWorkflowGuidesRaw),
    [quickChatWorkflowGuidesRaw]
  )

  const {
    messages,
    addMessage,
    updateLastMessage,
    clearMessages,
    isStreaming,
    setIsStreaming,
    isOpen,
    setIsOpen,
    modelOverride,
    setModelOverride
  } = useQuickChatStore()
  const activeModel = modelOverride || selectedModel || null

  const sendMessage = useCallback(
    async (content: string, sendOptions?: QuickChatSendOptions) => {
      const mode = sendOptions?.mode || "chat"
      const requiresChatModel = mode !== "docs_rag"
      if (!content.trim() || isStreaming || (requiresChatModel && !activeModel)) {
        return
      }

      // Add user message
      addMessage("user", content)

      // Add placeholder for assistant message
      addMessage("assistant", "")

      setIsStreaming(true)

      try {
        if (mode === "docs_rag") {
          await tldwClient.initialize()
          const ragController = new AbortController()
          ragControllerRef.current = ragController
          const docsProfile = buildQuickChatDocsRagProfile({
            query: content,
            currentRoute: sendOptions?.currentRoute ?? null,
            scope: {
              strictProjectDocsOnly: quickChatStrictDocsOnly !== false,
              projectDocsNamespace: quickChatDocsNamespace,
              projectDocsMediaIds: normalizeQuickChatDocsMediaIds(
                quickChatDocsMediaIdsRaw
              )
            }
          })
          const ragResponse = await tldwClient.ragSearch(docsProfile.query, {
            ...docsProfile.options,
            signal: ragController.signal
          })
          const ragReply = buildQuickChatRagReply(ragResponse, {
            query: content,
            currentRoute: sendOptions?.currentRoute ?? null,
            guides: quickChatWorkflowGuides
          })
          updateLastMessage(ragReply.message)
          return
        }

        // Build chat history for the API
        const currentMessages = useQuickChatStore.getState().messages
        const chatHistory: ChatMessage[] = currentMessages
          .slice(0, -1) // Exclude the empty assistant placeholder
          .map((msg: QuickChatMessage) =>
            msg.role === "user"
              ? { role: "user", content: msg.content }
              : { role: "assistant", content: msg.content }
          )

        const resolvedModel = activeModel
        if (!resolvedModel) {
          return
        }

        const options: TldwChatOptions = {
          model: resolvedModel,
          stream: true
        }

        const chunks: string[] = []

        // Stream the response
        for await (const chunk of quickChatService.streamMessage(
          chatHistory,
          options
        )) {
          if (typeof chunk !== "string") continue
          chunks.push(chunk)
          updateLastMessage(chunks.join(""))
        }
      } catch (error) {
        // Check if it's an abort error
        if (error instanceof Error && error.name === "AbortError") {
          // Stream was cancelled, don't show error
          return
        }

        if (mode === "docs_rag") {
          console.error("Quick chat docs RAG error:", error)
          updateLastMessage(
            "I could not query documentation right now. Try again, or switch to Browse Guides for curated workflow help."
          )
          return
        }

        console.error("Quick chat error:", error)
        const errorMessage =
          error instanceof Error ? error.message : "An error occurred"
        updateLastMessage(`Error: ${errorMessage}`)
      } finally {
        ragControllerRef.current = null
        setIsStreaming(false)
      }
    },
    [
      activeModel,
      isStreaming,
      addMessage,
      quickChatDocsMediaIdsRaw,
      quickChatDocsNamespace,
      quickChatStrictDocsOnly,
      quickChatWorkflowGuides,
      updateLastMessage,
      setIsStreaming
    ]
  )

  const cancelStream = useCallback(() => {
    quickChatService.cancelStream()
    if (ragControllerRef.current) {
      ragControllerRef.current.abort()
      ragControllerRef.current = null
    }
    setIsStreaming(false)
  }, [setIsStreaming])

  const openModal = useCallback(() => {
    setIsOpen(true)
  }, [setIsOpen])

  const closeModal = useCallback(() => {
    cancelStream()
    setIsOpen(false)
  }, [cancelStream, setIsOpen])

  // Abort in-flight streams on unmount to prevent state updates on unmounted component
  useEffect(() => {
    return () => {
      quickChatService.cancelStream()
      if (ragControllerRef.current) {
        ragControllerRef.current.abort()
        ragControllerRef.current = null
      }
    }
  }, [])

  return {
    messages,
    sendMessage,
    cancelStream,
    clearMessages,
    isStreaming,
    isOpen,
    openModal,
    closeModal,
    hasModel: !!activeModel,
    activeModel,
    currentModel: selectedModel || null,
    modelOverride,
    setModelOverride
  }
}
