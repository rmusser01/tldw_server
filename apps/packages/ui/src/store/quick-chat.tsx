import { createWithEqualityFn } from "zustand/traditional"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

export type QuickChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
}

export type QuickChatAssistantMode = "chat" | "docs_rag" | "browse_guides"

type QuickChatStore = {
  // Modal visibility
  isOpen: boolean
  setIsOpen: (open: boolean) => void

  // Chat messages (ephemeral, in-memory only)
  messages: QuickChatMessage[]
  addMessage: (role: "user" | "assistant", content: string) => void
  updateLastMessage: (content: string) => void
  clearMessages: () => void

  // Streaming state
  isStreaming: boolean
  setIsStreaming: (streaming: boolean) => void

  // Model override for quick chat
  modelOverride: string | null
  setModelOverride: (model: string | null) => void

  // Assistant mode (standard chat, docs-rag, or guide browsing)
  assistantMode: QuickChatAssistantMode
  setAssistantMode: (mode: QuickChatAssistantMode) => void

  // Pop-out window reference
  popoutWindow: Window | null
  setPopoutWindow: (win: Window | null) => void

  // For state transfer to pop-out
  getSerializableState: () => {
    messages: QuickChatMessage[]
    modelOverride: string | null
    assistantMode: QuickChatAssistantMode
  }
  restoreFromState: (state: {
    messages: QuickChatMessage[]
    modelOverride?: string | null
    assistantMode?: QuickChatAssistantMode
  }) => void
}

export const useQuickChatStore = createWithEqualityFn<QuickChatStore>((set, get) => ({
  isOpen: false,
  messages: [],
  isStreaming: false,
  modelOverride: null,
  assistantMode: "chat",
  popoutWindow: null,

  setIsOpen: (open) => {
    set({ isOpen: open })
    // Clear messages when closing
    if (!open) {
      set({ messages: [], isStreaming: false })
    }
  },

  addMessage: (role, content) => {
    const newMessage: QuickChatMessage = {
      id: crypto.randomUUID(),
      role,
      content,
      timestamp: Date.now()
    }
    set((state) => ({
      messages: [...state.messages, newMessage]
    }))
  },

  updateLastMessage: (content) => {
    set((state) => {
      const messages = [...state.messages]
      if (messages.length > 0) {
        const lastIndex = messages.length - 1
        messages[lastIndex] = {
          ...messages[lastIndex],
          content
        }
      }
      return { messages }
    })
  },

  clearMessages: () => {
    set({ messages: [], isStreaming: false })
  },

  setIsStreaming: (streaming) => {
    set({ isStreaming: streaming })
  },

  setModelOverride: (model) => {
    set({ modelOverride: model })
  },

  setAssistantMode: (mode) => {
    set({ assistantMode: mode })
  },

  setPopoutWindow: (win) => {
    set({ popoutWindow: win })
  },

  getSerializableState: () => {
    const state = get()
    return {
      messages: state.messages,
      modelOverride: state.modelOverride,
      assistantMode: state.assistantMode
    }
  },

  restoreFromState: (restoredState) => {
    const parsedMode = restoredState.assistantMode
    const assistantMode: QuickChatAssistantMode =
      parsedMode === "docs_rag" || parsedMode === "browse_guides"
        ? parsedMode
        : "chat"
    set({
      messages: restoredState.messages || [],
      modelOverride: restoredState.modelOverride ?? null,
      assistantMode
    })
  }
}))

// Expose for debugging in non-production builds
if (
  typeof window !== "undefined" &&
  process.env.NODE_ENV !== "production"
) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useQuickChatStore = useQuickChatStore
}


const stopWatchingAccount = watchChatAccountChanges((invalidated) => {
  if (!invalidated) return
  useQuickChatStore.setState({
    isOpen: false, messages: [], isStreaming: false, modelOverride: null,
    assistantMode: "chat", popoutWindow: null
  })
})
const hot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
hot?.dispose(stopWatchingAccount)
