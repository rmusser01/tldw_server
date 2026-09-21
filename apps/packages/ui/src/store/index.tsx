import { createWithEqualityFn } from "zustand/traditional"
import type { DiscoSkillComment } from "@/types/disco-skills"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

export type Message = {
  id?: string
  isBot: boolean
  name: string
  message: string
  role?: "user" | "assistant" | "system"
  sources: any[]
  images?: string[]
  messageType?: string
  modelName?: string
  modelImage?: string
  serverMessageId?: string
  serverMessageVersion?: number
  createdAt?: number
  discoSkillComment?: DiscoSkillComment
}

export type ChatHistory = {
  role: "user" | "assistant" | "system"
  content: string
  image?: string
  messageType?: string
}[]

export type QueuedMessage = {
  message: string
  image: string
}

type State = {
  messages: Message[]
  setMessages: (messages: Message[]) => void
  history: ChatHistory
  setHistory: (history: ChatHistory) => void
  streaming: boolean
  setStreaming: (streaming: boolean) => void
  isFirstMessage: boolean
  setIsFirstMessage: (isFirstMessage: boolean) => void
  historyId: string | null
  setHistoryId: (
    history_id: string | null,
    options?: { preserveServerChatId?: boolean }
  ) => void
  isLoading: boolean
  setIsLoading: (isLoading: boolean) => void
  isProcessing: boolean
  setIsProcessing: (isProcessing: boolean) => void
  selectedModel: string | null
  setSelectedModel: (selectedModel: string) => void
  chatMode: "normal" | "rag" | "vision"
  setChatMode: (chatMode: "normal" | "rag" | "vision") => void
  isEmbedding: boolean
  setIsEmbedding: (isEmbedding: boolean) => void
  speechToTextLanguage: string
  setSpeechToTextLanguage: (speechToTextLanguage: string) => void
  currentURL: string
  setCurrentURL: (currentURL: string) => void
  selectedSystemPrompt: string | null
  setSelectedSystemPrompt: (selectedSystemPrompt: string) => void

  selectedQuickPrompt: string | null
  setSelectedQuickPrompt: (selectedQuickPrompt: string) => void

  useOCR: boolean
  setUseOCR: (useOCR: boolean) => void
}

export const useStoreMessage = createWithEqualityFn<State>((set) => ({
  messages: [],
  setMessages: (messages) => set({ messages }),
  history: [],
  setHistory: (history) => set({ history }),
  streaming: false,
  setStreaming: (streaming) => set({ streaming }),
  isFirstMessage: true,
  setIsFirstMessage: (isFirstMessage) => set({ isFirstMessage }),
  historyId: null,
  setHistoryId: (historyId) => set({ historyId }),
  isLoading: false,
  setIsLoading: (isLoading) => set({ isLoading }),
  isProcessing: false,
  setIsProcessing: (isProcessing) => set({ isProcessing }),
  defaultSpeechToTextLanguage: "en-US",
  selectedModel: null,
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  chatMode: "normal",
  setChatMode: (chatMode) => set({ chatMode }),
  isEmbedding: false,
  setIsEmbedding: (isEmbedding) => set({ isEmbedding }),
  speechToTextLanguage: "en-US",
  setSpeechToTextLanguage: (speechToTextLanguage) =>
    set({ speechToTextLanguage }),
  currentURL: "",
  setCurrentURL: (currentURL) => set({ currentURL }),

  selectedSystemPrompt: null,
  setSelectedSystemPrompt: (selectedSystemPrompt) =>
    set({ selectedSystemPrompt }),
  selectedQuickPrompt: null,
  setSelectedQuickPrompt: (selectedQuickPrompt) => set({ selectedQuickPrompt }),

  useOCR: false,
  setUseOCR: (useOCR) => set({ useOCR })
}))

const stopWatchingAccount = watchChatAccountChanges((invalidated) => {
  if (!invalidated) return
  useStoreMessage.setState({
    messages: [], history: [], historyId: null, streaming: false,
    isFirstMessage: true, isLoading: false, isProcessing: false, isEmbedding: false,
    selectedSystemPrompt: null, selectedQuickPrompt: null, currentURL: ""
  })
})
const hot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
hot?.dispose(stopWatchingAccount)
