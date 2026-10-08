import type { StoreSlice } from "@/store/option/slices/types"
import { chatDraftAddressKey, freshChatDraftAddress } from "@/utils/chat-draft-address"
import {
  buildQueuedRequest,
  normalizeQueuedRequests
} from "@/utils/chat-request-queue"
import { isFireFoxPrivateMode } from "@/utils/is-private-mode"

export const createCoreSlice: StoreSlice<
  Pick<
    import("@/store/option/types").State,
    | "chatDraftAddress"
    | "chatDraftRevision"
    | "chatDraftTransition"
    | "startNewChatDraft"
    | "setChatDraftAddress"
    | "messages"
    | "setMessages"
    | "history"
    | "setHistory"
    | "streaming"
    | "detachedHistoryResponse"
    | "setStreaming"
    | "isFirstMessage"
    | "setIsFirstMessage"
    | "historyId"
    | "setHistoryId"
    | "isLoading"
    | "setIsLoading"
    | "isProcessing"
    | "setIsProcessing"
    | "selectedModel"
    | "selectedModelStorageHandled"
    | "selectedModelRevision"
    | "setSelectedModel"
    | "chatMode"
    | "setChatMode"
    | "isEmbedding"
    | "setIsEmbedding"
    | "webSearch"
    | "setWebSearch"
    | "toolChoice"
    | "setToolChoice"
    | "isSearchingInternet"
    | "setIsSearchingInternet"
    | "selectedSystemPrompt"
    | "setSelectedSystemPrompt"
    | "selectedQuickPrompt"
    | "quickPromptRevision"
    | "setSelectedQuickPrompt"
    | "messageSteeringMode"
    | "setMessageSteeringMode"
    | "messageSteeringForceNarrate"
    | "setMessageSteeringForceNarrate"
    | "clearMessageSteering"
    | "queuedMessages"
    | "addQueuedMessage"
    | "setQueuedMessages"
    | "clearQueuedMessages"
    | "selectedKnowledge"
    | "setSelectedKnowledge"
    | "speechToTextLanguage"
    | "setSpeechToTextLanguage"
    | "temporaryChat"
    | "setTemporaryChat"
    | "useOCR"
    | "setUseOCR"
    | "documentContext"
    | "setDocumentContext"
    | "uploadedFiles"
    | "setUploadedFiles"
    | "contextFiles"
    | "setContextFiles"
    | "actionInfo"
    | "setActionInfo"
    | "fileRetrievalEnabled"
    | "setFileRetrievalEnabled"
  >
> = (set, get) => ({
  chatDraftAddress: null,
  chatDraftRevision: 0,
  chatDraftTransition: null,
  startNewChatDraft: (temporary = get().temporaryChat) => {
    get().setChatDraftAddress(freshChatDraftAddress(temporary))
  },
  setChatDraftAddress: (address, options = {}) => {
    let accepted = false
    set((state) => {
      if (options.expectedRevision !== undefined && options.expectedRevision !== state.chatDraftRevision) return state
      if (options.sourceAddress !== undefined && options.sourceAddress !== state.chatDraftAddress) return state
      accepted = true
      const destinationKey = chatDraftAddressKey(address)
      if (state.chatDraftAddress && chatDraftAddressKey(state.chatDraftAddress) === destinationKey) return state
      const revision = state.chatDraftRevision + 1
      return { chatDraftAddress: address, chatDraftRevision: revision,
        ...(state.chatDraftAddress && options.kind !== "bind" ? { selectedQuickPrompt: null, quickPromptRevision: state.quickPromptRevision + 1 } : {}),
        temporaryChat: address.kind === "fresh" && address.temporary,
        chatDraftTransition: { kind: options.kind ?? "switch", sourceAddress: state.chatDraftAddress, destinationKey, revision } }
    })
    return accepted
  },
  messages: [],
  setMessages: (messagesOrUpdater) =>
    set({
      messages:
        typeof messagesOrUpdater === "function"
          ? messagesOrUpdater(get().messages)
          : messagesOrUpdater
    }),
  history: [],
  setHistory: (historyOrUpdater) =>
    set({
      history:
        typeof historyOrUpdater === "function"
          ? historyOrUpdater(get().history)
          : historyOrUpdater
    }),
  streaming: false,
  detachedHistoryResponse: null,
  setStreaming: (streaming) => set({ streaming }),
  isFirstMessage: true,
  setIsFirstMessage: (isFirstMessage) => set({ isFirstMessage }),
  historyId: null,
  setHistoryId: (historyId, options) =>
    set((state) => {
      if (options?.preserveServerChatId) {
        return { historyId }
      }
      return {
        historyId,
        // When switching to a local Dexie-backed chat, clear any active server-backed session id.
        serverChatId: historyId ? null : state.serverChatId,
        serverChatState: historyId ? null : state.serverChatState,
        serverChatVersion: historyId ? null : state.serverChatVersion,
        serverChatTitle: historyId ? null : state.serverChatTitle,
        serverChatCharacterId: historyId ? null : state.serverChatCharacterId,
        serverChatMetaLoaded: historyId ? false : state.serverChatMetaLoaded,
        serverChatLoadState: historyId ? "idle" : state.serverChatLoadState,
        serverChatLoadError: historyId ? null : state.serverChatLoadError,
        serverChatTopic: historyId ? null : state.serverChatTopic,
        serverChatClusterId: historyId ? null : state.serverChatClusterId,
        serverChatSource: historyId ? null : state.serverChatSource,
        serverChatExternalRef: historyId ? null : state.serverChatExternalRef
      }
    }),
  isLoading: false,
  setIsLoading: (isLoading) => set({ isLoading }),
  isProcessing: false,
  setIsProcessing: (isProcessing) => set({ isProcessing }),
  speechToTextLanguage: "en-US",
  setSpeechToTextLanguage: (language) =>
    set({ speechToTextLanguage: language }),
  selectedModel: null,
  selectedModelStorageHandled: false,
  selectedModelRevision: 0,
  setSelectedModel: (selectedModel, options) =>
    set((state) => ({
      selectedModel,
      selectedModelStorageHandled: options?.storageHandled === true,
      selectedModelRevision: state.selectedModelRevision + 1
    })),
  chatMode: "normal",
  setChatMode: (chatMode) => set({ chatMode }),
  isEmbedding: false,
  setIsEmbedding: (isEmbedding) => set({ isEmbedding }),
  webSearch: false,
  setWebSearch: (webSearch) => set({ webSearch }),
  toolChoice: "none",
  setToolChoice: (toolChoice) => set({ toolChoice }),
  isSearchingInternet: false,
  setIsSearchingInternet: (isSearchingInternet) => set({ isSearchingInternet }),
  selectedSystemPrompt: null,
  setSelectedSystemPrompt: (selectedSystemPrompt) =>
    set({ selectedSystemPrompt }),
  selectedQuickPrompt: null,
  quickPromptRevision: 0,
  setSelectedQuickPrompt: (selectedQuickPrompt, options) => {
    let accepted = false
    set((state) => {
      if (options?.expectedRevision !== undefined && options.expectedRevision !== state.quickPromptRevision) return state
      accepted = true
      return { selectedQuickPrompt: selectedQuickPrompt || null, quickPromptRevision: state.quickPromptRevision + 1 }
    })
    return accepted
  },
  messageSteeringMode: "none",
  setMessageSteeringMode: (mode) => set({ messageSteeringMode: mode }),
  messageSteeringForceNarrate: false,
  setMessageSteeringForceNarrate: (enabled) =>
    set({ messageSteeringForceNarrate: enabled }),
  clearMessageSteering: () =>
    set({
      messageSteeringMode: "none",
      messageSteeringForceNarrate: false
    }),
  queuedMessages: [],
  addQueuedMessage: (payload) =>
    set((state) => ({
      queuedMessages: [...state.queuedMessages, buildQueuedRequest(payload)]
    })),
  setQueuedMessages: (queuedMessagesOrUpdater) =>
    set((state) => ({
      queuedMessages: normalizeQueuedRequests(
        typeof queuedMessagesOrUpdater === "function"
          ? queuedMessagesOrUpdater(state.queuedMessages)
          : queuedMessagesOrUpdater
      )
    })),
  clearQueuedMessages: () => set({ queuedMessages: [] }),
  selectedKnowledge: null,
  setSelectedKnowledge: (selectedKnowledge) => set({ selectedKnowledge }),
  temporaryChat: isFireFoxPrivateMode,
  setTemporaryChat: (temporaryChat) => set((state) => ({ temporaryChat,
    ...(state.temporaryChat !== temporaryChat ? {
      chatDraftRevision: state.chatDraftRevision + 1,
      chatDraftTransition: null,
      chatDraftAddress: state.chatDraftAddress?.kind === "fresh" ? { ...state.chatDraftAddress, temporary: temporaryChat } : state.chatDraftAddress
    } : {})
  })),
  useOCR: false,
  setUseOCR: (useOCR) => set({ useOCR }),
  documentContext: null,
  setDocumentContext: (documentContext) => set({ documentContext }),
  uploadedFiles: [],
  setUploadedFiles: (uploadedFiles) => set({ uploadedFiles }),
  contextFiles: [],
  setContextFiles: (contextFiles) => set({ contextFiles }),
  actionInfo: null,
  setActionInfo: (actionInfo) => set({ actionInfo }),
  fileRetrievalEnabled: false,
  setFileRetrievalEnabled: (fileRetrievalEnabled) =>
    set({ fileRetrievalEnabled })
})
