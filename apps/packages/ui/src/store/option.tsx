import { createWithEqualityFn } from "zustand/traditional"
import { createCompareSlice } from "@/store/option/slices/compare-slice"
import { createCoreSlice } from "@/store/option/slices/core-slice"
import { createRagSlice } from "@/store/option/slices/rag-slice"
import { createReplySlice } from "@/store/option/slices/reply-slice"
import { createServerChatSlice } from "@/store/option/slices/server-chat-slice"
import type { State } from "@/store/option/types"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { updatePageTitle } from "@/utils/update-page-title"

export type {
  ChatHistory,
  Knowledge,
  Message,
  MessageMetadataExtra,
  MessageVariant,
  ReplyTarget,
  State,
  ToolChoice,
  WebSearch
} from "@/store/option/types"

export const useStoreMessageOption = createWithEqualityFn<State>()((set, get) => ({
  ...createCoreSlice(set, get),
  ...createRagSlice(set, get),
  ...createServerChatSlice(set, get),
  ...createCompareSlice(set, get),
  ...createReplySlice(set, get)
}))

const stopWatchingAccount = watchChatAccountChanges((invalidated) => {
  if (!invalidated) return
  useStoreMessageOption.getState().setServerChatId(null)
  useStoreMessageOption.setState({
    messages: [], history: [], historyId: null, queuedMessages: [],
    isFirstMessage: true, isLoading: false, isProcessing: false, streaming: false,
    isEmbedding: false, isSearchingInternet: false,
    selectedKnowledge: null, selectedSystemPrompt: null, selectedQuickPrompt: null,
    documentContext: null, uploadedFiles: [], contextFiles: [], actionInfo: null,
    fileRetrievalEnabled: false, ragMediaIds: null, ragSources: [], ragPinnedResults: [],
    compareMode: false, compareSelectedModels: [], compareSelectionByCluster: {},
    compareActiveModelsByCluster: {}, compareParentByHistory: {},
    compareCanonicalByCluster: {}, compareContinuationModeByCluster: {}, compareSplitChats: {},
    replyTarget: null, messageSteeringMode: "none", messageSteeringForceNarrate: false
  })
  updatePageTitle()
})
const hot = (import.meta as { hot?: { dispose: (callback: () => void) => void } }).hot
hot?.dispose(stopWatchingAccount)

if (typeof window !== "undefined") {
  // Expose for Playwright tests and debugging only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useStoreMessageOption = useStoreMessageOption
}
