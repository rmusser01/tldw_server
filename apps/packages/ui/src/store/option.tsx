import { createWithEqualityFn } from "zustand/traditional"
import { createCompareSlice } from "@/store/option/slices/compare-slice"
import { createCoreSlice } from "@/store/option/slices/core-slice"
import { createRagSlice } from "@/store/option/slices/rag-slice"
import { createReplySlice } from "@/store/option/slices/reply-slice"
import { createServerChatSlice } from "@/store/option/slices/server-chat-slice"
import type { State } from "@/store/option/types"

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

if (typeof window !== "undefined") {
  // Expose for Playwright tests and debugging only.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useStoreMessageOption = useStoreMessageOption
}
