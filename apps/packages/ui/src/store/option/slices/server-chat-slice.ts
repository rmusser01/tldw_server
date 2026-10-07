import type { StoreSlice } from "@/store/option/slices/types"

export const createServerChatSlice: StoreSlice<
  Pick<
    import("@/store/option/types").State,
    | "serverChatId"
    | "setServerChatId"
    | "serverChatTitle"
    | "setServerChatTitle"
    | "serverChatCharacterId"
    | "setServerChatCharacterId"
    | "serverChatAssistantKind"
    | "setServerChatAssistantKind"
    | "serverChatAssistantId"
    | "setServerChatAssistantId"
    | "serverChatPersonaMemoryMode"
    | "setServerChatPersonaMemoryMode"
    | "serverChatMetaLoaded"
    | "setServerChatMetaLoaded"
    | "serverChatLoadState"
    | "setServerChatLoadState"
    | "serverChatLoadError"
    | "setServerChatLoadError"
    | "serverChatState"
    | "setServerChatState"
    | "serverChatVersion"
    | "setServerChatVersion"
    | "serverChatTopic"
    | "setServerChatTopic"
    | "serverChatClusterId"
    | "setServerChatClusterId"
    | "serverChatSource"
    | "setServerChatSource"
    | "serverChatExternalRef"
    | "setServerChatExternalRef"
  >
> = (set) => ({
  serverChatId: null,
  setServerChatId: (id, options) =>
    set(() => ({
      serverChatId: id,
      // Saved selection retires a draft; creating a temporary session preserves it.
      ...(id?.trim() && !options?.preserveTemporaryChat ? { temporaryChat: false } : {}),
      serverChatState: id ? "in-progress" : null,
      serverChatVersion: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: false,
      serverChatLoadState: id ? "loading" : "idle",
      serverChatLoadError: null,
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null
    })),
  serverChatTitle: null,
  setServerChatTitle: (serverChatTitle) =>
    set({ serverChatTitle: serverChatTitle != null ? serverChatTitle : null }),
  serverChatCharacterId: null,
  setServerChatCharacterId: (serverChatCharacterId) =>
    set({
      serverChatCharacterId:
        serverChatCharacterId != null ? serverChatCharacterId : null
    }),
  serverChatAssistantKind: null,
  setServerChatAssistantKind: (serverChatAssistantKind) =>
    set({
      serverChatAssistantKind: serverChatAssistantKind ?? null
    }),
  serverChatAssistantId: null,
  setServerChatAssistantId: (serverChatAssistantId) =>
    set({
      serverChatAssistantId:
        serverChatAssistantId != null ? serverChatAssistantId : null
    }),
  serverChatPersonaMemoryMode: null,
  setServerChatPersonaMemoryMode: (serverChatPersonaMemoryMode) =>
    set({
      serverChatPersonaMemoryMode: serverChatPersonaMemoryMode ?? null
    }),
  serverChatMetaLoaded: false,
  setServerChatMetaLoaded: (serverChatMetaLoaded) =>
    set({ serverChatMetaLoaded }),
  serverChatLoadState: "idle",
  setServerChatLoadState: (serverChatLoadState) =>
    set({ serverChatLoadState }),
  serverChatLoadError: null,
  setServerChatLoadError: (serverChatLoadError) =>
    set({ serverChatLoadError: serverChatLoadError != null ? serverChatLoadError : null }),
  serverChatState: null,
  setServerChatState: (state) =>
    set({ serverChatState: state ?? null }),
  serverChatVersion: null,
  setServerChatVersion: (version) =>
    set({ serverChatVersion: version != null ? version : null }),
  serverChatTopic: null,
  setServerChatTopic: (topic) =>
    set({ serverChatTopic: topic != null ? topic : null }),
  serverChatClusterId: null,
  setServerChatClusterId: (clusterId) =>
    set({ serverChatClusterId: clusterId != null ? clusterId : null }),
  serverChatSource: null,
  setServerChatSource: (source) =>
    set({ serverChatSource: source != null ? source : null }),
  serverChatExternalRef: null,
  setServerChatExternalRef: (ref) =>
    set({ serverChatExternalRef: ref != null ? ref : null })
})
