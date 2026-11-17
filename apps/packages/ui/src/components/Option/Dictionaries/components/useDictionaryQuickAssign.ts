import type { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import React from "react"

import {
  formatDictionaryChatReferenceTitle,
  normalizeDictionaryChatState
} from "../listUtils"

type Params = {
  isOnline: boolean
  owner: ReturnType<typeof useChatDraftOwner>
  dictionaries: Array<{ id: number; name?: string }>
  notification: {
    warning: (config: { message: string; description?: string }) => void
  }
  queryClient: {
    invalidateQueries: (input: {
      queryKey: readonly unknown[]
    }) => Promise<unknown>
  }
  isDictionaryCurrent: (dictionary: unknown) => boolean
  onOpenChat?: (chat: unknown, isCurrent: () => boolean) => Promise<boolean>
}
type AssignModal = {
  key: string
  dictionary: Params["dictionaries"][number]
  owner: Params["owner"]
}
const unavailable =
  "This server version does not support dictionary assignment to saved chats across the chat interfaces. Existing links are shown for reference."

export function useDictionaryQuickAssign({
  isOnline,
  owner,
  dictionaries,
  notification,
  onOpenChat,
  isDictionaryCurrent
}: Params) {
  const [modal, setModal] = React.useState<AssignModal | null>(null)
  const [assignSearch, setSearch] = React.useState("")
  const currentModal = React.useRef(modal)
  const currentSource = React.useRef({ owner, dictionaries, isOnline })
  const mounted = React.useRef(true)
  const isModalCurrent = (captured: AssignModal | null) =>
    Boolean(
      captured &&
      mounted.current &&
      currentModal.current === captured &&
      captured.owner.isCurrent() &&
      currentSource.current.owner.isCurrent === captured.owner.isCurrent &&
      currentSource.current.dictionaries.includes(captured.dictionary)
    )
  React.useLayoutEffect(() => {
    currentSource.current = { owner, dictionaries, isOnline }
    if (currentModal.current && !isModalCurrent(currentModal.current)) {
      currentModal.current = null
      setModal(null)
      setSearch("")
    }
  }, [owner, dictionaries, isOnline])
  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      currentModal.current = null
    }
  }, [])

  const cache = useQueryClient()
  const queryKey = [
    "tldw:listChatsForDictionaryAssign",
    owner.serverScopeKey,
    modal?.dictionary.id ?? null,
    modal?.key ?? null
  ]
  const query = useQuery({
    queryKey,
    enabled: Boolean(modal && owner.isCurrent() && isOnline),
    retry: false,
    queryFn: async ({ signal }) => {
      const isCurrent = () =>
        isModalCurrent(modal) &&
        isDictionaryCurrent(modal?.dictionary) &&
        currentSource.current.isOnline &&
        !signal.aborted
      if (!isCurrent() || !modal?.owner.requestScope)
        throw new Error("The dictionary account changed. Reopen this panel.")
      const snapshot = await loadServicePromptSnapshot([], {
        requestScope: modal.owner.requestScope,
        signal
      })
      try {
        if (!isCurrent() || snapshot.scopeSignal.aborted)
          throw new Error("The dictionary account changed. Reopen this panel.")
        await tldwClient.initialize()
        const chats: Awaited<
          ReturnType<typeof tldwClient.listChatsWithMeta>
        >["chats"] = []
        for (let page = 0; page < 100; page++) {
          if (!isCurrent() || snapshot.scopeSignal.aborted)
            throw new Error(
              "The dictionary account changed. Reopen this panel."
            )
          const response = await tldwClient.listChatsWithMeta(
            { limit: 100, offset: page * 100, include_deleted: false },
            {
              requestScope: modal.owner.requestScope,
              signal: snapshot.scopeSignal,
              scope: { type: "global" }
            }
          )
          if (!isCurrent() || snapshot.scopeSignal.aborted)
            throw new Error(
              "The dictionary account changed. Reopen this panel."
            )
          if (
            !response ||
            !Array.isArray(response.chats) ||
            response.chats.length > 100
          )
            throw new Error(
              "The saved chat list could not be verified. Retry loading it."
            )
          if (page === 99 && response.chats.length === 100)
            throw new Error(
              "The saved chat list is too large to load completely here. Open conversation history to find the chat."
            )
          chats.push(...response.chats)
          if (response.chats.length < 100) return { modalKey: modal.key, chats }
        }
        throw new Error(
          "The saved chat list could not be verified. Retry loading it."
        )
      } finally {
        snapshot.release()
      }
    }
  })
  const hasCurrentCatalog = Boolean(
    isModalCurrent(modal) &&
    query.data?.modalKey === modal?.key &&
    query.status === "success" &&
    !query.isFetching &&
    query.fetchStatus !== "paused"
  )
  const currentCatalog = React.useRef({
    data: query.data,
    active: hasCurrentCatalog
  })
  React.useLayoutEffect(() => {
    currentCatalog.current = { data: query.data, active: hasCurrentCatalog }
  }, [query.data, hasCurrentCatalog])
  const catalog = hasCurrentCatalog ? query.data!.chats : []
  const quickAssignChatOptions = catalog
    .filter((chat) => {
      const search = assignSearch.trim().toLowerCase()
      return (
        !search ||
        chat.id?.toLowerCase().includes(search) ||
        formatDictionaryChatReferenceTitle(chat).toLowerCase().includes(search)
      )
    })
    .map((chat) => ({
      chat,
      chatId: chat.id,
      title: formatDictionaryChatReferenceTitle(chat),
      state: normalizeDictionaryChatState(chat.state),
      disabledReason: "Assignment unavailable on this server version."
    }))

  const openQuickAssignModal = (dictionary: Params["dictionaries"][number]) => {
    if (
      !mounted.current ||
      currentModal.current !== modal ||
      !owner.isCurrent() ||
      currentSource.current.owner.isCurrent !== owner.isCurrent ||
      currentSource.current.dictionaries !== dictionaries ||
      !dictionaries.includes(dictionary) ||
      !isDictionaryCurrent(dictionary) ||
      !Number.isSafeInteger(dictionary.id) ||
      dictionary.id <= 0
    )
      return
    const next = { key: crypto.randomUUID(), dictionary, owner }
    currentModal.current = next
    setModal(next)
    setSearch("")
  }
  const closeQuickAssignModal = () => {
    if (!isModalCurrent(modal)) return
    currentModal.current = null
    setModal(null)
    setSearch("")
  }
  const refetchAssignableChats = async () => {
    if (!isModalCurrent(modal) || !currentSource.current.isOnline) return
    return await query.refetch()
  }
  const openQuickAssignChat = async (chat: unknown) => {
    const isCurrent = () => {
      const state = cache.getQueryState<typeof query.data>(queryKey)
      return (
        isModalCurrent(modal) &&
        isDictionaryCurrent(modal?.dictionary) &&
        currentCatalog.current.active &&
        currentCatalog.current.data === query.data &&
        state?.status === "success" &&
        state.fetchStatus === "idle" &&
        state.data === query.data
      )
    }
    if (
      !hasCurrentCatalog ||
      !isCurrent() ||
      !catalog.includes(chat as (typeof catalog)[number]) ||
      !onOpenChat
    )
      return false
    return await onOpenChat(chat, isCurrent)
  }
  const handleConfirmQuickAssign = async () => {
    if (isModalCurrent(modal))
      notification.warning({
        message: "Dictionary assignment unavailable",
        description: unavailable
      })
  }
  const setAssignSearch: React.Dispatch<React.SetStateAction<string>> = (
    value
  ) => {
    if (isModalCurrent(modal)) setSearch(value)
  }
  return {
    assignFor: modal?.dictionary ?? null,
    assignChatIds: [] as string[],
    assignSearch,
    setAssignSearch,
    assignSaving: false,
    assignableChatsStatus:
      query.status === "error" || query.fetchStatus === "paused" || !isOnline
        ? ("error" as const)
        : hasCurrentCatalog
          ? ("success" as const)
          : ("pending" as const),
    assignableChatsError:
      query.fetchStatus === "paused" || !isOnline
        ? new Error("You are offline. Reconnect and retry loading chats.")
        : query.error,
    refetchAssignableChats,
    quickAssignChatOptions,
    openQuickAssignModal,
    closeQuickAssignModal,
    openQuickAssignChat,
    toggleAssignChatSelection: (_chatId: string) => undefined,
    handleConfirmQuickAssign
  }
}
