import React from "react"
import type { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { useDictionaryEntriesPanel } from "./useDictionaryEntriesPanel"
import { useDictionaryListData } from "./useDictionaryListData"
import { useDictionaryManagerWorkflows } from "./useDictionaryManagerWorkflows"

type UseDictionaryManagerDataFlowsParams = {
  owner: ReturnType<typeof useChatDraftOwner>
  onOpenChat: (chat: unknown, isCurrent: () => boolean) => Promise<boolean>
  isOnline: boolean
  dictionarySearch: string
  dictionaryCategoryFilter: string
  dictionaryTagFilters: string[]
  notification: {
    error: (config: { message: string; description?: string }) => void
    warning: (config: { message: string; description?: string }) => void
    success: (config: { message: string; description?: string }) => void
    info: (config: { message: string; description?: string }) => void
  }
  queryClient: {
    invalidateQueries: (input: { queryKey: readonly unknown[] }) => Promise<unknown>
  }
  confirmDanger: (config: {
    title: string
    content: string
    okText: string
    cancelText: string
  }) => Promise<boolean>
  t: (key: string, fallbackOrOptions?: any) => string
  entriesDrawerFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  quickAssignFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  importDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
}

export function useDictionaryManagerDataFlows({
  owner,
  onOpenChat,
  isOnline,
  dictionarySearch,
  dictionaryCategoryFilter,
  dictionaryTagFilters,
  notification,
  queryClient,
  confirmDanger,
  t,
  entriesDrawerFocusReturnRef,
  quickAssignFocusReturnRef,
  importDialogFocusReturnRef,
}: UseDictionaryManagerDataFlowsParams) {
  const { openEntries, openDictionaryEntriesPanel, closeDictionaryEntriesPanel } =
    useDictionaryEntriesPanel({
      entriesDrawerFocusReturnRef,
    })

  const {
    data,
    status,
    error,
    refetch,
    filteredDictionaries,
    categoryFilterOptions,
    tagFilterOptions,
    activeEntriesDictionary,
    dictionariesById,
    isDictionaryCurrent,
    duplicateDictionary,
  } = useDictionaryListData({
    owner,
    isOnline,
    dictionarySearch,
    dictionaryCategoryFilter,
    dictionaryTagFilters,
    openEntries,
    notification,
    queryClient,
  })

  const dictionaries = Array.isArray(data) ? data : []
  const workflows = useDictionaryManagerWorkflows({
    owner,
    onOpenChat,
    isOnline,
    dictionaries,
    isDictionaryCurrent,
    queryClient,
    notification,
    confirmDanger,
    t,
    quickAssignFocusReturnRef,
    importDialogFocusReturnRef,
  })

  const openLinkedDictionaryChat = async (chat: unknown, dictionary: unknown) => {
    const isCurrent = () => isDictionaryCurrent(dictionary) && Boolean(dictionary && typeof dictionary === "object" &&
      Array.isArray((dictionary as { used_by_chat_refs?: unknown[] }).used_by_chat_refs) &&
      (dictionary as { used_by_chat_refs: unknown[] }).used_by_chat_refs.includes(chat))
    if (!isCurrent()) return false
    return await onOpenChat(chat, isCurrent)
  }

  return {
    openLinkedDictionaryChat,
    openEntries,
    openDictionaryEntriesPanel,
    closeDictionaryEntriesPanel,
    data,
    status,
    error,
    refetch,
    filteredDictionaries,
    categoryFilterOptions,
    tagFilterOptions,
    activeEntriesDictionary,
    dictionariesById,
    isDictionaryCurrent,
    duplicateDictionary,
    ...workflows,
  }
}
