import React from "react"
import type { useChatDraftOwner } from "@/hooks/useChatDraftOwner"
import { useDictionaryImportFlow } from "./useDictionaryImportFlow"
import { useDictionaryQuickAssign } from "./useDictionaryQuickAssign"
import { useRestoreFocusOnClose } from "./useRestoreFocusOnClose"

type UseDictionaryManagerWorkflowsParams = {
  owner: ReturnType<typeof useChatDraftOwner>
  onOpenChat: (chat: unknown, isCurrent: () => boolean) => Promise<boolean>
  isOnline: boolean
  dictionaries: any[]
  isDictionaryCurrent: (dictionary: unknown) => boolean
  queryClient: {
    invalidateQueries: (input: { queryKey: readonly unknown[] }) => Promise<unknown>
  }
  notification: {
    error: (config: { message: string; description?: string }) => void
    warning: (config: { message: string; description?: string }) => void
    success: (config: { message: string; description?: string }) => void
    info: (config: { message: string; description?: string }) => void
  }
  confirmDanger: (config: {
    title: string
    content: string
    okText: string
    cancelText: string
  }) => Promise<boolean>
  t: (key: string, fallbackOrOptions?: any) => string
  quickAssignFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  importDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
}

export function useDictionaryManagerWorkflows({
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
}: UseDictionaryManagerWorkflowsParams) {
  const quickAssignFlow = useDictionaryQuickAssign({
    owner,
    dictionaries,
    isDictionaryCurrent,
    onOpenChat,
    isOnline,
    notification,
    queryClient,
  })
  const importFlow = useDictionaryImportFlow({
    dictionaries,
    queryClient,
    notification,
    confirmDanger,
    t,
  })

  useRestoreFocusOnClose(Boolean(quickAssignFlow.assignFor), quickAssignFocusReturnRef)
  useRestoreFocusOnClose(importFlow.openImport, importDialogFocusReturnRef)

  return {
    ...quickAssignFlow,
    ...importFlow,
  }
}
