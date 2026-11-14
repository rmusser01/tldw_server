import React from "react"
import { useDictionaryManagerAvailabilityColumns } from "./useDictionaryManagerAvailabilityColumns"
import { useDictionaryManagerFormDialogs } from "./useDictionaryManagerFormDialogs"
import { useDictionaryManagerRowActionControls } from "./useDictionaryManagerRowActionControls"

type UseDictionaryManagerWorkspaceControlsParams = {
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
  dictionariesById: Map<number, any>
  duplicateDictionary: (record: any) => Promise<void>
  setStatsFor: React.Dispatch<React.SetStateAction<any | null>>
  setVersionHistoryFor: React.Dispatch<React.SetStateAction<any | null>>
  statsDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  versionHistoryDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  createDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  editDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  importDialogFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  quickAssignFocusReturnRef: React.MutableRefObject<HTMLElement | null>
  openQuickAssignModalInternal: (record: any) => void
  closeQuickAssignModalInternal: () => void
  openImportModal: () => void
  capsLoading: boolean
  capabilities: any
  useMobileEntriesDrawer: boolean
  openChatContextFromDictionary: (chatRef: unknown, dictionary: unknown) => void
  openDictionaryEntriesPanel: (dictionaryId: number) => void
}

export function useDictionaryManagerWorkspaceControls({
  queryClient,
  notification,
  confirmDanger,
  t,
  dictionariesById,
  duplicateDictionary,
  setStatsFor,
  setVersionHistoryFor,
  statsDialogFocusReturnRef,
  versionHistoryDialogFocusReturnRef,
  createDialogFocusReturnRef,
  editDialogFocusReturnRef,
  importDialogFocusReturnRef,
  quickAssignFocusReturnRef,
  openQuickAssignModalInternal,
  closeQuickAssignModalInternal,
  openImportModal,
  capsLoading,
  capabilities,
  useMobileEntriesDrawer,
  openChatContextFromDictionary,
  openDictionaryEntriesPanel,
}: UseDictionaryManagerWorkspaceControlsParams) {
  const {
    exportDictionaryAsJson,
    exportDictionaryAsMarkdown,
    openDictionaryStatsModal,
    openDictionaryVersionHistoryModal,
    confirmAndDeleteDictionary,
    handleDictionaryVersionHistoryReverted,
  } = useDictionaryManagerRowActionControls({
    queryClient,
    notification,
    confirmDanger,
    t,
    setStatsFor,
    setVersionHistoryFor,
    statsDialogFocusReturnRef,
    versionHistoryDialogFocusReturnRef,
  })

  const {
    confirmDeactivationIfNeeded,
    openCreate,
    openEdit,
    createForm,
    editForm,
    creating,
    updating,
    closeCreateModal,
    handleCreateSubmit,
    handleEditSubmit,
    openQuickAssignModal,
    closeQuickAssignModal,
    openCreateDictionaryModal,
    openImportDictionaryModal,
    openDictionaryEditModal,
    closeDictionaryEditModal,
  } = useDictionaryManagerFormDialogs({
    queryClient,
    notification,
    confirmDanger,
    t,
    dictionariesById,
    createDialogFocusReturnRef,
    editDialogFocusReturnRef,
    importDialogFocusReturnRef,
    quickAssignFocusReturnRef,
    openQuickAssignModalInternal,
    closeQuickAssignModalInternal,
    openImportModal,
  })

  const {
    dictionariesUnsupported,
    dictionariesUnsupportedTitle,
    dictionariesUnsupportedDescription,
    dictionariesUnsupportedPrimaryActionLabel,
    openHealthDiagnostics,
    columns,
  } = useDictionaryManagerAvailabilityColumns({
    queryClient,
    notification,
    confirmDanger,
    t,
    confirmDeactivationIfNeeded,
    useMobileEntriesDrawer,
    openChatContextFromDictionary,
    openDictionaryEditModal,
    openDictionaryEntriesPanel,
    openQuickAssignModal,
    exportDictionaryAsJson,
    exportDictionaryAsMarkdown,
    openDictionaryStatsModal,
    openDictionaryVersionHistoryModal,
    duplicateDictionary,
    confirmAndDeleteDictionary,
    capsLoading,
    capabilities,
  })

  return {
    openCreate,
    openEdit,
    createForm,
    editForm,
    creating,
    updating,
    closeCreateModal,
    handleCreateSubmit,
    handleEditSubmit,
    openQuickAssignModal,
    closeQuickAssignModal,
    openCreateDictionaryModal,
    openImportDictionaryModal,
    openDictionaryEditModal,
    closeDictionaryEditModal,
    dictionariesUnsupported,
    dictionariesUnsupportedTitle,
    dictionariesUnsupportedDescription,
    dictionariesUnsupportedPrimaryActionLabel,
    openHealthDiagnostics,
    columns,
    handleDictionaryVersionHistoryReverted,
  }
}
