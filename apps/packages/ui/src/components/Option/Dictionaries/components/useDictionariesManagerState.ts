import React from "react"
import { DictionaryManagerOverlays } from "./DictionaryManagerOverlays"
import { DictionaryListSection } from "./DictionaryListSection"
import { useDictionaryManagerEnvironment } from "./useDictionaryManagerEnvironment"
import { useDictionaryManagerWorkspaceControls } from "./useDictionaryManagerWorkspaceControls"
import { useDictionaryManagerShortcuts } from "./useDictionaryManagerShortcuts"
import { useDictionaryManagerDataFlows } from "./useDictionaryManagerDataFlows"
import { useDictionaryManagerRenderProps } from "./useDictionaryManagerRenderProps"

type DictionariesManagerState = {
  listSectionProps: React.ComponentProps<typeof DictionaryListSection>
  overlayProps: React.ComponentProps<typeof DictionaryManagerOverlays>
}

export function useDictionariesManagerState(): DictionariesManagerState {
  const environment = useDictionaryManagerEnvironment()

  const dataFlows = useDictionaryManagerDataFlows({
    owner: environment.owner,
    onOpenChat: environment.openChatContextFromDictionary,
    isOnline: environment.isOnline,
    dictionarySearch: environment.dictionarySearch,
    dictionaryCategoryFilter: environment.dictionaryCategoryFilter,
    dictionaryTagFilters: environment.dictionaryTagFilters,
    notification: environment.notification,
    queryClient: environment.queryClient,
    confirmDanger: environment.confirmDanger,
    t: environment.t,
    entriesDrawerFocusReturnRef: environment.entriesDrawerFocusReturnRef,
    quickAssignFocusReturnRef: environment.quickAssignFocusReturnRef,
    importDialogFocusReturnRef: environment.importDialogFocusReturnRef,
  })

  const workspaceControls = useDictionaryManagerWorkspaceControls({
    queryClient: environment.queryClient,
    notification: environment.notification,
    confirmDanger: environment.confirmDanger,
    t: environment.t,
    dictionariesById: dataFlows.dictionariesById,
    duplicateDictionary: dataFlows.duplicateDictionary,
    setStatsFor: environment.setStatsFor,
    setVersionHistoryFor: environment.setVersionHistoryFor,
    statsDialogFocusReturnRef: environment.statsDialogFocusReturnRef,
    versionHistoryDialogFocusReturnRef:
      environment.versionHistoryDialogFocusReturnRef,
    createDialogFocusReturnRef: environment.createDialogFocusReturnRef,
    editDialogFocusReturnRef: environment.editDialogFocusReturnRef,
    importDialogFocusReturnRef: environment.importDialogFocusReturnRef,
    quickAssignFocusReturnRef: environment.quickAssignFocusReturnRef,
    openQuickAssignModalInternal: dataFlows.openQuickAssignModal,
    closeQuickAssignModalInternal: dataFlows.closeQuickAssignModal,
    openImportModal: dataFlows.openImportModal,
    capsLoading: environment.capsLoading,
    capabilities: environment.capabilities,
    useMobileEntriesDrawer: environment.useMobileEntriesDrawer,
    openChatContextFromDictionary: dataFlows.openLinkedDictionaryChat,
    openDictionaryEntriesPanel: dataFlows.openDictionaryEntriesPanel,
  })

  useDictionaryManagerShortcuts({
    openCreate: workspaceControls.openCreate,
    openEdit: workspaceControls.openEdit,
    openCreateDictionaryModal: workspaceControls.openCreateDictionaryModal,
    createForm: workspaceControls.createForm,
    editForm: workspaceControls.editForm,
  })

  return useDictionaryManagerRenderProps({
    environment,
    dataFlows,
    workspaceControls,
  })
}
