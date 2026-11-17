import React from "react"
import { DictionaryManagerOverlays } from "./DictionaryManagerOverlays"
import { useDictionaryManagerDataFlows } from "./useDictionaryManagerDataFlows"
import { useDictionaryManagerEnvironment } from "./useDictionaryManagerEnvironment"
import { useDictionaryManagerOverlayProps } from "./useDictionaryManagerOverlayProps"
import { useDictionaryManagerWorkspaceControls } from "./useDictionaryManagerWorkspaceControls"

type UseDictionaryManagerOverlayRenderPropsParams = {
  environment: ReturnType<typeof useDictionaryManagerEnvironment>
  dataFlows: ReturnType<typeof useDictionaryManagerDataFlows>
  workspaceControls: ReturnType<typeof useDictionaryManagerWorkspaceControls>
}

export function useDictionaryManagerOverlayRenderProps({
  environment,
  dataFlows,
  workspaceControls,
}: UseDictionaryManagerOverlayRenderPropsParams): React.ComponentProps<
  typeof DictionaryManagerOverlays
> {
  return useDictionaryManagerOverlayProps({
    assignFor: dataFlows.assignFor,
    closeQuickAssignModal: workspaceControls.closeQuickAssignModal,
    handleConfirmQuickAssign: dataFlows.handleConfirmQuickAssign,
    assignChatIds: dataFlows.assignChatIds,
    assignSaving: dataFlows.assignSaving,
    assignSearch: dataFlows.assignSearch,
    setAssignSearch: dataFlows.setAssignSearch,
    assignableChatsStatus: dataFlows.assignableChatsStatus,
    assignableChatsError: dataFlows.assignableChatsError,
    quickAssignChatOptions: dataFlows.quickAssignChatOptions,
    refetchAssignableChats: dataFlows.refetchAssignableChats,
    toggleAssignChatSelection: dataFlows.toggleAssignChatSelection,
    openChatContextFromDictionary: dataFlows.openQuickAssignChat,
    openCreate: workspaceControls.openCreate,
    closeCreateModal: workspaceControls.closeCreateModal,
    createForm: workspaceControls.createForm,
    handleCreateSubmit: workspaceControls.handleCreateSubmit,
    creating: workspaceControls.creating,
    openEdit: workspaceControls.openEdit,
    closeDictionaryEditModal: workspaceControls.closeDictionaryEditModal,
    editForm: workspaceControls.editForm,
    handleEditSubmit: workspaceControls.handleEditSubmit,
    updating: workspaceControls.updating,
    activeEntriesDictionary: dataFlows.activeEntriesDictionary,
    openEntries: dataFlows.openEntries,
    closeDictionaryEntriesPanel: dataFlows.closeDictionaryEntriesPanel,
    useMobileEntriesDrawer: environment.useMobileEntriesDrawer,
    entryForm: environment.entryForm,
    openImport: dataFlows.openImport,
    closeImportModal: dataFlows.closeImportModal,
    importFormat: dataFlows.importFormat,
    handleImportFormatChange: dataFlows.handleImportFormatChange,
    importMode: dataFlows.importMode,
    handleImportModeChange: dataFlows.handleImportModeChange,
    importSourceContent: dataFlows.importSourceContent,
    handleImportSourceContentChange: dataFlows.handleImportSourceContentChange,
    importMarkdownName: dataFlows.importMarkdownName,
    handleImportMarkdownNameChange: dataFlows.handleImportMarkdownNameChange,
    importFileName: dataFlows.importFileName,
    handleImportFileSelection: dataFlows.handleImportFileSelection,
    activateOnImport: dataFlows.activateOnImport,
    handleActivateOnImportChange: dataFlows.handleActivateOnImportChange,
    buildImportPreview: dataFlows.buildImportPreview,
    importValidationErrors: dataFlows.importValidationErrors,
    importPreview: dataFlows.importPreview,
    handleConfirmImport: dataFlows.handleConfirmImport,
    importing: dataFlows.importing,
    importConflictResolution: dataFlows.importConflictResolution,
    closeImportConflictResolution: dataFlows.closeImportConflictResolution,
    resolveImportConflictRename: dataFlows.resolveImportConflictRename,
    resolveImportConflictReplace: dataFlows.resolveImportConflictReplace,
    statsFor: environment.statsFor,
    setStatsFor: environment.setStatsFor,
    versionHistoryFor: environment.versionHistoryFor,
    setVersionHistoryFor: environment.setVersionHistoryFor,
    handleVersionHistoryReverted:
      workspaceControls.handleDictionaryVersionHistoryReverted,
  })
}
