import { Drawer } from "antd"
import React from "react"
import { DictionaryFormModal } from "./DictionaryFormModal"
import { DictionaryEntryManager } from "./DictionaryEntryManager"

const LazyDictionaryQuickAssignModal = React.lazy(() =>
  import("./DictionaryQuickAssignModal").then((module) => ({
    default: module.DictionaryQuickAssignModal,
  }))
)

const LazyDictionaryImportModal = React.lazy(() =>
  import("./DictionaryImportModal").then((module) => ({
    default: module.DictionaryImportModal,
  }))
)

const LazyDictionaryStatsModal = React.lazy(() =>
  import("../DictionaryStatsModal").then((module) => ({
    default: module.DictionaryStatsModal,
  }))
)

const LazyDictionaryVersionHistoryModal = React.lazy(() =>
  import("../DictionaryVersionHistoryModal").then((module) => ({
    default: module.DictionaryVersionHistoryModal,
  }))
)

type DictionaryManagerOverlaysProps = {
  assignFor: any
  closeQuickAssignModal: () => void
  handleConfirmQuickAssign: () => void
  assignChatIds: string[]
  assignSaving: boolean
  assignSearch: string
  setAssignSearch: React.Dispatch<React.SetStateAction<string>>
  assignableChatsStatus: "pending" | "error" | "success"
  assignableChatsError: unknown
  quickAssignChatOptions: Array<{
    chat: any
    chatId: string
    title: string
    state: string
  }>
  refetchAssignableChats: () => Promise<unknown>
  toggleAssignChatSelection: (chatId: string) => void
  openChatContextFromDictionary: (chatRef: any) => void
  openCreate: boolean
  closeCreateModal: () => void
  createForm: any
  handleCreateSubmit: (values: any) => void
  creating: boolean
  openEdit: boolean
  closeDictionaryEditModal: () => void
  editForm: any
  handleEditSubmit: (values: any) => Promise<void>
  updating: boolean
  activeEntriesDictionary: any | null
  openEntries: number | null
  closeDictionaryEntriesPanel: () => void
  useMobileEntriesDrawer: boolean
  entryForm: any
  openImport: boolean
  closeImportModal: () => void
  importFormat: "json" | "markdown"
  handleImportFormatChange: (value: "json" | "markdown") => void
  importMode: "file" | "paste"
  handleImportModeChange: (value: "file" | "paste") => void
  importSourceContent: string
  handleImportSourceContentChange: (value: string) => void
  importMarkdownName: string
  handleImportMarkdownNameChange: (value: string) => void
  importFileName: string | null
  handleImportFileSelection: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>
  activateOnImport: boolean
  handleActivateOnImportChange: (value: boolean) => void
  buildImportPreview: () => void
  importValidationErrors: string[]
  importPreview: any
  handleConfirmImport: () => void
  importing: boolean
  importConflictResolution: any
  closeImportConflictResolution: () => void
  resolveImportConflictRename: () => Promise<void>
  resolveImportConflictReplace: () => Promise<void>
  statsFor: any | null
  setStatsFor: React.Dispatch<React.SetStateAction<any | null>>
  versionHistoryFor: any | null
  setVersionHistoryFor: React.Dispatch<React.SetStateAction<any | null>>
  handleVersionHistoryReverted: () => void
}

export const DictionaryManagerOverlays: React.FC<DictionaryManagerOverlaysProps> = ({
  assignFor,
  closeQuickAssignModal,
  handleConfirmQuickAssign,
  assignChatIds,
  assignSaving,
  assignSearch,
  setAssignSearch,
  assignableChatsStatus,
  assignableChatsError,
  quickAssignChatOptions,
  refetchAssignableChats,
  toggleAssignChatSelection,
  openChatContextFromDictionary,
  openCreate,
  closeCreateModal,
  createForm,
  handleCreateSubmit,
  creating,
  openEdit,
  closeDictionaryEditModal,
  editForm,
  handleEditSubmit,
  updating,
  activeEntriesDictionary,
  openEntries,
  closeDictionaryEntriesPanel,
  useMobileEntriesDrawer,
  entryForm,
  openImport,
  closeImportModal,
  importFormat,
  handleImportFormatChange,
  importMode,
  handleImportModeChange,
  importSourceContent,
  handleImportSourceContentChange,
  importMarkdownName,
  handleImportMarkdownNameChange,
  importFileName,
  handleImportFileSelection,
  activateOnImport,
  handleActivateOnImportChange,
  buildImportPreview,
  importValidationErrors,
  importPreview,
  handleConfirmImport,
  importing,
  importConflictResolution,
  closeImportConflictResolution,
  resolveImportConflictRename,
  resolveImportConflictReplace,
  statsFor,
  setStatsFor,
  versionHistoryFor,
  setVersionHistoryFor,
  handleVersionHistoryReverted,
}) => {
  return (
    <>
      <React.Suspense fallback={null}>
        <LazyDictionaryQuickAssignModal
          dictionaryName={assignFor?.name}
          open={Boolean(assignFor)}
          onCancel={closeQuickAssignModal}
          onConfirm={() => void handleConfirmQuickAssign()}
          selectedChatIds={assignChatIds}
          assignSaving={assignSaving}
          searchValue={assignSearch}
          onSearchChange={setAssignSearch}
          chatsStatus={assignableChatsStatus}
          chatsError={assignableChatsError}
          chatOptions={quickAssignChatOptions}
          onRetry={() => void refetchAssignableChats()}
          onToggleChatSelection={toggleAssignChatSelection}
          onOpenChat={openChatContextFromDictionary}
        />
      </React.Suspense>

      <DictionaryFormModal
        title="Create Dictionary"
        open={openCreate}
        onCancel={closeCreateModal}
        form={createForm}
        onFinish={handleCreateSubmit}
        submitLabel="Create"
        submitLoading={creating}
      />

      <DictionaryFormModal
        title="Edit Dictionary"
        open={openEdit}
        onCancel={closeDictionaryEditModal}
        form={editForm}
        onFinish={handleEditSubmit}
        submitLabel="Save"
        submitLoading={updating}
        includeActiveField
      />

      <Drawer
        title={
          activeEntriesDictionary?.name
            ? `Manage Entries: ${activeEntriesDictionary.name}`
            : "Manage Entries"
        }
        open={!!openEntries}
        onClose={closeDictionaryEntriesPanel}
        placement="right"
        destroyOnClose
        size={useMobileEntriesDrawer ? "100vw" : 1040}
      >
        {openEntries ? <DictionaryEntryManager dictionaryId={openEntries} form={entryForm} /> : null}
      </Drawer>

      <React.Suspense fallback={null}>
        <LazyDictionaryImportModal
          open={openImport}
          onCancel={closeImportModal}
          importFormat={importFormat}
          onImportFormatChange={handleImportFormatChange}
          importMode={importMode}
          onImportModeChange={handleImportModeChange}
          importSourceContent={importSourceContent}
          onImportSourceContentChange={handleImportSourceContentChange}
          importMarkdownName={importMarkdownName}
          onImportMarkdownNameChange={handleImportMarkdownNameChange}
          importFileName={importFileName}
          onImportFileSelection={handleImportFileSelection}
          activateOnImport={activateOnImport}
          onActivateOnImportChange={handleActivateOnImportChange}
          onBuildImportPreview={buildImportPreview}
          importValidationErrors={importValidationErrors}
          importPreview={importPreview}
          onConfirmImport={() => void handleConfirmImport()}
          importing={importing}
          importConflictResolution={importConflictResolution}
          onCloseConflictResolution={closeImportConflictResolution}
          onResolveConflictRename={() => void resolveImportConflictRename()}
          onResolveConflictReplace={() => void resolveImportConflictReplace()}
        />
      </React.Suspense>

      {statsFor && (
        <React.Suspense fallback={null}>
          <LazyDictionaryStatsModal
            open={Boolean(statsFor)}
            stats={statsFor}
            onClose={() => setStatsFor(null)}
          />
        </React.Suspense>
      )}

      {versionHistoryFor && (
        <React.Suspense fallback={null}>
          <LazyDictionaryVersionHistoryModal
            open={Boolean(versionHistoryFor)}
            dictionary={versionHistoryFor}
            onClose={() => setVersionHistoryFor(null)}
            onReverted={handleVersionHistoryReverted}
          />
        </React.Suspense>
      )}
    </>
  )
}
