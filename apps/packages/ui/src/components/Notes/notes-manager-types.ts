export type { NoteListItem } from './types'

export type {
  NoteTask,
  NoteTaskActivityEvent,
  NoteTaskReconciliationSummary,
  NoteTaskStatus,
} from '@/services/notes-tasks'

export type {
  KeywordSyncWarning,
  SaveNoteOptions,
  SaveIndicatorState,
  SaveRecoveryNotice,
  NotesEditorMode,
  NotesInputMode,
  NotesSortOption,
  NotesBrowseSortOption,
  KeywordPickerSortMode,
  KeywordFrequencyTone,
  KeywordManagementItem,
  KeywordRenameDraft,
  KeywordMergeDraft,
  MarkdownToolbarAction,
  OfflineDraftSyncState,
  OfflineDraftEntry,
  OfflineDraftSyncResult,
  RemoteVersionInfo,
  NotesAssistAction,
  EditProvenanceState,
  MonitoringAlertSeverity,
  MonitoringNoticeState,
  ExportFormat,
  ExportProgressState,
  NotesListViewMode,
  MoodboardSummary,
  NotebookFilterOption,
  ImportFormat,
  ImportDuplicateStrategy,
  PendingImportFile,
  NotesImportResponsePayload,
  NotesTitleSettingsResponse,
  NoteTemplateDefinition,
  NotesTocEntry,
  NoteWithKeywords,
} from './notes-manager-utils'

/** Options for fetching one page of the notes list or of a search. */
export type NotesPageFetchOptions = {
  /** False to skip match excerpts: gathering ids for select-all or export needs none. */
  includeSnippets?: boolean
}
