import React from 'react'
import type { InputRef } from 'antd'
import { Input, Typography, Select, Button, Tooltip, Spin } from 'antd'
import {
  Sparkles as SparklesIcon,
  Bold as BoldIcon,
  Italic as ItalicIcon,
  Heading1 as HeadingIcon,
  List as ListIcon,
  Link2 as LinkIcon,
  Code2 as CodeIcon,
  Paperclip as PaperclipIcon
} from 'lucide-react'
import { QuestionCircleOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import NotesEditorHeader from '@/components/Notes/NotesEditorHeader'
import NotesStudioView from '@/components/Notes/NotesStudioView'
import CollapsibleSection from '@/components/Notes/CollapsibleSection'
import type { ActiveWikilinkQuery, WikilinkCandidate } from '@/components/Notes/wikilinks'
import type {
  SaveIndicatorState,
  NotesEditorMode,
  NotesInputMode,
  NotesAssistAction,
  NotesTocEntry,
  MonitoringNoticeState,
  RemoteVersionInfo,
  OfflineDraftEntry,
  MarkdownToolbarAction,
} from './notes-manager-types'
import type { SingleNoteCopyMode, SingleNoteExportFormat } from './export-utils'
import type { NoteStudioState, NotesStudioPaperSize } from './notes-studio-types'
import type { NotesTitleSuggestStrategy } from '@/services/settings/ui-settings'
import {
  NOTES_EDITOR_REGION_ID,
  NOTES_SHORTCUTS_SUMMARY_ID,
  NOTE_TEMPLATES,
  normalizeNotesTitleStrategy,
} from './notes-manager-utils'
import { NOTES_TITLE_SUGGEST_STRATEGY_SETTING } from '@/services/settings/ui-settings'
import { setSetting } from '@/services/settings/registry'

const LazyMarkdownPreview = React.lazy(() =>
  import('@/components/Common/MarkdownPreview').then((module) => ({
    default: module.MarkdownPreview,
  }))
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NoteRelationsShape {
  sources: Array<{ id: string; label: string }>
  manualLinks: Array<{
    edgeId: string
    noteId: string
    title: string
    directed: boolean
    outgoing: boolean
  }>
  related: Array<{ id: string; title: string }>
  backlinks: Array<{ id: string; title: string }>
}

export interface NotesEditorPaneProps {
  // Layout
  isMobileViewport: boolean
  setMobileSidebarOpen: (open: boolean) => void

  // Selection / identity
  selectedId: string | number | null
  showEmptyState: boolean
  title: string
  content: string
  editorDisabled: boolean
  isDirty: boolean
  isOnline: boolean
  loadingDetail: boolean
  saving: boolean

  // Backlinks
  backlinkConversationId: string | null
  backlinkConversationLabel: string | null
  backlinkMessageId: string | null

  // Relations
  noteRelations: NoteRelationsShape
  noteNeighborsLoading: boolean
  noteNeighborsError: boolean
  onRetryNeighbors?: () => void

  // Pinning
  selectedNotePinned: boolean

  // Editor state
  editorMode: NotesEditorMode
  editorInputMode: NotesInputMode
  openingLinkedChat: boolean
  editorKeywords: string[]
  keywordOptions: string[]
  saveIndicator: SaveIndicatorState
  saveIndicatorText: string | null
  selectedLastSavedAt: string | null
  offlineStatusText: string | null
  currentOfflineDraft: OfflineDraftEntry | null
  remoteVersionInfo: RemoteVersionInfo | null
  monitoringNotice: MonitoringNoticeState | null
  monitoringNoticeClasses: string

  // Title suggestion
  titleSuggestionLoading: boolean
  canSwitchTitleStrategy: boolean
  effectiveTitleSuggestStrategy: NotesTitleSuggestStrategy
  titleStrategyOptions: Array<{ label: string; value: string }>
  studioBadgeLabel?: string | null
  showStudioMarkdownOnlyNotice?: boolean
  selectedStudioState?: NoteStudioState | null
  studioPaperSize?: NotesStudioPaperSize
  onStudioPaperSizeChange?: (paperSize: NotesStudioPaperSize) => void
  onRegenerateStudioView?: () => void
  studioRegenerating?: boolean

  // Title strategy state setter
  setTitleSuggestStrategy: (strategy: NotesTitleSuggestStrategy) => void

  // Manual links
  manualLinkTargetId: string | null
  setManualLinkTargetId: (id: string | null) => void
  manualLinkSaving: boolean
  manualLinkOptions: Array<{ value: string; label: string }>
  manualLinkDeletingEdgeId: string | null

  // Assist
  assistLoadingAction: NotesAssistAction | null
  canUndoAssist: boolean
  undoAssist: () => void

  // TOC
  shouldShowToc: boolean
  tocEntries: NotesTocEntry[]

  // Preview
  previewContent: string
  usesLargePreviewGuardrails: boolean
  largePreviewReady: boolean

  // WYSIWYG
  wysiwygHtml: string

  // Wikilinks
  activeWikilinkQuery: ActiveWikilinkQuery | null
  wikilinkSuggestions: WikilinkCandidate[]
  wikilinkSuggestionDisplayCounts: Map<string, number>
  wikilinkSelectionIndex: number

  // Metrics
  metricSummaryText: string
  revisionSummaryText: string
  provenanceSummaryText: string
  queuedOfflineDraftCount: number

  // Refs
  titleInputRef: React.Ref<InputRef>
  contentTextareaRef: React.Ref<HTMLTextAreaElement>
  richEditorRef: React.Ref<HTMLDivElement>
  attachmentInputRef: React.Ref<HTMLInputElement>

  // Setters used in inline handlers
  setTitle: (title: string) => void
  setIsDirty: (dirty: boolean) => void
  setSaveIndicator: (state: SaveIndicatorState) => void
  setMonitoringNotice: (notice: MonitoringNoticeState | null) => void
  setEditorMode: (mode: NotesEditorMode) => void
  setEditorKeywords: (keywords: string[]) => void
  setEditorCursorIndex: (index: number | null) => void
  setShortcutHelpOpen: (open: boolean) => void

  // Callbacks
  markManualEdit: () => void
  suggestTitle: () => Promise<void>
  openLinkedConversation: () => Promise<void>
  openLinkedSource: (sourceId: string, sourceLabel: string) => void
  handleNewNote: (templateId?: string) => Promise<void>
  duplicateSelectedNote: () => Promise<void>
  toggleNotePinned: (id: string | number) => Promise<void>
  copySelected: (mode: SingleNoteCopyMode) => Promise<void>
  handleGenerateFlashcardsFromNote: () => void
  handleCreateStudyPackFromNote?: () => void
  handleOpenNotesStudio: () => void
  exportSelected: (format: SingleNoteExportFormat) => void
  saveNote: () => Promise<void>
  deleteNote: () => Promise<void>
  handleSelectNote: (id: string | number) => Promise<void>
  openGraphModal: () => void
  createManualLink: () => Promise<void>
  removeManualLink: (edgeId: string) => Promise<void>
  debouncedLoadKeywordSuggestions: (text: string) => void
  renderKeywordLabelWithFrequency: (
    keyword: string,
    options: { includeCount: boolean; testIdPrefix: string }
  ) => React.ReactNode
  handleEditorInputModeChange: (mode: NotesInputMode) => void
  applyMarkdownToolbarAction: (action: MarkdownToolbarAction) => void
  openAttachmentPicker: () => void
  handleAttachmentInputChange: (event: React.ChangeEvent<HTMLInputElement>) => void
  runAssistAction: (action: NotesAssistAction) => Promise<void>
  switchStudioNoticeToMarkdown: () => void
  dismissStudioMarkdownOnlyNotice: () => void
  handleTocJump: (entry: NotesTocEntry) => void
  handlePreviewLinkClick: (event: React.MouseEvent<HTMLDivElement>) => void
  handleWysiwygInput: (event: React.FormEvent<HTMLDivElement>) => void
  handleWysiwygPaste: (event: React.ClipboardEvent<HTMLDivElement>) => void
  handleEditorChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
  handleEditorKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void
  handleEditorSelectionUpdate: (
    event: React.SyntheticEvent<HTMLTextAreaElement>
  ) => void
  applyWikilinkSuggestion: (candidate: WikilinkCandidate) => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const NotesEditorPane: React.FC<NotesEditorPaneProps> = ({
  isMobileViewport,
  setMobileSidebarOpen,
  selectedId,
  showEmptyState,
  title,
  content,
  editorDisabled,
  isDirty,
  isOnline,
  loadingDetail,
  saving,
  backlinkConversationId,
  backlinkConversationLabel,
  backlinkMessageId,
  noteRelations,
  noteNeighborsLoading,
  noteNeighborsError,
  onRetryNeighbors,
  selectedNotePinned,
  editorMode,
  editorInputMode,
  openingLinkedChat,
  editorKeywords,
  keywordOptions,
  saveIndicator,
  saveIndicatorText,
  selectedLastSavedAt,
  offlineStatusText,
  currentOfflineDraft,
  remoteVersionInfo,
  monitoringNotice,
  monitoringNoticeClasses,
  titleSuggestionLoading,
  canSwitchTitleStrategy,
  effectiveTitleSuggestStrategy,
  titleStrategyOptions,
  studioBadgeLabel = null,
  showStudioMarkdownOnlyNotice = false,
  selectedStudioState = null,
  studioPaperSize = 'A4',
  onStudioPaperSizeChange = () => {},
  onRegenerateStudioView = () => {},
  studioRegenerating = false,
  setTitleSuggestStrategy,
  manualLinkTargetId,
  setManualLinkTargetId,
  manualLinkSaving,
  manualLinkOptions,
  manualLinkDeletingEdgeId,
  assistLoadingAction,
  canUndoAssist,
  undoAssist,
  shouldShowToc,
  tocEntries,
  previewContent,
  usesLargePreviewGuardrails,
  largePreviewReady,
  wysiwygHtml,
  activeWikilinkQuery,
  wikilinkSuggestions,
  wikilinkSuggestionDisplayCounts,
  wikilinkSelectionIndex,
  metricSummaryText,
  revisionSummaryText,
  provenanceSummaryText,
  queuedOfflineDraftCount,
  titleInputRef,
  contentTextareaRef,
  richEditorRef,
  attachmentInputRef,
  setTitle,
  setIsDirty,
  setSaveIndicator,
  setMonitoringNotice,
  setEditorMode,
  setEditorKeywords,
  setEditorCursorIndex,
  setShortcutHelpOpen,
  markManualEdit,
  suggestTitle,
  openLinkedConversation,
  openLinkedSource,
  handleNewNote,
  duplicateSelectedNote,
  toggleNotePinned,
  copySelected,
  handleGenerateFlashcardsFromNote,
  handleCreateStudyPackFromNote,
  handleOpenNotesStudio,
  exportSelected,
  saveNote,
  deleteNote,
  handleSelectNote,
  openGraphModal,
  createManualLink,
  removeManualLink,
  debouncedLoadKeywordSuggestions,
  renderKeywordLabelWithFrequency,
  handleEditorInputModeChange,
  applyMarkdownToolbarAction,
  openAttachmentPicker,
  handleAttachmentInputChange,
  runAssistAction,
  switchStudioNoticeToMarkdown,
  dismissStudioMarkdownOnlyNotice,
  handleTocJump,
  handlePreviewLinkClick,
  handleWysiwygInput,
  handleWysiwygPaste,
  handleEditorChange,
  handleEditorKeyDown,
  handleEditorSelectionUpdate,
  applyWikilinkSuggestion,
}) => {
  const { t } = useTranslation(['option', 'common'])
  const renderHelpButton = React.useCallback(
    (label: string) => (
      <button
        type="button"
        className="inline-flex items-center rounded-sm text-text-muted transition-colors hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        aria-label={label}
      >
        <QuestionCircleOutlined className="text-[11px]" />
      </button>
    ),
    []
  )

  const renderMarkdownPreviewSurface = (
    testId: string,
  ) => (
    <div
      className="w-full flex-1 text-sm p-4 rounded-lg border border-border bg-surface2 overflow-auto"
      onClick={handlePreviewLinkClick}
      data-testid={testId}
    >
      <React.Suspense fallback={null}>
        <LazyMarkdownPreview content={previewContent} size="sm" />
      </React.Suspense>
    </div>
  )

  return (
    <section
      id={NOTES_EDITOR_REGION_ID}
      tabIndex={-1}
      role="region"
      aria-label={t('option:notesSearch.editorRegionLabel', {
        defaultValue: 'Note editor'
      })}
      aria-describedby={NOTES_SHORTCUTS_SUMMARY_ID}
      aria-busy={loadingDetail}
      data-testid="notes-editor-region"
      className={`flex-1 flex flex-col overflow-hidden rounded-lg border border-border bg-surface ${
        isMobileViewport ? 'ml-0' : 'ml-4'
      }`}
      aria-disabled={editorDisabled}
    >
      {isMobileViewport && (
        <div className="border-b border-border px-4 py-2">
          <Button
            size="large"
            onClick={() => setMobileSidebarOpen(true)}
            data-testid="notes-mobile-open-list-button"
            className="min-h-[44px]"
          >
            {t('option:notesSearch.openMobileSidebar', {
              defaultValue: 'Browse notes'
            })}
          </Button>
        </div>
      )}
      <NotesEditorHeader
        title={title}
        selectedId={selectedId}
        backlinkConversationId={backlinkConversationId}
        backlinkConversationLabel={backlinkConversationLabel}
        backlinkMessageId={backlinkMessageId}
        sourceLinks={noteRelations.sources}
        editorDisabled={editorDisabled}
        openingLinkedChat={openingLinkedChat}
        editorMode={editorMode}
        hasContent={content.trim().length > 0}
        canSave={
          !editorDisabled &&
          (title.trim().length > 0 || content.trim().length > 0)
        }
        canGenerateFlashcards={!editorDisabled && content.trim().length > 0}
        canOpenNotesStudio={!editorDisabled && selectedId != null && content.trim().length > 0}
        canExport={Boolean(title || content)}
        canDuplicate={!editorDisabled && (title.trim().length > 0 || content.trim().length > 0)}
        canPin={!editorDisabled && selectedId != null}
        isPinned={selectedNotePinned}
        templateOptions={NOTE_TEMPLATES.map((template) => ({
          id: template.id,
          label: template.label
        }))}
        isSaving={saving}
        canDelete={!editorDisabled && isOnline && selectedId != null}
        isDirty={isDirty}
        saveIndicator={saveIndicator}
        lastSavedAt={selectedLastSavedAt}
        onOpenLinkedConversation={() => {
          void openLinkedConversation()
        }}
        onOpenSourceLink={(sourceId, sourceLabel) => {
          openLinkedSource(sourceId, sourceLabel)
        }}
        onApplyTemplate={(templateId) => {
          void handleNewNote(templateId)
        }}
        onDuplicate={() => {
          void duplicateSelectedNote()
        }}
        onTogglePin={() => {
          if (selectedId == null) return
          void toggleNotePinned(selectedId)
        }}
        onChangeEditorMode={(nextMode) => {
          setEditorMode(nextMode)
        }}
        onCopy={(mode) => {
          void copySelected(mode)
        }}
        canCreateStudyPack={!editorDisabled && selectedId != null && !isDirty && Boolean(title.trim())}
        onGenerateFlashcards={handleGenerateFlashcardsFromNote}
        onCreateStudyPack={handleCreateStudyPackFromNote}
        onOpenNotesStudio={handleOpenNotesStudio}
        onExport={(format) => {
          exportSelected(format)
        }}
        onSave={() => {
          void saveNote()
        }}
        onDelete={() => {
          void deleteNote()
        }}
        studioBadgeLabel={studioBadgeLabel}
      />
      <div className="flex-1 flex flex-col px-4 py-3 overflow-auto">
        {showEmptyState && !loadingDetail && (
          <div className="flex flex-col items-center justify-center gap-4 px-8 py-12 text-center" data-testid="notes-editor-empty-state">
            <div className="text-lg font-medium text-text">
              {t('option:notesSearch.editorEmptyTitle', {
                defaultValue: 'Select or create a note'
              })}
            </div>
            <div className="max-w-sm text-sm text-text-muted">
              {t('option:notesSearch.editorEmptyDescription', {
                defaultValue: 'Choose a note from the list to start editing, or create a new one.'
              })}
            </div>
            <div className="flex gap-2">
              <Button
                type="primary"
                onClick={() => {
                  if (!editorDisabled) {
                    void handleNewNote()
                  }
                }}
                disabled={editorDisabled}
                data-testid="notes-editor-empty-create"
              >
                {t('option:notesSearch.editorEmptyCreateAction', {
                  defaultValue: 'Create note'
                })}
              </Button>
            </div>
            <div className="mt-2 text-xs text-text-muted">
              {t('option:notesSearch.editorEmptyHint', {
                defaultValue: 'Tip: Type [[ in a note to link to another note.'
              })}
            </div>
          </div>
        )}
        {showStudioMarkdownOnlyNotice ? (
          <div className="mb-3 flex items-center justify-between gap-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-sm text-warn">
            <span>
              {t('option:notesSearch.notesStudioMarkdownOnlyNotice', {
                defaultValue: 'Notes Studio works from Markdown selections only.'
              })}
            </span>
            <div className="flex items-center gap-2">
              <Button size="small" onClick={switchStudioNoticeToMarkdown}>
                {t('option:notesSearch.notesStudioSwitchToMarkdown', {
                  defaultValue: 'Switch to Markdown'
                })}
              </Button>
              <Button size="small" type="text" onClick={dismissStudioMarkdownOnlyNotice}>
                {t('common:close', { defaultValue: 'Close' })}
              </Button>
            </div>
          </div>
        ) : null}
        {loadingDetail && (
          <div
            className="mb-3 inline-flex w-fit items-center gap-2 rounded border border-border bg-surface2 px-3 py-1.5 text-[12px] text-text-muted"
            role="status"
            aria-live="polite"
            data-testid="notes-editor-loading-detail"
          >
            <Spin size="small" />
            <span>
              {t('option:notesSearch.loadingDetail', {
                defaultValue: 'Loading note details...'
              })}
            </span>
          </div>
        )}
        {selectedStudioState && editorMode !== 'edit' ? (
          <NotesStudioView
            note={selectedStudioState.note}
            studioDocument={selectedStudioState.studio_document}
            isStale={selectedStudioState.is_stale}
            staleReason={selectedStudioState.stale_reason}
            paperSize={studioPaperSize}
            onPaperSizeChange={onStudioPaperSizeChange}
            onRegenerate={onRegenerateStudioView}
            regenerating={studioRegenerating}
            onContinueEditingPlainNote={() => {
              handleEditorInputModeChange('markdown')
              setEditorMode('edit')
            }}
          />
        ) : (
          <>
        <div className="flex items-center gap-2">
          <Input
            placeholder={t('option:notesSearch.titlePlaceholder', {
              defaultValue: 'Title'
            })}
            value={title}
            onChange={(e) => {
              setTitle(e.target.value)
              setIsDirty(true)
              setSaveIndicator('dirty')
              setMonitoringNotice(null)
              markManualEdit()
            }}
            disabled={editorDisabled}
            ref={titleInputRef}
            className="bg-transparent hover:bg-surface2 focus:bg-surface2 transition-colors"
          />
          <Tooltip
            title={t('option:notesSearch.generateTitleTooltip', {
              defaultValue: 'Generate title from content'
            })}
          >
            <Button
              size="small"
              onClick={() => {
                void suggestTitle()
              }}
              disabled={editorDisabled || !isOnline || content.trim().length === 0}
              loading={titleSuggestionLoading}
              icon={(<SparklesIcon className="w-4 h-4" />)}
              aria-label={t('option:notesSearch.generateTitleTooltip', {
                defaultValue: 'Generate title from content'
              })}
              data-testid="notes-generate-title-button"
            >
              {t('option:notesSearch.generateTitleAction', {
                defaultValue: 'Generate title'
              })}
            </Button>
          </Tooltip>
          {canSwitchTitleStrategy ? (
            <Select
              size="small"
              className="min-w-[170px]"
              value={effectiveTitleSuggestStrategy}
              options={titleStrategyOptions}
              onChange={(value) => {
                const normalized = normalizeNotesTitleStrategy(value)
                if (!normalized) return
                setTitleSuggestStrategy(normalized)
                void setSetting(NOTES_TITLE_SUGGEST_STRATEGY_SETTING, normalized)
              }}
              disabled={editorDisabled || !isOnline || titleSuggestionLoading}
              aria-label={t('option:notesSearch.titleStrategyLabel', {
                defaultValue: 'Title generation strategy'
              })}
              data-testid="notes-title-strategy-select"
            />
          ) : null}
          <Tooltip
            title={t('option:notesSearch.shortcutHelpTooltip', {
              defaultValue: 'Keyboard shortcuts'
            })}
          >
            <Button
              size="small"
              type="text"
              onClick={() => setShortcutHelpOpen(true)}
              aria-label={t('option:notesSearch.shortcutHelpTooltip', {
                defaultValue: 'Keyboard shortcuts'
              })}
              data-testid="notes-shortcuts-help-button"
            >
              {t('option:notesSearch.shortcutHelpLabel', {
                defaultValue: 'Keyboard shortcuts'
              })}
            </Button>
          </Tooltip>
        </div>
        <div className="mt-3">
          <Select
            mode="tags"
            allowClear
            placeholder={t('option:notesSearch.keywordsEditorPlaceholder', {
              defaultValue: 'Tags'
            })}
            data-testid="notes-keywords-editor"
            className="w-full"
            value={editorKeywords}
            onSearch={(txt) => {
              if (isOnline) void debouncedLoadKeywordSuggestions(txt)
            }}
            onChange={(vals) => {
              setEditorKeywords(vals as string[])
              setIsDirty(true)
              setSaveIndicator('dirty')
              setMonitoringNotice(null)
              markManualEdit()
            }}
            options={keywordOptions.map((keyword) => ({
              label: renderKeywordLabelWithFrequency(keyword, {
                includeCount: true,
                testIdPrefix: 'notes-keyword-editor-option-label'
              }),
              value: keyword
            }))}
            disabled={editorDisabled}
          />
          {saveIndicatorText && (
            <Typography.Text
              type={saveIndicator === 'error' ? 'danger' : 'secondary'}
              className="block text-[11px] mt-1 text-text-muted"
              aria-live="polite"
            >
              {saveIndicatorText}
            </Typography.Text>
          )}
          {offlineStatusText && (
            <Typography.Text
              type={currentOfflineDraft?.syncState === 'conflict' ? 'danger' : 'secondary'}
              className="block text-[11px] mt-1 text-text-muted"
              aria-live="polite"
              data-testid="notes-offline-sync-status"
            >
              {offlineStatusText}
            </Typography.Text>
          )}
          {remoteVersionInfo && (
            <div
              className="mt-2 rounded border border-warn/50 bg-warn/10 px-2 py-1 text-[12px] text-warn"
              role="status"
              data-testid="notes-stale-version-warning"
            >
              <span>
                {t('option:notesSearch.staleVersionWarning', {
                  defaultValue:
                    'This note was updated elsewhere. Reload to see the latest version.'
                })}
              </span>
              <Button
                type="link"
                size="small"
                className="!px-1"
                onClick={() => {
                  if (selectedId == null) return
                  void handleSelectNote(selectedId)
                }}
                data-testid="notes-stale-version-reload"
              >
                {t('option:notesSearch.reloadNoteAction', {
                  defaultValue: 'Reload note'
                })}
              </Button>
            </div>
          )}
          {monitoringNotice && (
            <div
              className={`mt-2 rounded border px-2 py-2 text-[12px] ${monitoringNoticeClasses}`}
              role="alert"
              aria-live="polite"
              data-testid="notes-monitoring-alert"
            >
              <div className="font-medium">{monitoringNotice.title}</div>
              <div className="mt-1">{monitoringNotice.guidance}</div>
            </div>
          )}
          {canUndoAssist && (
            <div className="mt-2">
              <Button
                size="small"
                onClick={undoAssist}
                className="text-xs"
                data-testid="notes-undo-assist"
              >
                {t('option:notesSearch.undoAssistAction', {
                  defaultValue: 'Undo AI change'
                })}
              </Button>
            </div>
          )}
        </div>
        {selectedId != null && (
          <CollapsibleSection
            title={t('option:notesSearch.connectionsSectionTitle', {
              defaultValue: 'Connections'
            })}
            titleAccessory={
              <Tooltip
                title={t('option:notesSearch.connectionsHelpTooltip', {
                  defaultValue: 'Links between this note and others — created by you, by [[ ]] note links, or automatically.'
                })}
              >
                {renderHelpButton(
                  t('option:notesSearch.connectionsHelpLabel', {
                    defaultValue: 'Connections help'
                  })
                )}
              </Tooltip>
            }
            defaultOpen
            storageKey="connections"
            testId="notes-section-connections"
          >
          <div
            className="grid grid-cols-1 gap-3 xl:grid-cols-2"
            data-testid="notes-graph-relation-panels"
          >
            <div className="rounded-lg border border-border bg-surface2 p-3">
              <Typography.Text
                className="text-[11px] uppercase tracking-[0.08em] text-text-muted"
                data-testid="notes-related-heading"
              >
                {t('option:notesSearch.relatedNotesHeading', {
                  defaultValue: 'Related notes'
                })}
              </Typography.Text>
              <Button
                size="small"
                className="mt-2"
                onClick={openGraphModal}
                data-testid="notes-open-graph-view"
              >
                {t('option:notesSearch.graphOpenButton', {
                  defaultValue: 'Open graph view'
                })}
              </Button>
              <div className="mt-2 flex items-center gap-2">
                <Select
                  className="flex-1"
                  size="small"
                  value={manualLinkTargetId ?? undefined}
                  onChange={(value) => {
                    setManualLinkTargetId(value || null)
                  }}
                  disabled={manualLinkSaving || editorDisabled || manualLinkOptions.length === 0}
                  placeholder={t('option:notesSearch.manualLinkTargetPlaceholder', {
                    defaultValue: 'Select a note to link'
                  })}
                  allowClear
                  showSearch
                  optionFilterProp="label"
                  options={manualLinkOptions.map((option) => ({
                    value: option.value,
                    label: option.label
                  }))}
                  data-testid="notes-manual-link-target-select"
                />
                <Button
                  size="small"
                  type="primary"
                  onClick={() => {
                    void createManualLink()
                  }}
                  disabled={!manualLinkTargetId || editorDisabled}
                  loading={manualLinkSaving}
                  data-testid="notes-manual-link-add"
                >
                  {t('option:notesSearch.manualLinkAdd', {
                    defaultValue: 'Add link'
                  })}
                </Button>
              </div>
              <Typography.Text
                type="secondary"
                className="block mt-2 text-[11px] text-text-muted"
              >
                {t('option:notesSearch.manualLinksHeading', {
                  defaultValue: 'Manual links'
                })}
              </Typography.Text>
              {noteRelations.manualLinks.length === 0 ? (
                <Typography.Text
                  type="secondary"
                  className="block mt-1 text-[12px] text-text-muted"
                  data-testid="notes-manual-links-empty"
                >
                  {t('option:notesSearch.manualLinksEmpty', {
                    defaultValue: 'No manual links yet.'
                  })}
                </Typography.Text>
              ) : (
                <div className="mt-1 flex flex-wrap gap-1.5" data-testid="notes-manual-links-list">
                  {noteRelations.manualLinks.map((link) => (
                    <div
                      key={link.edgeId}
                      className="inline-flex items-center gap-1 rounded border border-border bg-surface px-2 py-1"
                    >
                      <button
                        type="button"
                        className="text-xs text-text hover:underline"
                        onClick={() => {
                          void handleSelectNote(link.noteId)
                        }}
                      >
                        {link.title}
                      </button>
                      <button
                        type="button"
                        className="text-xs text-danger hover:underline"
                        onClick={() => {
                          void removeManualLink(link.edgeId)
                        }}
                        disabled={manualLinkDeletingEdgeId === link.edgeId}
                        aria-label={t('option:notesSearch.manualLinkRemoveAria', {
                          defaultValue: `Remove manual link ${link.title}`
                        })}
                        data-testid={`notes-manual-link-remove-${link.edgeId.replace(/[^a-z0-9_-]/gi, '_')}`}
                      >
                        {manualLinkDeletingEdgeId === link.edgeId
                          ? t('option:notesSearch.manualLinkRemoving', {
                              defaultValue: 'Removing...'
                            })
                          : t('option:notesSearch.manualLinkRemove', {
                              defaultValue: 'Remove'
                            })}
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {noteNeighborsLoading ? (
                <Typography.Text
                  type="secondary"
                  className="block mt-2 text-[12px] text-text-muted"
                >
                  {t('option:notesSearch.relatedNotesLoading', {
                    defaultValue: 'Loading related notes...'
                  })}
                </Typography.Text>
              ) : noteNeighborsError ? (
                <div className="mt-2" data-testid="notes-related-error">
                  <Typography.Text type="danger" className="text-[12px]">
                    {t('option:notesSearch.relatedNotesError', {
                      defaultValue: 'Could not load related notes.'
                    })}
                  </Typography.Text>
                  {onRetryNeighbors && (
                    <Button
                      size="small"
                      type="link"
                      className="ml-1 !px-0 text-[12px]"
                      onClick={onRetryNeighbors}
                      data-testid="notes-related-retry"
                    >
                      {t('option:notesSearch.relatedNotesRetry', {
                        defaultValue: 'Retry'
                      })}
                    </Button>
                  )}
                </div>
              ) : noteRelations.related.length === 0 ? (
                <Typography.Text
                  type="secondary"
                  className="block mt-2 text-[12px] text-text-muted"
                  data-testid="notes-related-empty"
                >
                  {t('option:notesSearch.relatedNotesEmpty', {
                    defaultValue: 'No related notes yet.'
                  })}
                </Typography.Text>
              ) : (
                <div className="mt-2 flex flex-wrap gap-1.5" data-testid="notes-related-list">
                  {noteRelations.related.map((note) => (
                    <button
                      key={`related-${note.id}`}
                      type="button"
                      className="rounded border border-border bg-surface px-2 py-1 text-left text-xs text-text hover:bg-surface3"
                      onClick={() => {
                        void handleSelectNote(note.id)
                      }}
                    >
                      {note.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="rounded-lg border border-border bg-surface2 p-3">
              <Typography.Text
                className="text-[11px] uppercase tracking-[0.08em] text-text-muted"
                data-testid="notes-backlinks-heading"
              >
                {t('option:notesSearch.backlinksHeading', {
                  defaultValue: 'Backlinks'
                })}
              </Typography.Text>
              {noteNeighborsLoading ? (
                <Typography.Text
                  type="secondary"
                  className="block mt-2 text-[12px] text-text-muted"
                >
                  {t('option:notesSearch.backlinksLoading', {
                    defaultValue: 'Loading backlinks...'
                  })}
                </Typography.Text>
              ) : noteNeighborsError ? (
                <Typography.Text
                  type="danger"
                  className="block mt-2 text-[12px]"
                  data-testid="notes-backlinks-error"
                >
                  {t('option:notesSearch.backlinksError', {
                    defaultValue: 'Could not load backlinks.'
                  })}
                </Typography.Text>
              ) : noteRelations.backlinks.length === 0 ? (
                <Typography.Text
                  type="secondary"
                  className="block mt-2 text-[12px] text-text-muted"
                  data-testid="notes-backlinks-empty"
                >
                  {t('option:notesSearch.backlinksEmpty', {
                    defaultValue: 'No backlinks yet.'
                  })}
                </Typography.Text>
              ) : (
                <div className="mt-2 flex flex-wrap gap-1.5" data-testid="notes-backlinks-list">
                  {noteRelations.backlinks.map((note) => (
                    <button
                      key={`backlink-${note.id}`}
                      type="button"
                      className="rounded border border-border bg-surface px-2 py-1 text-left text-xs text-text hover:bg-surface3"
                      onClick={() => {
                        void handleSelectNote(note.id)
                      }}
                    >
                      {note.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
          </CollapsibleSection>
        )}
        {editorMode !== 'preview' && (
          <div className="mt-3 flex items-center flex-wrap gap-1 rounded-lg border border-border bg-surface2 p-2">
            <div
              className="mr-2 inline-flex items-center gap-1 rounded-md border border-border bg-surface px-1 py-0.5"
              role="group"
              aria-label={t('option:notesSearch.inputModeGroup', {
                defaultValue: 'Input mode'
              })}
              data-testid="notes-input-mode-toggle"
            >
              <Button
                size="small"
                type={editorInputMode === 'markdown' ? 'primary' : 'text'}
                onClick={() => handleEditorInputModeChange('markdown')}
                disabled={editorDisabled}
                data-testid="notes-input-mode-markdown"
              >
                {t('option:notesSearch.inputModeMarkdown', {
                  defaultValue: 'Markdown'
                })}
              </Button>
              <Button
                size="small"
                type={editorInputMode === 'wysiwyg' ? 'primary' : 'text'}
                onClick={() => handleEditorInputModeChange('wysiwyg')}
                disabled={editorDisabled}
                data-testid="notes-input-mode-wysiwyg"
              >
                {t('option:notesSearch.inputModeWysiwyg', {
                  defaultValue: 'WYSIWYG'
                })}
              </Button>
            </div>
            <Typography.Text
              type="secondary"
              className="text-[11px] mr-1 uppercase tracking-[0.08em]"
            >
              {t('option:notesSearch.formattingLabel', {
                defaultValue: 'Formatting'
              })}
            </Typography.Text>
            <Tooltip title={t('option:notesSearch.toolbarBoldTooltip', { defaultValue: 'Bold' })}>
              <Button
                size="small"
                type="text"
                icon={(<BoldIcon className="w-4 h-4" />)}
                onClick={() => applyMarkdownToolbarAction('bold')}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarBoldTooltip', { defaultValue: 'Bold' })}
                data-testid="notes-toolbar-bold"
              />
            </Tooltip>
            <Tooltip title={t('option:notesSearch.toolbarItalicTooltip', { defaultValue: 'Italic' })}>
              <Button
                size="small"
                type="text"
                icon={(<ItalicIcon className="w-4 h-4" />)}
                onClick={() => applyMarkdownToolbarAction('italic')}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarItalicTooltip', { defaultValue: 'Italic' })}
                data-testid="notes-toolbar-italic"
              />
            </Tooltip>
            <Tooltip title={t('option:notesSearch.toolbarHeadingTooltip', { defaultValue: 'Heading' })}>
              <Button
                size="small"
                type="text"
                icon={(<HeadingIcon className="w-4 h-4" />)}
                onClick={() => applyMarkdownToolbarAction('heading')}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarHeadingTooltip', { defaultValue: 'Heading' })}
                data-testid="notes-toolbar-heading"
              />
            </Tooltip>
            <Tooltip title={t('option:notesSearch.toolbarListTooltip', { defaultValue: 'List' })}>
              <Button
                size="small"
                type="text"
                icon={(<ListIcon className="w-4 h-4" />)}
                onClick={() => applyMarkdownToolbarAction('list')}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarListTooltip', { defaultValue: 'List' })}
                data-testid="notes-toolbar-list"
              />
            </Tooltip>
            <Tooltip title={t('option:notesSearch.toolbarLinkTooltip', { defaultValue: 'Link' })}>
              <Button
                size="small"
                type="text"
                icon={(<LinkIcon className="w-4 h-4" />)}
                onClick={() => applyMarkdownToolbarAction('link')}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarLinkTooltip', { defaultValue: 'Link' })}
                data-testid="notes-toolbar-link"
              />
            </Tooltip>
            <Tooltip title={t('option:notesSearch.toolbarAttachmentTooltip', { defaultValue: 'Attachment' })}>
              <Button
                size="small"
                type="text"
                icon={(<PaperclipIcon className="w-4 h-4" />)}
                onClick={openAttachmentPicker}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarAttachmentTooltip', { defaultValue: 'Attachment' })}
                data-testid="notes-toolbar-attachment"
              />
            </Tooltip>
            <Tooltip title={t('option:notesSearch.toolbarCodeTooltip', { defaultValue: 'Code' })}>
              <Button
                size="small"
                type="text"
                icon={(<CodeIcon className="w-4 h-4" />)}
                onClick={() => applyMarkdownToolbarAction('code')}
                disabled={editorDisabled}
                aria-label={t('option:notesSearch.toolbarCodeTooltip', { defaultValue: 'Code' })}
                data-testid="notes-toolbar-code"
              />
            </Tooltip>
            <span className="mx-1 h-4 w-px bg-border" aria-hidden="true" />
            <Typography.Text
              type="secondary"
              className="text-[11px] mr-1 uppercase tracking-[0.08em]"
            >
              {t('option:notesSearch.assistLabel', {
                defaultValue: 'Assist'
              })}
            </Typography.Text>
            <Tooltip
              title={t('option:notesSearch.assistSummarizeTooltip', {
                defaultValue: 'Generate a concise summary draft'
              })}
            >
              <Button
                size="small"
                type="text"
                icon={(<SparklesIcon className="w-4 h-4" />)}
                onClick={() => {
                  void runAssistAction('summarize')
                }}
                disabled={editorDisabled || content.trim().length === 0}
                loading={assistLoadingAction === 'summarize'}
                data-testid="notes-assist-summarize"
              >
                {t('option:notesSearch.assistSummarizeAction', {
                  defaultValue: 'Summarize'
                })}
              </Button>
            </Tooltip>
            <Tooltip
              title={t('option:notesSearch.assistExpandOutlineTooltip', {
                defaultValue: 'Generate an expanded outline draft'
              })}
            >
              <Button
                size="small"
                type="text"
                icon={(<SparklesIcon className="w-4 h-4" />)}
                onClick={() => {
                  void runAssistAction('expand_outline')
                }}
                disabled={editorDisabled || content.trim().length === 0}
                loading={assistLoadingAction === 'expand_outline'}
                data-testid="notes-assist-expand-outline"
              >
                {t('option:notesSearch.assistExpandOutlineAction', {
                  defaultValue: 'Expand outline'
                })}
              </Button>
            </Tooltip>
            <Tooltip
              title={t('option:notesSearch.assistSuggestKeywordsTooltip', {
                defaultValue: 'Suggest tags from note content'
              })}
            >
              <Button
                size="small"
                type="text"
                icon={(<SparklesIcon className="w-4 h-4" />)}
                onClick={() => {
                  void runAssistAction('suggest_keywords')
                }}
                disabled={editorDisabled || content.trim().length === 0}
                loading={assistLoadingAction === 'suggest_keywords'}
                data-testid="notes-assist-suggest-keywords"
              >
                {t('option:notesSearch.assistSuggestKeywordsAction', {
                  defaultValue: 'Suggest tags'
                })}
              </Button>
            </Tooltip>
            <input
              ref={attachmentInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={handleAttachmentInputChange}
              data-testid="notes-attachment-input"
            />
          </div>
        )}
        {shouldShowToc && (
          <div
            className="mt-3 rounded-lg border border-border bg-surface2 p-2"
            data-testid="notes-toc-panel"
          >
            <Typography.Text
              type="secondary"
              className="block text-[11px] uppercase tracking-[0.08em] text-text-muted"
            >
              {t('option:notesSearch.tocHeading', {
                defaultValue: 'Table of contents'
              })}
            </Typography.Text>
            <div className="mt-2 space-y-1">
              {tocEntries.map((entry) => (
                <button
                  key={`toc-${entry.id}-${entry.offset}`}
                  type="button"
                  className="block w-full rounded px-2 py-1 text-left text-xs text-text hover:bg-surface3"
                  style={{ paddingLeft: `${8 + (entry.level - 1) * 12}px` }}
                  onClick={() => handleTocJump(entry)}
                  data-testid={`notes-toc-item-${entry.id}`}
                >
                  {entry.text}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="mt-2 flex-1 min-h-0">
          {editorMode === 'preview' ? (
            content.trim().length > 0 ? (
              <div className="h-full flex flex-col">
                <Typography.Text
                  type="secondary"
                  className="block text-[11px] mb-2 text-text-muted"
                >
                  {t('option:notesSearch.previewTitle', {
                    defaultValue: 'Preview (Markdown + LaTeX)'
                  })}
                </Typography.Text>
                {usesLargePreviewGuardrails && !largePreviewReady ? (
                  <div
                    className="w-full flex-1 rounded-lg border border-border bg-surface2 p-4"
                    role="status"
                    aria-live="polite"
                    data-testid="notes-large-preview-loading"
                  >
                    <div className="inline-flex items-center gap-2 text-sm text-text-muted">
                      <Spin size="small" />
                      <span>
                        {t('option:notesSearch.largePreviewLoadingLabel', {
                          defaultValue: 'Rendering preview for large note'
                        })}
                        {`: ${previewContent.length} chars`}
                      </span>
                    </div>
                  </div>
                ) : (
                  renderMarkdownPreviewSurface('notes-preview-surface')
                )}
              </div>
            ) : (
              <Typography.Text
                type="secondary"
                className="block text-[11px] mt-1 text-text-muted"
              >
                {t('option:notesSearch.emptyPreview', {
                  defaultValue:
                    'Start typing to see a live preview of your note.'
                })}
              </Typography.Text>
            )
          ) : editorMode === 'split' ? (
            <div className="grid h-full min-h-0 grid-cols-1 gap-3 lg:grid-cols-2">
              <div className="flex min-h-0 flex-col">
                <Typography.Text
                  type="secondary"
                  className="block text-[11px] mb-2 text-text-muted"
                >
                  {t('option:notesSearch.editModeLabel', {
                    defaultValue: 'Edit'
                  })}
                </Typography.Text>
                {editorInputMode === 'wysiwyg' ? (
                  <div
                    ref={richEditorRef}
                    role="textbox"
                    aria-multiline="true"
                    contentEditable={!editorDisabled}
                    suppressContentEditableWarning
                    className="w-full min-h-[220px] text-sm p-4 rounded-lg border border-border bg-surface2 text-text overflow-auto leading-relaxed focus:outline-none focus:ring-2 focus:ring-focus"
                    onInput={handleWysiwygInput}
                    onPaste={handleWysiwygPaste}
                    onBlur={() => setEditorCursorIndex(null)}
                    aria-label={t('option:notesSearch.editorAriaLabel', {
                      defaultValue: 'Note content'
                    })}
                    data-testid="notes-wysiwyg-editor"
                    dangerouslySetInnerHTML={{ __html: wysiwygHtml }}
                  />
                ) : (
                  <>
                    <textarea
                      ref={contentTextareaRef}
                      className="w-full min-h-[220px] text-sm p-4 rounded-lg border border-border bg-surface2 text-text resize-none leading-relaxed focus:outline-none focus:ring-2 focus:ring-focus"
                      value={content}
                      onChange={handleEditorChange}
                      onKeyDown={handleEditorKeyDown}
                      onSelect={handleEditorSelectionUpdate}
                      onClick={handleEditorSelectionUpdate}
                      onKeyUp={handleEditorSelectionUpdate}
                      onFocus={handleEditorSelectionUpdate}
                      onBlur={() => setEditorCursorIndex(null)}
                      placeholder={t('option:notesSearch.editorPlaceholder', {
                        defaultValue: 'Write your note here... (Markdown supported)'
                      })}
                      readOnly={editorDisabled}
                      aria-label={t('option:notesSearch.editorAriaLabel', {
                        defaultValue: 'Note content'
                      })}
                    />
                    {activeWikilinkQuery && wikilinkSuggestions.length > 0 && (
                      <div
                        className="mt-2 rounded-lg border border-border bg-surface p-1"
                        role="listbox"
                        aria-label={t('option:notesSearch.wikilinkSuggestionsLabel', {
                          defaultValue: 'Wikilink suggestions'
                        })}
                        data-testid="notes-wikilink-suggestions"
                      >
                        {wikilinkSuggestions.map((candidate, index) => {
                          const duplicateCount =
                            wikilinkSuggestionDisplayCounts.get(candidate.title.toLowerCase()) || 0
                          const label =
                            duplicateCount > 1 ? `${candidate.title} (${candidate.id})` : candidate.title
                          return (
                            <button
                              key={`${candidate.id}-${candidate.title}`}
                              type="button"
                              className={`block w-full rounded px-2 py-1 text-left text-xs ${
                                index === wikilinkSelectionIndex
                                  ? 'bg-surface2 text-text'
                                  : 'text-text-muted hover:bg-surface2 hover:text-text'
                              }`}
                              aria-selected={index === wikilinkSelectionIndex}
                              onMouseDown={(event) => {
                                event.preventDefault()
                                applyWikilinkSuggestion(candidate)
                              }}
                              data-testid={`notes-wikilink-suggestion-${candidate.id.replace(/[^a-z0-9_-]/gi, '_')}`}
                            >
                              {label}
                            </button>
                          )
                        })}
                      </div>
                    )}
                  </>
                )}
                <Typography.Text
                  type="secondary"
                  className="block text-[11px] mt-1 text-text-muted"
                >
                  {editorInputMode === 'wysiwyg'
                    ? t('option:notesSearch.wysiwygSupportHint', {
                        defaultValue: 'WYSIWYG mode keeps markdown structure while you edit.'
                      })
                    : t('option:notesSearch.editorSupportHint', {
                        defaultValue: 'Markdown + LaTeX supported'
                      })}
                </Typography.Text>
              </div>
              <div className="flex min-h-0 flex-col">
                {content.trim().length > 0 ? (
                  <>
                    <Typography.Text
                      type="secondary"
                      className="block text-[11px] mb-2 text-text-muted"
                    >
                      {t('option:notesSearch.previewTitle', {
                        defaultValue: 'Preview (Markdown + LaTeX)'
                      })}
                    </Typography.Text>
                    {usesLargePreviewGuardrails && !largePreviewReady ? (
                      <div
                        className="w-full flex-1 rounded-lg border border-border bg-surface2 p-4"
                        role="status"
                        aria-live="polite"
                        data-testid="notes-large-preview-loading"
                      >
                        <div className="inline-flex items-center gap-2 text-sm text-text-muted">
                          <Spin size="small" />
                          <span>
                            {t('option:notesSearch.largePreviewLoadingLabel', {
                              defaultValue: 'Rendering preview for large note'
                            })}
                            {`: ${previewContent.length} chars`}
                          </span>
                        </div>
                      </div>
                    ) : (
                      renderMarkdownPreviewSurface('notes-split-preview-surface')
                    )}
                  </>
                ) : (
                  <Typography.Text
                    type="secondary"
                    className="block text-[11px] mt-1 text-text-muted"
                  >
                    {t('option:notesSearch.emptyPreview', {
                      defaultValue:
                        'Start typing to see a live preview of your note.'
                    })}
                  </Typography.Text>
                )}
              </div>
            </div>
          ) : (
            <div className="flex h-full min-h-0 flex-col">
              {editorInputMode === 'wysiwyg' ? (
                <div
                  ref={richEditorRef}
                  role="textbox"
                  aria-multiline="true"
                  contentEditable={!editorDisabled}
                  suppressContentEditableWarning
                  className="w-full min-h-[280px] text-sm p-4 rounded-lg border border-border bg-surface2 text-text overflow-auto leading-relaxed focus:outline-none focus:ring-2 focus:ring-focus"
                  onInput={handleWysiwygInput}
                  onPaste={handleWysiwygPaste}
                  onBlur={() => setEditorCursorIndex(null)}
                  aria-label={t('option:notesSearch.editorAriaLabel', {
                    defaultValue: 'Note content'
                  })}
                  data-testid="notes-wysiwyg-editor"
                  dangerouslySetInnerHTML={{ __html: wysiwygHtml }}
                />
              ) : (
                <>
                  <textarea
                    ref={contentTextareaRef}
                    className="w-full min-h-[280px] text-sm p-4 rounded-lg border border-border bg-surface2 text-text resize-none leading-relaxed focus:outline-none focus:ring-2 focus:ring-focus"
                    value={content}
                    onChange={handleEditorChange}
                    onKeyDown={handleEditorKeyDown}
                    onSelect={handleEditorSelectionUpdate}
                    onClick={handleEditorSelectionUpdate}
                    onKeyUp={handleEditorSelectionUpdate}
                    onFocus={handleEditorSelectionUpdate}
                    onBlur={() => setEditorCursorIndex(null)}
                    placeholder={t('option:notesSearch.editorPlaceholder', {
                      defaultValue: 'Write your note here... (Markdown supported)'
                    })}
                    readOnly={editorDisabled}
                    aria-label={t('option:notesSearch.editorAriaLabel', {
                      defaultValue: 'Note content'
                    })}
                  />
                  {activeWikilinkQuery && wikilinkSuggestions.length > 0 && (
                    <div
                      className="mt-2 rounded-lg border border-border bg-surface p-1"
                      role="listbox"
                      aria-label={t('option:notesSearch.wikilinkSuggestionsLabel', {
                        defaultValue: 'Wikilink suggestions'
                      })}
                      data-testid="notes-wikilink-suggestions"
                    >
                      {wikilinkSuggestions.map((candidate, index) => {
                        const duplicateCount =
                          wikilinkSuggestionDisplayCounts.get(candidate.title.toLowerCase()) || 0
                        const label =
                          duplicateCount > 1 ? `${candidate.title} (${candidate.id})` : candidate.title
                        return (
                          <button
                            key={`${candidate.id}-${candidate.title}`}
                            type="button"
                            className={`block w-full rounded px-2 py-1 text-left text-xs ${
                              index === wikilinkSelectionIndex
                                ? 'bg-surface2 text-text'
                                : 'text-text-muted hover:bg-surface2 hover:text-text'
                            }`}
                            aria-selected={index === wikilinkSelectionIndex}
                            onMouseDown={(event) => {
                              event.preventDefault()
                              applyWikilinkSuggestion(candidate)
                            }}
                            data-testid={`notes-wikilink-suggestion-${candidate.id.replace(/[^a-z0-9_-]/gi, '_')}`}
                          >
                            {label}
                          </button>
                        )
                      })}
                    </div>
                  )}
                </>
              )}
              <Typography.Text
                type="secondary"
                className="block text-[11px] mt-1 text-text-muted"
              >
                {editorInputMode === 'wysiwyg'
                  ? t('option:notesSearch.wysiwygSupportHint', {
                      defaultValue: 'WYSIWYG mode keeps markdown structure while you edit.'
                    })
                  : t('option:notesSearch.editorSupportHint', {
                      defaultValue: 'Markdown + LaTeX supported'
                    })}
              </Typography.Text>
            </div>
          )}
        </div>
          </>
        )}
        <div className="mt-2 border-t border-border pt-2">
          <Typography.Text
            type="secondary"
            className="text-[11px] text-text-muted"
            data-testid="notes-editor-metrics"
          >
            {metricSummaryText}
          </Typography.Text>
          <Typography.Text
            type="secondary"
            className="block text-[11px] text-text-muted mt-1"
            data-testid="notes-editor-revision-meta"
          >
            {revisionSummaryText}
          </Typography.Text>
          <Typography.Text
            type="secondary"
            className="block text-[11px] text-text-muted mt-1"
            data-testid="notes-editor-provenance"
          >
            {provenanceSummaryText}
          </Typography.Text>
          {queuedOfflineDraftCount > 0 && (
            <Typography.Text
              type="secondary"
              className="block text-[11px] text-text-muted mt-1"
              data-testid="notes-editor-offline-queue-meta"
            >
              {t('option:notesSearch.offlineQueueFooterMeta', {
                defaultValue: 'Queued offline drafts: {{count}}',
                count: queuedOfflineDraftCount
              })}
            </Typography.Text>
          )}
        </div>
      </div>
    </section>
  )
}

export default React.memo(NotesEditorPane)
