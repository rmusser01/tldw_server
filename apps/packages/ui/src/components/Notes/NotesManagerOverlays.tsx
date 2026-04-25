import React from "react"
import { Button, Checkbox, Input, Modal, Select, Typography } from "antd"

import KeywordPickerModal from "@/components/Notes/KeywordPickerModal"
import NotesGraphModal from "@/components/Notes/NotesGraphModal"

import { useTutorialStore } from "@/store/tutorials"

import type {
  ImportDuplicateStrategy,
  KeywordManagementItem,
  KeywordMergeDraft,
  KeywordPickerSortMode,
  KeywordRenameDraft,
  PendingImportFile,
} from "./notes-manager-types"
import {
  NOTES_TUTORIAL_SHOWN_STORAGE_KEY,
  notesUiStorage,
  toKeywordTestIdSegment
} from "./notes-manager-utils"

type TranslationFn = (
  key: string,
  defaultValueOrOptions?:
    | string
    | {
        defaultValue?: string
        [key: string]: unknown
      },
) => string

type KeywordState = {
  keywordSuggestionOptions: string[]
  closeKeywordSuggestionModal: () => void
  applySelectedSuggestedKeywords: () => void
  setKeywordSuggestionSelection: React.Dispatch<React.SetStateAction<string[]>>
  keywordSuggestionSelection: string[]
  renderKeywordLabelWithFrequency: (
    keyword: string,
    options?: {
      includeCount?: boolean
      testIdPrefix?: string
    },
  ) => React.ReactNode
  keywordPickerOpen: boolean
  availableKeywords: string[]
  sortedKeywordPickerOptions: string[]
  recentKeywordPickerOptions: string[]
  keywordNoteCountByKey: Record<string, number>
  keywordPickerSortMode: KeywordPickerSortMode
  keywordPickerQuery: string
  keywordPickerSelection: string[]
  openKeywordManagerFromPicker: () => void
  keywordManagerOpen: boolean
  closeKeywordManager: () => void
  keywordManagerQuery: string
  setKeywordManagerQuery: React.Dispatch<React.SetStateAction<string>>
  keywordManagerLoading: boolean
  keywordManagerVisibleItems: KeywordManagementItem[]
  keywordManagerActionLoading: boolean
  setKeywordRenameDraft: React.Dispatch<React.SetStateAction<KeywordRenameDraft | null>>
  setKeywordMergeDraft: React.Dispatch<React.SetStateAction<KeywordMergeDraft | null>>
  handleKeywordManagerDelete: (item: KeywordManagementItem) => Promise<void> | void
  keywordRenameDraft: KeywordRenameDraft | null
  submitKeywordRename: () => Promise<void> | void
  keywordMergeDraft: KeywordMergeDraft | null
  submitKeywordMerge: () => Promise<void> | void
  keywordMergeTargetOptions: KeywordManagementItem[]
}

type ImportState = {
  importModalOpen: boolean
  closeImportModal: () => void
  confirmImport: () => Promise<void> | void
  importSubmitting: boolean
  importDuplicateStrategy: ImportDuplicateStrategy
  setImportDuplicateStrategy: React.Dispatch<
    React.SetStateAction<ImportDuplicateStrategy>
  >
  pendingImportFiles: PendingImportFile[]
}

type GraphState = {
  graphModalOpen: boolean
  selectedId: string | number | null
  graphMutationTick: number
}

type NotesManagerOverlaysProps = {
  kw: KeywordState
  imp: ImportState
  graph: GraphState
  isOnline: boolean
  shortcutHelpOpen: boolean
  setShortcutHelpOpen: React.Dispatch<React.SetStateAction<boolean>>
  closeGraphModal: () => void
  handleSelectNote: (noteId: string) => Promise<void> | void
  handleKeywordPickerCancel: () => void
  handleKeywordPickerApply: () => void
  handleKeywordPickerSortModeChange: (value: KeywordPickerSortMode) => void
  handleToggleRecentKeyword: (keyword: string) => void
  handleKeywordPickerQueryChange: (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => void
  handleKeywordPickerSelectionChange: (values: string[]) => void
  handleKeywordPickerSelectAll: () => void
  handleKeywordPickerClear: () => void
  t: TranslationFn
}

const NotesManagerOverlays: React.FC<NotesManagerOverlaysProps> = ({
  kw,
  imp,
  graph,
  isOnline,
  shortcutHelpOpen,
  setShortcutHelpOpen,
  closeGraphModal,
  handleSelectNote,
  handleKeywordPickerCancel,
  handleKeywordPickerApply,
  handleKeywordPickerSortModeChange,
  handleToggleRecentKeyword,
  handleKeywordPickerQueryChange,
  handleKeywordPickerSelectionChange,
  handleKeywordPickerSelectAll,
  handleKeywordPickerClear,
  t,
}) => {
  return (
    <>
      <Modal
        open={kw.keywordSuggestionOptions.length > 0}
        onCancel={kw.closeKeywordSuggestionModal}
        onOk={kw.applySelectedSuggestedKeywords}
        okText={t("option:notesSearch.assistKeywordsApplySelectedAction", {
          defaultValue: "Apply selected",
        })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        destroyOnHidden
        title={t("option:notesSearch.assistKeywordsReviewTitle", {
          defaultValue: "Review suggested tags",
        })}
      >
        <div
          className="space-y-3"
          data-testid="notes-assist-keyword-suggestions-modal"
        >
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.assistKeywordsReviewHelp", {
              defaultValue: "Select which suggested tags to add to this note.",
            })}
          </Typography.Text>
          <div className="flex items-center gap-2">
            <Button
              size="small"
              onClick={() =>
                kw.setKeywordSuggestionSelection([...kw.keywordSuggestionOptions])
              }
              disabled={kw.keywordSuggestionOptions.length === 0}
              data-testid="notes-assist-keyword-select-all"
            >
              {t("option:notesSearch.keywordPickerSelectAll", {
                defaultValue: "Select all",
              })}
            </Button>
            <Button
              size="small"
              onClick={() => kw.setKeywordSuggestionSelection([])}
              disabled={kw.keywordSuggestionSelection.length === 0}
              data-testid="notes-assist-keyword-clear-all"
            >
              {t("option:notesSearch.keywordPickerClear", {
                defaultValue: "Clear",
              })}
            </Button>
          </div>
          <Checkbox.Group
            value={kw.keywordSuggestionSelection}
            onChange={(values) =>
              kw.setKeywordSuggestionSelection((values as string[]).map(String))
            }
            className="w-full"
          >
            <div className="grid grid-cols-1 gap-2 rounded-lg border border-border bg-surface2 p-3 sm:grid-cols-2">
              {kw.keywordSuggestionOptions.map((keyword) => (
                <Checkbox
                  key={`assist-keyword-${keyword}`}
                  value={keyword}
                  data-testid={`notes-assist-keyword-option-${toKeywordTestIdSegment(keyword)}`}
                >
                  {kw.renderKeywordLabelWithFrequency(keyword, {
                    includeCount: true,
                    testIdPrefix: "notes-assist-keyword-label",
                  })}
                </Checkbox>
              ))}
            </div>
          </Checkbox.Group>
        </div>
      </Modal>

      <KeywordPickerModal
        open={kw.keywordPickerOpen}
        availableKeywords={kw.availableKeywords}
        filteredKeywordPickerOptions={kw.sortedKeywordPickerOptions}
        recentKeywordPickerOptions={kw.recentKeywordPickerOptions}
        keywordNoteCountByKey={kw.keywordNoteCountByKey}
        sortMode={kw.keywordPickerSortMode}
        keywordPickerQuery={kw.keywordPickerQuery}
        keywordPickerSelection={kw.keywordPickerSelection}
        onCancel={handleKeywordPickerCancel}
        onApply={handleKeywordPickerApply}
        onSortModeChange={handleKeywordPickerSortModeChange}
        onToggleRecentKeyword={handleToggleRecentKeyword}
        onQueryChange={handleKeywordPickerQueryChange}
        onSelectionChange={handleKeywordPickerSelectionChange}
        onSelectAll={handleKeywordPickerSelectAll}
        onClear={handleKeywordPickerClear}
        onOpenManager={kw.openKeywordManagerFromPicker}
        managerDisabled={!isOnline}
        t={t}
      />

      <Modal
        open={kw.keywordManagerOpen}
        onCancel={kw.closeKeywordManager}
        title={t("option:notesSearch.keywordManagerTitle", {
          defaultValue: "Manage tags",
        })}
        destroyOnHidden
        footer={[
          <Button key="close" onClick={kw.closeKeywordManager}>
            {t("common:close", { defaultValue: "Close" })}
          </Button>,
        ]}
      >
        <div className="space-y-3" data-testid="notes-keyword-manager-modal">
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.keywordManagerHelp", {
              defaultValue: "Rename, merge, or delete tags from one place.",
            })}
          </Typography.Text>
          <Input
            allowClear
            value={kw.keywordManagerQuery}
            onChange={(event) => kw.setKeywordManagerQuery(event.target.value)}
            placeholder={t("option:notesSearch.keywordManagerSearchPlaceholder", {
              defaultValue: "Filter tags",
            })}
            data-testid="notes-keyword-manager-search"
          />
          <div className="max-h-80 overflow-auto rounded-lg border border-border bg-surface2 p-2">
            {kw.keywordManagerLoading ? (
              <Typography.Text type="secondary" className="text-xs text-text-muted">
                {t("option:notesSearch.keywordManagerLoading", {
                  defaultValue: "Loading tags...",
                })}
              </Typography.Text>
            ) : kw.keywordManagerVisibleItems.length === 0 ? (
              <Typography.Text type="secondary" className="text-xs text-text-muted">
                {t("option:notesSearch.keywordManagerEmpty", {
                  defaultValue: "No tags found.",
                })}
              </Typography.Text>
            ) : (
              <div className="space-y-2">
                {kw.keywordManagerVisibleItems.map((item) => (
                  <div
                    key={`manager-${item.id}`}
                    className="rounded border border-border bg-surface px-2 py-2"
                    data-testid={`notes-keyword-manager-item-${item.id}`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-sm text-text">{item.keyword}</div>
                        <div className="text-[11px] text-text-muted">
                          {t("option:notesSearch.keywordManagerUsage", {
                            defaultValue: "{{count}} linked notes",
                            count: item.noteCount,
                          })}
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-1">
                        <Button
                          size="small"
                          onClick={() =>
                            kw.setKeywordRenameDraft({
                              id: item.id,
                              currentKeyword: item.keyword,
                              expectedVersion: item.version,
                              nextKeyword: item.keyword,
                            })
                          }
                          disabled={kw.keywordManagerActionLoading}
                          data-testid={`notes-keyword-manager-rename-${item.id}`}
                        >
                          {t("option:notesSearch.keywordManagerRenameAction", {
                            defaultValue: "Rename",
                          })}
                        </Button>
                        <Button
                          size="small"
                          onClick={() =>
                            kw.setKeywordMergeDraft({
                              source: item,
                              targetKeywordId: null,
                            })
                          }
                          disabled={kw.keywordManagerActionLoading}
                          data-testid={`notes-keyword-manager-merge-${item.id}`}
                        >
                          {t("option:notesSearch.keywordManagerMergeAction", {
                            defaultValue: "Merge",
                          })}
                        </Button>
                        <Button
                          size="small"
                          danger
                          onClick={() => {
                            void kw.handleKeywordManagerDelete(item)
                          }}
                          disabled={kw.keywordManagerActionLoading}
                          data-testid={`notes-keyword-manager-delete-${item.id}`}
                        >
                          {t("option:notesSearch.keywordManagerDeleteAction", {
                            defaultValue: "Delete",
                          })}
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </Modal>

      <Modal
        open={kw.keywordRenameDraft != null}
        onCancel={() => kw.setKeywordRenameDraft(null)}
        onOk={() => {
          void kw.submitKeywordRename()
        }}
        okText={t("option:notesSearch.keywordManagerRenameAction", {
          defaultValue: "Rename",
        })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={kw.keywordManagerActionLoading}
        destroyOnHidden
        title={t("option:notesSearch.keywordManagerRenameTitle", {
          defaultValue: "Rename tag",
        })}
      >
        <div className="space-y-2">
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.keywordManagerRenameHelp", {
              defaultValue: "Choose a new name for this tag.",
            })}
          </Typography.Text>
          <Input
            autoFocus
            value={kw.keywordRenameDraft?.nextKeyword ?? ""}
            onChange={(event) =>
              kw.setKeywordRenameDraft((current) =>
                current
                  ? {
                      ...current,
                      nextKeyword: event.target.value,
                    }
                  : current,
              )
            }
            data-testid="notes-keyword-manager-rename-input"
          />
        </div>
      </Modal>

      <Modal
        open={kw.keywordMergeDraft != null}
        onCancel={() => kw.setKeywordMergeDraft(null)}
        onOk={() => {
          void kw.submitKeywordMerge()
        }}
        okText={t("option:notesSearch.keywordManagerMergeAction", {
          defaultValue: "Merge",
        })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={kw.keywordManagerActionLoading}
        okButtonProps={{ danger: true }}
        destroyOnHidden
        title={t("option:notesSearch.keywordManagerMergeTitle", {
          defaultValue: "Merge tag",
        })}
      >
        <div className="space-y-2">
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.keywordManagerMergeHelp", {
              defaultValue:
                "Move all links from the source tag to the selected target tag.",
            })}
          </Typography.Text>
          <Typography.Text
            type="danger"
            className="block text-xs"
            data-testid="notes-keyword-merge-warning"
          >
            {t("option:notesSearch.keywordManagerMergeWarning", {
              defaultValue:
                "This will permanently combine these tags. The source tag will be deleted. This cannot be undone.",
            })}
          </Typography.Text>
          <div className="text-xs text-text-muted">
            {t("option:notesSearch.keywordManagerMergeSourceLabel", {
              defaultValue: "Source",
            })}
            :{" "}
            <span className="font-medium text-text">
              {kw.keywordMergeDraft?.source.keyword ?? ""}
            </span>
          </div>
          <Select
            className="w-full"
            showSearch
            allowClear
            optionFilterProp="label"
            placeholder={t("option:notesSearch.keywordManagerMergeTargetPlaceholder", {
              defaultValue: "Select target tag",
            })}
            value={kw.keywordMergeDraft?.targetKeywordId ?? undefined}
            onChange={(value: number | undefined) => {
              kw.setKeywordMergeDraft((current) =>
                current
                  ? {
                      ...current,
                      targetKeywordId:
                        value != null && Number.isFinite(value) && value > 0 ? value : null,
                    }
                  : current,
              )
            }}
            options={kw.keywordMergeTargetOptions.map((item) => ({
              value: item.id,
              label: `${item.keyword} (${item.noteCount})`,
            }))}
            data-testid="notes-keyword-manager-merge-target"
          />
        </div>
      </Modal>

      <Modal
        open={imp.importModalOpen}
        onCancel={imp.closeImportModal}
        onOk={() => {
          void imp.confirmImport()
        }}
        okText={t("option:notesSearch.importConfirmAction", {
          defaultValue: "Import notes",
        })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={imp.importSubmitting}
        destroyOnHidden
        title={t("option:notesSearch.importModalTitle", {
          defaultValue: "Import notes",
        })}
      >
        <div className="space-y-3" data-testid="notes-import-modal">
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.importModalHelp", {
              defaultValue:
                "Upload JSON exports or markdown files. Choose how to handle imported IDs that already exist.",
            })}
          </Typography.Text>
          <div className="space-y-1">
            <label
              htmlFor="notes-import-strategy"
              className="text-xs font-medium text-text"
            >
              {t("option:notesSearch.importDuplicateStrategyLabel", {
                defaultValue: "Duplicate handling",
              })}
            </label>
            <select
              id="notes-import-strategy"
              className="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-text"
              value={imp.importDuplicateStrategy}
              onChange={(event) =>
                imp.setImportDuplicateStrategy(
                  event.target.value as ImportDuplicateStrategy,
                )
              }
              data-testid="notes-import-duplicate-strategy"
            >
              <option value="create_copy">
                {t("option:notesSearch.importDuplicateCreateCopy", {
                  defaultValue: "Create copy",
                })}
              </option>
              <option value="skip">
                {t("option:notesSearch.importDuplicateSkip", {
                  defaultValue: "Skip duplicate IDs",
                })}
              </option>
              <option value="overwrite">
                {t("option:notesSearch.importDuplicateOverwrite", {
                  defaultValue: "Overwrite duplicate IDs",
                })}
              </option>
            </select>
            {imp.importDuplicateStrategy === 'overwrite' && (
              <Typography.Text
                type="danger"
                className="block mt-1 text-xs"
                data-testid="notes-import-overwrite-warning"
              >
                {t("option:notesSearch.importOverwriteWarning", {
                  defaultValue: "Warning: Existing notes with matching IDs will be permanently replaced. This cannot be undone.",
                })}
              </Typography.Text>
            )}
          </div>
          <div
            className="rounded border border-border bg-surface2 px-2 py-2 text-xs text-text-muted"
            data-testid="notes-import-preview-summary"
          >
            {`Files: ${imp.pendingImportFiles.length} · Estimated notes: ${imp.pendingImportFiles.reduce((sum, item) => sum + item.detectedNotes, 0)}`}
          </div>
          <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
            {imp.pendingImportFiles.map((file) => (
              <div
                key={`import-file-${file.fileName}`}
                className="rounded border border-border bg-surface px-2 py-2"
                data-testid={`notes-import-file-${file.fileName.toLowerCase().replace(/[^a-z0-9_-]/g, "_")}`}
              >
                <div className="truncate text-sm text-text">{file.fileName}</div>
                <div className="text-[11px] text-text-muted">
                  {`${file.format.toUpperCase()} · ${file.detectedNotes} note${file.detectedNotes === 1 ? "" : "s"} detected`}
                </div>
                {file.parseError && (
                  <div className="mt-1 text-[11px] text-warn">
                    {file.parseError}
                    <span className="block mt-0.5 text-text-muted">
                      {t("option:notesSearch.importParseErrorHint", {
                        defaultValue: "Check that JSON files match the tldw export format, or use plain Markdown (.md) files.",
                      })}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </Modal>

      <NotesGraphModal
        open={graph.graphModalOpen}
        noteId={graph.selectedId}
        refreshToken={graph.graphMutationTick}
        onClose={closeGraphModal}
        onOpenNote={(noteId) => {
          void handleSelectNote(noteId)
        }}
      />

      <Modal
        open={shortcutHelpOpen}
        onCancel={() => setShortcutHelpOpen(false)}
        footer={null}
        title={t("option:notesSearch.shortcutHelpTitle", {
          defaultValue: "Keyboard shortcuts",
        })}
        destroyOnHidden
      >
        <div className="space-y-4 text-sm text-text" data-testid="notes-shortcuts-modal">
          <div>
            <div className="text-[11px] uppercase tracking-[0.08em] text-text-muted mb-1">
              {t("option:notesSearch.shortcutGroupGeneral", { defaultValue: "General" })}
            </div>
            <div className="space-y-1">
              <div><strong>Ctrl/Cmd + S</strong>: {t("option:notesSearch.shortcutSaveDescription", { defaultValue: "Save the current note." })}</div>
              <div><strong>Alt + N</strong>: {t("option:notesSearch.shortcutNewNoteDescription", { defaultValue: "Create a new note." })}</div>
              <div><strong>Ctrl/Cmd + K</strong>: {t("option:notesSearch.shortcutFocusSearchDescription", { defaultValue: "Focus the search input." })}</div>
              <div><strong>Ctrl/Cmd + Shift + E/S/P</strong>: {t("option:notesSearch.shortcutEditorModesDescription", { defaultValue: "Switch editor mode (Edit / Split / Preview)." })}</div>
              <div><strong>?</strong>: {t("option:notesSearch.shortcutOpenHelpDescription", { defaultValue: "Open keyboard shortcut help." })}</div>
              <div><strong>Esc</strong>: {t("option:notesSearch.shortcutCloseDialogDescription", { defaultValue: "Close the current dialog." })}</div>
            </div>
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.08em] text-text-muted mb-1">
              {t("option:notesSearch.shortcutGroupEditing", { defaultValue: "Editing" })}
            </div>
            <div className="space-y-1">
              <div><strong>[[</strong>: {t("option:notesSearch.shortcutWikilinkDescription", { defaultValue: "Start a note link — type a note title to search, then select." })}</div>
              <div><strong>Ctrl/Cmd + B</strong>: {t("option:notesSearch.shortcutBoldDescription", { defaultValue: "Bold text (Markdown mode)." })}</div>
              <div><strong>Ctrl/Cmd + I</strong>: {t("option:notesSearch.shortcutItalicDescription", { defaultValue: "Italic text (Markdown mode)." })}</div>
            </div>
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.08em] text-text-muted mb-1">
              {t("option:notesSearch.shortcutGroupSearch", { defaultValue: "Search" })}
            </div>
            <div className="space-y-1">
              <div><strong>Ctrl/Cmd + F</strong>: {t("option:notesSearch.shortcutBrowserFindDescription", { defaultValue: "Find text in the current note (browser find)." })}</div>
              <div><strong>{'"exact phrase"'}</strong>: {t("option:notesSearch.shortcutExactMatchDescription", { defaultValue: "Search for an exact phrase in the notes search box." })}</div>
            </div>
          </div>
          <div className="border-t border-border pt-3">
            <button
              type="button"
              className="text-xs text-primary hover:underline"
              data-testid="notes-restart-tutorial"
              onClick={() => {
                void notesUiStorage.remove(NOTES_TUTORIAL_SHOWN_STORAGE_KEY).catch(() => undefined)
                setShortcutHelpOpen(false)
                useTutorialStore.getState().startTutorial('notes-basics')
              }}
            >
              {t("option:notesSearch.restartTutorialAction", {
                defaultValue: "Restart tutorial",
              })}
            </button>
          </div>
        </div>
      </Modal>
    </>
  )
}

export default NotesManagerOverlays
