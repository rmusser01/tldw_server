import React from "react"
import { Button, Checkbox, Input, Modal, Typography } from "antd"

import KeywordPickerModal from "@/components/Notes/KeywordPickerModal"
import NotesGraphModal from "@/components/Notes/NotesGraphModal"

import type {
  ImportDuplicateStrategy,
  KeywordManagementItem,
  KeywordMergeDraft,
  KeywordPickerSortMode,
  KeywordRenameDraft,
  PendingImportFile,
} from "./notes-manager-types"
import { toKeywordTestIdSegment } from "./notes-manager-utils"

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
          defaultValue: "Review suggested keywords",
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
              defaultValue: "Select which suggested keywords to add to this note.",
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
          defaultValue: "Manage keywords",
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
              defaultValue: "Rename, merge, or delete keywords from one place.",
            })}
          </Typography.Text>
          <Input
            allowClear
            value={kw.keywordManagerQuery}
            onChange={(event) => kw.setKeywordManagerQuery(event.target.value)}
            placeholder={t("option:notesSearch.keywordManagerSearchPlaceholder", {
              defaultValue: "Filter keywords",
            })}
            data-testid="notes-keyword-manager-search"
          />
          <div className="max-h-80 overflow-auto rounded-lg border border-border bg-surface2 p-2">
            {kw.keywordManagerLoading ? (
              <Typography.Text type="secondary" className="text-xs text-text-muted">
                {t("option:notesSearch.keywordManagerLoading", {
                  defaultValue: "Loading keywords...",
                })}
              </Typography.Text>
            ) : kw.keywordManagerVisibleItems.length === 0 ? (
              <Typography.Text type="secondary" className="text-xs text-text-muted">
                {t("option:notesSearch.keywordManagerEmpty", {
                  defaultValue: "No keywords found.",
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
          defaultValue: "Rename keyword",
        })}
      >
        <div className="space-y-2">
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.keywordManagerRenameHelp", {
              defaultValue: "Choose a new name for this keyword.",
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
        destroyOnHidden
        title={t("option:notesSearch.keywordManagerMergeTitle", {
          defaultValue: "Merge keyword",
        })}
      >
        <div className="space-y-2">
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted"
          >
            {t("option:notesSearch.keywordManagerMergeHelp", {
              defaultValue:
                "Move all links from the source keyword to the selected target keyword.",
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
          <select
            className="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-text"
            value={kw.keywordMergeDraft?.targetKeywordId ?? ""}
            onChange={(event) => {
              const parsed = Number(event.target.value)
              kw.setKeywordMergeDraft((current) =>
                current
                  ? {
                      ...current,
                      targetKeywordId:
                        Number.isFinite(parsed) && parsed > 0 ? parsed : null,
                    }
                  : current,
              )
            }}
            data-testid="notes-keyword-manager-merge-target"
          >
            <option value="">
              {t("option:notesSearch.keywordManagerMergeTargetPlaceholder", {
                defaultValue: "Select target keyword",
              })}
            </option>
            {kw.keywordMergeTargetOptions.map((item) => (
              <option key={`keyword-merge-target-${item.id}`} value={item.id}>
                {item.keyword} ({item.noteCount})
              </option>
            ))}
          </select>
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
                  <div className="mt-1 text-[11px] text-warn">{file.parseError}</div>
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
        <div className="space-y-2 text-sm text-text" data-testid="notes-shortcuts-modal">
          <div>
            <strong>Ctrl/Cmd + S</strong>:{" "}
            {t("option:notesSearch.shortcutSaveDescription", {
              defaultValue: "Save the current note.",
            })}
          </div>
          <div>
            <strong>?</strong>:{" "}
            {t("option:notesSearch.shortcutOpenHelpDescription", {
              defaultValue: "Open keyboard shortcut help.",
            })}
          </div>
          <div>
            <strong>Esc</strong>:{" "}
            {t("option:notesSearch.shortcutCloseDialogDescription", {
              defaultValue: "Close the current dialog.",
            })}
          </div>
        </div>
      </Modal>
    </>
  )
}

export default NotesManagerOverlays
