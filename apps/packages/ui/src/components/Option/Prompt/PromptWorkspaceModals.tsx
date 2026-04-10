import React, { Suspense } from "react"
import { useTranslation } from "react-i18next"
import { Modal, Input, Form, Tag } from "antd"
import type { FormInstance } from "antd"
import { Computer, Zap, Layers, Play } from "lucide-react"
import type {
  FewShotExample,
  PromptFormat,
  PromptSourceSystem,
  PromptSyncStatus,
  StructuredPromptDefinition
} from "@/db/dexie/types"
import type { ConflictInfo, ConflictResolution } from "@/services/prompt-sync"
import type { PromptRowVM } from "./prompt-workspace-types"

type PromptEditorInitialValues = {
  id?: string
  name?: string
  title?: string
  author?: string
  details?: string
  content?: string
  system_prompt?: string
  user_prompt?: string
  promptFormat?: PromptFormat
  promptSchemaVersion?: number | null
  structuredPromptDefinition?: StructuredPromptDefinition | null
  keywords?: string[]
  tags?: string[]
  is_system?: boolean
  serverId?: number | null
  syncStatus?: PromptSyncStatus
  sourceSystem?: PromptSourceSystem
  studioProjectId?: number | null
  lastSyncedAt?: number | null
  fewShotExamples?:
    | Array<
        | {
            input?: string
            output?: string
            explanation?: string | null
            inputs?: Record<string, unknown>
            outputs?: Record<string, unknown>
          }
        | FewShotExample
      >
    | null
  modulesConfig?: Array<{ name: string; enabled: boolean }> | null
  changeDescription?: string | null
  versionNumber?: number | null
}

// ---------------------------------------------------------------------------
// Lazy-loaded heavy sub-surfaces (mirrors index.tsx pattern)
// ---------------------------------------------------------------------------
const PromptDrawer = React.lazy(() =>
  import("./PromptDrawer").then((module) => ({ default: module.PromptDrawer }))
)
const PromptFullPageEditor = React.lazy(() =>
  import("./PromptFullPageEditor").then((module) => ({
    default: module.PromptFullPageEditor
  }))
)
const PromptInspectorPanel = React.lazy(() =>
  import("./PromptInspectorPanel").then((module) => ({
    default: module.PromptInspectorPanel
  }))
)
const ConflictResolutionModal = React.lazy(() =>
  import("./ConflictResolutionModal").then((module) => ({
    default: module.ConflictResolutionModal
  }))
)
const ProjectSelector = React.lazy(() =>
  import("./ProjectSelector").then((module) => ({ default: module.ProjectSelector }))
)

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface PromptWorkspaceModalsProps {
  // Copilot edit modal
  copilotEditOpen: boolean
  onCopilotEditCancel: () => void
  copilotEditForm: FormInstance
  copilotEditId: string | null
  copilotPromptIncludesTextPlaceholder: boolean
  onCopilotEditSubmit: (values: { key: string | null; prompt: string }) => void
  isUpdatingCopilotPrompt: boolean

  // Bulk keyword modal
  bulkKeywordModalOpen: boolean
  onBulkKeywordCancel: () => void
  bulkKeywordValue: string
  onBulkKeywordValueChange: (value: string) => void
  onBulkKeywordSubmit: () => void
  isBulkAddingKeyword: boolean

  // Quick test modal
  quickTestPrompt: { id: string; name: string; systemText?: string; userText?: string } | null
  onQuickTestClose: () => void
  quickTestInput: string
  onQuickTestInputChange: (value: string) => void
  quickTestOutput: string | null
  isRunningQuickTest: boolean
  quickTestRunInfo: { provider?: string; model?: string } | null
  onRunQuickTest: () => void

  // Collection create modal
  collectionModalOpen: boolean
  onCollectionModalCancel: () => void
  collectionName: string
  onCollectionNameChange: (value: string) => void
  collectionDescription: string
  onCollectionDescriptionChange: (value: string) => void
  onCollectionCreate: () => void
  isCreatingCollection: boolean

  // Shortcuts modal
  shortcutsOpen: boolean
  onShortcutsClose: () => void

  // Insert prompt modal
  insertPrompt: { systemText?: string; userText?: string } | null
  onInsertCancel: () => void
  onInsertChoice: (choice: "system" | "quick" | "both") => void

  // Project selector
  projectSelectorOpen: boolean
  onProjectSelectorClose: () => void
  onProjectSelect: (projectId: number) => void
  isPushing: boolean

  // Conflict resolution
  conflictModalOpen: boolean
  conflictLoading: boolean
  conflictInfo: ConflictInfo | null
  onConflictClose: () => void
  onConflictResolve: (resolution: ConflictResolution) => void

  // Drawer
  drawerOpen: boolean
  onDrawerClose: () => void
  drawerMode: "create" | "edit"
  drawerInitialValues: PromptEditorInitialValues | null
  onDrawerSubmit: (values: PromptEditorInitialValues) => void
  drawerLoading: boolean
  allTags: string[]

  // Full page editor
  fullEditorOpen: boolean
  onFullEditorClose: () => void
  fullEditorMode: "create" | "edit"
  fullEditorInitialValues: PromptEditorInitialValues | null
  onFullEditorSubmit: (values: PromptEditorInitialValues) => void
  fullEditorLoading: boolean

  // Inspector panel
  inspectorOpen: boolean
  inspectorPrompt: PromptRowVM | null
  onInspectorClose: () => void
  onInspectorEdit: (promptId: string) => void
  onInspectorUseInChat: (promptId: string) => void
  onInspectorDuplicate: (promptId: string) => void
  onInspectorDelete: (promptId: string) => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export const PromptWorkspaceModals: React.FC<PromptWorkspaceModalsProps> = (props) => {
  const { t } = useTranslation(["option", "common"])

  const {
    // Copilot edit
    copilotEditOpen,
    onCopilotEditCancel,
    copilotEditForm,
    copilotEditId,
    copilotPromptIncludesTextPlaceholder,
    onCopilotEditSubmit,
    isUpdatingCopilotPrompt,
    // Bulk keyword
    bulkKeywordModalOpen,
    onBulkKeywordCancel,
    bulkKeywordValue,
    onBulkKeywordValueChange,
    onBulkKeywordSubmit,
    isBulkAddingKeyword,
    // Quick test
    quickTestPrompt,
    onQuickTestClose,
    quickTestInput,
    onQuickTestInputChange,
    quickTestOutput,
    isRunningQuickTest,
    quickTestRunInfo,
    onRunQuickTest,
    // Collection create
    collectionModalOpen,
    onCollectionModalCancel,
    collectionName,
    onCollectionNameChange,
    collectionDescription,
    onCollectionDescriptionChange,
    onCollectionCreate,
    isCreatingCollection,
    // Shortcuts
    shortcutsOpen,
    onShortcutsClose,
    // Insert prompt
    insertPrompt,
    onInsertCancel,
    onInsertChoice,
    // Project selector
    projectSelectorOpen,
    onProjectSelectorClose,
    onProjectSelect,
    isPushing,
    // Conflict resolution
    conflictModalOpen,
    conflictLoading,
    conflictInfo,
    onConflictClose,
    onConflictResolve,
    // Drawer
    drawerOpen,
    onDrawerClose,
    drawerMode,
    drawerInitialValues,
    onDrawerSubmit,
    drawerLoading,
    allTags,
    // Full page editor
    fullEditorOpen,
    onFullEditorClose,
    fullEditorMode,
    fullEditorInitialValues,
    onFullEditorSubmit,
    fullEditorLoading,
    // Inspector panel
    inspectorOpen,
    inspectorPrompt,
    onInspectorClose,
    onInspectorEdit,
    onInspectorUseInChat,
    onInspectorDuplicate,
    onInspectorDelete
  } = props

  return (
    <>
      {/* PromptDrawer */}
      <Suspense fallback={null}>
        <PromptDrawer
          open={drawerOpen}
          onClose={onDrawerClose}
          mode={drawerMode}
          initialValues={drawerInitialValues}
          onSubmit={onDrawerSubmit}
          isLoading={drawerLoading}
          allTags={allTags}
        />
      </Suspense>

      {/* PromptFullPageEditor */}
      <Suspense fallback={null}>
        <PromptFullPageEditor
          open={fullEditorOpen}
          onClose={onFullEditorClose}
          mode={fullEditorMode}
          initialValues={fullEditorInitialValues}
          onSubmit={onFullEditorSubmit}
          isLoading={fullEditorLoading}
          allTags={allTags}
        />
      </Suspense>

      {/* PromptInspectorPanel */}
      <Suspense fallback={null}>
        <PromptInspectorPanel
          open={inspectorOpen}
          prompt={inspectorPrompt}
          onClose={onInspectorClose}
          onEdit={onInspectorEdit}
          onUseInChat={onInspectorUseInChat}
          onDuplicate={onInspectorDuplicate}
          onDelete={onInspectorDelete}
        />
      </Suspense>

      {/* Copilot edit modal */}
      <Modal
        title={t("managePrompts.modal.editTitle")}
        open={copilotEditOpen}
        onCancel={onCopilotEditCancel}
        footer={null}>
        <Form
          onFinish={(values) =>
            onCopilotEditSubmit({
              key: copilotEditId,
              prompt: values.prompt
            })
          }
          layout="vertical"
          form={copilotEditForm}>
          <Form.Item
            name="prompt"
            label={t("managePrompts.form.prompt.label")}
            extra={
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span>
                  {t("managePrompts.form.prompt.copilotPlaceholderHint", {
                    defaultValue: "Must include placeholder"
                  })}
                </span>
                <Tag color={copilotPromptIncludesTextPlaceholder ? "green" : "orange"}>
                  {"{text}"}
                </Tag>
                <span
                  data-testid="copilot-text-placeholder-status"
                  className={
                    copilotPromptIncludesTextPlaceholder
                      ? "text-success"
                      : "text-warn"
                  }
                >
                  {copilotPromptIncludesTextPlaceholder
                    ? t("managePrompts.form.prompt.copilotPlaceholderPresent", {
                        defaultValue: "placeholder detected"
                      })
                    : t("managePrompts.form.prompt.copilotPlaceholderMissing", {
                        defaultValue: "missing placeholder"
                      })}
                </span>
              </div>
            }
            rules={[
              {
                required: true,
                message: t("managePrompts.form.prompt.required")
              },
              {
                validator: (_, value) => {
                  if (value && value.includes("{text}")) {
                    return Promise.resolve()
                  }
                  return Promise.reject(
                    new Error(
                      t("managePrompts.form.prompt.missingTextPlaceholder")
                    )
                  )
                }
              }
            ]}>
            <Input.TextArea
              placeholder={t("managePrompts.form.prompt.placeholder")}
              autoSize={{ minRows: 3, maxRows: 10 }}
              data-testid="copilot-edit-prompt-input"
            />
          </Form.Item>

          <Form.Item>
            <button
              data-testid="copilot-edit-save"
              disabled={isUpdatingCopilotPrompt}
              className="inline-flex justify-center w-full text-center mt-4 items-center rounded-md border border-transparent bg-primary px-2 py-2 text-sm font-medium leading-4 text-white shadow-sm hover:bg-primaryStrong focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 disabled:opacity-50">
              {isUpdatingCopilotPrompt
                ? t("managePrompts.form.btnEdit.saving")
                : t("managePrompts.form.btnEdit.save")}
            </button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Bulk keyword modal */}
      <Modal
        title={t("managePrompts.bulk.addKeyword", { defaultValue: "Add keyword" })}
        open={bulkKeywordModalOpen}
        onCancel={onBulkKeywordCancel}
        onOk={onBulkKeywordSubmit}
        okText={t("common:add", { defaultValue: "Add" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        okButtonProps={{
          disabled:
            bulkKeywordValue.trim().length === 0 || isBulkAddingKeyword,
          loading: isBulkAddingKeyword
        }}
      >
        <Input
          autoFocus
          value={bulkKeywordValue}
          onChange={(event) => onBulkKeywordValueChange(event.target.value)}
          onPressEnter={() => {
            if (
              bulkKeywordValue.trim().length === 0 ||
              isBulkAddingKeyword
            ) {
              return
            }
            onBulkKeywordSubmit()
          }}
          placeholder={t("managePrompts.tags.addPlaceholder", {
            defaultValue: "Enter keyword"
          })}
          data-testid="prompts-bulk-keyword-input"
        />
      </Modal>

      {/* Quick test modal */}
      <Modal
        title={t("managePrompts.quickTest.modalTitle", {
          defaultValue: "Quick test prompt"
        })}
        open={!!quickTestPrompt}
        onCancel={onQuickTestClose}
        footer={[
          <button
            key="cancel"
            type="button"
            data-testid="prompts-local-quick-test-cancel"
            className="inline-flex items-center justify-center rounded-md border border-border bg-bg px-3 py-2 text-sm text-text hover:bg-surface2"
            onClick={onQuickTestClose}
            disabled={isRunningQuickTest}
          >
            {t("common:cancel", { defaultValue: "Cancel" })}
          </button>,
          <button
            key="run"
            type="button"
            data-testid="prompts-local-quick-test-run"
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-primary px-3 py-2 text-sm text-white hover:bg-primaryStrong disabled:opacity-50"
            onClick={() => {
              void onRunQuickTest()
            }}
            disabled={isRunningQuickTest}
          >
            <Play className="mr-1 size-4" />
            {isRunningQuickTest
              ? t("managePrompts.quickTest.running", { defaultValue: "Running..." })
              : t("managePrompts.quickTest.runAction", { defaultValue: "Run test" })}
          </button>
        ]}
        width={720}
      >
        {quickTestPrompt && (
          <div className="space-y-3">
            <div className="rounded border border-border bg-surface2 p-3">
              <div className="text-sm font-medium text-text">
                {quickTestPrompt.name}
              </div>
              {quickTestPrompt.systemText && (
                <div className="mt-2 text-xs text-text-muted">
                  <span className="font-medium">
                    {t("managePrompts.systemPrompt", {
                      defaultValue: "AI Instructions"
                    })}
                    :
                  </span>{" "}
                  <span className="line-clamp-2">{quickTestPrompt.systemText}</span>
                </div>
              )}
              {quickTestPrompt.userText && (
                <div className="mt-2 text-xs text-text-muted">
                  <span className="font-medium">
                    {t("managePrompts.quickPrompt", {
                      defaultValue: "Message Template"
                    })}
                    :
                  </span>{" "}
                  <span className="line-clamp-3">{quickTestPrompt.userText}</span>
                </div>
              )}
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium text-text">
                {t("managePrompts.quickTest.inputLabel", {
                  defaultValue: "Sample input"
                })}
              </label>
              <Input.TextArea
                value={quickTestInput}
                onChange={(event) => onQuickTestInputChange(event.target.value)}
                placeholder={t("managePrompts.quickTest.inputPlaceholder", {
                  defaultValue:
                    "Optional input text. Used for {{text}} templates or appended to quick prompts."
                })}
                autoSize={{ minRows: 3, maxRows: 8 }}
                data-testid="prompts-local-quick-test-input"
              />
            </div>

            {quickTestOutput && (
              <div
                className="rounded border border-border bg-bg p-3"
                data-testid="prompts-local-quick-test-output"
              >
                <div className="mb-1 text-xs text-text-muted">
                  {quickTestRunInfo
                    ? t("managePrompts.quickTest.outputMeta", {
                        defaultValue: "Result ({{provider}} / {{model}})",
                        provider:
                          quickTestRunInfo.provider ||
                          t("managePrompts.quickTest.defaultProvider", {
                            defaultValue: "default provider"
                          }),
                        model: quickTestRunInfo.model
                      })
                    : t("managePrompts.quickTest.outputTitle", {
                        defaultValue: "Result"
                      })}
                </div>
                <pre className="whitespace-pre-wrap break-words text-sm text-text">
                  {quickTestOutput}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* Collection create modal */}
      <Modal
        title={t("managePrompts.collections.create", {
          defaultValue: "New collection"
        })}
        open={collectionModalOpen}
        onCancel={onCollectionModalCancel}
        onOk={onCollectionCreate}
        okText={t("common:create", { defaultValue: "Create" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        okButtonProps={{
          loading: isCreatingCollection,
          disabled: collectionName.trim().length === 0
        }}
        data-testid="prompts-collection-create-modal"
      >
        <div className="space-y-3">
          <Input
            value={collectionName}
            onChange={(event) => onCollectionNameChange(event.target.value)}
            placeholder={t("managePrompts.collections.namePlaceholder", {
              defaultValue: "Collection name"
            })}
            data-testid="prompts-collection-name-input"
          />
          <Input.TextArea
            value={collectionDescription}
            onChange={(event) => onCollectionDescriptionChange(event.target.value)}
            placeholder={t("managePrompts.collections.descriptionPlaceholder", {
              defaultValue: "Description (optional)"
            })}
            autoSize={{ minRows: 2, maxRows: 4 }}
            data-testid="prompts-collection-description-input"
          />
        </div>
      </Modal>

      {/* Keyboard shortcuts modal */}
      <Modal
        title={t("managePrompts.shortcuts.title", {
          defaultValue: "Keyboard shortcuts"
        })}
        open={shortcutsOpen}
        onCancel={onShortcutsClose}
        footer={null}
        data-testid="prompts-shortcuts-modal"
      >
        <p className="text-sm text-text-muted">
          {t("managePrompts.shortcuts.description", {
            defaultValue:
              "Shortcuts are available when focus is not inside an input field."
          })}
        </p>
        <div className="mt-3 space-y-2 text-sm">
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.newPrompt", {
                defaultValue: "Create new prompt"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">N</kbd>
          </div>
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.focusSearch", {
                defaultValue: "Focus search"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">/</kbd>
          </div>
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.closeDrawer", {
                defaultValue: "Close drawer / modal"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">Esc</kbd>
          </div>
          <div className="flex items-center justify-between rounded border border-border p-2">
            <span>
              {t("managePrompts.shortcuts.openHelp", {
                defaultValue: "Open shortcut help"
              })}
            </span>
            <kbd className="rounded border border-border px-1.5 py-0.5 text-xs">?</kbd>
          </div>
        </div>
      </Modal>

      {/* Insert prompt modal */}
      <Modal
        title={t("option:promptInsert.confirmTitle", {
          defaultValue: "Use prompt in chat?"
        })}
        open={!!insertPrompt}
        onCancel={onInsertCancel}
        footer={null}
        width={520}>
        <div className="space-y-3">
          {/* System option */}
          {insertPrompt?.systemText && (
            <button
              type="button"
              onClick={() => {
                void onInsertChoice("system")
              }}
              data-testid="prompt-insert-system"
              className="w-full text-left p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <Computer className="size-5 text-warn" />
                <span className="font-medium">
                  {t("option:promptInsert.useAsSystem", {
                    defaultValue: "Use as System Instruction"
                  })}
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">
                {t("option:promptInsert.systemDescription", {
                  defaultValue: "Sets the AI's behavior and persona for the conversation."
                })}
              </p>
              <div className="bg-surface2 rounded p-2 text-xs line-clamp-3 font-mono text-text-muted">
                {insertPrompt.systemText}
              </div>
            </button>
          )}

          {/* Quick/User option */}
          {insertPrompt?.userText && (
            <button
              type="button"
              onClick={() => {
                void onInsertChoice("quick")
              }}
              data-testid="prompt-insert-quick"
              className="w-full text-left p-4 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="size-5 text-primary" />
                <span className="font-medium">
                  {t("option:promptInsert.useAsTemplate", {
                    defaultValue: "Insert as Message Template"
                  })}
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">
                {t("option:promptInsert.templateDescription", {
                  defaultValue: "Adds this text to your message composer."
                })}
              </p>
              <div className="bg-surface2 rounded p-2 text-xs line-clamp-3 font-mono text-text-muted">
                {insertPrompt.userText}
              </div>
            </button>
          )}

          {/* Use Both option - shown when prompt has both system and user */}
          {insertPrompt?.systemText && insertPrompt?.userText && (
            <button
              type="button"
              onClick={() => {
                void onInsertChoice("both")
              }}
              data-testid="prompt-insert-both"
              className="w-full text-left p-4 rounded-lg border-2 border-primary/50 bg-primary/5 hover:border-primary hover:bg-primary/10 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <Layers className="size-5 text-primary" />
                <span className="font-medium text-primary">
                  {t("option:promptInsert.useBoth", {
                    defaultValue: "Use Both (Recommended)"
                  })}
                </span>
              </div>
              <p className="text-xs text-text-muted mb-2">
                {t("option:promptInsert.bothDescription", {
                  defaultValue: "Sets the system instruction AND inserts the message template. Best for prompts designed to work together."
                })}
              </p>
            </button>
          )}
        </div>
      </Modal>

      {/* Project Selector for Push to Server */}
      <Suspense fallback={null}>
        <ProjectSelector
          open={projectSelectorOpen}
          onClose={onProjectSelectorClose}
          onSelect={onProjectSelect}
          loading={isPushing}
        />
      </Suspense>

      {/* Conflict Resolution */}
      <Suspense fallback={null}>
        <ConflictResolutionModal
          open={conflictModalOpen}
          loading={conflictLoading}
          conflictInfo={conflictInfo}
          onClose={onConflictClose}
          onResolve={onConflictResolve}
        />
      </Suspense>
    </>
  )
}
