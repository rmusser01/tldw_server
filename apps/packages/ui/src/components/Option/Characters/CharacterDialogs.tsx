import React from "react"
import {
  Button,
  Drawer,
  Modal,
  Input,
  Skeleton,
  Tag,
  Tooltip,
  Select,
  Segmented
} from "antd"
import type { FormInstance } from "antd"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives/Alert"
import {
  Copy,
  Download,
  Clock3,
  ChevronUp,
  ChevronDown
} from "lucide-react"
import {
  tldwClient,
  type ServerChatSummary
} from "@/services/tldw/TldwApiClient"
import { UserCircle2, MessageCircle } from "lucide-react"
import { GenerateCharacterPanel, GenerationPreviewModal } from "./GenerateCharacterPanel"
import { CHARACTER_TEMPLATES, type CharacterTemplate } from "@/data/character-templates"
import {
  DEFAULT_CHARACTER_PROMPT_PRESET
} from "@/data/character-prompt-presets"
import type { GeneratedCharacter } from "@/services/character-generation"
import {
  CHARACTER_VERSION_DIFF_FIELD_KEYS,
  CHARACTER_VERSION_FIELD_LABELS,
  normalizeVersionSnapshotValue,
  formatVersionSnapshotValue,
  getCharacterVisibleTags,
  normalizeCharacterFolderId,
  getCharacterFolderIdFromTags,
  hasAdvancedData,
  type CharacterWorldBookOption,
  type AdvancedSectionKey
} from "./utils"
import type { CharacterTagOperation } from "./tag-manager-utils"
import { formatDraftAge } from "@/hooks/useFormDraft"
import { validateAndCreateImageDataUrl } from "@/utils/image-utils"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { updatePageTitle } from "@/utils/update-page-title"
import { focusComposer } from "@/hooks/useComposerFocus"
import {
  buildCharacterChatReadiness,
  CHARACTER_CHAT_MODEL_SETTINGS_PATH,
  getCharacterChatReadinessCopy
} from "@/utils/chat-model-availability"
import type { TFunction } from "i18next"
import type { NavigateFunction } from "react-router-dom"
import type { CharacterChatIntentBlocker } from "./hooks/useCharacterCrud"
import { buildCharacterChatPath } from "@/routes/route-paths"

// ---------------------------------------------------------------------------
// Shared props type for the CharacterDialogs component
// ---------------------------------------------------------------------------

type SharedCharacterFormRenderer = (props: {
  form: FormInstance
  mode: "create" | "edit"
  initialValues?: Record<string, any>
  worldBookFieldContext: {
    options: CharacterWorldBookOption[]
    loading: boolean
    editCharacterNumericId: number | null
  }
  isSubmitting: boolean
  submitButtonClassName?: string
  submitPendingLabel: string
  submitIdleLabel: string
  showPreview: boolean
  onTogglePreview: () => void
  onValuesChange: (allValues: Record<string, any>) => void
  onFinish: (values: Record<string, any>) => void
}) => React.ReactNode

export type CharacterDialogsProps = {
  t: TFunction
  navigate: NavigateFunction

  // --- import preview ---
  importPreviewOpen: boolean
  resetImportPreview: () => void
  importPreviewHasSuccessfulCompletion: boolean
  retryableFailedPreviewItems: any[]
  importPreviewProcessing: boolean
  handleRetryFailedImportPreview: () => Promise<void>
  importPreviewLoading: boolean
  importablePreviewItems: any[]
  importQueueSummary: {
    total: number
    queued: number
    processing: number
    success: number
    failure: number
    complete: boolean
  }
  importPreviewItems: any[]
  importQueueItemsById: Map<string, any>
  importing: boolean
  handleConfirmImportPreview: () => Promise<void>
  getImportQueueStateLabel: (state: string) => string
  getImportQueueStateColor: (state: string) => string

  // --- quick chat ---
  quickChatCharacter: any | null
  closeQuickChat: () => Promise<void>
  quickChatModelOptions: Array<{ value: string; label: string }>
  activeQuickChatModel: string | null
  isServerConnected: boolean
  setQuickChatModelOverride: (v: string | null) => void
  quickChatError: string | null
  quickChatMessages: Array<{ id: string; role: string; content: string }>
  quickChatDraft: string
  setQuickChatDraft: (v: string) => void
  quickChatSending: boolean
  sendQuickChatMessage: () => Promise<void>
  handlePromoteQuickChat: () => Promise<void>
  chatIntentBlocker: CharacterChatIntentBlocker | null
  clearChatIntentBlocker: () => Promise<void>
  retryChatIntentBlocker: () => Promise<void>

  // --- conversations ---
  conversationsOpen: boolean
  setConversationsOpen: (v: boolean) => void
  conversationCharacter: any | null
  setConversationCharacter: (v: any | null) => void
  characterChats: ServerChatSummary[]
  setCharacterChats: (v: ServerChatSummary[]) => void
  chatsError: string | null
  setChatsError: (v: string | null) => void
  loadingChats: boolean
  setLoadingChats: (v: boolean) => void
  resumingChatId: string | null
  setResumingChatId: (v: string | null) => void
  // store setters for conversation resume
  setSelectedCharacter: (v: any) => void
  setHistory: (v: any[]) => void
  setMessages: (v: any[]) => void
  setHistoryId: (v: string | null) => void
  setServerChatId: (v: string) => void
  setServerChatState: (v: string) => void
  setServerChatTopic: (v: string | null) => void
  setServerChatClusterId: (v: string | null) => void
  setServerChatSource: (v: string | null) => void
  setServerChatExternalRef: (v: string | null) => void

  // --- create drawer ---
  open: boolean
  setOpen: (v: boolean) => void
  createForm: FormInstance
  createFormDirty: boolean
  setCreateFormDirty: (v: boolean) => void
  creating: boolean
  createCharacter: (values: Record<string, any>) => void
  showCreatePreview: boolean
  setShowCreatePreview: React.Dispatch<React.SetStateAction<boolean>>
  setShowCreateAdvanced: (v: boolean) => void
  setShowCreateSystemPromptExample: (v: boolean) => void
  setShowTemplates: (v: boolean) => void
  showTemplates: boolean
  markTemplateChooserSeen: () => void
  applyTemplateToCreateForm: (template: CharacterTemplate) => void
  newButtonRef: React.RefObject<HTMLButtonElement | null>
  // draft
  hasCreateDraft: boolean
  createDraftData: any
  saveCreateDraft: (v: Record<string, any>) => void
  clearCreateDraft: () => void
  applyCreateDraft: () => Record<string, any> | null
  dismissCreateDraft: () => void

  // --- edit drawer ---
  openEdit: boolean
  setOpenEdit: (v: boolean) => void
  editForm: FormInstance
  editFormDirty: boolean
  setEditFormDirty: (v: boolean) => void
  editId: string | null
  setEditId: (v: string | null) => void
  editVersion: number | null
  setEditVersion: (v: number | null) => void
  editCharacterNumericId: number | null
  updating: boolean
  updateCharacter: (values: Record<string, any>) => void
  showEditPreview: boolean
  setShowEditPreview: React.Dispatch<React.SetStateAction<boolean>>
  setShowEditAdvanced: (v: boolean) => void
  setShowEditSystemPromptExample: (v: boolean) => void
  lastEditTriggerRef: React.RefObject<Element | null>
  editWorldBooksInitializedRef: React.MutableRefObject<boolean>
  // draft
  hasEditDraft: boolean
  editDraftData: any
  saveEditDraft: (v: Record<string, any>) => void
  clearEditDraft: () => void
  applyEditDraft: () => Record<string, any> | null
  dismissEditDraft: () => void

  // world book options (shared by create + edit)
  worldBookOptions: CharacterWorldBookOption[]
  worldBookOptionsLoading: boolean

  // --- generation preview ---
  isGenerating: boolean
  generatingField: string | null
  generationError: string | null
  handleGenerateFullCharacter: (concept: string, model: string, apiProvider?: string) => Promise<void>
  cancelGeneration: () => void
  clearGenerationError: () => void
  generationPreviewOpen: boolean
  setGenerationPreviewOpen: (v: boolean) => void
  generationPreviewData: GeneratedCharacter | null
  setGenerationPreviewData: (v: GeneratedCharacter | null) => void
  generationPreviewField: string | null
  setGenerationPreviewField: (v: string | null) => void
  applyGenerationPreview: () => void

  // --- version history ---
  versionHistoryOpen: boolean
  setVersionHistoryOpen: (v: boolean) => void
  versionHistoryCharacter: any | null
  setVersionHistoryCharacter: (v: any | null) => void
  versionHistoryCharacterId: number | null | undefined
  versionHistoryCharacterName: string
  versionFrom: number | null
  setVersionFrom: (v: number | null) => void
  versionTo: number | null
  setVersionTo: (v: number | null) => void
  versionRevertTarget: number | null
  setVersionRevertTarget: (v: number | null) => void
  versionHistoryItems: any[]
  versionHistoryLoading: boolean
  versionHistoryFetching: boolean
  versionSelectOptions: Array<{ value: number; label: string }>
  versionDiffResponse: any
  versionDiffLoading: boolean
  versionDiffFetching: boolean
  revertingCharacterVersion: boolean
  openVersionHistory: (record: any) => void
  revertCharacterVersion: (opts: { characterId: number; targetVersion: number }) => void

  // --- compare ---
  compareModalOpen: boolean
  closeCompareModal: () => void
  compareCharacters: [any, any] | null
  comparisonRows: Array<{ field: string; label: string; leftValue: string; rightValue: string; different: boolean }>
  changedComparisonRows: Array<{ field: string; label: string; leftValue: string; rightValue: string; different: boolean }>
  handleCopyComparisonSummary: () => Promise<void>
  handleExportComparisonSummary: () => void

  // --- tag manager ---
  tagManagerOpen: boolean
  closeTagManager: () => void
  tagManagerLoading: boolean
  tagManagerSubmitting: boolean
  tagManagerOperation: CharacterTagOperation
  setTagManagerOperation: (v: CharacterTagOperation) => void
  tagManagerSourceTag: string | undefined
  setTagManagerSourceTag: (v: string | undefined) => void
  tagManagerTargetTag: string
  setTagManagerTargetTag: (v: string) => void
  tagManagerTagUsageData: Array<{ tag: string; count: number }>
  handleApplyTagManagerOperation: () => Promise<void>

  // --- bulk tags ---
  bulkTagModalOpen: boolean
  setBulkTagModalOpen: (v: boolean) => void
  bulkTagsToAdd: string[]
  setBulkTagsToAdd: (v: string[]) => void
  bulkOperationLoading: boolean
  handleBulkAddTagsForSelection: () => void
  selectedCount: number
  popularTags: Array<{ tag: string; count: number }>
  tagOptionsWithCounts: Array<{ value: string; label: React.ReactNode }>

  // --- confirm ---
  confirmDanger: (opts: {
    title: string
    content: string
    okText: string
    cancelText: string
  }) => Promise<boolean>

  // shared form renderer
  renderSharedCharacterForm: SharedCharacterFormRenderer

  // data for version history inline link
  data: any[] | undefined
}

export const CharacterDialogs: React.FC<CharacterDialogsProps> = (props) => {
  const {
    t,
    navigate,
    // import preview
    importPreviewOpen,
    resetImportPreview,
    importPreviewHasSuccessfulCompletion,
    retryableFailedPreviewItems,
    importPreviewProcessing,
    handleRetryFailedImportPreview,
    importPreviewLoading,
    importablePreviewItems,
    importQueueSummary,
    importPreviewItems,
    importQueueItemsById,
    importing,
    handleConfirmImportPreview,
    getImportQueueStateLabel,
    getImportQueueStateColor,
    // quick chat
    quickChatCharacter,
    closeQuickChat,
    quickChatModelOptions,
    activeQuickChatModel,
    isServerConnected,
    setQuickChatModelOverride,
    quickChatError,
    quickChatMessages,
    quickChatDraft,
    setQuickChatDraft,
    quickChatSending,
    sendQuickChatMessage,
    handlePromoteQuickChat,
    chatIntentBlocker,
    clearChatIntentBlocker,
    retryChatIntentBlocker,
    // conversations
    conversationsOpen,
    setConversationsOpen,
    conversationCharacter,
    setConversationCharacter,
    characterChats,
    setCharacterChats,
    chatsError,
    setChatsError,
    loadingChats,
    setLoadingChats,
    resumingChatId,
    setResumingChatId,
    setSelectedCharacter,
    setHistory,
    setMessages,
    setHistoryId,
    setServerChatId,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef,
    // create
    open,
    setOpen,
    createForm,
    createFormDirty,
    setCreateFormDirty,
    creating,
    createCharacter,
    showCreatePreview,
    setShowCreatePreview,
    setShowCreateAdvanced,
    setShowCreateSystemPromptExample,
    setShowTemplates,
    showTemplates,
    markTemplateChooserSeen,
    applyTemplateToCreateForm,
    newButtonRef,
    hasCreateDraft,
    createDraftData,
    saveCreateDraft,
    clearCreateDraft,
    applyCreateDraft,
    dismissCreateDraft,
    // edit
    openEdit,
    setOpenEdit,
    editForm,
    editFormDirty,
    setEditFormDirty,
    editId,
    setEditId,
    editVersion,
    setEditVersion,
    editCharacterNumericId,
    updating,
    updateCharacter,
    showEditPreview,
    setShowEditPreview,
    setShowEditAdvanced,
    setShowEditSystemPromptExample,
    lastEditTriggerRef,
    editWorldBooksInitializedRef,
    hasEditDraft,
    editDraftData,
    saveEditDraft,
    clearEditDraft,
    applyEditDraft,
    dismissEditDraft,
    worldBookOptions,
    worldBookOptionsLoading,
    // generation
    isGenerating,
    generatingField,
    generationError,
    handleGenerateFullCharacter,
    cancelGeneration,
    clearGenerationError,
    generationPreviewOpen,
    setGenerationPreviewOpen,
    generationPreviewData,
    setGenerationPreviewData,
    generationPreviewField,
    setGenerationPreviewField,
    applyGenerationPreview,
    // version history
    versionHistoryOpen,
    setVersionHistoryOpen,
    versionHistoryCharacter,
    setVersionHistoryCharacter,
    versionHistoryCharacterId,
    versionHistoryCharacterName,
    versionFrom,
    setVersionFrom,
    versionTo,
    setVersionTo,
    versionRevertTarget,
    setVersionRevertTarget,
    versionHistoryItems,
    versionHistoryLoading,
    versionHistoryFetching,
    versionSelectOptions,
    versionDiffResponse,
    versionDiffLoading,
    versionDiffFetching,
    revertingCharacterVersion,
    openVersionHistory,
    revertCharacterVersion,
    // compare
    compareModalOpen,
    closeCompareModal,
    compareCharacters,
    comparisonRows,
    changedComparisonRows,
    handleCopyComparisonSummary,
    handleExportComparisonSummary,
    // tag manager
    tagManagerOpen,
    closeTagManager,
    tagManagerLoading,
    tagManagerSubmitting,
    tagManagerOperation,
    setTagManagerOperation,
    tagManagerSourceTag,
    setTagManagerSourceTag,
    tagManagerTargetTag,
    setTagManagerTargetTag,
    tagManagerTagUsageData,
    handleApplyTagManagerOperation,
    // bulk tags
    bulkTagModalOpen,
    setBulkTagModalOpen,
    bulkTagsToAdd,
    setBulkTagsToAdd,
    bulkOperationLoading,
    handleBulkAddTagsForSelection,
    selectedCount,
    popularTags,
    tagOptionsWithCounts,
    // confirm
    confirmDanger,
    // shared form
    renderSharedCharacterForm,
    data
  } = props

  const quickChatReadiness = React.useMemo(
    () =>
      buildCharacterChatReadiness({
        isServerConnected,
        selectedCharacter: quickChatCharacter,
        selectedModel: activeQuickChatModel,
        availableModels: quickChatModelOptions.map((option) => ({
          model: option.value
        })),
        isSendBlocked: quickChatSending
      }),
    [
      activeQuickChatModel,
      isServerConnected,
      quickChatCharacter,
      quickChatModelOptions,
      quickChatSending
    ]
  )
  const quickChatReadinessCopy = React.useMemo(
    () =>
      getCharacterChatReadinessCopy(quickChatReadiness, t, {
        characterName:
          quickChatCharacter?.name ||
          quickChatCharacter?.title ||
          quickChatCharacter?.slug ||
          null
      }),
    [quickChatCharacter, quickChatReadiness, t]
  )
  const showQuickChatModelBlocker =
    quickChatReadiness.missingRequirement === "chat-model"
  const chatIntentReadinessCopy = React.useMemo(() => {
    if (!chatIntentBlocker) return null
    const characterSelection = chatIntentBlocker.characterSelection
    const readiness = chatIntentBlocker.readiness
    return getCharacterChatReadinessCopy(readiness, t, {
      characterName:
        characterSelection?.name ||
        characterSelection?.title ||
        characterSelection?.slug ||
        null
    })
  }, [chatIntentBlocker, t])

  const characterIdentifierFn = (record: any): string =>
    String(record?.id ?? record?.slug ?? record?.name ?? "")

  const conversationLoadErrorMessage = React.useMemo(
    () =>
      t("settings:manageCharacters.conversations.error", {
        defaultValue: "Unable to load conversations for this character."
      }),
    [t]
  )

  const loadConversationChats = React.useCallback(async () => {
    if (!conversationCharacter) return

    setLoadingChats(true)
    setChatsError(null)
    setCharacterChats([])

    try {
      await tldwClient.initialize()
      const characterId = characterIdentifierFn(conversationCharacter)
      const chats = await tldwClient.listChats({
        character_id: characterId || undefined,
        limit: 100,
        ordering: "-updated_at"
      })
      const filtered = Array.isArray(chats)
        ? chats.filter(
            (chat) =>
              characterId &&
              String(chat.character_id ?? "") === String(characterId)
          )
        : []
      setCharacterChats(filtered)
    } catch {
      setChatsError(conversationLoadErrorMessage)
    } finally {
      setLoadingChats(false)
    }
  }, [
    conversationLoadErrorMessage,
    conversationCharacter,
    setCharacterChats,
    setChatsError,
    setLoadingChats
  ])

  React.useEffect(() => {
    if (!conversationsOpen || !conversationCharacter) return
    void loadConversationChats()
  }, [conversationCharacter, conversationsOpen, loadConversationChats])

  const formatUpdatedLabel = (value?: string | null) => {
    const fallback = t("settings:manageCharacters.conversations.unknownTime", {
      defaultValue: "Unknown"
    })
    let formatted = fallback
    if (value) {
      try {
        formatted = new Date(value).toLocaleString()
      } catch {
        formatted = String(value)
      }
    }
    return t("settings:manageCharacters.conversations.updated", {
      defaultValue: "Updated {{time}}",
      time: formatted
    })
  }

  const conversationInsights = React.useMemo(() => {
    if (!Array.isArray(characterChats) || characterChats.length === 0) {
      return {
        lastActive: null as string | null,
        averageMessageCount: 0
      }
    }

    let newestTimestamp = 0
    let messageCountTotal = 0
    let messageCountSamples = 0

    for (const chat of characterChats) {
      const timestamp = Date.parse(
        String(chat.last_active || chat.updated_at || chat.created_at || "")
      )
      if (Number.isFinite(timestamp) && timestamp > newestTimestamp) {
        newestTimestamp = timestamp
      }

      const count =
        typeof chat.message_count === "number"
          ? chat.message_count
          : Number.NaN
      if (Number.isFinite(count)) {
        messageCountTotal += count
        messageCountSamples += 1
      }
    }

    return {
      lastActive:
        newestTimestamp > 0 ? new Date(newestTimestamp).toISOString() : null,
      averageMessageCount:
        messageCountSamples > 0 ? messageCountTotal / messageCountSamples : 0
    }
  }, [characterChats])

  const averageConversationMessageCountLabel = React.useMemo(() => {
    const average = conversationInsights.averageMessageCount
    return Number.isInteger(average) ? String(average) : average.toFixed(1)
  }, [conversationInsights.averageMessageCount])

  return (
    <>
      {/* Full character-chat intent blocker */}
      <Modal
        title={t("settings:manageCharacters.chatIntent.title", {
          defaultValue: "Character chat setup"
        })}
        open={!!chatIntentBlocker}
        onCancel={() => {
          void clearChatIntentBlocker()
        }}
        destroyOnHidden
        width={560}
        rootClassName="characters-motion-modal"
        footer={[
          <Button
            key="return"
            onClick={() => {
              void clearChatIntentBlocker()
            }}>
            {t("settings:manageCharacters.chatIntent.return", {
              defaultValue: "Return to character"
            })}
          </Button>,
          <Button
            key="retry"
            onClick={() => {
              void retryChatIntentBlocker()
            }}>
            {t("settings:manageCharacters.chatIntent.retry", {
              defaultValue: "Retry character chat"
            })}
          </Button>,
          <Button
            key="settings"
            type="primary"
            href={CHARACTER_CHAT_MODEL_SETTINGS_PATH}>
            {chatIntentReadinessCopy?.actionLabel ||
              t("settings:manageCharacters.chatIntent.openModelSettings", {
                defaultValue: "Open model settings"
              })}
          </Button>
        ]}>
        {chatIntentReadinessCopy && (
          <DesignSystemAlert
            variant="warning"
            title={chatIntentReadinessCopy.title}
          >
            {chatIntentReadinessCopy.description}
          </DesignSystemAlert>
        )}
      </Modal>

      {/* Import Preview Modal */}
      <Modal
        title={t("settings:manageCharacters.import.previewTitle", {
          defaultValue: "Import preview"
        })}
        open={importPreviewOpen}
        onCancel={resetImportPreview}
        destroyOnHidden
        footer={
          importPreviewHasSuccessfulCompletion
            ? [
                retryableFailedPreviewItems.length > 0 ? (
                  <Button
                    key="retry-failed"
                    disabled={importPreviewProcessing}
                    onClick={() => {
                      void handleRetryFailedImportPreview()
                    }}>
                    {t("settings:manageCharacters.import.retryFailed", {
                      defaultValue: "Retry failed"
                    })}
                  </Button>
                ) : null,
                <Button
                  key="ok"
                  type="primary"
                  onClick={resetImportPreview}>
                  {t("settings:manageCharacters.import.dismissOk", {
                    defaultValue: "OK"
                  })}
                </Button>
              ].filter(Boolean)
            : [
                <Button
                  key="cancel"
                  disabled={importPreviewProcessing}
                  onClick={resetImportPreview}>
                  {t("common:cancel", { defaultValue: "Cancel" })}
                </Button>,
                retryableFailedPreviewItems.length > 0 ? (
                  <Button
                    key="retry-failed"
                    disabled={importPreviewProcessing}
                    onClick={() => {
                      void handleRetryFailedImportPreview()
                    }}>
                    {t("settings:manageCharacters.import.retryFailed", {
                      defaultValue: "Retry failed"
                    })}
                  </Button>
                ) : null,
                <Button
                  key="confirm"
                  type="primary"
                  loading={importPreviewProcessing || importing}
                  disabled={
                    importablePreviewItems.length === 0 ||
                    importPreviewLoading ||
                    importPreviewProcessing ||
                    importQueueSummary.complete
                  }
                  onClick={() => {
                    void handleConfirmImportPreview()
                  }}>
                  {t("settings:manageCharacters.import.confirmPreview", {
                    defaultValue: "Confirm import"
                  })}
                </Button>
              ].filter(Boolean)
        }
        rootClassName="characters-motion-modal">
        {importPreviewLoading ? (
          <Skeleton active paragraph={{ rows: 3 }} />
        ) : (
          <div className="space-y-3">
            {importQueueSummary.total > 0 && (
              <div
                className="rounded-md border border-border bg-surface2/40 p-2 text-xs text-text-subtle"
                data-testid="character-import-progress-summary">
                {t("settings:manageCharacters.import.progressSummary", {
                  defaultValue:
                    "Queued {{queued}} · Processing {{processing}} · Success {{success}} · Failed {{failed}}",
                  queued: importQueueSummary.queued,
                  processing: importQueueSummary.processing,
                  success: importQueueSummary.success,
                  failed: importQueueSummary.failure
                })}
              </div>
            )}
            {importPreviewItems.map((item) => {
              const queueItem = importQueueItemsById.get(item.id)
              const runtimeState =
                queueItem?.state || (item.parseError ? "failure" : "queued")
              const runtimeMessage = queueItem?.message
              return (
                <div
                  key={item.id}
                  className="rounded-md border border-border bg-surface2/40 p-3"
                  data-testid="character-import-preview-item">
                  <div className="flex items-start gap-3">
                    {item.avatarUrl ? (
                      <img
                        src={item.avatarUrl}
                        alt={item.name}
                        className="h-12 w-12 rounded-md object-cover border border-border"
                      />
                    ) : (
                      <div className="flex h-12 w-12 items-center justify-center rounded-md border border-border bg-surface">
                        <UserCircle2 className="h-6 w-6 text-text-muted" />
                      </div>
                    )}
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-sm text-text">
                          {item.name}
                        </span>
                        <span className="text-xs text-text-subtle uppercase">
                          {item.format}
                        </span>
                        <Tag
                          color={getImportQueueStateColor(runtimeState)}
                          className="m-0"
                          data-testid={`character-import-status-${runtimeState}`}>
                          {getImportQueueStateLabel(runtimeState)}
                        </Tag>
                      </div>
                      <div className="text-xs text-text-muted">{item.fileName}</div>
                      {item.description && (
                        <div className="text-sm text-text line-clamp-2">
                          {item.description}
                        </div>
                      )}
                      <div className="text-xs text-text-subtle">
                        {t("settings:manageCharacters.import.previewMeta", {
                          defaultValue: "Fields: {{fields}} · Tags: {{tags}}",
                          fields: item.fieldCount,
                          tags: item.tagCount
                        })}
                      </div>
                      {item.parseError && (
                        <div className="text-xs text-danger">
                          {String(t(item.parseError.key, {
                            defaultValue: item.parseError.fallback,
                            ...(item.parseError.values || {})
                          }))}
                        </div>
                      )}
                      {runtimeMessage && !item.parseError && (
                        <div
                          className={`text-xs ${
                            runtimeState === "failure"
                              ? "text-danger"
                              : runtimeState === "success"
                                ? "text-text"
                                : "text-text-subtle"
                          }`}>
                          {runtimeMessage}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Modal>

      {/* Quick Chat Modal */}
      <Modal
        title={t("settings:manageCharacters.quickChat.title", {
          defaultValue: "Quick chat: {{name}}",
          name:
            quickChatCharacter?.name ||
            quickChatCharacter?.title ||
            quickChatCharacter?.slug ||
            t("settings:manageCharacters.preview.untitled", {
              defaultValue: "Untitled character"
            })
        })}
        open={!!quickChatCharacter}
        onCancel={() => {
          void closeQuickChat()
        }}
        footer={null}
        destroyOnHidden
        width={560}
        rootClassName="characters-motion-modal">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted">
              {t("settings:manageCharacters.quickChat.modelLabel", {
                defaultValue: "Model"
              })}
            </span>
            <Select
              className="flex-1"
              size="small"
              showSearch
              placeholder={t("settings:manageCharacters.quickChat.modelPlaceholder", {
                defaultValue: "Select a model"
              })}
              options={quickChatModelOptions}
              value={activeQuickChatModel || undefined}
              optionFilterProp="label"
              onChange={(value) => setQuickChatModelOverride(value || null)}
              allowClear
            />
          </div>

          {showQuickChatModelBlocker && (
            <DesignSystemAlert
              variant="warning"
              title={quickChatReadinessCopy.title}
            >
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <span>{quickChatReadinessCopy.description}</span>
                <a
                  href={CHARACTER_CHAT_MODEL_SETTINGS_PATH}
                  className="shrink-0 text-primary hover:underline"
                >
                  {quickChatReadinessCopy.actionLabel}
                </a>
              </div>
            </DesignSystemAlert>
          )}

          {quickChatError && (
            <DesignSystemAlert
              variant="warning"
              title={quickChatError}
            />
          )}

          <div
            className="max-h-72 min-h-[12rem] overflow-y-auto rounded-md border border-border bg-surface2/40 p-3 space-y-2"
            role="log"
            aria-live="polite">
            {quickChatMessages.length === 0 ? (
              <div className="text-sm text-text-subtle">
                {t("settings:manageCharacters.quickChat.emptyState", {
                  defaultValue:
                    "Send a message to quickly test this character without leaving the page."
                })}
              </div>
            ) : (
              quickChatMessages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}>
                  <div
                    className={`max-w-[85%] whitespace-pre-wrap rounded-md px-3 py-2 text-sm ${
                      message.role === "user"
                        ? "bg-primary text-white"
                        : "bg-surface text-text border border-border"
                    }`}>
                    {message.content}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="flex gap-2">
            <Input.TextArea
              value={quickChatDraft}
              onChange={(event) => setQuickChatDraft(event.target.value)}
              placeholder={t("settings:manageCharacters.quickChat.placeholder", {
                defaultValue: "Ask this character a quick question..."
              })}
              autoSize={{ minRows: 1, maxRows: 4 }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault()
                  void sendQuickChatMessage()
                }
              }}
              disabled={quickChatSending}
            />
            <Button
              type="primary"
              loading={quickChatSending}
              onClick={() => {
                void sendQuickChatMessage()
              }}
              disabled={!quickChatDraft.trim() || !quickChatReadiness.canStart}>
              {t("settings:manageCharacters.quickChat.send", {
                defaultValue: "Send"
              })}
            </Button>
          </div>
          <div className="flex justify-end">
            <Button
              onClick={() => {
                void handlePromoteQuickChat()
              }}
              disabled={!quickChatCharacter || quickChatSending}>
              {t("settings:manageCharacters.quickChat.openFullChat", {
                defaultValue: "Open full chat"
              })}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Conversations Modal */}
      <Modal
        title={
          conversationCharacter
            ? t("settings:manageCharacters.conversations.title", {
                defaultValue: "Conversations for {{name}}",
                name:
                  conversationCharacter.name ||
                  conversationCharacter.title ||
                  conversationCharacter.slug ||
                  ""
              })
            : t("settings:manageCharacters.conversations.titleGeneric", {
                defaultValue: "Character conversations"
              })
        }
        open={conversationsOpen}
        onCancel={() => {
          setConversationsOpen(false)
          setConversationCharacter(null)
          setCharacterChats([])
          setChatsError(null)
          setResumingChatId(null)
        }}
        footer={null}
        destroyOnHidden
        rootClassName="characters-motion-modal">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3 text-xs text-text-subtle">
            <span>
              {t("settings:manageCharacters.conversations.lastActive", {
                defaultValue: "Last active: {{time}}",
                time: conversationInsights.lastActive
                  ? new Date(conversationInsights.lastActive).toLocaleString()
                  : t("settings:manageCharacters.conversations.unknownTime", {
                      defaultValue: "Unknown"
                    })
              })}
            </span>
            <span>
              {t("settings:manageCharacters.conversations.avgMessages", {
                defaultValue: "Avg messages: {{count}}",
                count: Number(averageConversationMessageCountLabel)
              })}
            </span>
          </div>
          <p className="text-sm text-text-muted">
            {t("settings:manageCharacters.conversations.subtitle", {
              defaultValue:
                "Select a conversation to continue as this character."
            })}
          </p>
          {chatsError && (
            <DesignSystemAlert
              variant="error"
              title={chatsError}
              action={{
                label: t("common:retry", { defaultValue: "Retry" }),
                onClick: () => {
                  void loadConversationChats()
                }
              }}
            />
          )}
          {loadingChats && <Skeleton active title paragraph={{ rows: 4 }} />}
          {!loadingChats && !chatsError && (
            <>
              {characterChats.length === 0 ? (
                <div className="rounded-md border border-dashed border-border bg-surface2 p-3 text-sm text-text-muted">
                  {t("settings:manageCharacters.conversations.empty", {
                    defaultValue: "No conversations found for this character yet."
                  })}
                </div>
              ) : (
                <div className="space-y-2">
                  {characterChats.map((chat, index) => (
                    <div
                      key={chat.id}
                      className={`flex items-start justify-between gap-3 rounded-md border p-3 shadow-sm ${
                        index === 0
                          ? "border-primary/30 bg-primary/10"
                          : "border-border bg-surface"
                      }`}>
                      <div className="min-w-0 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-text truncate">
                            {chat.title ||
                              t("settings:manageCharacters.conversations.untitled", {
                                defaultValue: "Untitled"
                              })}
                          </span>
                          {index === 0 && (
                            <span className="inline-flex items-center rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                              {t("settings:manageCharacters.conversations.mostRecent", {
                                defaultValue: "Most recent"
                              })}
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-text-subtle">
                          {formatUpdatedLabel(chat.updated_at || chat.created_at)}
                        </div>
                        <div className="flex flex-wrap items-center gap-2 text-xs text-text-subtle">
                          <span className="inline-flex items-center rounded-full bg-surface2 px-2 py-0.5 lowercase text-text-muted">
                            {(chat.state as string) || "in-progress"}
                          </span>
                          {chat.topic_label && (
                            <span
                              className="truncate max-w-[14rem]"
                              title={String(chat.topic_label)}
                            >
                              {String(chat.topic_label)}
                            </span>
                          )}
                        </div>
                      </div>
                      <Tooltip
                        title={t("settings:manageCharacters.conversations.resumeTooltip", {
                          defaultValue: "Load chat history and continue this conversation"
                        })}>
                        <Button
                          type="primary"
                          size="small"
                          loading={resumingChatId === chat.id}
                          onClick={async () => {
                          if (!conversationCharacter) return
                          setResumingChatId(chat.id)
                          try {
                            await tldwClient.initialize()

                            const assistantName =
                              conversationCharacter.name ||
                              conversationCharacter.title ||
                              conversationCharacter.slug ||
                              t("common:assistant", {
                                defaultValue: "Assistant"
                              })

                            const messages = await tldwClient.listChatMessages(
                              chat.id,
                              { include_deleted: "false" } as any
                            )
                            const history = messages.map((m) => ({
                              role: normalizeChatRole(m.role),
                              content: m.content
                            }))
                            const mappedMessages = messages.map((m) => {
                              const createdAt = Date.parse(m.created_at)
                              const normalized = normalizeChatRole(m.role)
                              return {
                                createdAt: Number.isNaN(createdAt)
                                  ? undefined
                                  : createdAt,
                                isBot: normalized === "assistant",
                                role: normalized,
                                name:
                                  normalized === "assistant"
                                    ? assistantName
                                    : normalized === "system"
                                      ? t("common:system", {
                                          defaultValue: "System"
                                        })
                                      : t("common:you", { defaultValue: "You" }),
                                message: m.content,
                                sources: [],
                                images: [],
                                serverMessageId: m.id,
                                serverMessageVersion: m.version
                              }
                            })

                            const id = characterIdentifierFn(conversationCharacter)
                            setSelectedCharacter({
                              id,
                              name:
                                conversationCharacter.name ||
                                conversationCharacter.title ||
                                conversationCharacter.slug,
                              system_prompt:
                                conversationCharacter.system_prompt ||
                                conversationCharacter.systemPrompt ||
                                conversationCharacter.instructions ||
                                "",
                              greeting:
                                conversationCharacter.greeting ||
                                conversationCharacter.first_message ||
                                conversationCharacter.greet ||
                                "",
                              avatar_url:
                                conversationCharacter.avatar_url ||
                                validateAndCreateImageDataUrl(
                                  conversationCharacter.image_base64
                                ) ||
                                ""
                            })

                            setHistoryId(null)
                            setServerChatId(chat.id)
                            setServerChatState(
                              (chat as any)?.state ??
                                (chat as any)?.conversation_state ??
                                "in-progress"
                            )
                            setServerChatTopic(
                              (chat as any)?.topic_label ?? null
                            )
                            setServerChatClusterId(
                              (chat as any)?.cluster_id ?? null
                            )
                            setServerChatSource(
                              (chat as any)?.source ?? null
                            )
                            setServerChatExternalRef(
                              (chat as any)?.external_ref ?? null
                            )
                            setHistory(history)
                            setMessages(mappedMessages)
                            updatePageTitle(chat.title)
                            setConversationsOpen(false)
                            setConversationCharacter(null)
                            navigate(buildCharacterChatPath({ characterId: id }))
                            setTimeout(() => {
                              focusComposer()
                            }, 0)
                          } catch (e) {
                            setChatsError(
                              t("settings:manageCharacters.conversations.error", {
                                defaultValue:
                                  "Unable to load conversations for this character."
                              })
                            )
                          } finally {
                            setResumingChatId(null)
                          }
                        }}>
                          {t("settings:manageCharacters.conversations.resume", {
                            defaultValue: "Continue chat"
                          })}
                        </Button>
                      </Tooltip>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </Modal>

      {/* Create Character Drawer */}
      <Drawer
        title={t("settings:manageCharacters.modal.addTitle", {
          defaultValue: "New character"
        })}
        open={open}
        onClose={() => {
          if (createFormDirty) {
            Modal.confirm({
              title: t("settings:manageCharacters.modal.unsavedTitle", {
                defaultValue: "Discard changes?"
              }),
              content: t("settings:manageCharacters.modal.unsavedContent", {
                defaultValue: "You have unsaved changes. Are you sure you want to close?"
              }),
              okText: t("common:discard", { defaultValue: "Discard" }),
              cancelText: t("common:cancel", { defaultValue: "Cancel" }),
              onOk: () => {
                setOpen(false)
                createForm.resetFields()
                setCreateFormDirty(false)
                setShowCreateAdvanced(false)
                setShowCreateSystemPromptExample(false)
                setTimeout(() => {
                  newButtonRef.current?.focus()
                }, 0)
              }
            })
          } else {
            setOpen(false)
            createForm.resetFields()
            setShowCreateAdvanced(false)
            setShowCreateSystemPromptExample(false)
            setTimeout(() => {
              newButtonRef.current?.focus()
            }, 0)
          }
        }}
        size={520}
        placement="right"
        rootClassName="characters-motion-modal">
        <p className="text-sm text-text-muted mb-4">
          {t("settings:manageCharacters.modal.description", {
            defaultValue: "Define a reusable character you can chat with in the sidebar."
          })}
        </p>

        {/* Draft Recovery Banner (H4) */}
        {hasCreateDraft && createDraftData && (
          <DesignSystemAlert
            variant="info"
            className="mb-4"
            title={
              <span>
                {t("settings:manageCharacters.draft.found", {
                  defaultValue: "Resume unsaved character?"
                })}
                {createDraftData.formData?.name && (
                  <strong className="ml-1">"{createDraftData.formData.name}"</strong>
                )}
                <span className="text-text-muted ml-1">
                  ({formatDraftAge(createDraftData.savedAt)})
                </span>
              </span>
            }
            action={{
              label: t("settings:manageCharacters.draft.restore", { defaultValue: "Restore" }),
              variant: "primary",
              onClick: () => {
                const draft = applyCreateDraft()
                if (draft) {
                  createForm.setFieldsValue({
                    ...draft,
                    tags: getCharacterVisibleTags(draft.tags),
                    folder_id:
                      normalizeCharacterFolderId(draft.folder_id) ??
                      getCharacterFolderIdFromTags(draft.tags)
                  })
                  setCreateFormDirty(true)
                  if (hasAdvancedData(draft, draft.extensions || '')) {
                    setShowCreateAdvanced(true)
                  }
                }
              }
            }}
            secondaryAction={{
              label: t("settings:manageCharacters.draft.discard", { defaultValue: "Discard" }),
              onClick: dismissCreateDraft
            }}
          />
        )}

        {/* Template Selection (M4) */}
        {!showTemplates ? (
          <div className="mb-4">
            <Button
              type="link"
              size="small"
              className="p-0"
              onClick={() => {
                setShowTemplates(true)
                markTemplateChooserSeen()
              }}>
              {t("settings:manageCharacters.templates.startFrom", {
                defaultValue: "Start from a template..."
              })}
            </Button>
          </div>
        ) : (
          <div className="mb-4 p-3 bg-surface rounded-lg border border-border">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">
                {t("settings:manageCharacters.templates.title", {
                  defaultValue: "Choose a template"
                })}
              </span>
              <Button
                type="link"
                size="small"
                onClick={() => {
                  setShowTemplates(false)
                  markTemplateChooserSeen()
                }}>
                {t("common:cancel", { defaultValue: "Cancel" })}
              </Button>
            </div>
            <div className="grid grid-cols-1 gap-2">
              {CHARACTER_TEMPLATES.map((template) => (
                <button
                  key={template.id}
                  type="button"
                  className="text-left p-2 rounded border border-border hover:border-primary hover:bg-surface-hover transition-colors motion-reduce:transition-none"
                  onClick={() => applyTemplateToCreateForm(template)}>
                  <div className="font-medium text-sm">{template.name}</div>
                  <div className="text-xs text-text-muted">{template.description}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* AI Character Generation Panel */}
        <GenerateCharacterPanel
          isGenerating={isGenerating && generatingField === 'all'}
          error={generationError}
          onGenerate={handleGenerateFullCharacter}
          onCancel={cancelGeneration}
          onClearError={clearGenerationError}
        />

        {renderSharedCharacterForm({
          form: createForm,
          mode: "create",
          initialValues: { prompt_preset: DEFAULT_CHARACTER_PROMPT_PRESET },
          worldBookFieldContext: {
            options: worldBookOptions,
            loading: worldBookOptionsLoading,
            editCharacterNumericId
          },
          isSubmitting: creating,
          submitButtonClassName: "mt-4",
          submitPendingLabel: t("settings:manageCharacters.form.btnSave.saving", {
            defaultValue: "Creating character..."
          }),
          submitIdleLabel: t("settings:manageCharacters.form.btnSave.save", {
            defaultValue: "Create character"
          }),
          showPreview: showCreatePreview,
          onTogglePreview: () => setShowCreatePreview((v) => !v),
          onValuesChange: (allValues) => {
            setCreateFormDirty(true)
            saveCreateDraft(allValues)
          },
          onFinish: (values) => {
            createCharacter(values)
            clearCreateDraft()
            setCreateFormDirty(false)
            setShowCreateAdvanced(false)
          }
        })}
      </Drawer>

      {/* Edit Character Drawer */}
      <Drawer
        title={t("settings:manageCharacters.modal.editTitle", {
          defaultValue: "Edit character"
        })}
        open={openEdit}
        onClose={() => {
          if (editFormDirty) {
            Modal.confirm({
              title: t("settings:manageCharacters.modal.unsavedTitle", {
                defaultValue: "Discard changes?"
              }),
              content: t("settings:manageCharacters.modal.unsavedContent", {
                defaultValue: "You have unsaved changes. Are you sure you want to close?"
              }),
              okText: t("common:discard", { defaultValue: "Discard" }),
              cancelText: t("common:cancel", { defaultValue: "Cancel" }),
              onOk: () => {
                setOpenEdit(false)
                editForm.resetFields()
                setEditId(null)
                setEditVersion(null)
                editWorldBooksInitializedRef.current = false
                setEditFormDirty(false)
                setShowEditSystemPromptExample(false)
                setTimeout(() => {
                  if (lastEditTriggerRef.current instanceof HTMLElement) {
                    lastEditTriggerRef.current.focus()
                  }
                }, 0)
              }
            })
          } else {
            setOpenEdit(false)
            editForm.resetFields()
            setEditId(null)
            setEditVersion(null)
            editWorldBooksInitializedRef.current = false
            setShowEditSystemPromptExample(false)
            setTimeout(() => {
              if (lastEditTriggerRef.current instanceof HTMLElement) {
                lastEditTriggerRef.current.focus()
              }
            }, 0)
          }
        }}
        size={520}
        placement="right"
        rootClassName="characters-motion-modal">
        <p className="text-sm text-text-muted mb-4">
          {t("settings:manageCharacters.modal.editDescription", {
            defaultValue: "Update the character's name, behavior, and other settings."
          })}
        </p>
        {!!editId && (
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface2/40 px-3 py-2">
            <span className="text-xs text-text-muted">
              {t("settings:manageCharacters.versionHistory.inlineHint", {
                defaultValue:
                  "Inspect revision metadata, compare fields, or restore an earlier version."
              })}
            </span>
            <Button
              size="small"
              icon={<Clock3 className="w-4 h-4" />}
              onClick={() => {
                const existingRecord =
                  (Array.isArray(data) ? data : []).find((character: any) => {
                    const candidate = String(
                      character?.id || character?.slug || character?.name || ""
                    )
                    return candidate === String(editId)
                  }) ?? {
                    id: editId,
                    name: editForm.getFieldValue("name")
                  }
                openVersionHistory(existingRecord)
              }}>
              {t("settings:manageCharacters.actions.versionHistory", {
                defaultValue: "Version history"
              })}
            </Button>
          </div>
        )}

        {/* Draft Recovery Banner for Edit Form (H4) */}
        {hasEditDraft && editDraftData && (
          <DesignSystemAlert
            variant="info"
            className="mb-4"
            title={
              <span>
                {t("settings:manageCharacters.draft.resumeEdit", {
                  defaultValue: "Resume unsaved changes?"
                })}
                <span className="text-text-muted ml-1">
                  ({formatDraftAge(editDraftData.savedAt)})
                </span>
              </span>
            }
            action={{
              label: t("settings:manageCharacters.draft.restore", { defaultValue: "Restore" }),
              variant: "primary",
              onClick: () => {
                const draft = applyEditDraft()
                if (draft) {
                  editForm.setFieldsValue({
                    ...draft,
                    tags: getCharacterVisibleTags(draft.tags),
                    folder_id:
                      normalizeCharacterFolderId(draft.folder_id) ??
                      getCharacterFolderIdFromTags(draft.tags)
                  })
                  setEditFormDirty(true)
                  if (hasAdvancedData(draft, draft.extensions || '')) {
                    setShowEditAdvanced(true)
                  }
                }
              }
            }}
            secondaryAction={{
              label: t("settings:manageCharacters.draft.discard", { defaultValue: "Discard" }),
              onClick: dismissEditDraft
            }}
          />
        )}

        {renderSharedCharacterForm({
          form: editForm,
          mode: "edit",
          worldBookFieldContext: {
            options: worldBookOptions,
            loading: worldBookOptionsLoading,
            editCharacterNumericId
          },
          isSubmitting: updating,
          submitButtonClassName: "w-full",
          submitPendingLabel: t("settings:manageCharacters.form.btnEdit.saving", {
            defaultValue: "Saving changes..."
          }),
          submitIdleLabel: t("settings:manageCharacters.form.btnEdit.save", {
            defaultValue: "Save changes"
          }),
          showPreview: showEditPreview,
          onTogglePreview: () => setShowEditPreview((v) => !v),
          onValuesChange: (allValues) => {
            setEditFormDirty(true)
            saveEditDraft(allValues)
          },
          onFinish: (values) => {
            updateCharacter(values)
            clearEditDraft()
            setEditFormDirty(false)
          }
        })}
      </Drawer>

      {/* Generation Preview Modal */}
      <GenerationPreviewModal
        open={generationPreviewOpen}
        generatedData={generationPreviewData}
        fieldName={generationPreviewField}
        onApply={applyGenerationPreview}
        onCancel={() => {
          setGenerationPreviewOpen(false)
          setGenerationPreviewData(null)
          setGenerationPreviewField(null)
        }}
      />

      {/* Version History Modal */}
      <Modal
        title={t("settings:manageCharacters.versionHistory.title", {
          defaultValue: "Version history: {{name}}",
          name: versionHistoryCharacterName
        })}
        open={versionHistoryOpen}
        onCancel={() => {
          setVersionHistoryOpen(false)
          setVersionHistoryCharacter(null)
          setVersionFrom(null)
          setVersionTo(null)
          setVersionRevertTarget(null)
        }}
        footer={null}
        width={920}
        destroyOnHidden
        rootClassName="characters-motion-modal">
        <div className="space-y-4">
          <p className="text-sm text-text-muted">
            {t("settings:manageCharacters.versionHistory.description", {
              defaultValue:
                "Review revision metadata, compare field-level changes, and restore earlier versions safely."
            })}
          </p>

          {(versionHistoryLoading || versionHistoryFetching) && (
            <Skeleton active paragraph={{ rows: 6 }} />
          )}

          {!versionHistoryLoading &&
            !versionHistoryFetching &&
            versionHistoryItems.length === 0 && (
              <DesignSystemAlert
                variant="info"
                title={t("settings:manageCharacters.versionHistory.empty", {
                  defaultValue: "No version snapshots available yet."
                })}
              />
            )}

          {!versionHistoryLoading &&
            !versionHistoryFetching &&
            versionHistoryItems.length > 0 && (
              <div className="grid gap-4 md:grid-cols-[280px_minmax(0,1fr)]">
                <div className="space-y-2">
                  <div className="text-xs font-medium uppercase tracking-wide text-text-subtle">
                    {t("settings:manageCharacters.versionHistory.timeline", {
                      defaultValue: "Timeline"
                    })}
                  </div>
                  <div className="max-h-[26rem] overflow-y-auto rounded-md border border-border bg-surface2/40 p-2">
                    <div className="space-y-2">
                      {versionHistoryItems.map((entry) => {
                        const isFrom = entry.version === versionFrom
                        const isTo = entry.version === versionTo
                        const timestampLabel = entry.timestamp
                          ? new Date(entry.timestamp).toLocaleString()
                          : t(
                              "settings:manageCharacters.versionHistory.unknownTimestamp",
                              { defaultValue: "Unknown time" }
                            )
                        return (
                          <button
                            key={`${entry.change_id}-${entry.version}`}
                            type="button"
                            className={`w-full rounded-md border px-3 py-2 text-left transition motion-reduce:transition-none ${
                              isTo
                                ? "border-primary bg-primary/10"
                                : isFrom
                                  ? "border-warning/40 bg-warning/10"
                                  : "border-border bg-surface hover:border-primary/40"
                            }`}
                            onClick={() => {
                              setVersionTo(entry.version)
                            }}>
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-sm font-medium text-text">
                                {`v${entry.version}`}
                              </span>
                              <span className="text-xs uppercase text-text-subtle">
                                {entry.operation}
                              </span>
                            </div>
                            <div className="mt-1 text-xs text-text-muted">
                              {timestampLabel}
                            </div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {isFrom && (
                                <Tag color="gold">
                                  {t(
                                    "settings:manageCharacters.versionHistory.baseBadge",
                                    {
                                      defaultValue: "Base"
                                    }
                                  )}
                                </Tag>
                              )}
                              {isTo && (
                                <Tag color="blue">
                                  {t(
                                    "settings:manageCharacters.versionHistory.compareBadge",
                                    {
                                      defaultValue: "Compare"
                                    }
                                  )}
                                </Tag>
                              )}
                            </div>
                            {entry.client_id && (
                              <div className="mt-1 text-[11px] text-text-subtle">
                                {t(
                                  "settings:manageCharacters.versionHistory.clientId",
                                  {
                                    defaultValue: "Client: {{clientId}}",
                                    clientId: entry.client_id
                                  }
                                )}
                              </div>
                            )}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div>
                      <div className="mb-1 text-xs text-text-subtle">
                        {t("settings:manageCharacters.versionHistory.baseVersion", {
                          defaultValue: "Base version"
                        })}
                      </div>
                      <Select
                        className="w-full"
                        value={versionFrom ?? undefined}
                        options={versionSelectOptions}
                        onChange={(value) => setVersionFrom(Number(value))}
                      />
                    </div>
                    <div>
                      <div className="mb-1 text-xs text-text-subtle">
                        {t(
                          "settings:manageCharacters.versionHistory.compareVersion",
                          {
                            defaultValue: "Compare version"
                          }
                        )}
                      </div>
                      <Select
                        className="w-full"
                        value={versionTo ?? undefined}
                        options={versionSelectOptions}
                        onChange={(value) => setVersionTo(Number(value))}
                      />
                    </div>
                  </div>

                  {versionFrom === versionTo && (
                    <DesignSystemAlert
                      variant="info"
                      title={t("settings:manageCharacters.versionHistory.sameVersion", {
                        defaultValue: "Select two different versions to view a diff."
                      })}
                    />
                  )}

                  {versionFrom !== versionTo &&
                    (versionDiffLoading || versionDiffFetching) && (
                      <Skeleton active paragraph={{ rows: 5 }} />
                    )}

                  {versionFrom !== versionTo &&
                    !versionDiffLoading &&
                    !versionDiffFetching &&
                    !versionDiffResponse && (
                      <DesignSystemAlert
                        variant="warning"
                        title={t(
                          "settings:manageCharacters.versionHistory.diffUnavailable",
                          {
                            defaultValue:
                              "Unable to load version diff. Try selecting different revisions."
                          }
                        )}
                      />
                    )}

                  {versionFrom !== versionTo &&
                    !versionDiffLoading &&
                    !versionDiffFetching &&
                    versionDiffResponse && (
                      <div className="space-y-2 rounded-md border border-border bg-surface2/40 p-3">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-sm font-medium text-text">
                            {t("settings:manageCharacters.versionHistory.diffTitle", {
                              defaultValue: "Differences: v{{from}} -> v{{to}}",
                              from: versionFrom,
                              to: versionTo
                            })}
                          </span>
                          <span className="text-xs text-text-subtle">
                            {t("settings:manageCharacters.versionHistory.diffCount", {
                              defaultValue: "{{count}} fields changed",
                              count: versionDiffResponse.changed_count
                            })}
                          </span>
                        </div>
                        {versionDiffResponse.changed_fields.length === 0 ? (
                          <div className="text-sm text-text-muted">
                            {t(
                              "settings:manageCharacters.versionHistory.noFieldChanges",
                              {
                                defaultValue:
                                  "No tracked field changes were found between these versions."
                              }
                            )}
                          </div>
                        ) : (
                          <div className="max-h-[16rem] space-y-2 overflow-y-auto pr-1">
                            {versionDiffResponse.changed_fields.map((diffField: any) => {
                              const fieldKey =
                                diffField.field as (typeof CHARACTER_VERSION_DIFF_FIELD_KEYS)[number]
                              const fieldLabel =
                                CHARACTER_VERSION_FIELD_LABELS[fieldKey] || diffField.field
                              const hasValueChange =
                                normalizeVersionSnapshotValue(diffField.old_value) !==
                                normalizeVersionSnapshotValue(diffField.new_value)
                              if (!hasValueChange) return null
                              return (
                                <div
                                  key={`${diffField.field}-${versionFrom}-${versionTo}`}
                                  className="rounded-md border border-border bg-surface p-2">
                                  <div className="mb-2 text-sm font-medium text-text">
                                    {fieldLabel}
                                  </div>
                                  <div className="grid gap-2 sm:grid-cols-2">
                                    <div>
                                      <div className="mb-1 text-xs uppercase text-text-subtle">
                                        {t(
                                          "settings:manageCharacters.versionHistory.beforeLabel",
                                          { defaultValue: "Before" }
                                        )}
                                      </div>
                                      <pre className="max-h-24 overflow-auto whitespace-pre-wrap rounded bg-surface2 px-2 py-1 text-xs text-text-muted">
                                        {formatVersionSnapshotValue(diffField.old_value)}
                                      </pre>
                                    </div>
                                    <div>
                                      <div className="mb-1 text-xs uppercase text-text-subtle">
                                        {t(
                                          "settings:manageCharacters.versionHistory.afterLabel",
                                          { defaultValue: "After" }
                                        )}
                                      </div>
                                      <pre className="max-h-24 overflow-auto whitespace-pre-wrap rounded bg-surface2 px-2 py-1 text-xs text-text-muted">
                                        {formatVersionSnapshotValue(diffField.new_value)}
                                      </pre>
                                    </div>
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    )}

                  <div className="space-y-2 rounded-md border border-border bg-surface2/40 p-3">
                    <div className="text-sm font-medium text-text">
                      {t("settings:manageCharacters.versionHistory.restoreTitle", {
                        defaultValue: "Restore a previous version"
                      })}
                    </div>
                    <div className="flex flex-wrap items-end gap-2">
                      <div className="min-w-[180px] flex-1">
                        <div className="mb-1 text-xs text-text-subtle">
                          {t(
                            "settings:manageCharacters.versionHistory.restoreVersion",
                            { defaultValue: "Version to restore" }
                          )}
                        </div>
                        <Select
                          className="w-full"
                          value={versionRevertTarget ?? undefined}
                          options={versionSelectOptions}
                          onChange={(value) =>
                            setVersionRevertTarget(Number(value))
                          }
                        />
                      </div>
                      <Button
                        danger
                        type="primary"
                        loading={revertingCharacterVersion}
                        disabled={
                          versionHistoryCharacterId == null ||
                          versionRevertTarget == null
                        }
                        onClick={async () => {
                          if (
                            versionHistoryCharacterId == null ||
                            versionRevertTarget == null
                          ) {
                            return
                          }
                          const ok = await confirmDanger({
                            title: t("common:confirmTitle", {
                              defaultValue: "Please confirm"
                            }),
                            content: t(
                              "settings:manageCharacters.versionHistory.revertConfirm",
                              {
                                defaultValue:
                                  "Restore version {{version}}? This creates a new latest version.",
                                version: versionRevertTarget
                              }
                            ),
                            okText: t(
                              "settings:manageCharacters.versionHistory.revertAction",
                              {
                                defaultValue: "Revert to selected version"
                              }
                            ),
                            cancelText: t("common:cancel", {
                              defaultValue: "Cancel"
                            })
                          })
                          if (!ok) return
                          revertCharacterVersion({
                            characterId: versionHistoryCharacterId,
                            targetVersion: versionRevertTarget
                          })
                        }}>
                        {t("settings:manageCharacters.versionHistory.revertAction", {
                          defaultValue: "Revert to selected version"
                        })}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}
        </div>
      </Modal>

      {/* Compare Modal */}
      <Modal
        title={
          <h2 className="m-0 text-base font-semibold text-text">
            {t("settings:manageCharacters.compare.title", {
              defaultValue: "Compare characters"
            })}
          </h2>
        }
        open={compareModalOpen}
        onCancel={closeCompareModal}
        width={960}
        rootClassName="characters-motion-modal"
        footer={[
          <Button
            key="copy"
            icon={<Copy className="h-4 w-4" />}
            onClick={() => {
              void handleCopyComparisonSummary()
            }}
            disabled={!compareCharacters}>
            {t("settings:manageCharacters.compare.copySummary", {
              defaultValue: "Copy summary"
            })}
          </Button>,
          <Button
            key="export"
            icon={<Download className="h-4 w-4" />}
            onClick={handleExportComparisonSummary}
            disabled={!compareCharacters}>
            {t("settings:manageCharacters.compare.exportSummary", {
              defaultValue: "Export summary"
            })}
          </Button>,
          <Button key="close" onClick={closeCompareModal}>
            {t("common:close", { defaultValue: "Close" })}
          </Button>
        ]}>
        {!compareCharacters ? (
          <Skeleton active paragraph={{ rows: 6 }} />
        ) : (
          <div className="space-y-4">
            <div className="grid gap-2 sm:grid-cols-2">
              {compareCharacters.map((character, idx) => (
                <div
                  key={String(character?.id || character?.name || idx)}
                  className="rounded-md border border-border bg-surface2 p-3">
                  <div className="text-xs uppercase text-text-subtle">
                    {idx === 0
                      ? t("settings:manageCharacters.compare.left", {
                          defaultValue: "Left character"
                        })
                      : t("settings:manageCharacters.compare.right", {
                          defaultValue: "Right character"
                        })}
                  </div>
                  <div className="text-sm font-semibold text-text">
                    {String(character?.name || character?.id || "Untitled")}
                  </div>
                </div>
              ))}
            </div>

            <DesignSystemAlert
              variant="info"
              title={t("settings:manageCharacters.compare.changedCount", {
                defaultValue: "{{changed}} of {{total}} tracked fields differ",
                changed: changedComparisonRows.length,
                total: comparisonRows.length
              })}
            />

            <div className="grid grid-cols-[140px_1fr_1fr] gap-2 px-1 text-xs font-medium uppercase tracking-wide text-text-subtle">
              <span>
                {t("settings:manageCharacters.compare.fieldColumn", {
                  defaultValue: "Field"
                })}
              </span>
              <span>
                {t("settings:manageCharacters.compare.leftColumn", {
                  defaultValue: "Left"
                })}
              </span>
              <span>
                {t("settings:manageCharacters.compare.rightColumn", {
                  defaultValue: "Right"
                })}
              </span>
            </div>

            <div className="max-h-[55vh] overflow-y-auto rounded-md border border-border">
              {comparisonRows.map((row) => (
                <div
                  key={row.field}
                  className={`grid grid-cols-[140px_1fr_1fr] gap-2 border-b border-border p-2 text-sm last:border-b-0 ${
                    row.different ? "bg-primary/5" : "bg-surface"
                  }`}>
                  <div className="font-medium text-text-muted">{row.label}</div>
                  <pre className="whitespace-pre-wrap break-words font-sans text-text">
                    {row.leftValue}
                  </pre>
                  <pre className="whitespace-pre-wrap break-words font-sans text-text">
                    {row.rightValue}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}
      </Modal>

      {/* Tag Manager Modal */}
      <Modal
        title={t("settings:manageCharacters.tags.manageTitle", {
          defaultValue: "Manage tags"
        })}
        open={tagManagerOpen}
        onCancel={closeTagManager}
        onOk={() => {
          void handleApplyTagManagerOperation()
        }}
        okText={t("settings:manageCharacters.tags.applyOperation", {
          defaultValue: "Apply"
        })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={tagManagerSubmitting}
        rootClassName="characters-motion-modal"
        okButtonProps={{
          disabled:
            tagManagerLoading ||
            !tagManagerSourceTag ||
            ((tagManagerOperation === "rename" || tagManagerOperation === "merge") &&
              tagManagerTargetTag.trim().length === 0)
        }}>
        <div className="space-y-4">
          <p className="text-sm text-text-muted">
            {t("settings:manageCharacters.tags.manageDescription", {
              defaultValue:
                "Rename, merge, or delete tags across your character library."
            })}
          </p>
          <Segmented
            value={tagManagerOperation}
            onChange={(value) =>
              setTagManagerOperation(value as CharacterTagOperation)
            }
            options={[
              {
                value: "rename",
                label: t("settings:manageCharacters.tags.operation.rename", {
                  defaultValue: "Rename"
                })
              },
              {
                value: "merge",
                label: t("settings:manageCharacters.tags.operation.merge", {
                  defaultValue: "Merge"
                })
              },
              {
                value: "delete",
                label: t("settings:manageCharacters.tags.operation.delete", {
                  defaultValue: "Delete"
                })
              }
            ]}
          />
          <Select
            allowClear
            showSearch
            className="w-full"
            loading={tagManagerLoading}
            placeholder={t("settings:manageCharacters.tags.sourcePlaceholder", {
              defaultValue: "Select source tag"
            })}
            value={tagManagerSourceTag}
            options={tagManagerTagUsageData.map(({ tag, count }) => ({
              value: tag,
              label: `${tag} (${count})`
            }))}
            onChange={(value) => setTagManagerSourceTag(value || undefined)}
            filterOption={(input, option) =>
              option?.value?.toString().toLowerCase().includes(input.toLowerCase()) ?? false
            }
          />
          {(tagManagerOperation === "rename" || tagManagerOperation === "merge") && (
            <Input
              value={tagManagerTargetTag}
              onChange={(event) => setTagManagerTargetTag(event.target.value)}
              placeholder={t("settings:manageCharacters.tags.targetPlaceholder", {
                defaultValue: "Destination tag"
              })}
            />
          )}
          {tagManagerLoading ? (
            <Skeleton active paragraph={{ rows: 4 }} />
          ) : (
            <div className="max-h-60 overflow-y-auto rounded-md border border-border">
              {tagManagerTagUsageData.length === 0 ? (
                <div className="p-3 text-sm text-text-muted">
                  {t("settings:manageCharacters.tags.none", {
                    defaultValue: "No tags found."
                  })}
                </div>
              ) : (
                <ul className="divide-y divide-border">
                  {tagManagerTagUsageData.map(({ tag, count }) => (
                    <li
                      key={tag}
                      className="flex items-center justify-between px-3 py-2 text-sm">
                      <span className="font-medium">{tag}</span>
                      <span className="text-text-muted">{count}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </Modal>

      {/* Bulk Add Tags Modal (M5) */}
      <Modal
        title={t("settings:manageCharacters.bulk.addTagsTitle", {
          defaultValue: "Add tags to {{count}} characters",
          count: selectedCount
        })}
        open={bulkTagModalOpen}
        onCancel={() => {
          setBulkTagModalOpen(false)
          setBulkTagsToAdd([])
        }}
        onOk={handleBulkAddTagsForSelection}
        okText={t("settings:manageCharacters.bulk.addTagsConfirm", { defaultValue: "Add tags" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={bulkOperationLoading}
        rootClassName="characters-motion-modal"
        okButtonProps={{ disabled: bulkTagsToAdd.length === 0 }}>
        <div className="space-y-4">
          <p className="text-sm text-text-muted">
            {t("settings:manageCharacters.bulk.addTagsDescription", {
              defaultValue: "Select tags to add to all selected characters. Existing tags will be preserved."
            })}
          </p>
          {/* Popular tags suggestion chips */}
          {popularTags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              <span className="text-xs text-text-subtle mr-1">
                {t("settings:manageCharacters.tags.popular", { defaultValue: "Popular:" })}
              </span>
              {popularTags.map(({ tag, count }) => {
                const isSelected = bulkTagsToAdd.includes(tag)
                return (
                  <button
                    key={tag}
                    type="button"
                    className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full border transition-colors motion-reduce:transition-none ${
                      isSelected
                        ? 'bg-primary/10 border-primary text-primary'
                        : 'bg-surface border-border text-text-muted hover:border-primary/50 hover:text-primary'
                    }`}
                    onClick={() => {
                      if (isSelected) {
                        setBulkTagsToAdd(bulkTagsToAdd.filter((t) => t !== tag))
                      } else {
                        setBulkTagsToAdd([...bulkTagsToAdd, tag])
                      }
                    }}>
                    {tag}
                    <span className="text-text-subtle">({count})</span>
                  </button>
                )
              })}
            </div>
          )}
          <Select
            mode="tags"
            allowClear
            className="w-full"
            placeholder={t("settings:manageCharacters.bulk.selectTags", {
              defaultValue: "Select or type tags to add"
            })}
            value={bulkTagsToAdd}
            onChange={(values) => setBulkTagsToAdd(values)}
            options={tagOptionsWithCounts}
            filterOption={(input, option) =>
              option?.value?.toString().toLowerCase().includes(input.toLowerCase()) ?? false
            }
          />
        </div>
      </Modal>
    </>
  )
}
