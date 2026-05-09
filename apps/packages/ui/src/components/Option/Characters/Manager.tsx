import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Button,
  Form,
  Upload
} from "antd"
import type { InputRef, FormInstance } from "antd"
import React from "react"
import { fetchChatModels } from "@/services/tldw-server"
import {
  useCharacterFiltering,
  useCharacterInlineEdit,
  useCharacterQuickChat,
  useCharacterVersionHistory,
  useCharacterImportQueue,
  useCharacterTagManagement,
  useCharacterModalState,
  useCharacterData,
  useCharacterCrud,
  useCharacterBulkOps
} from "./hooks"
import type { CharacterChatIntentBlocker } from "./hooks/useCharacterCrud"
import { useCharacterGeneration } from "@/hooks/useCharacterGeneration"
import { useFormDraft } from "@/hooks/useFormDraft"
import { useCharacterShortcuts } from "@/hooks/useCharacterShortcuts"
import type { CharacterTemplate } from "@/data/character-templates"
import type { GeneratedCharacter, CharacterField } from "@/services/character-generation"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { useNavigate } from "react-router-dom"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import {
  DEFAULT_CHARACTER_STORAGE_KEY,
  defaultCharacterStorage,
  resolveCharacterSelectionId
} from "@/utils/default-character-preference"
import {
  IMPORT_UPLOAD_ACCEPT,
  TEMPLATE_CHOOSER_SEEN_KEY,
  parseCharacterImportPreview,
  type CharacterWorldBookOption,
} from "./utils"
export { withCharacterNameInLabel } from "./utils"

// --- Extracted sub-components ---
import { CharacterListToolbar } from "./CharacterListToolbar"
import { CharacterListContent } from "./CharacterListContent"

const loadCharacterDialogs = () =>
  import("./CharacterDialogs").then((module) => ({
    default: module.CharacterDialogs,
  }))
const LazyCharacterDialogs = React.lazy(() => loadCharacterDialogs())
const loadCharacterEditorForm = () =>
  import("./CharacterEditorForm").then((module) => ({
    default: module.CharacterEditorForm,
  }))
const LazyCharacterEditorForm = React.lazy(() =>
  loadCharacterEditorForm(),
)

type CharactersManagerProps = {
  forwardedNewButtonRef?: React.RefObject<HTMLButtonElement | null>
  autoOpenCreate?: boolean
}

type SharedCharacterFormProps = {
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
}

export const CharactersManager: React.FC<CharactersManagerProps> = ({
  forwardedNewButtonRef,
  autoOpenCreate = false
}) => {
  const { t } = useTranslation(["settings", "common"])
  const qc = useQueryClient()
  const navigate = useNavigate()
  const notification = useAntdNotification()
  const confirmDanger = useConfirmDanger()
  const [createForm] = Form.useForm()
  const [editForm] = Form.useForm()
  const [, setSelectedCharacter] = useSelectedCharacter<any>(null)
  const [defaultCharacterSelection, setDefaultCharacterSelection] =
    useStorage<any | null>(
      {
        key: DEFAULT_CHARACTER_STORAGE_KEY,
        instance: defaultCharacterStorage
      },
      null
    )
  const createNameRef = React.useRef<InputRef>(null)
  const editNameRef = React.useRef<InputRef>(null)
  const hasPreloadedCharacterEditorRef = React.useRef(false)

  // --- Extracted hooks ---
  const filtering = useCharacterFiltering({ t })
  const {
    searchInputRef,
    searchTerm, setSearchTerm,
    debouncedSearchTerm,
    filterTags, setFilterTags,
    folderFilterId, setFolderFilterId,
    matchAllTags, setMatchAllTags,
    creatorFilter, setCreatorFilter,
    createdFromDate, setCreatedFromDate,
    createdToDate, setCreatedToDate,
    updatedFromDate, setUpdatedFromDate,
    updatedToDate, setUpdatedToDate,
    hasConversationsOnly, setHasConversationsOnly,
    favoritesOnly, setFavoritesOnly,
    advancedFiltersOpen, setAdvancedFiltersOpen,
    characterListScope, setCharacterListScope,
    sortColumn, setSortColumn,
    sortOrder, setSortOrder,
    currentPage, setCurrentPage,
    pageSize, setPageSize,
    hasFilters,
    activeAdvancedFilterCount,
    clearFilters
  } = filtering

  const modalState = useCharacterModalState({ t, characterListScope })
  const {
    open, setOpen,
    openEdit, setOpenEdit,
    editId, setEditId,
    editVersion, setEditVersion,
    editCharacterNumericId,
    conversationsOpen, setConversationsOpen,
    conversationCharacter, setConversationCharacter,
    previewCharacter, setPreviewCharacter,
    compareModalOpen, setCompareModalOpen,
    compareCharacters, setCompareCharacters,
    closeCompareModal,
    createFormDirty, setCreateFormDirty,
    editFormDirty, setEditFormDirty,
    markModeDirty,
    showCreateSystemPromptExample, setShowCreateSystemPromptExample,
    showEditSystemPromptExample, setShowEditSystemPromptExample,
    showCreatePreview, setShowCreatePreview,
    showEditPreview, setShowEditPreview,
    showEditAdvanced, setShowEditAdvanced,
    showCreateAdvanced, setShowCreateAdvanced,
    createAdvancedSections, setCreateAdvancedSections,
    editAdvancedSections, setEditAdvancedSections,
    viewMode, setViewMode,
    galleryDensity, setGalleryDensity,
    tableDensity, setTableDensity,
    generationPreviewOpen, setGenerationPreviewOpen,
    generationTargetForm, setGenerationTargetForm,
    exporting, setExporting,
    newButtonRef, lastEditTriggerRef,
    editWorldBooksInitializedRef, autoOpenCreateHandledRef
  } = modalState

  const importQueue = useCharacterImportQueue({
    t,
    notification,
    qc,
    parseCharacterImportPreview
  })
  const {
    importButtonContainerRef,
    importing,
    importPreviewOpen, setImportPreviewOpen,
    importPreviewLoading,
    importPreviewItems,
    importPreviewProcessing,
    importQueueState, dispatchImportQueue,
    importablePreviewItems,
    importQueueItemsById,
    importQueueSummary,
    retryableFailedPreviewItems,
    importPreviewHasSuccessfulCompletion,
    isImportBusy,
    importCharacterFile,
    runBatchImport,
    openImportPreviewForBatch,
    handleConfirmImportPreview,
    handleRetryFailedImportPreview,
    handleImportUpload,
    handleImportDragEnter,
    handleImportDragLeave,
    handleImportDragOver,
    handleImportDrop,
    triggerImportPicker,
    resetImportPreview,
    getImportQueueStateLabel,
    getImportQueueStateColor
  } = importQueue

  // crossNavigationContext, previewCharacterId, selectedCharacterIds moved to useCharacterData / useCharacterBulkOps

  // Character generation state
  const {
    isGenerating,
    generatingField,
    error: generationError,
    generateFullCharacter,
    generateField,
    cancel: cancelGeneration,
    clearError: clearGenerationError
  } = useCharacterGeneration()
  const [selectedGenModel] = useStorage<string | null>("characterGenModel", null)
  const [selectedChatModel] = useStorage<string | null>("selectedModel", null)
  // serverQueryRolloutFlag moved to useCharacterData

  // Fetch models to get provider for selected model
  const { data: generationModels } = useQuery<Array<{ model: string; provider?: string }>>({
    queryKey: ["getModelsForFieldGeneration"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    staleTime: 5 * 60 * 1000 // 5 minutes
  })

  // Get the provider for the selected generation model
  const selectedGenModelProvider = React.useMemo(() => {
    if (!selectedGenModel || !generationModels) return undefined
    const modelData = generationModels.find((m) => m.model === selectedGenModel)
    return modelData?.provider
  }, [selectedGenModel, generationModels])

  const quickChatModelOptions = React.useMemo(
    () =>
      (generationModels || [])
        .filter((model) => typeof model.model === "string" && model.model.trim())
        .map((model) => {
          const modelName = model.model.trim()
          return {
            value: modelName,
            label: model.provider
              ? `${modelName} (${model.provider})`
              : modelName
          }
        }),
    [generationModels]
  )

  // Quick chat model override kept here since activeQuickChatModel depends on it
  const [quickChatModelOverride, setQuickChatModelOverride] = React.useState<string | null>(null)

  const activeQuickChatModel =
    quickChatModelOverride ||
    selectedChatModel ||
    quickChatModelOptions[0]?.value ||
    null

  const quickChat = useCharacterQuickChat({ t, activeQuickChatModel })
  const {
    quickChatCharacter, setQuickChatCharacter,
    quickChatMessages, setQuickChatMessages,
    quickChatDraft, setQuickChatDraft,
    quickChatSessionId, setQuickChatSessionId,
    quickChatSending,
    quickChatError, setQuickChatError,
    openQuickChat,
    closeQuickChat,
    sendQuickChatMessage,
    handlePromoteQuickChat
  } = quickChat

  const [chatIntentBlocker, setChatIntentBlocker] =
    React.useState<CharacterChatIntentBlocker | null>(null)
  const clearChatIntentBlocker = React.useCallback(async () => {
    setChatIntentBlocker(null)
    await setSelectedCharacter(null)
  }, [setSelectedCharacter])

  const versionHistory = useCharacterVersionHistory({ t, notification, qc, confirmDanger })
  const {
    versionHistoryOpen, setVersionHistoryOpen,
    versionHistoryCharacter, setVersionHistoryCharacter,
    versionHistoryCharacterId,
    versionHistoryCharacterName,
    versionFrom, setVersionFrom,
    versionTo, setVersionTo,
    versionRevertTarget, setVersionRevertTarget,
    versionHistoryItems,
    versionHistoryLoading,
    versionHistoryFetching,
    versionSelectOptions,
    versionDiffResponse,
    versionDiffLoading,
    versionDiffFetching,
    revertingCharacterVersion,
    openVersionHistory,
    revertCharacterVersion
  } = versionHistory

  const tagManagement = useCharacterTagManagement({ t, notification, qc, confirmDanger })
  const {
    tagManagerOpen, setTagManagerOpen,
    tagManagerLoading,
    tagManagerSubmitting,
    tagManagerCharacters,
    tagManagerOperation, setTagManagerOperation,
    tagManagerSourceTag, setTagManagerSourceTag,
    tagManagerTargetTag, setTagManagerTargetTag,
    bulkTagModalOpen, setBulkTagModalOpen,
    bulkTagsToAdd, setBulkTagsToAdd,
    bulkOperationLoading, setBulkOperationLoading,
    tagManagerTagUsageData,
    openTagManager,
    closeTagManager,
    handleApplyTagManagerOperation,
    handleBulkAddTags
  } = tagManagement

  const defaultCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(defaultCharacterSelection),
    [defaultCharacterSelection]
  )

  const [generationPreviewData, setGenerationPreviewData] = React.useState<GeneratedCharacter | null>(null)
  const [generationPreviewField, setGenerationPreviewField] = React.useState<string | null>(null)

  // Template selection state
  const [hasSeenTemplateChooser, setHasSeenTemplateChooser] = React.useState(() => {
    if (typeof window === "undefined") return false
    return localStorage.getItem(TEMPLATE_CHOOSER_SEEN_KEY) === "true"
  })
  const [showTemplates, setShowTemplates] = React.useState(() => {
    if (typeof window === "undefined") return true
    return localStorage.getItem(TEMPLATE_CHOOSER_SEEN_KEY) !== "true"
  })

  // Form draft autosave (H4)
  const {
    hasDraft: hasCreateDraft,
    draftData: createDraftData,
    saveDraft: saveCreateDraft,
    clearDraft: clearCreateDraft,
    applyDraft: applyCreateDraft,
    dismissDraft: dismissCreateDraft,
    lastSaved: createLastSaved
  } = useFormDraft<Record<string, any>>({
    storageKey: 'character-form-draft-create',
    formType: 'create',
    autoSaveInterval: 30000
  })

  const {
    hasDraft: hasEditDraft,
    draftData: editDraftData,
    saveDraft: saveEditDraft,
    clearDraft: clearEditDraft,
    applyDraft: applyEditDraft,
    dismissDraft: dismissEditDraft,
    lastSaved: editLastSaved
  } = useFormDraft<Record<string, any>>({
    storageKey: 'character-form-draft-edit',
    formType: 'edit',
    editId: editId ?? undefined,
    autoSaveInterval: 30000
  })

  const markTemplateChooserSeen = React.useCallback(() => {
    setHasSeenTemplateChooser(true)
    if (typeof window !== "undefined") {
      localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")
    }
  }, [])

  const openCreateModal = React.useCallback(() => {
    void loadCharacterDialogs()
    void loadCharacterEditorForm()
    if (!hasSeenTemplateChooser) {
      setShowTemplates(true)
      markTemplateChooserSeen()
    }
    setOpen(true)
  }, [hasSeenTemplateChooser, markTemplateChooserSeen])

  const applyTemplateToCreateForm = React.useCallback(
    (template: CharacterTemplate) => {
      createForm.setFieldsValue({
        name: template.name,
        description: template.description,
        system_prompt: template.system_prompt,
        greeting: template.greeting,
        tags: template.tags,
        folder_id: undefined
      })
      setCreateFormDirty(true)
      setShowTemplates(false)
      markTemplateChooserSeen()
      setOpen(true)
      notification.info({
        message: t("settings:manageCharacters.templates.applied", {
          defaultValue: "Template applied"
        }),
        description: t("settings:manageCharacters.templates.appliedDesc", {
          defaultValue: "You can customize all fields before saving."
        })
      })
    },
    [createForm, markTemplateChooserSeen, notification, t]
  )

  // Keyboard shortcuts (H1)
  const modalOpen =
    open ||
    openEdit ||
    conversationsOpen ||
    generationPreviewOpen ||
    Boolean(chatIntentBlocker) ||
    Boolean(quickChatCharacter)
  useCharacterShortcuts({
    modalOpen,
    onNewCharacter: () => {
      if (!modalOpen) openCreateModal()
    },
    onFocusSearch: () => {
      searchInputRef.current?.focus()
    },
    onCloseModal: () => {
      if (generationPreviewOpen) {
        setGenerationPreviewOpen(false)
      } else if (conversationsOpen) {
        setConversationsOpen(false)
      } else if (chatIntentBlocker) {
        void clearChatIntentBlocker()
      } else if (quickChatCharacter) {
        void closeQuickChat()
      } else if (openEdit) {
        setOpenEdit(false)
        editWorldBooksInitializedRef.current = false
        setShowEditSystemPromptExample(false)
      } else if (open) {
        setOpen(false)
        setShowCreateSystemPromptExample(false)
      }
    },
    onTableView: () => setViewMode("table"),
    onGalleryView: () => setViewMode("gallery"),
    enabled: true
  })

  const shortcutHelpItems = React.useMemo(
    () => [
      {
        id: "new",
        keys: ["N"],
        label: t("settings:manageCharacters.shortcuts.new", {
          defaultValue: "New character"
        })
      },
      {
        id: "search",
        keys: ["/"],
        label: t("settings:manageCharacters.shortcuts.search", {
          defaultValue: "Focus search"
        })
      },
      {
        id: "table",
        keys: ["G", "T"],
        label: t("settings:manageCharacters.shortcuts.tableView", {
          defaultValue: "Table view"
        })
      },
      {
        id: "gallery",
        keys: ["G", "G"],
        label: t("settings:manageCharacters.shortcuts.galleryView", {
          defaultValue: "Gallery view"
        })
      },
      {
        id: "close",
        keys: ["Esc"],
        label: t("settings:manageCharacters.shortcuts.close", {
          defaultValue: "Close modal"
        })
      }
    ],
    [t]
  )

  const shortcutSummaryText = React.useMemo(
    () =>
      shortcutHelpItems
        .map((item) => `${item.keys.join(" ")} ${item.label}`)
        .join(". "),
    [shortcutHelpItems]
  )

  // Helper to get current form values as GeneratedCharacter
  const getFormFieldsAsCharacter = (form: FormInstance): Partial<GeneratedCharacter> => {
    const values = form.getFieldsValue()
    return {
      name: values.name,
      description: values.description,
      personality: values.personality,
      scenario: values.scenario,
      system_prompt: values.system_prompt,
      first_message: values.greeting || values.first_message,
      message_example: values.message_example,
      creator_notes: values.creator_notes,
      tags: values.tags,
      alternate_greetings: values.alternate_greetings
    }
  }

  // Handle full character generation
  const handleGenerateFullCharacter = async (concept: string, model: string, apiProvider?: string) => {
    const result = await generateFullCharacter(concept, { model, apiProvider })
    if (result) {
      setGenerationPreviewData(result)
      setGenerationPreviewField(null)
      setGenerationTargetForm('create')
      setGenerationPreviewOpen(true)
    }
  }

  // Handle single field generation
  const handleGenerateField = async (
    field: CharacterField,
    form: FormInstance,
    targetForm: 'create' | 'edit'
  ) => {
    if (!selectedGenModel) {
      notification.warning({
        message: t("settings:manageCharacters.generate.noModelSelected", {
          defaultValue: "No model selected"
        }),
        description: t("settings:manageCharacters.generate.noModelSelectedDesc", {
          defaultValue: "Please select a model in the generation panel first."
        })
      })
      return
    }

    const existingFields = getFormFieldsAsCharacter(form)
    const currentValue = (existingFields as any)[field]

    const result = await generateField(field, existingFields, {
      model: selectedGenModel,
      apiProvider: selectedGenModelProvider
    })

    if (result !== null) {
      // If field has existing content, show preview
      if (currentValue && String(currentValue).trim().length > 0) {
        setGenerationPreviewData({ [field]: result } as GeneratedCharacter)
        setGenerationPreviewField(field)
        setGenerationTargetForm(targetForm)
        setGenerationPreviewOpen(true)
      } else {
        // Apply directly if field is empty
        applyGeneratedFieldToForm(field, result, form)
      }
    }
  }

  const renderSharedCharacterForm = ({
    form,
    mode,
    initialValues,
    worldBookFieldContext,
    isSubmitting,
    submitButtonClassName,
    submitPendingLabel,
    submitIdleLabel,
    showPreview,
    onTogglePreview,
    onValuesChange,
    onFinish
  }: SharedCharacterFormProps) => (
    <React.Suspense fallback={null}>
      <LazyCharacterEditorForm
        t={t}
        form={form}
        mode={mode}
        initialValues={initialValues}
        worldBookFieldContext={worldBookFieldContext}
        isSubmitting={isSubmitting}
        submitButtonClassName={submitButtonClassName}
        submitPendingLabel={submitPendingLabel}
        submitIdleLabel={submitIdleLabel}
        showPreview={showPreview}
        onTogglePreview={onTogglePreview}
        onValuesChange={onValuesChange}
        onFinish={onFinish}
        generatingField={generatingField}
        isGenerating={isGenerating}
        handleGenerateField={handleGenerateField}
        showSystemPromptExample={
          mode === "create"
            ? showCreateSystemPromptExample
            : showEditSystemPromptExample
        }
        setShowSystemPromptExample={
          mode === "create"
            ? setShowCreateSystemPromptExample
            : setShowEditSystemPromptExample
        }
        markModeDirty={markModeDirty}
        popularTags={popularTags}
        tagOptionsWithCounts={tagOptionsWithCounts}
        characterFolderOptions={characterFolderOptions}
        characterFolderOptionsLoading={characterFolderOptionsLoading}
        showAdvanced={mode === "create" ? showCreateAdvanced : showEditAdvanced}
        setShowAdvanced={
          mode === "create" ? setShowCreateAdvanced : setShowEditAdvanced
        }
        advancedSections={
          mode === "create" ? createAdvancedSections : editAdvancedSections
        }
        setAdvancedSections={
          mode === "create" ? setCreateAdvancedSections : setEditAdvancedSections
        }
        createNameRef={createNameRef}
        editNameRef={editNameRef}
      />
    </React.Suspense>
  )

  // Apply generated data to form
  const applyGeneratedFieldToForm = (
    field: string,
    value: any,
    form: FormInstance
  ) => {
    // Map field names to form field names
    const fieldMap: Record<string, string> = {
      first_message: 'greeting'
    }
    const formField = fieldMap[field] || field
    form.setFieldValue(formField, value)
    if (generationTargetForm === 'create') {
      setCreateFormDirty(true)
    } else {
      setEditFormDirty(true)
    }
  }

  // Apply full character generation preview
  const applyGenerationPreview = () => {
    if (!generationPreviewData) return

    const form = generationTargetForm === 'create' ? createForm : editForm

    if (generationPreviewField) {
      // Single field
      const value = (generationPreviewData as any)[generationPreviewField]
      if (value !== undefined) {
        applyGeneratedFieldToForm(generationPreviewField, value, form)
      }
    } else {
      // Full character - apply all non-empty fields
      const fieldMap: Record<string, string> = {
        first_message: 'greeting'
      }

      Object.entries(generationPreviewData).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
          const formField = fieldMap[key] || key
          form.setFieldValue(formField, value)
        }
      })

      // Auto-expand advanced section if advanced fields were generated
      const hasAdvanced = generationPreviewData.personality ||
        generationPreviewData.scenario ||
        generationPreviewData.message_example ||
        generationPreviewData.creator_notes ||
        (generationPreviewData.alternate_greetings && generationPreviewData.alternate_greetings.length > 0)

      if (hasAdvanced) {
        if (generationTargetForm === 'create') {
          setShowCreateAdvanced(true)
        } else {
          setShowEditAdvanced(true)
        }
      }

      if (generationTargetForm === 'create') {
        setCreateFormDirty(true)
      } else {
        setEditFormDirty(true)
      }
    }

    setGenerationPreviewOpen(false)
    setGenerationPreviewData(null)
    setGenerationPreviewField(null)
  }

  React.useEffect(() => {
    if (forwardedNewButtonRef && newButtonRef.current) {
      ;(forwardedNewButtonRef as any).current = newButtonRef.current
    }
  }, [forwardedNewButtonRef])

  React.useEffect(() => {
    if (!autoOpenCreate) return
    if (autoOpenCreateHandledRef.current) return
    autoOpenCreateHandledRef.current = true
    if (!openEdit && !conversationsOpen) {
      openCreateModal()
    }
  }, [autoOpenCreate, conversationsOpen, openCreateModal, openEdit])

  // Cleanup pending delete timeout on unmount
  React.useEffect(() => {
    return () => {
      if (undoDeleteRef.current) {
        clearTimeout(undoDeleteRef.current)
      }
      if (bulkUndoDeleteRef.current) {
        clearTimeout(bulkUndoDeleteRef.current)
      }
    }
  }, [])

  // --- useCharacterData hook ---
  const characterData = useCharacterData({
    t,
    notification,
    qc,
    searchTerm,
    debouncedSearchTerm,
    filterTags,
    folderFilterId,
    matchAllTags,
    creatorFilter,
    createdFromDate,
    createdToDate,
    updatedFromDate,
    updatedToDate,
    hasConversationsOnly,
    favoritesOnly,
    characterListScope,
    sortColumn,
    sortOrder,
    currentPage,
    pageSize,
    setCurrentPage,
    previewCharacter,
    setPreviewCharacter,
    editCharacterNumericId,
    editWorldBooksInitializedRef,
    open,
    openEdit,
    editForm,
    defaultCharacterSelection,
    setDefaultCharacterSelection,
    defaultCharacterId
  })
  const {
    status,
    error,
    refetch,
    data,
    totalCharacters,
    pagedGalleryData,
    effectiveDefaultCharacterId,
    conversationCounts,
    previewCharacterWorldBooks,
    previewCharacterWorldBooksLoading,
    worldBookOptionsLoading,
    worldBookOptions,
    tagUsageData,
    allTags,
    popularTags,
    tagOptionsWithCounts,
    tagFilterOptions,
    creatorFilterOptions,
    characterFolderOptions,
    characterFolderOptionsLoading,
    selectedFolderFilterLabel,
    crossNavigationContext
  } = characterData

  React.useEffect(() => {
    if (hasPreloadedCharacterEditorRef.current || totalCharacters < 1) {
      return
    }

    hasPreloadedCharacterEditorRef.current = true
    const preloadTimer = window.setTimeout(() => {
      void loadCharacterDialogs()
      void loadCharacterEditorForm()
    }, 0)

    return () => {
      window.clearTimeout(preloadTimer)
    }
  }, [totalCharacters])

  // --- useCharacterCrud hook ---
  const crud = useCharacterCrud({
    t,
    notification,
    qc,
    createForm,
    editForm,
    editId,
    setEditId,
    editVersion,
    editCharacterNumericId,
    setOpen,
    setOpenEdit,
    setConversationsOpen,
    setConversationCharacter,
    setPreviewCharacter,
    setShowTemplates: (v: boolean) => setShowTemplates(v),
    setShowCreateSystemPromptExample,
    setShowEditSystemPromptExample,
    setShowCreateAdvanced,
    setShowEditAdvanced,
    setCreateFormDirty,
    setEditFormDirty,
    setExporting,
    newButtonRef,
    lastEditTriggerRef,
    editWorldBooksInitializedRef,
    clearCreateDraft,
    clearEditDraft,
    data,
    effectiveDefaultCharacterId,
    defaultCharacterSelection,
    setDefaultCharacterSelection,
    activeChatModel: activeQuickChatModel,
    availableChatModels: generationModels ?? [],
    setChatIntentBlocker
  })
  const {
    createCharacter,
    creating,
    updateCharacter,
    updating,
    deleting,
    handleExport,
    handleChat,
    handleChatInNewTab,
    handleEdit,
    handleDuplicate,
    handleDelete,
    handleViewConversations,
    handleRestoreFromTrash,
    isDefaultCharacterRecord,
    handleSetDefaultCharacter,
    handleClearDefaultCharacter,
    isCharacterFavoriteRecord,
    handleToggleFavorite,
    isPersonaCreatePending,
    getCreatePersonaActionLabel,
    openPersonaGardenForCharacter,
    createPersonaFromCharacter,
    characterChats,
    setCharacterChats,
    chatsError,
    setChatsError,
    loadingChats,
    setLoadingChats,
    resumingChatId,
    setResumingChatId,
    setHistory,
    setMessages,
    setHistoryId,
    setServerChatId,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef,
    pendingDelete,
    undoDeleteRef,
    bulkUndoDeleteRef,
    conversationsLoadErrorMessageRef,
    restoreCharacter
  } = crud
  const retryChatIntentBlocker = React.useCallback(async () => {
    if (!chatIntentBlocker) return
    await handleChat(chatIntentBlocker.record)
  }, [chatIntentBlocker, handleChat])

  const handleEditWithPrefetch = React.useCallback(
    (record: any, triggerRef?: HTMLButtonElement | null) => {
      void loadCharacterDialogs()
      void loadCharacterEditorForm()
      handleEdit(record, triggerRef)
    },
    [handleEdit]
  )

  React.useEffect(() => {
    if (open) {
      setTimeout(() => {
        createNameRef.current?.focus()
      }, 0)
    }
  }, [open])

  React.useEffect(() => {
    if (openEdit) {
      setTimeout(() => {
        editNameRef.current?.focus()
      }, 0)
    }
  }, [openEdit])

  // Inline edit state, mutation, and handlers (M1) - now from hook
  const {
    inlineEdit,
    setInlineEdit,
    inlineUpdating,
    inlineEditInputRef,
    inlineEditTriggerRef,
    inlineEditFocusKeyRef,
    startInlineEdit,
    saveInlineEdit,
    cancelInlineEdit,
    restoreInlineEditFocus
  } = useCharacterInlineEdit({ t, notification, qc, data })

  // --- useCharacterBulkOps hook ---
  const bulkOps = useCharacterBulkOps({
    t,
    notification,
    qc,
    data,
    characterListScope,
    creatorFilter,
    currentPage,
    debouncedSearchTerm,
    filterTags,
    folderFilterId,
    favoritesOnly,
    hasConversationsOnly,
    matchAllTags,
    pageSize,
    sortColumn,
    sortOrder,
    bulkUndoDeleteRef,
    setBulkOperationLoading,
    compareModalOpen,
    setCompareModalOpen,
    compareCharacters,
    setCompareCharacters,
    closeCompareModal,
    handleBulkAddTags
  })
  const {
    selectedCharacterIds,
    setSelectedCharacterIds,
    toggleCharacterSelection,
    selectAllOnPage,
    clearSelection,
    selectedCount,
    hasSelection,
    selectedCharacters,
    allOnPageSelected,
    someOnPageSelected,
    handleBulkDelete,
    handleBulkExport,
    handleBulkAddTagsForSelection,
    comparisonRows,
    changedComparisonRows,
    handleOpenCompareModal,
    handleCopyComparisonSummary,
    handleExportComparisonSummary
  } = bulkOps
  const shouldRenderCharacterDialogs =
    importPreviewOpen ||
    open ||
    openEdit ||
    conversationsOpen ||
    Boolean(quickChatCharacter) ||
    Boolean(chatIntentBlocker) ||
    generationPreviewOpen ||
    versionHistoryOpen ||
    compareModalOpen ||
    tagManagerOpen ||
    bulkTagModalOpen

  return (
    <div className="characters-page" data-testid="characters-page">
      <a
        href="#characters-main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:border focus:border-border focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-text focus:shadow">
        {t("settings:manageCharacters.skipToContent", {
          defaultValue: "Skip to characters content"
        })}
      </a>
      <div
        id="characters-main-content"
        role="main"
        tabIndex={-1}
        aria-describedby="characters-shortcuts-summary"
        className="space-y-4"
        onDragEnter={handleImportDragEnter}
        onDragOver={handleImportDragOver}
        onDragLeave={handleImportDragLeave}
        onDrop={(event) => {
          void handleImportDrop(event)
        }}>
      <div
        ref={importButtonContainerRef}
        data-testid="character-import-dropzone"
        className="sr-only"
      >
        <Upload
          accept={IMPORT_UPLOAD_ACCEPT}
          multiple
          showUploadList={false}
          beforeUpload={handleImportUpload}
          disabled={isImportBusy}
        >
          <button type="button" tabIndex={-1} aria-hidden="true">
            {t("settings:manageCharacters.import.button", {
              defaultValue: "Upload character"
            })}
          </button>
        </Upload>
      </div>

      <CharacterListToolbar
        t={t}
        searchInputRef={searchInputRef}
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        viewMode={viewMode}
        setViewMode={setViewMode}
        characterListScope={characterListScope}
        setCharacterListScope={setCharacterListScope}
        advancedFiltersOpen={advancedFiltersOpen}
        setAdvancedFiltersOpen={setAdvancedFiltersOpen}
        activeAdvancedFilterCount={activeAdvancedFilterCount}
        hasFilters={hasFilters}
        clearFilters={clearFilters}
        filterTags={filterTags}
        setFilterTags={setFilterTags}
        folderFilterId={folderFilterId}
        setFolderFilterId={setFolderFilterId}
        creatorFilter={creatorFilter}
        setCreatorFilter={setCreatorFilter}
        createdFromDate={createdFromDate}
        setCreatedFromDate={setCreatedFromDate}
        createdToDate={createdToDate}
        setCreatedToDate={setCreatedToDate}
        updatedFromDate={updatedFromDate}
        setUpdatedFromDate={setUpdatedFromDate}
        updatedToDate={updatedToDate}
        setUpdatedToDate={setUpdatedToDate}
        matchAllTags={matchAllTags}
        setMatchAllTags={setMatchAllTags}
        hasConversationsOnly={hasConversationsOnly}
        setHasConversationsOnly={setHasConversationsOnly}
        favoritesOnly={favoritesOnly}
        setFavoritesOnly={setFavoritesOnly}
        tagFilterOptions={tagFilterOptions}
        creatorFilterOptions={creatorFilterOptions}
        characterFolderOptions={characterFolderOptions}
        characterFolderOptionsLoading={characterFolderOptionsLoading}
        selectedFolderFilterLabel={selectedFolderFilterLabel}
        galleryDensity={galleryDensity}
        setGalleryDensity={setGalleryDensity}
        tableDensity={tableDensity}
        setTableDensity={setTableDensity}
        shortcutHelpItems={shortcutHelpItems}
        shortcutSummaryText={shortcutSummaryText}
        isImportBusy={isImportBusy}
        triggerImportPicker={triggerImportPicker}
        preloadCreateEditor={loadCharacterEditorForm}
        newButtonRef={newButtonRef}
        openCreateModal={openCreateModal}
        openTagManager={openTagManager}
      />

      <CharacterListContent
        t={t}
        status={status}
        error={error}
        refetch={refetch}
        data={data}
        totalCharacters={totalCharacters}
        pagedGalleryData={pagedGalleryData}
        conversationCounts={conversationCounts}
        viewMode={viewMode}
        characterListScope={characterListScope}
        setCharacterListScope={setCharacterListScope}
        galleryDensity={galleryDensity}
        tableDensity={tableDensity}
        currentPage={currentPage}
        setCurrentPage={setCurrentPage}
        pageSize={pageSize}
        setPageSize={setPageSize}
        sortColumn={sortColumn}
        setSortColumn={setSortColumn}
        sortOrder={sortOrder}
        setSortOrder={setSortOrder}
        hasFilters={hasFilters}
        searchTerm={searchTerm}
        filterTags={filterTags}
        setFilterTags={setFilterTags}
        matchAllTags={matchAllTags}
        folderFilterId={folderFilterId}
        selectedFolderFilterLabel={selectedFolderFilterLabel}
        creatorFilter={creatorFilter}
        createdFromDate={createdFromDate}
        createdToDate={createdToDate}
        updatedFromDate={updatedFromDate}
        updatedToDate={updatedToDate}
        hasConversationsOnly={hasConversationsOnly}
        favoritesOnly={favoritesOnly}
        clearFilters={clearFilters}
        previewCharacter={previewCharacter}
        setPreviewCharacter={setPreviewCharacter}
        previewCharacterWorldBooks={previewCharacterWorldBooks}
        previewCharacterWorldBooksLoading={previewCharacterWorldBooksLoading}
        crossNavigationContext={crossNavigationContext}
        inlineEdit={inlineEdit}
        setInlineEdit={setInlineEdit}
        inlineUpdating={inlineUpdating}
        inlineEditInputRef={inlineEditInputRef}
        startInlineEdit={startInlineEdit}
        saveInlineEdit={saveInlineEdit}
        cancelInlineEdit={cancelInlineEdit}
        selectedCharacterIds={selectedCharacterIds}
        setSelectedCharacterIds={setSelectedCharacterIds}
        toggleCharacterSelection={toggleCharacterSelection}
        selectAllOnPage={selectAllOnPage}
        clearSelection={clearSelection}
        selectedCount={selectedCount}
        hasSelection={hasSelection}
        allOnPageSelected={allOnPageSelected}
        someOnPageSelected={someOnPageSelected}
        handleBulkDelete={handleBulkDelete}
        handleBulkExport={handleBulkExport}
        handleOpenCompareModal={handleOpenCompareModal}
        bulkOperationLoading={bulkOperationLoading}
        setBulkTagModalOpen={setBulkTagModalOpen}
        handleChat={handleChat}
        handleChatInNewTab={handleChatInNewTab}
        preloadCharacterEditor={loadCharacterEditorForm}
        handleEdit={handleEditWithPrefetch}
        handleDuplicate={handleDuplicate}
        handleDelete={handleDelete}
        handleExport={handleExport}
        handleViewConversations={handleViewConversations}
        handleRestoreFromTrash={handleRestoreFromTrash}
        handleToggleFavorite={handleToggleFavorite}
        handleSetDefaultCharacter={handleSetDefaultCharacter}
        handleClearDefaultCharacter={handleClearDefaultCharacter}
        isDefaultCharacterRecord={isDefaultCharacterRecord}
        isCharacterFavoriteRecord={isCharacterFavoriteRecord}
        isPersonaCreatePending={isPersonaCreatePending}
        getCreatePersonaActionLabel={getCreatePersonaActionLabel}
        openPersonaGardenForCharacter={openPersonaGardenForCharacter}
        createPersonaFromCharacter={createPersonaFromCharacter}
        openVersionHistory={openVersionHistory}
        openQuickChat={openQuickChat}
        deleting={deleting}
        exporting={exporting}
        setConversationCharacter={setConversationCharacter}
        setCharacterChats={setCharacterChats}
        setChatsError={setChatsError}
        setConversationsOpen={setConversationsOpen}
        openCreateModal={openCreateModal}
        setShowTemplates={setShowTemplates}
        markTemplateChooserSeen={markTemplateChooserSeen}
        isImportBusy={isImportBusy}
        triggerImportPicker={triggerImportPicker}
        confirmDanger={confirmDanger}
      />

      {shouldRenderCharacterDialogs ? (
        <React.Suspense fallback={null}>
          <LazyCharacterDialogs
        t={t}
        navigate={navigate}
        // import
        importPreviewOpen={importPreviewOpen}
        resetImportPreview={resetImportPreview}
        importPreviewHasSuccessfulCompletion={importPreviewHasSuccessfulCompletion}
        retryableFailedPreviewItems={retryableFailedPreviewItems}
        importPreviewProcessing={importPreviewProcessing}
        handleRetryFailedImportPreview={handleRetryFailedImportPreview}
        importPreviewLoading={importPreviewLoading}
        importablePreviewItems={importablePreviewItems}
        importQueueSummary={importQueueSummary}
        importPreviewItems={importPreviewItems}
        importQueueItemsById={importQueueItemsById}
        importing={importing}
        handleConfirmImportPreview={handleConfirmImportPreview}
        getImportQueueStateLabel={getImportQueueStateLabel}
        getImportQueueStateColor={getImportQueueStateColor}
        // quick chat
        quickChatCharacter={quickChatCharacter}
        closeQuickChat={closeQuickChat}
        quickChatModelOptions={quickChatModelOptions}
        activeQuickChatModel={activeQuickChatModel}
        setQuickChatModelOverride={setQuickChatModelOverride}
        quickChatError={quickChatError}
        quickChatMessages={quickChatMessages}
        quickChatDraft={quickChatDraft}
        setQuickChatDraft={setQuickChatDraft}
        quickChatSending={quickChatSending}
        sendQuickChatMessage={sendQuickChatMessage}
        handlePromoteQuickChat={handlePromoteQuickChat}
        // full-chat intent blocker
        chatIntentBlocker={chatIntentBlocker}
        clearChatIntentBlocker={clearChatIntentBlocker}
        retryChatIntentBlocker={retryChatIntentBlocker}
        // conversations
        conversationsOpen={conversationsOpen}
        setConversationsOpen={setConversationsOpen}
        conversationCharacter={conversationCharacter}
        setConversationCharacter={setConversationCharacter}
        characterChats={characterChats}
        setCharacterChats={setCharacterChats}
        chatsError={chatsError}
        setChatsError={setChatsError}
        loadingChats={loadingChats}
        setLoadingChats={setLoadingChats}
        resumingChatId={resumingChatId}
        setResumingChatId={setResumingChatId}
        setSelectedCharacter={setSelectedCharacter}
        setHistory={setHistory}
        setMessages={setMessages}
        setHistoryId={setHistoryId}
        setServerChatId={setServerChatId}
        setServerChatState={setServerChatState}
        setServerChatTopic={setServerChatTopic}
        setServerChatClusterId={setServerChatClusterId}
        setServerChatSource={setServerChatSource}
        setServerChatExternalRef={setServerChatExternalRef}
        // create
        open={open}
        setOpen={setOpen}
        createForm={createForm}
        createFormDirty={createFormDirty}
        setCreateFormDirty={setCreateFormDirty}
        creating={creating}
        createCharacter={createCharacter}
        showCreatePreview={showCreatePreview}
        setShowCreatePreview={setShowCreatePreview}
        setShowCreateAdvanced={setShowCreateAdvanced}
        setShowCreateSystemPromptExample={setShowCreateSystemPromptExample}
        setShowTemplates={setShowTemplates}
        showTemplates={showTemplates}
        markTemplateChooserSeen={markTemplateChooserSeen}
        applyTemplateToCreateForm={applyTemplateToCreateForm}
        newButtonRef={newButtonRef}
        hasCreateDraft={hasCreateDraft}
        createDraftData={createDraftData}
        saveCreateDraft={saveCreateDraft}
        clearCreateDraft={clearCreateDraft}
        applyCreateDraft={applyCreateDraft}
        dismissCreateDraft={dismissCreateDraft}
        // edit
        openEdit={openEdit}
        setOpenEdit={setOpenEdit}
        editForm={editForm}
        editFormDirty={editFormDirty}
        setEditFormDirty={setEditFormDirty}
        editId={editId}
        setEditId={setEditId}
        editVersion={editVersion}
        setEditVersion={setEditVersion}
        editCharacterNumericId={editCharacterNumericId}
        updating={updating}
        updateCharacter={updateCharacter}
        showEditPreview={showEditPreview}
        setShowEditPreview={setShowEditPreview}
        setShowEditAdvanced={setShowEditAdvanced}
        setShowEditSystemPromptExample={setShowEditSystemPromptExample}
        lastEditTriggerRef={lastEditTriggerRef}
        editWorldBooksInitializedRef={editWorldBooksInitializedRef}
        hasEditDraft={hasEditDraft}
        editDraftData={editDraftData}
        saveEditDraft={saveEditDraft}
        clearEditDraft={clearEditDraft}
        applyEditDraft={applyEditDraft}
        dismissEditDraft={dismissEditDraft}
        worldBookOptions={worldBookOptions}
        worldBookOptionsLoading={worldBookOptionsLoading}
        // generation
        isGenerating={isGenerating}
        generatingField={generatingField}
        generationError={generationError}
        handleGenerateFullCharacter={handleGenerateFullCharacter}
        cancelGeneration={cancelGeneration}
        clearGenerationError={clearGenerationError}
        generationPreviewOpen={generationPreviewOpen}
        setGenerationPreviewOpen={setGenerationPreviewOpen}
        generationPreviewData={generationPreviewData}
        setGenerationPreviewData={setGenerationPreviewData}
        generationPreviewField={generationPreviewField}
        setGenerationPreviewField={setGenerationPreviewField}
        applyGenerationPreview={applyGenerationPreview}
        // version history
        versionHistoryOpen={versionHistoryOpen}
        setVersionHistoryOpen={setVersionHistoryOpen}
        versionHistoryCharacter={versionHistoryCharacter}
        setVersionHistoryCharacter={setVersionHistoryCharacter}
        versionHistoryCharacterId={versionHistoryCharacterId}
        versionHistoryCharacterName={versionHistoryCharacterName}
        versionFrom={versionFrom}
        setVersionFrom={setVersionFrom}
        versionTo={versionTo}
        setVersionTo={setVersionTo}
        versionRevertTarget={versionRevertTarget}
        setVersionRevertTarget={setVersionRevertTarget}
        versionHistoryItems={versionHistoryItems}
        versionHistoryLoading={versionHistoryLoading}
        versionHistoryFetching={versionHistoryFetching}
        versionSelectOptions={versionSelectOptions}
        versionDiffResponse={versionDiffResponse}
        versionDiffLoading={versionDiffLoading}
        versionDiffFetching={versionDiffFetching}
        revertingCharacterVersion={revertingCharacterVersion}
        openVersionHistory={openVersionHistory}
        revertCharacterVersion={revertCharacterVersion}
        // compare
        compareModalOpen={compareModalOpen}
        closeCompareModal={closeCompareModal}
        compareCharacters={compareCharacters}
        comparisonRows={comparisonRows}
        changedComparisonRows={changedComparisonRows}
        handleCopyComparisonSummary={handleCopyComparisonSummary}
        handleExportComparisonSummary={handleExportComparisonSummary}
        // tag manager
        tagManagerOpen={tagManagerOpen}
        closeTagManager={closeTagManager}
        tagManagerLoading={tagManagerLoading}
        tagManagerSubmitting={tagManagerSubmitting}
        tagManagerOperation={tagManagerOperation}
        setTagManagerOperation={setTagManagerOperation}
        tagManagerSourceTag={tagManagerSourceTag}
        setTagManagerSourceTag={setTagManagerSourceTag}
        tagManagerTargetTag={tagManagerTargetTag}
        setTagManagerTargetTag={setTagManagerTargetTag}
        tagManagerTagUsageData={tagManagerTagUsageData}
        handleApplyTagManagerOperation={handleApplyTagManagerOperation}
        // bulk tags
        bulkTagModalOpen={bulkTagModalOpen}
        setBulkTagModalOpen={setBulkTagModalOpen}
        bulkTagsToAdd={bulkTagsToAdd}
        setBulkTagsToAdd={setBulkTagsToAdd}
        bulkOperationLoading={bulkOperationLoading}
        handleBulkAddTagsForSelection={handleBulkAddTagsForSelection}
        selectedCount={selectedCount}
        popularTags={popularTags}
        tagOptionsWithCounts={tagOptionsWithCounts}
        // confirm
        confirmDanger={confirmDanger}
        // shared form
        renderSharedCharacterForm={renderSharedCharacterForm}
        data={data}
          />
        </React.Suspense>
      ) : null}
      </div>
    </div>
  )
}
