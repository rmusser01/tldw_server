import React from "react"
import {
  Button,
  Card,
  Checkbox,
  Collapse,
  Dropdown,
  Empty,
  Input,
  InputNumber,
  Modal,
  Select,
  Segmented,
  Skeleton,
  Tag,
  Tooltip,
  Typography,
  message
} from "antd"
import type { MenuProps } from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"
import type { JSONContent } from "@tiptap/react"
import { useQuery } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import {
  Activity,
  Columns2,
  Copy,
  Download,
  Edit3,
  Eye,
  Menu,
  MoreHorizontal,
  Pencil,
  Redo2,
  Search,
  Settings,
  Square,
  Trash2,
  Undo2,
  X,
  Zap
} from "lucide-react"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { READY_STATE_LABEL } from "@/design-system"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { MarkdownPreview } from "@/components/Common/MarkdownPreview"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { TldwChatService } from "@/services/tldw/TldwChat"
import {
  getWritingCapabilities,
  type WritingSessionListItem,
  type WritingTemplateResponse,
  type WritingThemeResponse
} from "@/services/writing-playground"
import { useStoreChatModelSettings } from "@/store/model"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { cn } from "@/libs/utils"
import { markdownToText } from "@/utils/markdown-to-text"
import {
  buildExtraBodyPayload,
  parseStringListInput
} from "./extra-body-utils"
import {
  buildContextSystemMessages,
  composeContextPrompt,
  injectSystemMessages,
  parseWorldInfoKeysInput
} from "./writing-context-utils"
import {
  resolveGenerationStopStrings,
  type BasicStoppingModeType
} from "./writing-stop-mode-utils"
import {
  extractLogprobEntriesFromChunk,
  type WritingLogprobEntry
} from "./writing-logprob-utils"
import {
  parseLogitBiasInput,
  withLogitBiasEntry,
  withoutLogitBiasEntry
} from "./writing-logit-bias-utils"
import {
  computeTokensPerSecond,
  estimateTokenCountFromText
} from "./writing-generation-stats-utils"
import { buildDiagnosticsSummary } from "./writing-diagnostics-utils"
import { WritingPlaygroundActiveSessionGuard } from "./WritingPlaygroundActiveSessionGuard"
import { ManuscriptTreePanel } from "./ManuscriptTreePanel"
import { resolveTipTapDocument } from "./writing-tiptap-utils"
import { WritingPlaygroundShell } from "./WritingPlaygroundShell"
import { WritingPlaygroundLibraryPanel } from "./WritingPlaygroundLibraryPanel"
import { WritingPlaygroundEditorPanel } from "./WritingPlaygroundEditorPanel"
import { WritingPlaygroundInspectorPanel } from "./WritingPlaygroundInspectorPanel"
import { CharacterWorldTab } from "./CharacterWorldTab"
import { ResearchTab } from "./ResearchTab"
import { AIAgentTab } from "./AIAgentTab"
import { FeedbackTab } from "./FeedbackTab"
import { MOOD_COLORS } from "./feedback-constants"
import { WritingAnalysisModalHost } from "./WritingAnalysisModalHost"
import { WritingPlaygroundDiagnosticsPanel } from "./WritingPlaygroundDiagnosticsPanel"
import { WritingWorldInfoImportControls } from "./WritingWorldInfoImportControls"
import {
  WritingActionBar,
  type WritingActionBarRequest
} from "./WritingActionBar"
import { WritingRevisionQueue } from "./WritingRevisionQueue"
import {
  WRITING_REVISION_PRESETS,
  getWritingRevisionPreset
} from "./writing-revision-presets"
import {
  buildRevisionUserPrompt,
  parseRevisionModelResponse
} from "./writing-revision-prompt-utils"
import {
  confirmRevisionTarget,
  countWords,
  resolveRevisionTarget
} from "./writing-revision-utils"
import type {
  WritingRevisionAction,
  WritingRevisionOperation,
  WritingRevisionPresetId,
  WritingRevisionProposal,
  WritingRevisionTarget
} from "./writing-revision-types"
import {
  buildSpeechVoiceOptions,
  clampSpeechRate,
  resolvePauseResumeAction,
  resolveSpeechVoice
} from "./writing-speech-utils"
import {
  normalizeWritingSpeechPreferences,
  type WritingSpeechPreferences
} from "./writing-speech-settings-utils"
import {
  applyPlaceholderAtRange,
  applyTextAtRange
} from "./writing-editor-actions-utils"
import {
  createTextareaEditorAdapter,
  type WritingEditorAdapter,
  type WritingEditorSelection
} from "./writing-editor-adapter"
import {
  useWritingSessionManagement,
  useWritingTemplateLibrary,
  useWritingGenerationSettings,
  useWritingContextComposition,
  useWritingInspectorPanels,
  useWritingImportExport,
  useWritingFeedback
} from "./hooks"
import { useWritingRevisions } from "./hooks/useWritingRevisions"
import {
  ADVANCED_NUMBER_PARAMS,
  buildChatMessages,
  buildFillPrompt,
  buildRegex,
  DEFAULT_CONTEXT_ORDER,
  DEFAULT_SETTINGS,
  DEFAULT_TOP_LOGPROBS,
  escapeRegex,
  FILL_PLACEHOLDER,
  FILL_SYSTEM_PROMPT,
  isAbortError,
  isRecord,
  MAX_CHUNKS,
  MAX_MATCHES,
  normalizeStopStrings,
  PREDICT_PLACEHOLDER,
  PREDICT_SYSTEM_PROMPT,
  resolveGenerationPlan,
  applyFimTemplate,
  getPromptRichFromPayload,
  getRevisionPresetIdFromPayload,
  WRITING_SPEECH_PREFS_STORAGE_KEY,
  type EditorViewMode,
  type GenerationHistoryEntry,
  type LastGenerationContext,
  type NonToolMessage,
  type SessionUsageMap
} from "./hooks/utils"

const { Paragraph } = Typography

const SECRET_FIELD_PATTERN =
  /(api[_-]?key|authorization|auth[_-]?token|secret|password|token)/i

const sanitizeRevisionValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sanitizeRevisionValue)
  if (!value || typeof value !== "object") return value

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !SECRET_FIELD_PATTERN.test(key))
      .map(([key, entryValue]) => [key, sanitizeRevisionValue(entryValue)])
  )
}

const LazyWritingPlaygroundModalHost = React.lazy(() =>
  import("./WritingPlaygroundModalHost").then((module) => ({
    default: module.WritingPlaygroundModalHost,
  })),
)

const LazyWritingTipTapEditor = React.lazy(() =>
  import("./WritingTipTapEditor").then((module) => ({
    default: module.WritingTipTapEditor,
  })),
)

export const WritingPlayground = () => {
  const { t } = useTranslation(["option"])
  const isOnline = useServerOnline()
  const { capabilities } = useServerCapabilities()
  const {
    activeSessionId,
    activeSessionName,
    setActiveSessionId,
    setActiveSessionName,
    editorMode,
    setEditorMode,
    focusMode,
    setFocusMode,
    activeNodeId,
    setActiveNodeId,
    setAnalysisModalOpen,
  } = useWritingPlaygroundStore()
  // TODO Phase 2: React to activeNodeId changes to load scene content into editor
  const [selectedModel, setSelectedModel] = useStorage<string>("selectedModel")
  const apiProviderOverride = useStoreChatModelSettings(
    (state) => state.apiProvider
  )
  const setApiProvider = useStoreChatModelSettings((state) => state.setApiProvider)
  const [sessionUsageMap, setSessionUsageMap] = useStorage<SessionUsageMap>(
    "writing:session-usage",
    {}
  )
  const [storedSpeechPreferences, setStoredSpeechPreferences] =
    useStorage<WritingSpeechPreferences>(WRITING_SPEECH_PREFS_STORAGE_KEY, {
      rate: 1,
      voiceURI: null
    })

  // --- Local-only state (not managed by hooks) ---
  const [libraryView, setLibraryView] = React.useState<"sessions" | "manuscript">("sessions")
  const [tipTapContent, setTipTapContent] = React.useState<JSONContent | null>(null)
  const editorTextChangedByTipTap = React.useRef(false)
  const [editorView, setEditorView] = React.useState<EditorViewMode>("edit")
  const [searchOpen, setSearchOpen] = React.useState(false)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [replaceQuery, setReplaceQuery] = React.useState("")
  const [matchCase, setMatchCase] = React.useState(false)
  const [useRegex, setUseRegex] = React.useState(false)
  const [activeMatchIndex, setActiveMatchIndex] = React.useState(0)
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [isRevisionGenerating, setIsRevisionGenerating] = React.useState(false)
  const [canUndoGeneration, setCanUndoGeneration] = React.useState(false)
  const [canRedoGeneration, setCanRedoGeneration] = React.useState(false)
  const [isSpeaking, setIsSpeaking] = React.useState(false)
  const [isSpeechPaused, setIsSpeechPaused] = React.useState(false)
  const normalizedStoredSpeechPreferences = React.useMemo(
    () => normalizeWritingSpeechPreferences(storedSpeechPreferences),
    [storedSpeechPreferences]
  )
  const [speechRate, setSpeechRate] = React.useState(
    normalizedStoredSpeechPreferences.rate
  )
  const [speechVoiceURI, setSpeechVoiceURI] = React.useState<string | null>(
    normalizedStoredSpeechPreferences.voiceURI
  )
  const [speechVoices, setSpeechVoices] = React.useState<SpeechSynthesisVoice[]>([])
  const [extraBodyJsonModalOpen, setExtraBodyJsonModalOpen] = React.useState(false)
  const [extraBodyJsonDraft, setExtraBodyJsonDraft] = React.useState("{}")
  const [extraBodyJsonError, setExtraBodyJsonError] = React.useState<string | null>(null)
  const [selectedRevisionPresetId, setSelectedRevisionPresetId] =
    React.useState<WritingRevisionPresetId>(
      WRITING_REVISION_PRESETS[0]?.id ?? "polish_prose"
    )
  const [revisionSelection, setRevisionSelection] =
    React.useState<WritingEditorSelection | null>(null)

  // --- Refs (local only) ---
  const generationServiceRef = React.useRef(new TldwChatService())
  const generationUndoRef = React.useRef<GenerationHistoryEntry[]>([])
  const generationRedoRef = React.useRef<GenerationHistoryEntry[]>([])
  const lastGenerationContextRef = React.useRef<LastGenerationContext | null>(null)
  const generationSessionIdRef = React.useRef<string | null>(null)
  const generationCancelledRef = React.useRef(false)
  const speechUtteranceRef = React.useRef<SpeechSynthesisUtterance | null>(null)
  const editorRef = React.useRef<TextAreaRef | null>(null)
  const activeEditorAdapterRef = React.useRef<WritingEditorAdapter | null>(null)
  const pendingEditorSelectionRef =
    React.useRef<WritingEditorSelection | null>(null)
  const previewRef = React.useRef<HTMLDivElement | null>(null)
  const isSyncingScrollRef = React.useRef(false)

  // --- Capabilities ---
  const {
    data: writingCaps,
    isLoading: capsLoading,
    error: capsError
  } = useQuery({
    queryKey: ["writing-capabilities"],
    queryFn: () => getWritingCapabilities({ includeProviders: false }),
    enabled: isOnline,
    staleTime: 5 * 60 * 1000
  })
  const hasWriting = Boolean(writingCaps?.server?.sessions)
  const hasTemplates = Boolean(writingCaps?.server?.templates)
  const hasThemes = Boolean(writingCaps?.server?.themes)
  const hasServerDefaultsCatalog = Boolean(writingCaps?.server?.defaults_catalog)
  const hasSnapshots = Boolean(writingCaps?.server?.snapshots)
  const hasChat = capabilities?.hasChat !== false
  const isWritingRequestBusy = isGenerating || isRevisionGenerating

  // =====================================================================
  // Hook 1: Session Management
  // =====================================================================
  const sessionMgmt = useWritingSessionManagement({
    isOnline, hasWriting, activeSessionId, activeSessionName,
    setActiveSessionId, setActiveSessionName, sessionUsageMap, setSessionUsageMap,
    selectedModel, setSelectedModel, apiProviderOverride, setApiProvider,
    isGenerating: isWritingRequestBusy, t
  })
  const {
    sessions, sessionsLoading, sessionsFetching, sessionsError,
    activeSessionDetail, activeSessionLoading, activeSessionError,
    activeSession, sortedSessions,
    createModalOpen, setCreateModalOpen,
    newSessionName, setNewSessionName,
    renameModalOpen, setRenameModalOpen,
    renameSessionName, setRenameSessionName,
    renameTarget,
    isDirty, setIsDirty,
    lastSavedAt,
    editorText, setEditorText,
    settings,
    stopStringsInput,
    bannedTokensInput, setBannedTokensInput,
    drySequenceBreakersInput, setDrySequenceBreakersInput,
    logitBiasInput, setLogitBiasInput,
    logitBiasError, setLogitBiasError,
    logitBiasTokenInput, setLogitBiasTokenInput,
    logitBiasValueInput, setLogitBiasValueInput,
    selectedTemplateName, selectedThemeName,
    chatMode,
    createSessionMutation, renameSessionMutation, deleteSessionMutation,
    cloneSessionMutation, saveSessionMutation,
    savingSessionIdRef,
    handleSelectSession, openRenameModal,
    applyPromptValue, updateSetting,
    handleTemplateChange, handleThemeChange, handleChatModeChange,
    applySessionPayloadPatch,
    canCreateSession, canRenameSession
  } = sessionMgmt

  const textareaEditorAdapter = React.useMemo(
    () => createTextareaEditorAdapter(editorRef),
    []
  )

  const applyPendingEditorSelection = React.useCallback(
    (adapter: WritingEditorAdapter | null) => {
      const pendingSelection = pendingEditorSelectionRef.current
      if (!adapter || !pendingSelection) return
      if (
        adapter === textareaEditorAdapter &&
        !editorRef.current?.resizableTextArea?.textArea
      ) {
        return
      }
      adapter.focus()
      adapter.setSelection(pendingSelection)
      pendingEditorSelectionRef.current = null
    },
    [textareaEditorAdapter]
  )

  const updateRevisionSelection = React.useCallback(
    (selection: WritingEditorSelection | null) => {
      setRevisionSelection((current) => {
        if (
          current?.start === selection?.start &&
          current?.end === selection?.end
        ) {
          return current
        }
        return selection
      })
    },
    []
  )

  const setActiveEditorAdapter = React.useCallback(
    (adapter: WritingEditorAdapter | null) => {
      activeEditorAdapterRef.current = adapter
      applyPendingEditorSelection(adapter)
      updateRevisionSelection(adapter?.getSelection() ?? null)
    },
    [applyPendingEditorSelection, updateRevisionSelection]
  )

  // Sync TipTap content when editorText changes from an external source (e.g. session load, undo/redo, generate)
  React.useEffect(() => {
    if (editorMode !== "tiptap") {
      setActiveEditorAdapter(textareaEditorAdapter)
      setTipTapContent(null)
      editorTextChangedByTipTap.current = false
      return
    }

    if (!editorTextChangedByTipTap.current) {
      setTipTapContent(
        resolveTipTapDocument(
          editorText,
          getPromptRichFromPayload(activeSessionDetail?.payload)
        )
      )
    }
    editorTextChangedByTipTap.current = false
  }, [
    activeSessionDetail?.id,
    activeSessionDetail?.payload,
    editorMode,
    editorText,
    setActiveEditorAdapter,
    textareaEditorAdapter
  ])

  React.useEffect(() => {
    if (editorMode === "tiptap" || editorView === "preview") return
    applyPendingEditorSelection(activeEditorAdapterRef.current)
  }, [applyPendingEditorSelection, editorMode, editorView])

  const settingsDisabled =
    isGenerating || isRevisionGenerating || !activeSessionDetail

  React.useEffect(() => {
    const persistedPresetId = getRevisionPresetIdFromPayload(
      activeSessionDetail?.payload
    )
    setSelectedRevisionPresetId(
      persistedPresetId ?? WRITING_REVISION_PRESETS[0]?.id ?? "polish_prose"
    )
  }, [activeSessionDetail?.id, activeSessionDetail?.payload])

  React.useEffect(() => {
    setRevisionSelection(null)
  }, [activeSessionDetail?.id, editorMode])

  // =====================================================================
  // Hook 2: Template Library
  // =====================================================================
  const templateLib = useWritingTemplateLibrary({
    isOnline, hasWriting, hasTemplates, hasThemes, hasServerDefaultsCatalog,
    selectedTemplateName, selectedThemeName,
    handleTemplateChange, handleThemeChange, settingsDisabled, t
  })
  const {
    templates, templatesLoading, templatesError,
    themes, themesLoading, themesError,
    effectiveTemplate, effectiveTheme,
    templateOptions, themeOptions,
    activeThemeClassName, activeThemeCss,
    templatesModalOpen, setTemplatesModalOpen,
    templateForm, editingTemplate,
    templateImporting, templateRestoringDefaults,
    themesModalOpen, setThemesModalOpen,
    themeForm, editingTheme,
    themeImporting, themeRestoringDefaults,
    templateFileInputRef, themeFileInputRef,
    deleteTemplateMutation, deleteThemeMutation,
    updateTemplateForm, updateThemeForm,
    handleTemplateSelect, handleTemplateNew,
    handleTemplateDuplicate, handleTemplateRestoreDefaults,
    handleOpenTemplatesModal,
    handleThemeSelect, handleThemeNew,
    handleThemeDuplicate, handleThemeRestoreDefaults,
    handleOpenThemesModal,
    handleTemplateSave, handleThemeSave,
    exportTemplate, handleTemplateImport,
    exportTheme, handleThemeImport,
    templateSelectDisabled, templateSaveLoading, templateSaveDisabled,
    templateExportDisabled, templateDuplicateDisabled,
    templateRestoreDefaultsDisabled, templateDeleteDisabled,
    templateFormDisabled,
    themeSelectDisabled, themeSaveLoading, themeSaveDisabled,
    themeExportDisabled, themeDuplicateDisabled,
    themeRestoreDefaultsDisabled, themeDeleteDisabled,
    themeFormDisabled
  } = templateLib

  // =====================================================================
  // Hook 3: Generation Settings
  // =====================================================================
  const advancedExtraBody = React.useMemo(
    () => (isRecord(settings.advanced_extra_body) ? settings.advanced_extra_body : {}),
    [settings.advanced_extra_body]
  )
  const genSettings = useWritingGenerationSettings({
    isOnline, hasWriting, selectedModel, apiProviderOverride,
    settings, advancedExtraBody, updateSetting, settingsDisabled,
    logitBiasInput, setLogitBiasInput,
    logitBiasError, setLogitBiasError,
    logitBiasTokenInput, setLogitBiasTokenInput,
    logitBiasValueInput, setLogitBiasValueInput,
    bannedTokensInput, setBannedTokensInput,
    drySequenceBreakersInput, setDrySequenceBreakersInput,
    extraBodyJsonModalOpen, setExtraBodyJsonModalOpen,
    extraBodyJsonDraft, setExtraBodyJsonDraft,
    extraBodyJsonError, setExtraBodyJsonError, t
  })
  const {
    requestedCaps, requestedCapsLoading, extraBodyCompat,
    requestedLogprobsSupported,
    requestedLogprobsExplicitlyUnsupported,
    requestedTopLogprobsSupported,
    knownExtraBodyParams, supportsAdvancedCompat,
    shouldShowAdvancedParam, tokenInspectorSupportsLogitBias,
    updateAdvancedExtraBodyField, getAdvancedNumberValue,
    applyLogitBiasObject,
    applyTokenInspectorLogitBiasPreset,
    applyTokenInspectorLogitBiasPresetBatch,
    logitBiasEntries,
    openExtraBodyJsonEditor, applyExtraBodyJsonDraft,
    advancedExtraBodyUnknownKeys,
    hasAdvancedSettingsValues, showAdvancedSamplerControls,
    logprobsControlsDisabled, topLogprobsControlsDisabled
  } = genSettings

  // =====================================================================
  // Hook 4: Context Composition
  // =====================================================================
  const ctxComp = useWritingContextComposition({
    settings, editorText, chatMode, effectiveTemplate,
    settingsDisabled, updateSetting, t
  })
  const {
    contextPreviewModalOpen, setContextPreviewModalOpen,
    memoryBlock, authorNote, worldInfo, worldInfoEntries,
    contextPreviewMessages, contextPreviewJson,
    updateMemoryBlock, updateAuthorNote, updateWorldInfo,
    addWorldInfoEntry, updateWorldInfoEntry,
    removeWorldInfoEntry, moveWorldInfoEntryById,
    handleWorldInfoExport, handleWorldInfoImported, handleWorldInfoImportError,
    handleCopyContextPreview, handleExportContextPreview
  } = ctxComp

  // =====================================================================
  // Hook 6: Import/Export
  // =====================================================================
  const importExport = useWritingImportExport({
    isOnline, hasWriting, hasSnapshots, sessions, sessionsLoading,
    activeSessionId, setActiveSessionId, setActiveSessionName,
    selectedModel, setSelectedModel, apiProviderOverride, setApiProvider, t
  })
  const {
    sessionImporting, snapshotImporting, snapshotExporting,
    sessionFileInputRef, snapshotFileInputRef,
    exportSession, exportSnapshot,
    openSnapshotImportPicker,
    handleSnapshotImport, handleSessionImport,
    sessionImportDisabled, snapshotImportDisabled, snapshotExportDisabled
  } = importExport

  // =====================================================================
  // Hook 7: Writing Feedback (mood + echo chamber)
  // =====================================================================
  const feedback = useWritingFeedback({
    editorText,
    isOnline,
    isGenerating,
    selectedModel: selectedModel ?? undefined,
  })

  // --- Diagnostics ---
  const showOffline = !isOnline
  const showUnsupported =
    !showOffline && !capsLoading && (!hasWriting || Boolean(capsError))
  const diagnosticsSummary = buildDiagnosticsSummary({
    showOffline,
    showUnsupported,
    isGenerating
  })

  // =====================================================================
  // Hook 5: Inspector Panels (before handleGenerate to avoid forward ref)
  // =====================================================================
  const handleGenerateRef = React.useRef<(overrideText?: string) => void>(() => {})
  const stableHandleGenerate = React.useCallback(
    (overrideText?: string) => { handleGenerateRef.current(overrideText) },
    []
  )
  const inspectorPanels = useWritingInspectorPanels({
    isOnline,
    selectedModel,
    apiProviderOverride,
    activeSessionId,
    activeSessionDetail,
    settingsDisabled,
    isGenerating,
    writingCaps,
    requestedCaps,
    requestedCapsLoading,
    requestedLogprobsExplicitlyUnsupported,
    requestedLogprobsSupported,
    requestedTopLogprobsSupported,
    settings,
    lastGenerationContextRef,
    handleGenerate: stableHandleGenerate,
    t
  })
  const {
    libraryOpen, setLibraryOpen,
    inspectorOpen, setInspectorOpen,
    activeInspectorTab, setActiveInspectorTab,
    showPromptChunks, setShowPromptChunks,
    generationElapsed,
    tokenCountResult,
    tokenizeResult,
    tokenInspectorError,
    isCountingTokens,
    isTokenizingText,
    handleCountTokens,
    handleTokenizePreview,
    clearTokenInspector,
    tokenPreviewRows,
    tokenPreviewRawText,
    tokenPreviewTotal,
    tokenPreviewTruncated,
    tokenInspectorBusy,
    tokenInspectorUnavailableReason,
    tokenizerName,
    responseLogprobs, setResponseLogprobs,
    responseInspectorQuery, setResponseInspectorQuery,
    responseInspectorSort, setResponseInspectorSort,
    responseInspectorHideWhitespace, setResponseInspectorHideWhitespace,
    generationTokenCount, setGenerationTokenCount,
    generationTokensPerSec, setGenerationTokensPerSec,
    clearResponseInspector,
    handleCopyResponseInspectorJson,
    handleRerollFromResponseToken,
    handleExportResponseInspectorCsv,
    responseInspectorRowsAll,
    responseLogprobRows,
    responseLogprobTruncated,
    inlineResponseTokens,
    inlineResponseTokensTruncated,
    responseInspectorHasRows,
    wordcloudStatus,
    wordcloudWords,
    wordcloudMeta,
    wordcloudError,
    isGeneratingWordcloud,
    wordcloudMaxWords,
    wordcloudMinWordLength,
    wordcloudKeepNumbers,
    wordcloudStopwordsInput,
    handleGenerateWordcloud,
    clearWordcloud,
    wordcloudTopWeight,
    wordcloudStatusColor,
    canGenerateWordcloud,
    serverSupportsTokenize,
    serverSupportsTokenCount,
    serverSupportsWordclouds,
    showResponseInspectorPanel,
    showTokenInspectorPanel,
    showWordcloudPanel,
    canCountTokens,
    canTokenizePreview,
    logprobsUnavailableReason,
    topLogprobsHint
  } = inspectorPanels

  // =====================================================================
  // Generation logic (unique - not in hooks)
  // =====================================================================
  const handleGenerate = React.useCallback(
    async (overrideText?: string) => {
    if (isGenerating || isRevisionGenerating) return
    if (!activeSessionDetail) {
      message.info(
        t("option:writingPlayground.selectSession", "Select a session to begin.")
      )
      return
    }
    if (!isOnline || !hasChat) {
      message.error(
        t("option:writingPlayground.generateUnavailable", "Chat completions unavailable.")
      )
      return
    }
    if (!selectedModel) {
      message.info(
        t("option:writingPlayground.modelMissing", "Select a model in Settings to generate.")
      )
      return
    }

    const beforeText = typeof overrideText === "string" ? overrideText : editorText
    const plan = resolveGenerationPlan(beforeText)
    const fimPrompt =
      plan.mode === "fill"
        ? applyFimTemplate(effectiveTemplate, plan.prefix, plan.suffix)
        : null
    if (plan.mode === "fill" && !fimPrompt) {
      message.info(
        t("option:writingPlayground.fillFallbackNotice", "Fill template missing; using a basic fill prompt.")
      )
    }
    const promptText =
      plan.mode === "fill"
        ? fimPrompt ?? buildFillPrompt(plan.prefix, plan.suffix)
        : plan.prefix
    const contextSettings = {
      memory_block: settings.memory_block,
      author_note: settings.author_note,
      world_info: settings.world_info,
      context_order: settings.context_order,
      context_length: settings.context_length,
      author_note_depth_mode: settings.author_note_depth_mode
    }
    const contextComposedPrompt = chatMode
      ? promptText
      : composeContextPrompt(promptText, contextSettings)
    const baseMessages = buildChatMessages(
      contextComposedPrompt,
      effectiveTemplate,
      chatMode
    )
    const contextMessages = chatMode
      ? buildContextSystemMessages(beforeText, contextSettings)
      : []
    const messages = injectSystemMessages(baseMessages, contextMessages)
    const stopStrings = resolveGenerationStopStrings({
      useBasicMode: settings.use_basic_stopping_mode,
      basicModeType: settings.basic_stopping_mode_type,
      customStopStrings: settings.stop,
      fillSuffix: plan.suffix
    })
    const genAdvancedExtraBody =
      supportsAdvancedCompat ? settings.advanced_extra_body : {}
    const extraBody = buildExtraBodyPayload({
      top_k: settings.top_k,
      seed: settings.seed,
      stop: stopStrings,
      advanced_extra_body: genAdvancedExtraBody
    })
    const enableLogprobs =
      settings.logprobs && !requestedLogprobsExplicitlyUnsupported
    const requestedTopLogprobs = enableLogprobs
      ? settings.top_logprobs ?? undefined
      : undefined
    const generationRequestOptions = {
      model: selectedModel,
      temperature: settings.temperature,
      maxTokens: settings.max_tokens,
      topP: settings.top_p,
      frequencyPenalty: settings.frequency_penalty,
      presencePenalty: settings.presence_penalty,
      logprobs: enableLogprobs,
      topLogprobs: requestedTopLogprobs,
      systemPrompt: chatMode
        ? undefined
        : plan.mode === "fill"
          ? FILL_SYSTEM_PROMPT
          : PREDICT_SYSTEM_PROMPT,
      extraBody
    }

    generationSessionIdRef.current = activeSessionDetail.id
    generationCancelledRef.current = false
    setIsGenerating(true)
    setIsDirty(true)
    setResponseLogprobs([])
    setResponseInspectorQuery("")
    setResponseInspectorSort("sequence")
    setResponseInspectorHideWhitespace(false)
    setGenerationTokenCount(0)
    setGenerationTokensPerSec(0)
    if (plan.placeholder) {
      setEditorText(plan.prefix + plan.suffix)
    }

    let generated = ""
    let generatedTokenCount = 0
    let streamError: unknown = null
    const streamedLogprobs: WritingLogprobEntry[] = []
    const generationStartMs = performance.now()

    try {
      if (settings.token_streaming) {
        for await (const token of generationServiceRef.current.streamMessage(
          messages,
          generationRequestOptions,
          (chunk) => {
            if (!enableLogprobs) return
            const entries = extractLogprobEntriesFromChunk(chunk)
            if (entries.length === 0) return
            streamedLogprobs.push(...entries)
          }
        )) {
          if (generationCancelledRef.current) break
          generated += token
          generatedTokenCount += 1
          setGenerationTokenCount(generatedTokenCount)
          setGenerationTokensPerSec(
            computeTokensPerSecond(generatedTokenCount, performance.now() - generationStartMs)
          )
          setEditorText(plan.prefix + generated + plan.suffix)
        }
      } else {
        generated = await generationServiceRef.current.sendMessage(
          messages,
          generationRequestOptions,
          (response) => {
            if (!enableLogprobs) return
            const entries = extractLogprobEntriesFromChunk(response)
            if (entries.length === 0) return
            streamedLogprobs.push(...entries)
          }
        )
        generatedTokenCount = estimateTokenCountFromText(generated)
        setGenerationTokenCount(generatedTokenCount)
        setGenerationTokensPerSec(
          computeTokensPerSecond(generatedTokenCount, performance.now() - generationStartMs)
        )
        if (!generationCancelledRef.current) {
          setEditorText(plan.prefix + generated + plan.suffix)
        }
      }
    } catch (error) {
      streamError = error
    }

    const aborted = generationCancelledRef.current || isAbortError(streamError)
    const finalText =
      generated.length > 0 ? plan.prefix + generated + plan.suffix : beforeText
    if (generatedTokenCount > 0) {
      setGenerationTokensPerSec(
        computeTokensPerSecond(generatedTokenCount, performance.now() - generationStartMs)
      )
    }
    if (activeSessionDetail.id === generationSessionIdRef.current) {
      applyHistoryText(finalText)
      pushGenerationHistory(beforeText, finalText)
      setResponseLogprobs(streamedLogprobs)
      if (streamedLogprobs.length > 0) {
        lastGenerationContextRef.current = {
          prefix: plan.prefix,
          suffix: plan.suffix
        }
      } else {
        lastGenerationContextRef.current = null
      }
    }

    generationSessionIdRef.current = null
    generationCancelledRef.current = false
    setIsGenerating(false)

    if (streamError && !aborted) {
      const detail =
        streamError instanceof Error
          ? streamError.message
          : t("option:error", "Error")
      message.error(
        t("option:writingPlayground.generateError", "Generation failed: {{detail}}", { detail })
      )
    }
    },
    [
      activeSessionDetail,
      chatMode,
      editorText,
      effectiveTemplate,
      hasChat,
      isGenerating,
      isRevisionGenerating,
      isOnline,
      selectedModel,
      settings,
      requestedLogprobsExplicitlyUnsupported,
      supportsAdvancedCompat,
      t
    ]
  )

  // Keep handleGenerate ref in sync
  React.useEffect(() => {
    handleGenerateRef.current = handleGenerate
  }, [handleGenerate])

  // =====================================================================
  // Speech (unique - not in hooks)
  // =====================================================================
  const getCurrentEditorAdapter = React.useCallback((): WritingEditorAdapter | null => {
    return activeEditorAdapterRef.current
  }, [])

  const stopSpeech = React.useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    speechUtteranceRef.current = null
    setIsSpeaking(false)
    setIsSpeechPaused(false)
  }, [])

  const pauseSpeech = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return
    const synthesis = window.speechSynthesis
    if (!synthesis.speaking || synthesis.paused) return
    synthesis.pause()
    setIsSpeechPaused(true)
  }, [])

  const resumeSpeech = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return
    const synthesis = window.speechSynthesis
    if (!synthesis.paused) return
    synthesis.resume()
    setIsSpeechPaused(false)
  }, [])

  const updateSpeechPreferences = React.useCallback(
    (patch: Partial<WritingSpeechPreferences>) => {
      const next = normalizeWritingSpeechPreferences({
        rate: speechRate,
        voiceURI: speechVoiceURI,
        ...patch
      })
      setSpeechRate(next.rate)
      setSpeechVoiceURI(next.voiceURI)
      setStoredSpeechPreferences(next)
    },
    [setStoredSpeechPreferences, speechRate, speechVoiceURI]
  )

  const speechVoiceOptions = React.useMemo(
    () => buildSpeechVoiceOptions(speechVoices),
    [speechVoices]
  )
  const pauseResumeAction = resolvePauseResumeAction(isSpeaking, isSpeechPaused)

  const handleSpeakEditor = React.useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      message.warning(
        t("option:writingPlayground.speechUnavailable", "Browser speech synthesis is not available.")
      )
      return
    }
    const selectedText =
      getCurrentEditorAdapter()?.getSelectedText(editorText) ?? ""
    const sourceText = selectedText.trim().length > 0 ? selectedText : editorText
    const utteranceText = markdownToText(sourceText).trim()
    if (!utteranceText) {
      message.info(
        t("option:writingPlayground.speechEmpty", "Write or select text in the editor first.")
      )
      return
    }
    const selectedVoice = resolveSpeechVoice(speechVoices, speechVoiceURI)
    const utterance = new SpeechSynthesisUtterance(utteranceText)
    utterance.rate = clampSpeechRate(speechRate)
    if (selectedVoice) {
      utterance.voice = selectedVoice
    }
    utterance.onend = () => {
      if (speechUtteranceRef.current === utterance) {
        speechUtteranceRef.current = null
        setIsSpeaking(false)
        setIsSpeechPaused(false)
      }
    }
    utterance.onerror = () => {
      if (speechUtteranceRef.current === utterance) {
        speechUtteranceRef.current = null
        setIsSpeaking(false)
        setIsSpeechPaused(false)
      }
    }
    window.speechSynthesis.cancel()
    speechUtteranceRef.current = utterance
    setIsSpeaking(true)
    setIsSpeechPaused(false)
    window.speechSynthesis.speak(utterance)
  }, [editorText, getCurrentEditorAdapter, speechRate, speechVoiceURI, speechVoices, t])

  // --- Speech effects ---
  React.useEffect(() => {
    return () => { generationServiceRef.current.cancelStream() }
  }, [])

  React.useEffect(() => {
    return () => { stopSpeech() }
  }, [stopSpeech])

  React.useEffect(() => {
    const next = normalizeWritingSpeechPreferences(storedSpeechPreferences)
    setSpeechRate(next.rate)
    setSpeechVoiceURI(next.voiceURI)
  }, [storedSpeechPreferences])

  React.useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return
    const synthesis = window.speechSynthesis
    const syncVoices = () => {
      const voices = synthesis.getVoices()
      setSpeechVoices(voices)
      if (speechVoiceURI && voices.some((voice) => voice.voiceURI === speechVoiceURI)) {
        return
      }
      updateSpeechPreferences({
        voiceURI:
          voices.find((voice) => voice.default)?.voiceURI ?? voices[0]?.voiceURI ?? null
      })
    }
    syncVoices()
    synthesis.addEventListener("voiceschanged", syncVoices)
    return () => { synthesis.removeEventListener("voiceschanged", syncVoices) }
  }, [speechVoiceURI, updateSpeechPreferences])

  // --- Reset speech + generation undo on session change ---
  React.useEffect(() => {
    stopSpeech()
    generationUndoRef.current = []
    generationRedoRef.current = []
    generationSessionIdRef.current = null
    generationCancelledRef.current = false
    setCanUndoGeneration(false)
    setCanRedoGeneration(false)
  }, [activeSessionId, stopSpeech])

  // =====================================================================
  // Editor helpers (unique - not in hooks)
  // =====================================================================
  const confirmDeleteSession = React.useCallback(
    (session: WritingSessionListItem) => {
      Modal.confirm({
        title: t("option:writingPlayground.deleteSessionTitle", "Delete session?"),
        content: t("option:writingPlayground.deleteSessionBody", "This will permanently delete the session."),
        okText: t("common:delete", "Delete"),
        okButtonProps: { danger: true },
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => deleteSessionMutation.mutateAsync({ session })
      })
    },
    [deleteSessionMutation, t]
  )

  const focusEditorSelection = React.useCallback(
    (start: number, end: number) => {
      pendingEditorSelectionRef.current = { start, end }
      if (editorView === "preview") {
        setEditorView("edit")
      }
      window.setTimeout(() => {
        applyPendingEditorSelection(getCurrentEditorAdapter())
      }, 0)
    },
    [applyPendingEditorSelection, editorView, getCurrentEditorAdapter]
  )

  const refreshRevisionSelection = React.useCallback(() => {
    const selection = getCurrentEditorAdapter()?.getSelection() ?? null
    updateRevisionSelection(selection)
    return selection
  }, [getCurrentEditorAdapter, updateRevisionSelection])

  const syncScroll = React.useCallback((source: "editor" | "preview") => {
    if (editorView !== "split") return
    if (isSyncingScrollRef.current) return
    const editorEl = editorRef.current?.resizableTextArea?.textArea
    const previewEl = previewRef.current
    if (!editorEl || !previewEl) return
    const sourceEl = source === "editor" ? editorEl : previewEl
    const targetEl = source === "editor" ? previewEl : editorEl
    const maxSource = sourceEl.scrollHeight - sourceEl.clientHeight
    const maxTarget = targetEl.scrollHeight - targetEl.clientHeight
    if (maxSource <= 0 || maxTarget <= 0) return
    const ratio = sourceEl.scrollTop / maxSource
    isSyncingScrollRef.current = true
    targetEl.scrollTop = ratio * maxTarget
    window.setTimeout(() => { isSyncingScrollRef.current = false }, 0)
  }, [editorView])

  const insertPlaceholder = React.useCallback(
    (placeholder: "{predict}" | "{fill}") => {
      const currentValue = editorText
      const selection = getCurrentEditorAdapter()?.getSelection()
      if (!selection) {
        applyPromptValue(currentValue + placeholder, {
          start: currentValue.length + placeholder.length,
          end: currentValue.length + placeholder.length
        })
        return
      }
      const start = selection.start
      const end = selection.end
      const { nextValue, cursor } = applyPlaceholderAtRange(currentValue, start, end, placeholder)
      applyPromptValue(nextValue, { start: cursor, end: cursor })
    },
    [applyPromptValue, editorText, getCurrentEditorAdapter]
  )

  const fillSelectionAtCursor = React.useCallback(() => {
    const currentValue = editorText
    const selection = getCurrentEditorAdapter()?.getSelection()
    const start = selection?.start ?? currentValue.length
    const end = selection?.end ?? currentValue.length
    if (end <= start) {
      message.info(t("option:writingPlayground.fillSelectionRequired", "Select text to replace with {fill}."))
      return
    }
    const { nextValue, cursor } = applyPlaceholderAtRange(currentValue, start, end, "{fill}")
    applyPromptValue(nextValue, { start: cursor, end: cursor })
  }, [applyPromptValue, editorText, getCurrentEditorAdapter, t])

  const insertTokenTextAtCursor = React.useCallback(
    (tokenText: string) => {
      const currentValue = editorText
      const selection = getCurrentEditorAdapter()?.getSelection()
      if (!selection) {
        applyPromptValue(currentValue + tokenText, {
          start: currentValue.length + tokenText.length,
          end: currentValue.length + tokenText.length
        })
        return
      }
      const start = selection.start
      const end = selection.end
      const { nextValue, cursor } = applyTextAtRange(currentValue, start, end, tokenText)
      applyPromptValue(nextValue, { start: cursor, end: cursor })
      message.success(t("option:writingPlayground.tokenInspectorInsertSuccess", "Token text inserted."))
    },
    [applyPromptValue, editorText, getCurrentEditorAdapter, t]
  )

  const insertTemplateBlock = React.useCallback(
    (kind: "system" | "user" | "assistant") => {
      const blocks = {
        system: {
          prefix: effectiveTemplate.systemPrefix,
          suffix: effectiveTemplate.systemSuffix,
          label: t("option:writingPlayground.templateInsertSystem", "System")
        },
        user: {
          prefix: effectiveTemplate.userPrefix,
          suffix: effectiveTemplate.userSuffix,
          label: t("option:writingPlayground.templateInsertUser", "User")
        },
        assistant: {
          prefix: effectiveTemplate.assistantPrefix,
          suffix: effectiveTemplate.assistantSuffix,
          label: t("option:writingPlayground.templateInsertAssistant", "Assistant")
        }
      }
      const block = blocks[kind]
      if (!block.prefix && !block.suffix) {
        message.info(
          t("option:writingPlayground.templateInsertMissing", "{{label}} markers missing in template.", { label: block.label })
        )
        return
      }
      const currentValue = editorText
      const selection = getCurrentEditorAdapter()?.getSelection()
      const start = selection?.start ?? currentValue.length
      const end = selection?.end ?? currentValue.length
      const selected = currentValue.slice(start, end)
      const nextValue =
        currentValue.slice(0, start) + block.prefix + selected + block.suffix + currentValue.slice(end)
      const cursor = start + block.prefix.length + selected.length
      applyPromptValue(nextValue, { start: cursor, end: cursor })
    },
    [applyPromptValue, editorText, effectiveTemplate, getCurrentEditorAdapter, t]
  )

  // =====================================================================
  // Generation history (unique)
  // =====================================================================
  const syncGenerationHistory = React.useCallback(() => {
    setCanUndoGeneration(generationUndoRef.current.length > 0)
    setCanRedoGeneration(generationRedoRef.current.length > 0)
  }, [])

  const pushGenerationHistory = React.useCallback(
    (before: string, after: string) => {
      if (before === after) return
      generationUndoRef.current.push({ before, after })
      generationRedoRef.current = []
      syncGenerationHistory()
    },
    [syncGenerationHistory]
  )

  const applyHistoryText = React.useCallback(
    (nextText: string) => {
      if (activeSessionDetail) {
        applyPromptValue(nextText)
      } else {
        setEditorText(nextText)
      }
    },
    [activeSessionDetail, applyPromptValue]
  )

  const applyRevisionEditorText = React.useCallback(
    (nextText: string): { applied: true } | { applied: false; reason: string } => {
      if (editorMode === "tiptap") {
        return {
          applied: false,
          reason:
            "Rich editor manual apply is required because plain-text replacement does not preserve rich structure."
        }
      }
      applyHistoryText(nextText)
      pushGenerationHistory(editorText, nextText)
      return { applied: true }
    },
    [applyHistoryText, editorMode, editorText, pushGenerationHistory]
  )

  const revisionState = useWritingRevisions({
    activeSessionId,
    activeSessionPayload: activeSessionDetail?.payload,
    editorText,
    applyEditorText: applyRevisionEditorText,
    applySessionPayloadPatch
  })

  const buildRevisionGenerationSettingsSummary = React.useCallback(
    (): Record<string, unknown> => ({
      temperature: settings.temperature,
      topP: settings.top_p,
      topK: settings.top_k,
      maxTokens: settings.max_tokens,
      presencePenalty: settings.presence_penalty,
      frequencyPenalty: settings.frequency_penalty,
      seed: settings.seed,
      stop: settings.stop,
      logprobs: settings.logprobs,
      topLogprobs: settings.top_logprobs,
      tokenStreaming: false,
      useBasicStoppingMode: settings.use_basic_stopping_mode,
      basicStoppingModeType: settings.basic_stopping_mode_type,
      contextLength: settings.context_length,
      authorNoteDepthMode: settings.author_note_depth_mode,
      advancedExtraBody: sanitizeRevisionValue(settings.advanced_extra_body)
    }),
    [settings]
  )

  const buildRevisionContext = React.useCallback(
    (documentText: string) => {
      const contextSettings = {
        memory_block: settings.memory_block,
        author_note: settings.author_note,
        world_info: settings.world_info,
        context_order: settings.context_order,
        context_length: settings.context_length,
        author_note_depth_mode: settings.author_note_depth_mode
      }
      return {
        selectedTemplateName,
        selectedThemeName,
        chatMode,
        contextComposedPrompt: chatMode
          ? null
          : composeContextPrompt(documentText, contextSettings),
        contextMessages: chatMode
          ? buildContextSystemMessages(documentText, contextSettings)
          : null,
        memoryBlock,
        authorNote,
        worldInfoEntries,
        provider: apiProviderOverride ?? null,
        model: selectedModel ?? null,
        generationSettingsSummary: buildRevisionGenerationSettingsSummary()
      }
    },
    [
      apiProviderOverride,
      authorNote,
      buildRevisionGenerationSettingsSummary,
      chatMode,
      memoryBlock,
      selectedModel,
      selectedTemplateName,
      selectedThemeName,
      settings,
      worldInfoEntries
    ]
  )

  const buildRevisionRequestOptions = React.useCallback(() => {
    const stopStrings = resolveGenerationStopStrings({
      useBasicMode: settings.use_basic_stopping_mode,
      basicModeType: settings.basic_stopping_mode_type,
      customStopStrings: settings.stop
    })
    const genAdvancedExtraBody =
      supportsAdvancedCompat
        ? (sanitizeRevisionValue(settings.advanced_extra_body) as Record<
            string,
            unknown
          >)
        : {}
    return {
      model: selectedModel,
      temperature: settings.temperature,
      maxTokens: settings.max_tokens,
      topP: settings.top_p,
      frequencyPenalty: settings.frequency_penalty,
      presencePenalty: settings.presence_penalty,
      logprobs: false,
      topLogprobs: undefined,
      systemPrompt:
        "Return only the requested JSON object for the writing revision.",
      extraBody: buildExtraBodyPayload({
        top_k: settings.top_k,
        seed: settings.seed,
        stop: stopStrings,
        advanced_extra_body: genAdvancedExtraBody
      })
    }
  }, [selectedModel, settings, supportsAdvancedCompat])

  const persistRevisionPreset = React.useCallback(
    (presetId: WritingRevisionPresetId | null | undefined) => {
      if (!presetId) return
      setSelectedRevisionPresetId(presetId)
      applySessionPayloadPatch((payload) => ({
        ...payload,
        revision_preset_id: presetId
      }))
    },
    [applySessionPayloadPatch]
  )

  const createRevisionProposal = React.useCallback(
    async (input: {
      action: WritingRevisionAction
      operation: WritingRevisionOperation
      instruction: string
      target: WritingRevisionTarget
      presetId?: WritingRevisionPresetId | null
      presetInstruction?: string | null
    }): Promise<WritingRevisionProposal | null> => {
      if (!activeSessionDetail) {
        message.info(
          t("option:writingPlayground.selectSession", "Select a session to begin.")
        )
        return null
      }
      if (!isOnline || !hasChat) {
        message.error(
          t("option:writingPlayground.generateUnavailable", "Chat completions unavailable.")
        )
        return null
      }
      if (!selectedModel) {
        message.info(
          t("option:writingPlayground.modelMissing", "Select a model in Settings to generate.")
        )
        return null
      }

      const userPrompt = buildRevisionUserPrompt({
        action: input.action,
        operation: input.operation,
        instruction: input.instruction,
        documentText: editorText,
        target: input.target,
        presetInstruction: input.presetInstruction,
        writingContext: buildRevisionContext(editorText)
      })
      const messages: NonToolMessage[] = [
        {
          role: "user",
          content: userPrompt
        }
      ]
      const responseText = await generationServiceRef.current.sendMessage(
        messages,
        buildRevisionRequestOptions()
      )
      return parseRevisionModelResponse({
        responseText,
        sessionId: activeSessionDetail.id,
        action: input.action,
        operation: input.operation,
        instruction: input.instruction,
        target: input.target,
        presetId: input.presetId,
        presetInstruction: input.presetInstruction
      })
    },
    [
      activeSessionDetail,
      buildRevisionContext,
      buildRevisionRequestOptions,
      editorText,
      hasChat,
      isOnline,
      selectedModel,
      t
    ]
  )

  const resolveFreshRevisionTarget = React.useCallback(
    (request: WritingActionBarRequest): WritingRevisionTarget | null => {
      const adapterSelection = refreshRevisionSelection()
      const freshTarget = resolveRevisionTarget({
        text: editorText,
        action: request.action,
        operation: request.operation,
        selection: adapterSelection,
        cursor: adapterSelection?.end ?? editorText.length,
        preferredTargetMode:
          request.target.requiresConfirmation &&
          request.target.mode === "document"
            ? "document"
            : null
      })

      if (!freshTarget.requiresConfirmation) return freshTarget
      if (
        request.target.requiresConfirmation &&
        request.target.mode === freshTarget.mode
      ) {
        return confirmRevisionTarget(freshTarget)
      }

      message.warning(
        freshTarget.confirmationReason ??
          t(
            "option:writingPlayground.revisionConfirmBroadTarget",
            "Confirm before applying a broad text-changing request."
          )
      )
      return null
    },
    [editorText, refreshRevisionSelection, t]
  )

  const handleRevisionRequest = React.useCallback(
    async (request: WritingActionBarRequest) => {
      if (isGenerating || isRevisionGenerating) return
      const target = resolveFreshRevisionTarget(request)
      if (!target) return

      persistRevisionPreset(request.presetId)
      setIsRevisionGenerating(true)
      try {
        const proposal = await createRevisionProposal({
          action: request.action,
          operation: request.operation,
          instruction: request.instruction,
          target,
          presetId: request.presetId,
          presetInstruction: request.presetInstruction
        })
        if (proposal) {
          revisionState.addRevision(proposal)
        }
      } catch (error) {
        const detail =
          error instanceof Error
            ? error.message
            : t("option:error", "Error")
        message.error(
          t("option:writingPlayground.generateError", "Generation failed: {{detail}}", { detail })
        )
      } finally {
        setIsRevisionGenerating(false)
      }
    },
    [
      createRevisionProposal,
      isGenerating,
      isRevisionGenerating,
      persistRevisionPreset,
      resolveFreshRevisionTarget,
      revisionState,
      t
    ]
  )

  const handleRegenerateRevision = React.useCallback(
    async (proposal: WritingRevisionProposal) => {
      if (isGenerating || isRevisionGenerating) return
      setIsRevisionGenerating(true)
      try {
        await revisionState.regenerateRevision(proposal.id, async (source) => {
          const preset =
            getWritingRevisionPreset(source.presetId) ??
            getWritingRevisionPreset(selectedRevisionPresetId)
          const replacement = await createRevisionProposal({
            action: source.action,
            operation: source.operation,
            instruction: source.instruction,
            target: source.target.requiresConfirmation
              ? confirmRevisionTarget(source.target)
              : source.target,
            presetId: source.presetId ?? preset?.id ?? null,
            presetInstruction:
              source.presetInstruction ?? preset?.instruction ?? null
          })
          if (!replacement) {
            throw new Error("Revision regeneration did not return a proposal.")
          }
          return {
            ...replacement,
            notes: [
              ...(replacement.notes ?? []),
              `Regenerated from ${source.id}`
            ]
          }
        })
      } catch (error) {
        const detail =
          error instanceof Error
            ? error.message
            : t("option:error", "Error")
        message.error(
          t("option:writingPlayground.generateError", "Generation failed: {{detail}}", { detail })
        )
      } finally {
        setIsRevisionGenerating(false)
      }
    },
    [
      createRevisionProposal,
      isGenerating,
      isRevisionGenerating,
      revisionState,
      selectedRevisionPresetId,
      t
    ]
  )

  const displayedRevisionTarget = React.useMemo(
    () =>
      resolveRevisionTarget({
        text: editorText,
        action: "rewrite",
        operation: "replace",
        selection: revisionSelection,
        cursor: revisionSelection?.end ?? 0
      }),
    [editorText, revisionSelection]
  )
  const writingWordCount = React.useMemo(
    () => countWords(editorText),
    [editorText]
  )
  const selectedWordCount = React.useMemo(() => {
    if (!revisionSelection || revisionSelection.start === revisionSelection.end) {
      return 0
    }
    return countWords(
      editorText.slice(revisionSelection.start, revisionSelection.end)
    )
  }, [editorText, revisionSelection])
  const pendingRevisionCount = React.useMemo(
    () =>
      revisionState.revisions.filter((proposal) => proposal.status === "pending")
        .length,
    [revisionState.revisions]
  )

  const handleCopyRevision = React.useCallback(
    async (proposal: WritingRevisionProposal) => {
      const copyText =
        proposal.replacementText ??
        proposal.rawText ??
        proposal.rationale ??
        ""
      if (!copyText) return
      try {
        await navigator.clipboard?.writeText(copyText)
        message.success(
          t("option:writingPlayground.revisionCopySuccess", "Revision copied.")
        )
      } catch {
        message.info(
          t(
            "option:writingPlayground.revisionCopyManual",
            "Copy the suggestion from the revision queue."
          )
        )
      }
    },
    [t]
  )

  const handleUndoGeneration = React.useCallback(() => {
    if (isGenerating) return
    const entry = generationUndoRef.current.pop()
    if (!entry) return
    generationRedoRef.current.push(entry)
    syncGenerationHistory()
    applyHistoryText(entry.before)
  }, [applyHistoryText, isGenerating, syncGenerationHistory])

  const handleRedoGeneration = React.useCallback(() => {
    if (isGenerating) return
    const entry = generationRedoRef.current.pop()
    if (!entry) return
    generationUndoRef.current.push(entry)
    syncGenerationHistory()
    applyHistoryText(entry.after)
  }, [applyHistoryText, isGenerating, syncGenerationHistory])

  const handleCancelGeneration = React.useCallback(() => {
    if (!isGenerating) return
    generationCancelledRef.current = true
    generationServiceRef.current.cancelStream()
  }, [isGenerating])

  // --- Keyboard shortcut ---
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && isGenerating && settings.token_streaming) {
        handleCancelGeneration()
        return
      }
      if (event.key !== "Enter") return
      if (!event.ctrlKey && !event.metaKey) return
      const activeElement = document.activeElement
      const editorEl = editorRef.current?.resizableTextArea?.textArea
      const isPlainEditorFocused = Boolean(editorEl && activeElement === editorEl)
      const isTipTapFocused =
        activeElement instanceof HTMLElement &&
        Boolean(activeElement.closest(".ProseMirror"))
      const isEditorFocused =
        editorMode === "tiptap" ? editorView !== "preview" && isTipTapFocused : isPlainEditorFocused
      if (!isEditorFocused) return
      event.preventDefault()
      void handleGenerate()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => { window.removeEventListener("keydown", handleKeyDown) }
  }, [
    editorMode,
    editorView,
    handleCancelGeneration,
    handleGenerate,
    isGenerating,
    isRevisionGenerating,
    settings.token_streaming
  ])

  // =====================================================================
  // Search & replace (unique)
  // =====================================================================
  const searchData = React.useMemo(() => {
    if (!searchQuery.trim()) {
      return { matches: [], error: null }
    }
    if (useRegex) {
      const regex = buildRegex(searchQuery, { global: true, matchCase })
      if (!regex) {
        return { matches: [], error: t("option:writingPlayground.searchRegexError", "Invalid regex") }
      }
      const matches: Array<{ start: number; end: number }> = []
      let match: RegExpExecArray | null
      while ((match = regex.exec(editorText)) !== null) {
        matches.push({ start: match.index, end: match.index + match[0].length })
        if (match[0].length === 0) { regex.lastIndex += 1 }
        if (matches.length >= MAX_MATCHES) break
      }
      return { matches, error: null }
    }
    const source = matchCase ? editorText : editorText.toLowerCase()
    const query = matchCase ? searchQuery : searchQuery.toLowerCase()
    const matches: Array<{ start: number; end: number }> = []
    let idx = 0
    while (query && (idx = source.indexOf(query, idx)) !== -1) {
      matches.push({ start: idx, end: idx + query.length })
      idx += query.length || 1
      if (matches.length >= MAX_MATCHES) break
    }
    return { matches, error: null }
  }, [editorText, matchCase, searchQuery, t, useRegex])

  const searchMatches = searchData.matches
  const searchError = searchData.error

  React.useEffect(() => { setActiveMatchIndex(0) }, [searchQuery, useRegex, matchCase])
  React.useEffect(() => {
    if (activeMatchIndex >= searchMatches.length) { setActiveMatchIndex(0) }
  }, [activeMatchIndex, searchMatches.length])

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault()
        setFocusMode(!focusMode)
      }
      if (e.key === "Escape" && focusMode) {
        setFocusMode(false)
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [focusMode, setFocusMode])

  const navigateMatch = React.useCallback(
    (direction: "next" | "prev") => {
      if (!searchMatches.length) return
      const nextIndex =
        direction === "next"
          ? (activeMatchIndex + 1) % searchMatches.length
          : (activeMatchIndex - 1 + searchMatches.length) % searchMatches.length
      setActiveMatchIndex(nextIndex)
      const match = searchMatches[nextIndex]
      focusEditorSelection(match.start, match.end)
    },
    [activeMatchIndex, focusEditorSelection, searchMatches]
  )

  const replaceCurrent = React.useCallback(() => {
    if (!searchMatches.length) return
    const match = searchMatches[activeMatchIndex] ?? searchMatches[0]
    if (!match) return
    const matchText = editorText.slice(match.start, match.end)
    let replacement = replaceQuery
    if (useRegex) {
      const regex = buildRegex(searchQuery, { global: false, matchCase })
      if (!regex) return
      replacement = matchText.replace(regex, replaceQuery)
    }
    const nextValue =
      editorText.slice(0, match.start) + replacement + editorText.slice(match.end)
    const cursor = match.start + replacement.length
    applyPromptValue(nextValue, { start: cursor, end: cursor })
  }, [activeMatchIndex, applyPromptValue, editorText, matchCase, replaceQuery, searchMatches, searchQuery, useRegex])

  const replaceAll = React.useCallback(() => {
    if (!searchQuery.trim()) return
    if (useRegex) {
      const regex = buildRegex(searchQuery, { global: true, matchCase })
      if (!regex) return
      const nextValue = editorText.replace(regex, replaceQuery)
      applyPromptValue(nextValue)
      return
    }
    const source = matchCase ? editorText : editorText.toLowerCase()
    const query = matchCase ? searchQuery : searchQuery.toLowerCase()
    if (!query) return
    let idx = 0
    let result = ""
    while (idx < editorText.length) {
      const found = source.indexOf(query, idx)
      if (found === -1) {
        result += editorText.slice(idx)
        break
      }
      result += editorText.slice(idx, found) + replaceQuery
      idx = found + query.length
    }
    applyPromptValue(result)
  }, [applyPromptValue, editorText, matchCase, replaceQuery, searchQuery, useRegex])

  // =====================================================================
  // Menus (unique)
  // =====================================================================
  const templateInsertMenuItems: NonNullable<MenuProps["items"]> = React.useMemo(
    () => [
      {
        key: "template-system",
        label: t("option:writingPlayground.templateInsertSystem", "System"),
        disabled: !effectiveTemplate.systemPrefix && !effectiveTemplate.systemSuffix,
        onClick: () => insertTemplateBlock("system")
      },
      {
        key: "template-user",
        label: t("option:writingPlayground.templateInsertUser", "User"),
        disabled: !effectiveTemplate.userPrefix && !effectiveTemplate.userSuffix,
        onClick: () => insertTemplateBlock("user")
      },
      {
        key: "template-assistant",
        label: t("option:writingPlayground.templateInsertAssistant", "Assistant"),
        disabled: !effectiveTemplate.assistantPrefix && !effectiveTemplate.assistantSuffix,
        onClick: () => insertTemplateBlock("assistant")
      }
    ],
    [effectiveTemplate, insertTemplateBlock, t]
  )

  const insertMenuItems: NonNullable<MenuProps["items"]> = React.useMemo(
    () => [
      {
        key: "predict",
        label: t("option:writingPlayground.insertPredict", "Insert {predict}"),
        onClick: () => insertPlaceholder("{predict}")
      },
      {
        key: "fill",
        label: t("option:writingPlayground.insertFill", "Insert {fill}"),
        onClick: () => insertPlaceholder("{fill}")
      },
      { type: "divider" },
      ...templateInsertMenuItems
    ],
    [insertPlaceholder, t, templateInsertMenuItems]
  )

  const editorMenuItems: MenuProps["items"] = React.useMemo(
    () => [
      {
        key: "predict-here",
        label: t("option:writingPlayground.predictHereAction", "Predict here"),
        onClick: () => insertPlaceholder("{predict}")
      },
      {
        key: "fill-here",
        label: t("option:writingPlayground.fillInMiddleAction", "Fill in middle here"),
        onClick: fillSelectionAtCursor
      },
      { type: "divider" },
      ...templateInsertMenuItems,
      { type: "divider" },
      {
        key: "search",
        label: t("option:writingPlayground.searchReplace", "Search & replace"),
        onClick: () => setSearchOpen(true)
      }
    ],
    [fillSelectionAtCursor, insertPlaceholder, t, templateInsertMenuItems]
  )

  // =====================================================================
  // Derived values (unique)
  // =====================================================================
  const handlePromptChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      applyPromptValue(event.target.value)
    },
    [applyPromptValue]
  )

  const promptChunkData = React.useMemo(() => {
    if (!editorText) {
      return { chunks: [], total: 0, truncated: false }
    }
    const parts = editorText
      .split(
        new RegExp(
          `(${escapeRegex(PREDICT_PLACEHOLDER)}|${escapeRegex(FILL_PLACEHOLDER)})`,
          "g"
        )
      )
      .filter(Boolean)
    const chunks = parts.map((part, index) => ({
      key: `${index}-${part}`,
      type:
        part === PREDICT_PLACEHOLDER || part === FILL_PLACEHOLDER
          ? "placeholder"
          : "text",
      label: part
    }))
    return {
      chunks: chunks.slice(0, MAX_CHUNKS),
      total: chunks.length,
      truncated: chunks.length > MAX_CHUNKS
    }
  }, [editorText])

  const canGenerate =
    Boolean(activeSessionDetail) &&
    Boolean(selectedModel) &&
    hasChat &&
    !isGenerating &&
    !isRevisionGenerating
  const generateDisabledReason = React.useMemo(() => {
    if (isGenerating) return null
    if (isRevisionGenerating)
      return t("option:writingPlayground.disabledRevisionBusy", "Revision request in progress")
    if (!activeSessionDetail)
      return t("option:writingPlayground.disabledNoSession", "Select a session first")
    if (!selectedModel)
      return t("option:writingPlayground.disabledNoModel", "Set a model to generate")
    if (!isOnline)
      return t("option:writingPlayground.disabledOffline", "Server is offline")
    if (!hasChat)
      return t("option:writingPlayground.disabledNoChat", "Chat completions unavailable")
    return null
  }, [
    activeSessionDetail,
    hasChat,
    isGenerating,
    isOnline,
    isRevisionGenerating,
    selectedModel,
    t
  ])

  const saveStatusLabel = React.useMemo(() => {
    if (!activeSessionId) return null
    if (isGenerating) {
      return t("option:writingPlayground.generatingLabel", "Generating...")
    }
    if (
      saveSessionMutation.isPending &&
      savingSessionIdRef.current === activeSessionId
    ) {
      return t("option:writingPlayground.savingLabel", "Saving...")
    }
    if (isDirty) {
      return t("option:writingPlayground.unsavedLabel", "Unsaved changes")
    }
    if (lastSavedAt) {
      return t("option:writingPlayground.savedLabel", "Saved {{time}}", {
        time: formatRelativeTime(new Date(lastSavedAt).toISOString(), t)
      })
    }
    return null
  }, [activeSessionId, isDirty, isGenerating, lastSavedAt, saveSessionMutation.isPending, t])

  const shouldRenderWritingModalHost =
    extraBodyJsonModalOpen ||
    contextPreviewModalOpen ||
    templatesModalOpen ||
    themesModalOpen ||
    createModalOpen ||
    renameModalOpen

  const confirmDeleteTemplate = React.useCallback(
    (template: WritingTemplateResponse) => {
      Modal.confirm({
        title: t("option:writingPlayground.templateDeleteTitle", "Delete template?"),
        content: t("option:writingPlayground.templateDeleteBody", "This will permanently delete the template."),
        okText: t("common:delete", "Delete"),
        okButtonProps: { danger: true },
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => deleteTemplateMutation.mutateAsync({ template })
      })
    },
    [deleteTemplateMutation, t]
  )

  const confirmDeleteTheme = React.useCallback(
    (theme: WritingThemeResponse) => {
      Modal.confirm({
        title: t("option:writingPlayground.themeDeleteTitle", "Delete theme?"),
        content: t("option:writingPlayground.themeDeleteBody", "This will permanently delete the theme."),
        okText: t("common:delete", "Delete"),
        okButtonProps: { danger: true },
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => deleteThemeMutation.mutateAsync({ theme })
      })
    },
    [deleteThemeMutation, t]
  )

  // =====================================================================
  // JSX: Tab content blocks
  // =====================================================================
  const samplingTabContent = (
                  <Card
                    title={t("option:writingPlayground.samplingTitle", "Sampling")}>
                    <WritingPlaygroundActiveSessionGuard
                      hasActiveSession={Boolean(activeSession)}
                      isLoading={activeSessionLoading}
                      hasError={Boolean(activeSessionError)}
                      t={t}>
                      <div className="flex flex-col gap-4">
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <div className="flex flex-col gap-1">
                        <span className="text-xs text-text-muted">
                          {t("option:writingPlayground.topKLabel", "Top K")}
                        </span>
                        <InputNumber
                          size="small"
                          min={0}
                          max={2048}
                          step={1}
                          value={settings.top_k}
                          disabled={settingsDisabled}
                          onChange={(value) =>
                            updateSetting({
                              top_k: value == null ? DEFAULT_SETTINGS.top_k : value
                            })
                          }
                          className="w-full"
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-xs text-text-muted">
                          {t("option:writingPlayground.presencePenaltyLabel", "Presence penalty")}
                        </span>
                        <InputNumber
                          size="small"
                          min={-2}
                          max={2}
                          step={0.1}
                          value={settings.presence_penalty}
                          disabled={settingsDisabled}
                          onChange={(value) =>
                            updateSetting({
                              presence_penalty: value == null ? DEFAULT_SETTINGS.presence_penalty : value
                            })
                          }
                          className="w-full"
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-xs text-text-muted">
                          {t("option:writingPlayground.frequencyPenaltyLabel", "Frequency penalty")}
                        </span>
                        <InputNumber
                          size="small"
                          min={-2}
                          max={2}
                          step={0.1}
                          value={settings.frequency_penalty}
                          disabled={settingsDisabled}
                          onChange={(value) =>
                            updateSetting({
                              frequency_penalty: value == null ? DEFAULT_SETTINGS.frequency_penalty : value
                            })
                          }
                          className="w-full"
                        />
                      </div>
                      <div className="flex flex-col gap-1 sm:col-span-2">
                        <span className="text-xs text-text-muted">
                          {t("option:writingPlayground.seedLabel", "Seed")}
                        </span>
                        <InputNumber
                          size="small"
                          min={0}
                          step={1}
                          value={settings.seed ?? null}
                          disabled={settingsDisabled}
                          onChange={(value) =>
                            updateSetting({
                              seed: value == null ? null : Math.max(0, Math.floor(value))
                            })
                          }
                          className="w-full"
                        />
                      </div>
                      <div className="flex flex-col gap-2 sm:col-span-2">
                        <Checkbox
                          checked={settings.use_basic_stopping_mode}
                          disabled={settingsDisabled}
                          onChange={(event) =>
                            updateSetting({ use_basic_stopping_mode: event.target.checked })
                          }>
                          {t("option:writingPlayground.basicStoppingModeLabel", "Stop condition")}
                        </Checkbox>
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_180px]">
                          <span className="text-xs text-text-muted">
                            {t("option:writingPlayground.basicStoppingModeHint", "Mikupad-style quick stopping presets for generation.")}
                          </span>
                          <Select
                            size="small"
                            value={settings.basic_stopping_mode_type}
                            disabled={settingsDisabled || !settings.use_basic_stopping_mode}
                            options={[
                              { value: "max_tokens", label: "Max tokens" },
                              { value: "new_line", label: "New line" },
                              { value: "fill_suffix", label: "Fill suffix" }
                            ]}
                            onChange={(value) =>
                              updateSetting({ basic_stopping_mode_type: value as BasicStoppingModeType })
                            }
                          />
                        </div>
                      </div>
                      <div className="flex flex-col gap-2 sm:col-span-2">
                        <Checkbox
                          checked={settings.logprobs}
                          disabled={logprobsControlsDisabled}
                          onChange={(event) => {
                            const checked = event.target.checked
                            updateSetting({
                              logprobs: checked,
                              top_logprobs: checked ? settings.top_logprobs ?? DEFAULT_TOP_LOGPROBS : null
                            })
                          }}>
                          {t("option:writingPlayground.logprobsLabel", "Enable logprobs")}
                        </Checkbox>
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_180px]">
                          <span className="text-xs text-text-muted">{topLogprobsHint}</span>
                          <div className="flex flex-col gap-1">
                            <span className="text-xs text-text-muted">
                              {t("option:writingPlayground.topLogprobsLabel", "Top logprobs")}
                            </span>
                            <InputNumber
                              size="small"
                              min={1}
                              max={20}
                              step={1}
                              value={settings.top_logprobs ?? null}
                              disabled={topLogprobsControlsDisabled}
                              placeholder={String(DEFAULT_TOP_LOGPROBS)}
                              onChange={(value) =>
                                updateSetting({
                                  top_logprobs: value == null ? null : Math.max(1, Math.min(20, Math.floor(value)))
                                })
                              }
                              className="w-full"
                            />
                          </div>
                        </div>
                        {logprobsUnavailableReason ? (
                          <DesignSystemAlert variant="info" title={logprobsUnavailableReason} />
                        ) : null}
                      </div>
                    </div>
                    <div className="flex flex-col gap-1">
                      <span className="text-xs text-text-muted">
                        {t("option:writingPlayground.stopStringsLabel", "Stop strings")}
                      </span>
                      <Input.TextArea
                        value={stopStringsInput}
                        disabled={settingsDisabled || settings.use_basic_stopping_mode}
                        onChange={(event) => {
                          const nextInput = event.target.value
                          updateSetting({ stop: normalizeStopStrings(nextInput) }, nextInput)
                        }}
                        placeholder={t("option:writingPlayground.stopStringsPlaceholder", "One per line")}
                        rows={4}
                      />
                    </div>
                    {showAdvancedSamplerControls && (
                      <Collapse
                        ghost
                        size="small"
                        defaultActiveKey={supportsAdvancedCompat ? [] : ["advanced-extra-body"]}
                        items={[
                          {
                            key: "advanced-extra-body",
                            label: t("option:writingPlayground.advancedSamplerLabel", "Advanced sampler controls (extra_body)"),
                            children: (
                              <div className="flex flex-col gap-3">
                                {extraBodyCompat ? (
                                  <span className="text-xs text-text-muted">
                                    {extraBodyCompat.notes || t("option:writingPlayground.advancedSamplerHint", "Advanced controls are sent through extra_body for provider compatibility.")}
                                  </span>
                                ) : null}
                                {!supportsAdvancedCompat ? (
                                  <DesignSystemAlert
                                    variant="info"
                                    title={extraBodyCompat?.effective_reason || t("option:writingPlayground.advancedUnsupported", "Advanced sampler controls are disabled by runtime configuration.")}
                                  />
                                ) : null}
                                <div className="flex flex-wrap items-center gap-2">
                                  <Button size="small" disabled={settingsDisabled || !supportsAdvancedCompat} onClick={openExtraBodyJsonEditor}>
                                    {t("option:writingPlayground.extraBodyJsonEditorLabel", "Edit raw extra_body JSON")}
                                  </Button>
                                  {hasAdvancedSettingsValues ? (
                                    <Button
                                      size="small"
                                      disabled={settingsDisabled}
                                      onClick={() => {
                                        updateSetting({ advanced_extra_body: {} })
                                        setBannedTokensInput("")
                                        setDrySequenceBreakersInput("")
                                        setLogitBiasInput("")
                                        setLogitBiasError(null)
                                        setLogitBiasTokenInput("")
                                        setLogitBiasValueInput(null)
                                      }}>
                                      {t("option:writingPlayground.clearAdvancedSettings", "Clear advanced values")}
                                    </Button>
                                  ) : null}
                                </div>
                                {knownExtraBodyParams.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {knownExtraBodyParams.map((param) => (<Tag key={param}>{param}</Tag>))}
                                  </div>
                                ) : null}
                                {advancedExtraBodyUnknownKeys.length > 0 ? (
                                  <div className="flex flex-col gap-1">
                                    <span className="text-xs text-text-muted">
                                      {t("option:writingPlayground.extraBodyCustomKeys", "Custom keys from raw JSON:")}
                                    </span>
                                    <div className="flex flex-wrap gap-1">
                                      {advancedExtraBodyUnknownKeys.map((key) => (<Tag key={key}>{key}</Tag>))}
                                    </div>
                                  </div>
                                ) : null}
                                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                  {ADVANCED_NUMBER_PARAMS.filter((param) => shouldShowAdvancedParam(param.key)).map((param) => (
                                    <div key={param.key} className="flex flex-col gap-1">
                                      <span className="text-xs text-text-muted">{param.label}</span>
                                      <InputNumber
                                        size="small"
                                        min={param.min}
                                        max={param.max}
                                        step={param.step}
                                        value={getAdvancedNumberValue(param.key)}
                                        disabled={settingsDisabled || !supportsAdvancedCompat}
                                        onChange={(value) => {
                                          const parsed = value == null ? null : param.step === 1 ? Math.round(value) : value
                                          updateAdvancedExtraBodyField(param.key, parsed)
                                        }}
                                        className="w-full"
                                      />
                                    </div>
                                  ))}
                                </div>
                                {shouldShowAdvancedParam("ignore_eos") && (
                                  <Checkbox
                                    checked={Boolean(advancedExtraBody.ignore_eos)}
                                    disabled={settingsDisabled || !supportsAdvancedCompat}
                                    onChange={(event) => updateAdvancedExtraBodyField("ignore_eos", event.target.checked ? true : null)}>
                                    {t("option:writingPlayground.ignoreEosLabel", "Ignore EOS")}
                                  </Checkbox>
                                )}
                                {shouldShowAdvancedParam("penalize_nl") && (
                                  <Checkbox
                                    checked={Boolean(advancedExtraBody.penalize_nl)}
                                    disabled={settingsDisabled || !supportsAdvancedCompat}
                                    onChange={(event) => updateAdvancedExtraBodyField("penalize_nl", event.target.checked ? true : null)}>
                                    {t("option:writingPlayground.penalizeNewlineLabel", "Penalize newline")}
                                  </Checkbox>
                                )}
                                {shouldShowAdvancedParam("post_sampling_probs") && (
                                  <Checkbox
                                    checked={Boolean(advancedExtraBody.post_sampling_probs)}
                                    disabled={settingsDisabled || !supportsAdvancedCompat}
                                    onChange={(event) => updateAdvancedExtraBodyField("post_sampling_probs", event.target.checked ? true : null)}>
                                    {t("option:writingPlayground.postSamplingProbsLabel", "Post-sampling probabilities")}
                                  </Checkbox>
                                )}
                                {shouldShowAdvancedParam("banned_tokens") && (
                                  <div className="flex flex-col gap-1">
                                    <span className="text-xs text-text-muted">{t("option:writingPlayground.bannedTokensLabel", "Banned tokens (comma or newline separated)")}</span>
                                    <Input.TextArea
                                      value={bannedTokensInput}
                                      rows={3}
                                      disabled={settingsDisabled || !supportsAdvancedCompat}
                                      onChange={(event) => {
                                        const nextInput = event.target.value
                                        setBannedTokensInput(nextInput)
                                        updateAdvancedExtraBodyField("banned_tokens", parseStringListInput(nextInput))
                                      }}
                                    />
                                  </div>
                                )}
                                {shouldShowAdvancedParam("dry_sequence_breakers") && (
                                  <div className="flex flex-col gap-1">
                                    <span className="text-xs text-text-muted">{t("option:writingPlayground.drySequenceBreakersLabel", "DRY sequence breakers (comma or newline separated)")}</span>
                                    <Input.TextArea
                                      value={drySequenceBreakersInput}
                                      rows={3}
                                      disabled={settingsDisabled || !supportsAdvancedCompat}
                                      onChange={(event) => {
                                        const nextInput = event.target.value
                                        setDrySequenceBreakersInput(nextInput)
                                        updateAdvancedExtraBodyField("dry_sequence_breakers", parseStringListInput(nextInput))
                                      }}
                                    />
                                  </div>
                                )}
                                {shouldShowAdvancedParam("grammar") && (
                                  <div className="flex flex-col gap-1">
                                    <span className="text-xs text-text-muted">{t("option:writingPlayground.grammarLabel", "Grammar")}</span>
                                    <Input.TextArea
                                      value={typeof advancedExtraBody.grammar === "string" ? advancedExtraBody.grammar : ""}
                                      rows={4}
                                      disabled={settingsDisabled || !supportsAdvancedCompat}
                                      onChange={(event) => updateAdvancedExtraBodyField("grammar", event.target.value)}
                                    />
                                  </div>
                                )}
                                {shouldShowAdvancedParam("logit_bias") && (
                                  <div className="flex flex-col gap-1">
                                    <span className="text-xs text-text-muted">{t("option:writingPlayground.logitBiasLabel", "Logit bias (JSON object)")}</span>
                                    <Input.TextArea
                                      value={logitBiasInput}
                                      rows={4}
                                      disabled={settingsDisabled || !supportsAdvancedCompat}
                                      placeholder='{"50256": -100, "198": -1.5}'
                                      onChange={(event) => {
                                        const nextInput = event.target.value
                                        setLogitBiasInput(nextInput)
                                        const parsed = parseLogitBiasInput(nextInput)
                                        if (parsed.error) { setLogitBiasError(parsed.error); return }
                                        applyLogitBiasObject(parsed.value ?? {})
                                      }}
                                    />
                                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_140px_auto]">
                                      <Input
                                        value={logitBiasTokenInput}
                                        disabled={settingsDisabled || !supportsAdvancedCompat}
                                        placeholder={t("option:writingPlayground.logitBiasTokenPlaceholder", "Token id (e.g. 50256)")}
                                        onChange={(event) => setLogitBiasTokenInput(event.target.value)}
                                      />
                                      <InputNumber
                                        min={-100}
                                        max={100}
                                        step={0.1}
                                        value={logitBiasValueInput}
                                        disabled={settingsDisabled || !supportsAdvancedCompat}
                                        placeholder={"0"}
                                        onChange={(value) => setLogitBiasValueInput(value == null ? null : value)}
                                      />
                                      <Button
                                        disabled={settingsDisabled || !supportsAdvancedCompat || !logitBiasTokenInput.trim() || logitBiasValueInput == null}
                                        onClick={() => {
                                          const next = withLogitBiasEntry(advancedExtraBody.logit_bias, logitBiasTokenInput, logitBiasValueInput)
                                          applyLogitBiasObject(next)
                                          setLogitBiasTokenInput("")
                                          setLogitBiasValueInput(null)
                                        }}>
                                        {t("option:writingPlayground.logitBiasAddAction", "Add")}
                                      </Button>
                                    </div>
                                    {logitBiasEntries.length > 0 ? (
                                      <div className="flex flex-wrap gap-1">
                                        {logitBiasEntries.map(([token, bias]) => (
                                          <Tag
                                            key={token}
                                            closable
                                            onClose={(event) => {
                                              event.preventDefault()
                                              const next = withoutLogitBiasEntry(advancedExtraBody.logit_bias, token)
                                              applyLogitBiasObject(next)
                                            }}>
                                            {`${token}: ${bias}`}
                                          </Tag>
                                        ))}
                                      </div>
                                    ) : null}
                                    {logitBiasError ? (
                                      <DesignSystemAlert variant="error" title={logitBiasError} />
                                    ) : null}
                                  </div>
                                )}
                              </div>
                            )
                          }
                        ]}
                      />
                    )}
                      </div>
                    </WritingPlaygroundActiveSessionGuard>
                  </Card>
  )

  const contextTabContent = (
                  <Card
                    title={t("option:writingPlayground.sidebarContext", "Context")}
                    extra={
                      <Button size="small" disabled={settingsDisabled} onClick={() => setContextPreviewModalOpen(true)}>
                        {t("option:writingPlayground.contextPreviewAction", "Preview")}
                      </Button>
                    }>
                    <WritingPlaygroundActiveSessionGuard
                      hasActiveSession={Boolean(activeSession)}
                      isLoading={activeSessionLoading}
                      hasError={Boolean(activeSessionError)}
                      t={t}>
                      <div className="flex flex-col gap-4">
                                    <div className="rounded-md border border-border bg-surface p-3">
                                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                        <div className="flex flex-col gap-1">
                                          <span className="text-xs text-text-muted">{t("option:writingPlayground.contextLengthLabel", "Context length (tokens)")}</span>
                                          <InputNumber size="small" min={0} step={128} value={settings.context_length} disabled={settingsDisabled} onChange={(value) => updateSetting({ context_length: value == null ? DEFAULT_SETTINGS.context_length : Math.max(0, Math.floor(value)) })} />
                                        </div>
                                        <div className="flex flex-col gap-1">
                                          <span className="text-xs text-text-muted">{t("option:writingPlayground.authorDepthModeLabel", "Author note depth mode")}</span>
                                          <Select size="small" value={settings.author_note_depth_mode} disabled={settingsDisabled} onChange={(value) => updateSetting({ author_note_depth_mode: value === "annotation" ? "annotation" : "insertion" })} options={[{ value: "insertion", label: t("option:writingPlayground.authorDepthModeInsertion", "Insertion") }, { value: "annotation", label: t("option:writingPlayground.authorDepthModeAnnotation", "Annotation") }]} />
                                        </div>
                                      </div>
                                      <div className="mt-3 flex flex-col gap-1">
                                        <span className="text-xs text-text-muted">{t("option:writingPlayground.contextOrderLabel", "Context order")}</span>
                                        <Input.TextArea value={settings.context_order} rows={2} disabled={settingsDisabled} placeholder={DEFAULT_CONTEXT_ORDER} onChange={(event) => updateSetting({ context_order: event.target.value })} />
                                        <span className="text-[11px] text-text-muted">{t("option:writingPlayground.contextOrderHint", "Placeholders: {memPrefix}, {wiPrefix}, {wiText}, {wiSuffix}, {memText}, {memSuffix}, {prompt}")}</span>
                                      </div>
                                    </div>
                                    <div className="rounded-md border border-border bg-surface p-3">
                                      <div className="flex items-center justify-between gap-2">
                                        <span className="text-xs font-medium text-text">{t("option:writingPlayground.memoryBlockLabel", "Memory block")}</span>
                                        <Checkbox checked={memoryBlock.enabled} disabled={settingsDisabled} onChange={(event) => updateMemoryBlock({ enabled: event.target.checked })}>{t("common:enabled", "Enabled")}</Checkbox>
                                      </div>
                                      <div className="mt-3 grid grid-cols-1 gap-3">
                                        <Input value={memoryBlock.prefix} disabled={settingsDisabled} placeholder={t("option:writingPlayground.prefixLabel", "Prefix")} onChange={(event) => updateMemoryBlock({ prefix: event.target.value })} />
                                        <Input.TextArea value={memoryBlock.text} rows={3} disabled={settingsDisabled} placeholder={t("option:writingPlayground.memoryTextPlaceholder", "Facts and reminders to keep consistent.")} onChange={(event) => updateMemoryBlock({ text: event.target.value })} />
                                        <Input value={memoryBlock.suffix} disabled={settingsDisabled} placeholder={t("option:writingPlayground.suffixLabel", "Suffix")} onChange={(event) => updateMemoryBlock({ suffix: event.target.value })} />
                                      </div>
                                    </div>
                                    <div className="rounded-md border border-border bg-surface p-3">
                                      <div className="flex flex-wrap items-center justify-between gap-2">
                                        <span className="text-xs font-medium text-text">{t("option:writingPlayground.authorNoteLabel", "Author note")}</span>
                                        <div className="flex items-center gap-3">
                                          <span className="text-xs text-text-muted">{t("option:writingPlayground.authorDepthLabel", "Depth")}</span>
                                          <InputNumber size="small" min={1} step={1} value={authorNote.insertion_depth} disabled={settingsDisabled} onChange={(value) => updateAuthorNote({ insertion_depth: value == null ? 1 : Math.max(1, Math.floor(value)) })} />
                                          <Checkbox checked={authorNote.enabled} disabled={settingsDisabled} onChange={(event) => updateAuthorNote({ enabled: event.target.checked })}>{t("common:enabled", "Enabled")}</Checkbox>
                                        </div>
                                      </div>
                                      <div className="mt-3 grid grid-cols-1 gap-3">
                                        <Input value={authorNote.prefix} disabled={settingsDisabled} placeholder={t("option:writingPlayground.prefixLabel", "Prefix")} onChange={(event) => updateAuthorNote({ prefix: event.target.value })} />
                                        <Input.TextArea value={authorNote.text} rows={3} disabled={settingsDisabled} placeholder={t("option:writingPlayground.authorNotePlaceholder", "Guidance to inject during generation.")} onChange={(event) => updateAuthorNote({ text: event.target.value })} />
                                        <Input value={authorNote.suffix} disabled={settingsDisabled} placeholder={t("option:writingPlayground.suffixLabel", "Suffix")} onChange={(event) => updateAuthorNote({ suffix: event.target.value })} />
                                      </div>
                                    </div>
                                    <div className="rounded-md border border-border bg-surface p-3">
                                      <div className="flex flex-wrap items-center justify-between gap-2">
                                        <span className="text-xs font-medium text-text">{t("option:writingPlayground.worldInfoLabel", "World info")}</span>
                                        <div className="flex items-center gap-2">
                                          <Checkbox checked={worldInfo.enabled} disabled={settingsDisabled} onChange={(event) => updateWorldInfo({ enabled: event.target.checked })}>{t("common:enabled", "Enabled")}</Checkbox>
                                          <Button size="small" disabled={settingsDisabled} onClick={addWorldInfoEntry}>{t("option:writingPlayground.addWorldInfo", "Add entry")}</Button>
                                          <WritingWorldInfoImportControls disabled={settingsDisabled} worldInfo={worldInfo} onImported={handleWorldInfoImported} onImportError={handleWorldInfoImportError} t={t} />
                                          <Button size="small" disabled={settingsDisabled} onClick={handleWorldInfoExport}>{t("option:writingPlayground.worldInfoExportAction", "Export")}</Button>
                                        </div>
                                      </div>
                                      <div className="mt-3 flex flex-col gap-3">
                                        <div className="flex items-center gap-2">
                                          <span className="text-xs text-text-muted">{t("option:writingPlayground.worldInfoSearchRange", "Search range (chars)")}</span>
                                          <InputNumber size="small" min={0} step={100} value={worldInfo.search_range} disabled={settingsDisabled} onChange={(value) => updateWorldInfo({ search_range: value == null ? 0 : Math.max(0, Math.floor(value)) })} />
                                        </div>
                                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                                          <Input value={worldInfo.prefix} disabled={settingsDisabled} placeholder={t("option:writingPlayground.worldInfoPrefixLabel", "World info prefix")} onChange={(event) => updateWorldInfo({ prefix: event.target.value })} />
                                          <Input value={worldInfo.suffix} disabled={settingsDisabled} placeholder={t("option:writingPlayground.worldInfoSuffixLabel", "World info suffix")} onChange={(event) => updateWorldInfo({ suffix: event.target.value })} />
                                        </div>
                                        {worldInfoEntries.length === 0 ? (
                                          <span className="text-xs text-text-muted">{t("option:writingPlayground.worldInfoEmpty", "No world info entries yet.")}</span>
                                        ) : (
                                          <div className="flex flex-col gap-3">
                                            {worldInfoEntries.map((entry, index) => (
                                              <div key={entry.id} className="rounded-md border border-border bg-background p-3">
                                                <div className="flex items-center justify-between gap-2">
                                                  <span className="text-xs font-medium text-text">{entry.display_name?.trim() ? entry.display_name : t("option:writingPlayground.worldInfoEntryLabel", "Entry {{index}}", { index: index + 1 })}</span>
                                                  <div className="flex items-center gap-2">
                                                    <Button size="small" disabled={settingsDisabled || index === 0} onClick={() => moveWorldInfoEntryById(entry.id, "up")}>{t("option:writingPlayground.worldInfoMoveUp", "Move up")}</Button>
                                                    <Button size="small" disabled={settingsDisabled || index >= worldInfoEntries.length - 1} onClick={() => moveWorldInfoEntryById(entry.id, "down")}>{t("option:writingPlayground.worldInfoMoveDown", "Move down")}</Button>
                                                    <Button size="small" danger disabled={settingsDisabled} onClick={() => removeWorldInfoEntry(entry.id)}>{t("common:delete", "Delete")}</Button>
                                                  </div>
                                                </div>
                                                <div className="mt-3 flex flex-col gap-3">
                                                  <Input value={entry.display_name || ""} disabled={settingsDisabled} placeholder={t("option:writingPlayground.worldInfoDisplayNamePlaceholder", "Display name (optional)")} onChange={(event) => updateWorldInfoEntry(entry.id, { display_name: event.target.value })} />
                                                  <div className="flex flex-wrap items-center gap-3">
                                                    <Checkbox checked={entry.enabled} disabled={settingsDisabled} onChange={(event) => updateWorldInfoEntry(entry.id, { enabled: event.target.checked })}>{t("common:enabled", "Enabled")}</Checkbox>
                                                    <Checkbox checked={entry.use_regex} disabled={settingsDisabled} onChange={(event) => updateWorldInfoEntry(entry.id, { use_regex: event.target.checked })}>{t("option:writingPlayground.worldInfoRegex", "Regex")}</Checkbox>
                                                    <Checkbox checked={entry.case_sensitive} disabled={settingsDisabled} onChange={(event) => updateWorldInfoEntry(entry.id, { case_sensitive: event.target.checked })}>{t("option:writingPlayground.worldInfoCaseSensitive", "Case sensitive")}</Checkbox>
                                                  </div>
                                                  <div className="flex flex-wrap items-center gap-2">
                                                    <span className="text-xs text-text-muted">{t("option:writingPlayground.worldInfoEntrySearchRange", "Entry search range (chars)")}</span>
                                                    <InputNumber size="small" min={0} step={100} value={entry.search_range ?? worldInfo.search_range} disabled={settingsDisabled} onChange={(value) => updateWorldInfoEntry(entry.id, { search_range: value == null ? undefined : Math.max(0, Math.floor(value)) })} />
                                                    <Button size="small" disabled={settingsDisabled} onClick={() => updateWorldInfoEntry(entry.id, { search_range: undefined })}>{t("option:writingPlayground.worldInfoUseGlobalRange", "Use global")}</Button>
                                                  </div>
                                                  <Input.TextArea value={entry.keys.join("\n")} rows={2} disabled={settingsDisabled} placeholder={t("option:writingPlayground.worldInfoKeysPlaceholder", "Trigger keys (comma or newline separated)")} onChange={(event) => updateWorldInfoEntry(entry.id, { keys: parseWorldInfoKeysInput(event.target.value) })} />
                                                  <Input.TextArea value={entry.content} rows={3} disabled={settingsDisabled} placeholder={t("option:writingPlayground.worldInfoContentPlaceholder", "Context to inject when triggered.")} onChange={(event) => updateWorldInfoEntry(entry.id, { content: event.target.value })} />
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    </div>
                      </div>
                    </WritingPlaygroundActiveSessionGuard>
                  </Card>
  )

  const setupTabContent = (
                  <Card
                    title={t("option:writingPlayground.sidebarSetup", "Setup")}
                    extra={
                      <div className="flex items-center gap-2">
                        <Button size="small" onClick={handleOpenTemplatesModal} disabled={templateSelectDisabled}>{t("option:writingPlayground.manageTemplates", "Manage templates")}</Button>
                        <Button size="small" onClick={handleOpenThemesModal} disabled={themeSelectDisabled}>{t("option:writingPlayground.manageThemes", "Manage themes")}</Button>
                      </div>
                    }>
                    <WritingPlaygroundActiveSessionGuard hasActiveSession={Boolean(activeSession)} isLoading={activeSessionLoading} hasError={Boolean(activeSessionError)} t={t}>
                      <div className="flex flex-col gap-3">
                          <div className="flex flex-col gap-1">
                            <span className="text-xs text-text-muted">{t("option:writingPlayground.templateLabel", "Template")}</span>
                            <Select allowClear size="small" options={templateOptions} loading={templatesLoading} value={selectedTemplateName ?? undefined} disabled={templateSelectDisabled} placeholder={t("option:writingPlayground.templatePlaceholder", "Server default")} onChange={(value) => handleTemplateChange(value ? String(value) : null)} />
                            <span className="text-xs text-text-muted">{templatesError ? t("option:writingPlayground.templateError", "Unable to load templates.") : !hasTemplates ? t("option:writingPlayground.templateUnavailable", "Templates unavailable.") : t("option:writingPlayground.templateHint", "Choose an instruct template for chat parsing and FIM.")}</span>
                          </div>
                          <div className="flex flex-col gap-1">
                            <span className="text-xs text-text-muted">{t("option:writingPlayground.themeLabel", "Theme")}</span>
                            <Select allowClear size="small" options={themeOptions} loading={themesLoading} value={selectedThemeName ?? undefined} disabled={themeSelectDisabled} placeholder={t("option:writingPlayground.themePlaceholder", "Server default")} onChange={(value) => handleThemeChange(value ? String(value) : null)} />
                            <span className="text-xs text-text-muted">{themesError ? t("option:writingPlayground.themeError", "Unable to load themes.") : !hasThemes ? t("option:writingPlayground.themeUnavailable", "Themes unavailable.") : t("option:writingPlayground.themeHint", "Apply a theme to style the editor.")}</span>
                          </div>
                          <div className="flex flex-col gap-1">
                            <Checkbox checked={chatMode} disabled={settingsDisabled} onChange={(event) => handleChatModeChange(event.target.checked)}>{t("option:writingPlayground.chatModeLabel", "Chat mode")}</Checkbox>
                            <span className="text-xs text-text-muted">{t("option:writingPlayground.chatModeHint", "Parse prompt text into messages using the selected template.")}</span>
                          </div>
                      </div>
                    </WritingPlaygroundActiveSessionGuard>
                  </Card>
  )

  const inspectTabContent = (
                  <WritingPlaygroundDiagnosticsPanel
                    title={t("option:writingPlayground.sidebarInspect", "Analysis")}
                    t={t}
                    status={diagnosticsSummary.status}
                    showOffline={showOffline}
                    showUnsupported={showUnsupported}
                    hasActiveSession={Boolean(activeSession)}
                    response={{
                      enabled: showResponseInspectorPanel,
                      responseInspectorRowsCount: responseInspectorRowsAll.length,
                      responseLogprobsCount: responseLogprobs.length,
                      settingsLogprobsEnabled: settings.logprobs,
                      settingsDisabled,
                      responseLogprobRowsCount: responseLogprobRows.length,
                      responseLogprobTruncated,
                      onCopyResponseInspectorJson: handleCopyResponseInspectorJson,
                      onExportResponseInspectorCsv: handleExportResponseInspectorCsv,
                      onClearResponseInspector: clearResponseInspector
                    }}
                    token={{
                      enabled: showTokenInspectorPanel,
                      tokenizerName,
                      serverSupportsTokenCount,
                      canCountTokens,
                      isCountingTokens,
                      onCountTokens: () => handleCountTokens(editorText),
                      serverSupportsTokenize,
                      canTokenizePreview,
                      isTokenizingText,
                      onTokenizePreview: () => handleTokenizePreview(editorText),
                      hasTokenCountResult: Boolean(tokenCountResult),
                      tokenCountValue: tokenCountResult?.count ?? null,
                      hasTokenizeResult: Boolean(tokenizeResult),
                      tokenInspectorError,
                      tokenInspectorBusy,
                      tokenInspectorUnavailableReason,
                      onClearTokenInspector: clearTokenInspector,
                      tokenPreviewRowsCount: tokenPreviewRows.length,
                      tokenPreviewTotal
                    }}
                    wordcloud={{
                      enabled: showWordcloudPanel,
                      wordcloudStatus,
                      wordcloudStatusColor,
                      canGenerateWordcloud,
                      isGeneratingWordcloud,
                      onGenerateWordcloud: () => handleGenerateWordcloud(editorText),
                      wordcloudError,
                      onClearWordcloud: clearWordcloud,
                      wordcloudWords
                    }}
                  />
  )

  const libraryDrawerContent = (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-border">
        <span className="text-sm font-medium">{t("option:writingPlayground.sessionsTitle", "Sessions")}</span>
        <div className="flex items-center gap-1">
          <Button size="small" type="primary" onClick={() => setCreateModalOpen(true)}>{t("option:writingPlayground.newSession", "New session")}</Button>
          <Dropdown
            menu={{
              items: [
                { key: "import", label: t("option:writingPlayground.importSession", "Import"), disabled: sessionImportDisabled, onClick: () => sessionFileInputRef.current?.click() },
                ...(hasSnapshots ? [
                  { type: "divider" as const },
                  { key: "export-all", label: t("option:writingPlayground.exportSnapshot", "Export all"), disabled: snapshotExportDisabled, onClick: () => { void exportSnapshot() } },
                  { key: "import-all", label: t("option:writingPlayground.importSnapshot", "Import all"), disabled: snapshotImportDisabled, onClick: () => openSnapshotImportPicker("merge") },
                  { key: "replace-all", label: t("option:writingPlayground.replaceSnapshot", "Replace all"), danger: true, disabled: snapshotImportDisabled, onClick: () => openSnapshotImportPicker("replace") }
                ] : [])
              ]
            }}
            trigger={["click"]}>
            <Button size="small" icon={<MoreHorizontal className="h-3.5 w-3.5" />} />
          </Dropdown>
        </div>
      </div>
      <div className="px-3 py-2 border-b border-border">
        <Segmented
          block
          size="small"
          value={libraryView}
          onChange={(v) => setLibraryView(v as "sessions" | "manuscript")}
          options={[
            { value: "sessions", label: "Sessions" },
            { value: "manuscript", label: "Manuscript" },
          ]}
        />
      </div>
      <div className="flex-1 overflow-y-auto px-2 py-1">
        {libraryView === "manuscript" ? (
          <ManuscriptTreePanel isOnline={isOnline} />
        ) : (
        sessionsLoading ? (<Skeleton active />) : sessionsError ? (
          <DesignSystemAlert
            variant="error"
            title={t("option:writingPlayground.sessionsError", "Unable to load sessions.")}
          />
        ) : sortedSessions.length === 0 ? (
          <Empty description={t("option:writingPlayground.sessionsEmpty", "Create your first session to start writing.")} />
        ) : (
          <div className="flex flex-col gap-1">
            {sortedSessions.map(({ session, lastUsedAt }) => {
              const isActive = activeSessionId === session.id
              const lastUsedLabel = lastUsedAt
                ? t("option:writingPlayground.lastOpenedLabel", "Last opened {{time}}", { time: formatRelativeTime(new Date(lastUsedAt).toISOString(), t) })
                : t("option:writingPlayground.notOpened", "Not opened yet")
              const menuItems: MenuProps["items"] = [
                { key: "rename", icon: <Pencil className="h-4 w-4" />, label: t("option:writingPlayground.renameSession", "Rename session"), onClick: () => openRenameModal(session) },
                { key: "clone", icon: <Copy className="h-4 w-4" />, label: t("option:writingPlayground.cloneSession", "Clone session"), onClick: () => cloneSessionMutation.mutate({ session }) },
                { key: "export", icon: <Download className="h-4 w-4" />, label: t("option:writingPlayground.exportSession", "Export session"), onClick: () => exportSession(session) },
                { type: "divider" },
                { key: "delete", icon: <Trash2 className="h-4 w-4" />, label: t("option:writingPlayground.deleteSession", "Delete session"), danger: true, onClick: () => confirmDeleteSession(session) }
              ]
              return (
                <div
                  key={session.id}
                  className={`cursor-pointer rounded-md px-2 py-3 transition ${isActive ? "bg-surface-hover" : "hover:bg-surface-hover/60"}`}
                  role="button" tabIndex={0}
                  onClick={() => handleSelectSession(session)}
                  onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); handleSelectSession(session) } }}>
                  <div className="flex w-full items-center justify-between gap-3">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-text">{session.name}</span>
                      <span className="text-xs text-text-muted">{lastUsedLabel}</span>
                    </div>
                    <div className="flex items-center gap-2" onClick={(event) => event.stopPropagation()}>
                      {isActive ? (<Tag color="blue">{t("option:writingPlayground.active", "Active")}</Tag>) : null}
                      <Dropdown menu={{ items: menuItems }} trigger={["click"]}>
                        <Button type="text" size="small" aria-label={t("option:writingPlayground.sessionActionsAria", "Session actions")} icon={<MoreHorizontal className="h-4 w-4" />} loading={deleteSessionMutation.isPending || renameSessionMutation.isPending || cloneSessionMutation.isPending || sessionImporting} />
                      </Dropdown>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )
        )}
      </div>
    </div>
  )

  const charactersTabContent = <CharacterWorldTab isOnline={isOnline} />
  const researchTabContent = <ResearchTab isOnline={isOnline} />
  const agentTabContent = <AIAgentTab isOnline={isOnline} />
  const feedbackTabContent = <FeedbackTab {...feedback} />

  const inspectorDrawerContent = (
    <div className="p-3">
      <WritingPlaygroundInspectorPanel
        activeTab={activeInspectorTab}
        onTabChange={setActiveInspectorTab}
        tabLabels={{
          sampling: t("option:writingPlayground.sidebarSampling", "Sampling"),
          context: t("option:writingPlayground.sidebarContext", "Context"),
          setup: t("option:writingPlayground.sidebarSetup", "Setup"),
          inspect: t("option:writingPlayground.sidebarInspect", "Analysis"),
          characters: t("option:writingPlayground.sidebarCharacters", "Characters"),
          research: t("option:writingPlayground.sidebarResearch", "Research"),
          agent: t("option:writingPlayground.sidebarAgent", "Agent"),
          feedback: t("option:writingPlayground.sidebarFeedback", "Feedback")
        }}
        tabBadges={{
          inspect: responseInspectorRowsAll.length > 0 ? (<Tag color="blue" className="!m-0 !px-1 !text-[10px]">{responseInspectorRowsAll.length}</Tag>) : null
        }}
        essentialsStrip={(
          <Card data-testid="writing-playground-settings-card" size="small" className="!border-border">
            <div className="flex flex-col gap-2">
              <Tooltip title={t("option:writingPlayground.temperatureTooltip", "Controls randomness. Higher = more creative, lower = more focused. Typical range: 0.1-1.0")}>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">{t("option:writingPlayground.temperatureLabel", "Temperature")}</span>
                  <InputNumber size="small" min={0} max={2} step={0.01} value={settings.temperature} disabled={settingsDisabled} onChange={(value) => updateSetting({ temperature: value == null ? DEFAULT_SETTINGS.temperature : value })} className="w-full" />
                </div>
              </Tooltip>
              <Tooltip title={t("option:writingPlayground.topPTooltip", "Nucleus sampling. Only consider tokens with cumulative probability above this threshold. Typical: 0.9-1.0")}>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">{t("option:writingPlayground.topPLabel", "Top P")}</span>
                  <InputNumber size="small" min={0} max={1} step={0.01} value={settings.top_p} disabled={settingsDisabled} onChange={(value) => updateSetting({ top_p: value == null ? DEFAULT_SETTINGS.top_p : value })} className="w-full" />
                </div>
              </Tooltip>
              <Tooltip title={t("option:writingPlayground.maxTokensTooltip", "Maximum number of tokens to generate in the response")}>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-text-muted">{t("option:writingPlayground.maxTokensLabel", "Max tokens")}</span>
                  <InputNumber size="small" min={1} max={8192} step={1} value={settings.max_tokens} disabled={settingsDisabled} onChange={(value) => updateSetting({ max_tokens: value == null ? DEFAULT_SETTINGS.max_tokens : Math.max(1, Math.round(value)) })} className="w-full" />
                </div>
              </Tooltip>
              <Checkbox checked={settings.token_streaming} disabled={settingsDisabled} onChange={(event) => updateSetting({ token_streaming: event.target.checked })}>{t("option:writingPlayground.tokenStreamingLabel", "Streaming")}</Checkbox>
              {selectedTemplateName ? (
                <button type="button" className="truncate text-xs text-text-muted hover:text-text transition-colors text-left" onClick={() => setActiveInspectorTab("setup")}>
                  {t("option:writingPlayground.essentialsTpl", "tpl: {{name}}", { name: selectedTemplateName })}
                </button>
              ) : null}
            </div>
          </Card>
        )}
        sampling={samplingTabContent}
        context={contextTabContent}
        setup={setupTabContent}
        inspect={inspectTabContent}
        characters={charactersTabContent}
        research={researchTabContent}
        agent={agentTabContent}
        feedback={feedbackTabContent}
      />
    </div>
  )

  return (
    <div className={cn("writing-playground flex flex-col h-full", activeThemeClassName)}>
      {activeThemeCss ? <style>{activeThemeCss}</style> : null}
      {showOffline && (
        <div className="p-4">
          <DesignSystemAlert
            variant="warning"
            title={t("option:writingPlayground.offlineTitle", "Server required")}
          >
            {t("option:writingPlayground.offlineBody", "Connect to your tldw server to load writing sessions and generate.")}
          </DesignSystemAlert>
        </div>
      )}
      {showUnsupported && (
        <div className="p-4">
          <DesignSystemAlert
            variant="info"
            title={t("option:writingPlayground.unavailableTitle", "Playground unavailable")}
          >
            {t("option:writingPlayground.unavailableBody", "This server does not advertise writing playground support yet.")}
          </DesignSystemAlert>
        </div>
      )}
      {!showOffline && !showUnsupported && (
        <WritingPlaygroundShell
          focusMode={focusMode}
          libraryOpen={libraryOpen}
          inspectorOpen={inspectorOpen}
          onLibraryToggle={() => setLibraryOpen((prev) => !prev)}
          onInspectorToggle={() => setInspectorOpen((prev) => !prev)}
          onLibraryClose={() => setLibraryOpen(false)}
          onInspectorClose={() => setInspectorOpen(false)}
          libraryContent={libraryDrawerContent}
          inspectorContent={inspectorDrawerContent}>
          <div data-testid="writing-playground-topbar" className="flex items-center gap-3 px-4 py-2 border-b border-border bg-surface flex-shrink-0">
            <Button type="text" size="small" icon={<Menu className="h-4 w-4" />} onClick={() => setLibraryOpen((prev) => !prev)} aria-label={t("option:writingPlayground.toggleLibrary", "Toggle sessions")} className={libraryOpen ? "text-primary" : ""} />
            <span className="text-sm font-medium truncate max-w-[200px]">{activeSessionName || t("option:writingPlayground.noSession", "No session")}</span>
            <div className="flex-1" />
            <Input size="small" value={selectedModel || ""} placeholder={t("option:writingPlayground.modelPlaceholder", "e.g. gpt-4o")} onChange={(event) => { void setSelectedModel(event.target.value) }} data-testid="writing-topbar-model" className="!w-[200px]" />
            <Tooltip title={generateDisabledReason}>
              <Button
                type="primary"
                icon={isGenerating ? <Square className="h-3.5 w-3.5" /> : <Zap className="h-3.5 w-3.5" />}
                onClick={() => { if (isGenerating && settings.token_streaming) { handleCancelGeneration() } else { void handleGenerate() } }}
                loading={isGenerating && !settings.token_streaming}
                disabled={isGenerating ? !settings.token_streaming : !canGenerate || isRevisionGenerating}
                data-testid="writing-topbar-generate">
                {isGenerating ? t("option:writingPlayground.stopAction", "Stop") : t("option:writingPlayground.generateAction", "Generate")}
              </Button>
            </Tooltip>
            <Tag color={diagnosticsSummary.status === "warning" ? "gold" : diagnosticsSummary.status === "busy" ? "blue" : "green"} className="!m-0">
              {diagnosticsSummary.status === "warning" ? t("option:writingPlayground.diagnosticsWarning", "Warning") : diagnosticsSummary.status === "busy" ? t("option:writingPlayground.diagnosticsBusy", "Busy") : t("option:writingPlayground.diagnosticsReady", READY_STATE_LABEL)}
            </Tag>
            <Button type="text" size="small" icon={<Settings className="h-4 w-4" />} onClick={() => setInspectorOpen((prev) => !prev)} aria-label={t("option:writingPlayground.toggleInspector", "Toggle settings")} className={inspectorOpen ? "text-primary" : ""} />
          </div>
          <div className="flex-1 min-h-0">
          <div data-testid="writing-playground-main-grid" className="flex flex-col h-full">
            <div className="flex-1 overflow-y-auto px-4 py-3">
              <WritingPlaygroundEditorPanel>
              {activeSession ? (
                activeSessionLoading ? (<Skeleton active />) : activeSessionError ? (
                  <DesignSystemAlert
                    variant="error"
                    title={t("option:writingPlayground.editorError", "Unable to load this session.")}
                  />
                ) : (
                  <div className="flex flex-col gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="flex items-center gap-1">
                        <Button size="small" icon={<Undo2 className="h-3.5 w-3.5" />} disabled={isGenerating || !canUndoGeneration} onClick={handleUndoGeneration} title={t("option:writingPlayground.undoGeneration", "Undo generation")} />
                        <Button size="small" icon={<Redo2 className="h-3.5 w-3.5" />} disabled={isGenerating || !canRedoGeneration} onClick={handleRedoGeneration} title={t("option:writingPlayground.redoGeneration", "Redo generation")} />
                      </div>
                      <div className="h-4 w-px bg-border" />
                      <Segmented
                        size="small"
                        value={editorMode}
                        onChange={(value) => {
                          setEditorMode(value as "plain" | "tiptap")
                        }}
                        options={[
                          { value: "plain", label: "Plain" },
                          { value: "tiptap", label: "Rich" },
                        ]}
                      />
                      <div className="h-4 w-px bg-border" />
                      <Segmented
                        size="small"
                        value={editorView}
                        onChange={(value) => setEditorView(value as EditorViewMode)}
                        options={[
                          { value: "edit", icon: <Edit3 className="h-3.5 w-3.5" />, label: t("option:writingPlayground.editorModeEdit", "Edit") },
                          { value: "preview", icon: <Eye className="h-3.5 w-3.5" />, label: t("option:writingPlayground.editorModePreview", "Preview") },
                          { value: "split", icon: <Columns2 className="h-3.5 w-3.5" />, label: t("option:writingPlayground.editorModeSplit", "Split") }
                        ]}
                      />
                      <div className="h-4 w-px bg-border" />
                      <Dropdown
                        menu={{
                          items: [
                            { key: "read-aloud", label: isSpeaking ? t("option:writingPlayground.speechStopAction", "Stop reading") : t("option:writingPlayground.speechReadAction", "Read aloud"), disabled: isGenerating && !isSpeaking, onClick: () => { if (isSpeaking) { stopSpeech() } else { handleSpeakEditor() } } },
                            ...(pauseResumeAction ? [{ key: "pause-resume", label: pauseResumeAction === "pause" ? t("option:writingPlayground.speechPauseAction", "Pause") : t("option:writingPlayground.speechResumeAction", "Resume"), onClick: () => { if (pauseResumeAction === "pause") { pauseSpeech() } else { resumeSpeech() } } }] : []),
                            { type: "divider" as const },
                            { key: "chunks", label: showPromptChunks ? t("option:writingPlayground.hidePromptChunks", "Hide chunks") : t("option:writingPlayground.viewPromptChunks", "View chunks"), onClick: () => setShowPromptChunks((prev) => !prev) },
                            ...insertMenuItems
                          ]
                        }}
                        trigger={["click"]}
                        disabled={isGenerating}>
                        <Button size="small" icon={<MoreHorizontal className="h-3.5 w-3.5" />}>{t("option:writingPlayground.moreActions", "More")}</Button>
                      </Dropdown>
                      <Dropdown menu={{
                        items: [
                          { key: "pulse", label: t("option:writingPlayground.analysisStoryPulse", "Story Pulse"), onClick: () => setAnalysisModalOpen("pulse") },
                          { key: "plot", label: t("option:writingPlayground.analysisPlotTracker", "Plot Tracker"), onClick: () => setAnalysisModalOpen("plot") },
                          { key: "timeline", label: t("option:writingPlayground.analysisEventLine", "Event Line"), onClick: () => setAnalysisModalOpen("timeline") },
                          { key: "web", label: t("option:writingPlayground.analysisConnectionWeb", "Connection Web"), onClick: () => setAnalysisModalOpen("web") },
                        ]
                      }} trigger={["click"]}>
                        <Button size="small" icon={<Activity className="h-3.5 w-3.5" />}>{t("option:writingPlayground.analysisButton", "Analysis")}</Button>
                      </Dropdown>
                      <Button size="small" icon={searchOpen ? <X className="h-3.5 w-3.5" /> : <Search className="h-3.5 w-3.5" />} onClick={() => setSearchOpen((open) => !open)} title={searchOpen ? t("option:writingPlayground.searchClose", "Close search") : t("option:writingPlayground.searchToggle", "Find")} />
                    </div>
                    <WritingActionBar
                      generationAvailable={canGenerate && !isRevisionGenerating}
                      target={displayedRevisionTarget}
                      selectedPresetId={selectedRevisionPresetId}
                      isGenerating={isRevisionGenerating}
                      onPresetChange={persistRevisionPreset}
                      onRequest={(request) => {
                        void handleRevisionRequest(request)
                      }}
                    />
                    {searchOpen && (
                      <div className="rounded-md border border-border bg-surface p-3">
                        <div className="flex flex-col gap-3">
                          <div className="flex flex-wrap gap-2">
                            <Input value={searchQuery} allowClear placeholder={t("option:writingPlayground.searchPlaceholder", "Find text")} onChange={(event) => setSearchQuery(event.target.value)} className="min-w-[200px] flex-1" />
                            <Input value={replaceQuery} allowClear placeholder={t("option:writingPlayground.replacePlaceholder", "Replace with")} onChange={(event) => setReplaceQuery(event.target.value)} className="min-w-[200px] flex-1" />
                            <div className="flex items-center gap-2">
                              <Button size="small" onClick={() => navigateMatch("prev")} disabled={!searchMatches.length}>{t("option:writingPlayground.searchPrev", "Prev")}</Button>
                              <Button size="small" onClick={() => navigateMatch("next")} disabled={!searchMatches.length}>{t("option:writingPlayground.searchNext", "Next")}</Button>
                            </div>
                          </div>
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex items-center gap-3">
                              <Checkbox checked={matchCase} onChange={(event) => setMatchCase(event.target.checked)}>{t("option:writingPlayground.searchMatchCase", "Match case")}</Checkbox>
                              <Checkbox checked={useRegex} onChange={(event) => setUseRegex(event.target.checked)}>{t("option:writingPlayground.searchRegex", "Regex")}</Checkbox>
                            </div>
                            <div className="flex items-center gap-2">
                              <Button size="small" onClick={replaceCurrent} disabled={!searchMatches.length}>{t("option:writingPlayground.searchReplaceAction", "Replace")}</Button>
                              <Button size="small" onClick={replaceAll} disabled={!searchQuery.trim()}>{t("option:writingPlayground.searchReplaceAll", "Replace all")}</Button>
                            </div>
                          </div>
                          <div className="text-xs text-text-muted">
                            {searchError ? searchError : searchQuery.trim() ? searchMatches.length ? t("option:writingPlayground.searchMatchCount", "{{current}} of {{total}} matches", { current: Math.min(activeMatchIndex + 1, searchMatches.length), total: searchMatches.length }) : t("option:writingPlayground.searchNoMatches", "No matches") : t("option:writingPlayground.searchHint", "Enter text to search")}
                          </div>
                        </div>
                      </div>
                    )}
                  {editorView === "edit" && (
                    editorMode === "tiptap" ? (
                      <React.Suspense fallback={<div className="p-4 text-sm text-gray-400">Loading editor...</div>}>
                        <LazyWritingTipTapEditor
                          content={tipTapContent}
                          onContentChange={(json, plain) => {
                            editorTextChangedByTipTap.current = true
                            setTipTapContent(json)
                            applyPromptValue(plain, { promptRich: json })
                          }}
                          onAdapterReady={(adapter) => {
                            setActiveEditorAdapter(adapter)
                          }}
                          onSelectionChange={updateRevisionSelection}
                          editable={!isGenerating}
                          placeholder={t("option:writingPlayground.editorPlaceholder", "Start writing your prompt...")}
                          className={cn("flex-1 transition-all", isGenerating && "ring-2 ring-primary/50 ring-offset-1 animate-pulse rounded-md")}
                        />
                      </React.Suspense>
                    ) : (
                      <Dropdown menu={{ items: editorMenuItems }} trigger={["contextMenu"]}>
                        <div className={cn("flex-1 transition-all", isGenerating && "ring-2 ring-primary/50 ring-offset-1 animate-pulse rounded-md")}>
                            <Input.TextArea ref={editorRef} value={editorText} onChange={handlePromptChange} onClick={refreshRevisionSelection} onKeyUp={refreshRevisionSelection} onSelect={refreshRevisionSelection} onScroll={() => syncScroll("editor")} placeholder={t("option:writingPlayground.editorPlaceholder", "Start writing your prompt...")} autoSize={{ minRows: 12 }} disabled={isGenerating} className="!resize-y" />
                        </div>
                      </Dropdown>
                    )
                  )}
                    {editorView === "preview" && (
                      <div ref={previewRef} className="flex-1 overflow-y-auto rounded-md border border-border bg-surface p-4" onScroll={() => syncScroll("preview")}>
                        {editorText.trim() ? (<MarkdownPreview content={editorText} size="sm" />) : (<Paragraph type="secondary" className="!mb-0 italic">{t("option:writingPlayground.editorEmptyPreview", "Nothing to preview yet.")}</Paragraph>)}
                      </div>
                    )}
                    {editorView === "split" && (
                      <div className="flex flex-1 flex-col gap-4 lg:flex-row">
                        <div className={cn("flex-1", isGenerating && "ring-2 ring-primary/50 ring-offset-1 animate-pulse rounded-md")}>
                          {editorMode === "tiptap" ? (
                            <React.Suspense fallback={<div className="p-4 text-sm text-gray-400">Loading editor...</div>}>
                              <LazyWritingTipTapEditor
                                content={tipTapContent}
                                onContentChange={(json, plain) => {
                                  editorTextChangedByTipTap.current = true
                                  setTipTapContent(json)
                                  applyPromptValue(plain, { promptRich: json })
                                }}
                                onAdapterReady={(adapter) => {
                                  setActiveEditorAdapter(adapter)
                                }}
                                onSelectionChange={updateRevisionSelection}
                                editable={!isGenerating}
                                placeholder={t("option:writingPlayground.editorPlaceholder", "Start writing your prompt...")}
                                className={cn("flex-1 transition-all", isGenerating && "ring-2 ring-primary/50 ring-offset-1 animate-pulse rounded-md")}
                              />
                            </React.Suspense>
                          ) : (
                            <Dropdown menu={{ items: editorMenuItems }} trigger={["contextMenu"]}>
                              <div>
                                <Input.TextArea ref={editorRef} value={editorText} onChange={handlePromptChange} onClick={refreshRevisionSelection} onKeyUp={refreshRevisionSelection} onSelect={refreshRevisionSelection} onScroll={() => syncScroll("editor")} placeholder={t("option:writingPlayground.editorPlaceholder", "Start writing your prompt...")} autoSize={{ minRows: 12 }} disabled={isGenerating} className="!resize-y" />
                              </div>
                            </Dropdown>
                          )}
                        </div>
                        <div ref={previewRef} className="flex-1 overflow-y-auto rounded-md border border-border bg-surface p-4" onScroll={() => syncScroll("preview")}>
                          {editorText.trim() ? (<MarkdownPreview content={editorText} size="sm" />) : (<Paragraph type="secondary" className="!mb-0 italic">{t("option:writingPlayground.editorEmptyPreview", "Nothing to preview yet.")}</Paragraph>)}
                        </div>
                      </div>
                    )}
                    <WritingRevisionQueue
                      proposals={revisionState.revisions}
                      onApply={(proposal) =>
                        revisionState.applyRevision(proposal.id)
                      }
                      onReject={(proposal) =>
                        revisionState.rejectRevision(proposal.id)
                      }
                      onCopy={(proposal) => {
                        void handleCopyRevision(proposal)
                      }}
                      onRegenerate={(proposal) => {
                        void handleRegenerateRevision(proposal)
                      }}
                    />
                    {responseLogprobs.length > 0 ? (
                      <div className="rounded-md border border-border bg-surface p-3">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-xs text-text-muted">{t("option:writingPlayground.inlineTokenInspectorHint", "Inline token probabilities: hover token chips to inspect alternatives, click a token or alternative to reroll from that point.")}</span>
                          <Tag color="blue">{t("option:writingPlayground.inlineTokenInspectorCount", "{{count}} tokens", { count: inlineResponseTokens.length })}</Tag>
                        </div>
                        <div className="mt-2 max-h-40 overflow-y-auto rounded-md border border-border bg-background p-2">
                          <div className="flex flex-wrap gap-1">
                            {inlineResponseTokens.map((row) => (
                              <Tooltip
                                key={`inline-token-${row.sequence}`}
                                placement="top"
                                title={
                                  <div className="flex max-w-[360px] flex-col gap-1">
                                    <span className="text-[11px] text-text-muted">{t("option:writingPlayground.inlineTokenAlternativesLabel", "Top alternatives")}</span>
                                    {row.topLogprobs.length === 0 ? (<span className="text-[11px] text-text-muted">{t("option:writingPlayground.inlineTokenAlternativesEmpty", "No alternatives.")}</span>) : (
                                      row.topLogprobs.map((alt, idx) => (
                                        <Button key={`inline-alt-${row.sequence}-${idx}`} size="small" type="text" className="!h-auto !justify-start !px-1.5 !py-0.5 font-mono text-[11px]" onClick={(event) => { event.preventDefault(); event.stopPropagation(); handleRerollFromResponseToken(row.sequence, alt.token) }}>
                                          {`${alt.displayToken} (${alt.probability >= 0.001 ? alt.probability.toFixed(3) : alt.probability.toExponential(2)})`}
                                        </Button>
                                      ))
                                    )}
                                  </div>
                                }>
                                <button type="button" className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-[11px] text-text transition hover:bg-surface-hover" onClick={() => handleRerollFromResponseToken(row.sequence)}>
                                  {row.displayToken}
                                </button>
                              </Tooltip>
                            ))}
                          </div>
                        </div>
                        {inlineResponseTokensTruncated ? (<span className="mt-2 block text-xs text-text-muted">{t("option:writingPlayground.inlineTokenInspectorTruncated", "Showing first {{count}} tokens.", { count: inlineResponseTokens.length })}</span>) : null}
                      </div>
                    ) : null}
                    {showPromptChunks ? (
                      <div data-testid="writing-section-prompt-chunks">
                        <Collapse ghost size="small" defaultActiveKey={["chunks"]} items={[{
                          key: "chunks",
                          label: t("option:writingPlayground.promptChunksTitle", "Prompt chunks ({{count}})", { count: promptChunkData.total }),
                          children: promptChunkData.total === 0 ? (<Paragraph type="secondary" className="!mb-0">{t("option:writingPlayground.promptChunksEmpty", "No chunks yet.")}</Paragraph>) : (
                            <div className="flex flex-col gap-2">
                              {promptChunkData.chunks.map((chunk) => (
                                <div key={chunk.key} className="flex flex-col gap-1 rounded-md border border-border bg-surface p-2">
                                  <Tag className="mr-auto" color={chunk.type === "placeholder" ? "gold" : "default"}>{chunk.type === "placeholder" ? t("option:writingPlayground.chunkPlaceholder", "Placeholder") : t("option:writingPlayground.chunkText", "Text")}</Tag>
                                  <pre className="m-0 whitespace-pre-wrap break-words font-mono text-xs text-text">{chunk.label}</pre>
                                </div>
                              ))}
                              {promptChunkData.truncated ? (<Paragraph type="secondary" className="!mb-0">{t("option:writingPlayground.promptChunksTruncated", "Showing first {{count}} chunks.", { count: promptChunkData.chunks.length })}</Paragraph>) : null}
                            </div>
                          )
                        }]} />
                      </div>
                    ) : null}
                  </div>
                )
              ) : (
                <Empty description={t("option:writingPlayground.selectSession", "Select a session to begin.")} />
              )}
              </WritingPlaygroundEditorPanel>
            </div>
            <div data-testid="writing-playground-statusbar" className="flex items-center gap-3 px-4 py-1.5 border-t border-border bg-surface text-xs text-text-muted flex-shrink-0">
              <span data-testid="writing-status-word-count">
                {t(
                  "option:writingPlayground.wordCountLabel",
                  writingWordCount === 1
                    ? `${writingWordCount} word`
                    : `${writingWordCount} words`,
                  { count: writingWordCount }
                )}
              </span>
              {selectedWordCount > 0 ? (
                <span data-testid="writing-status-selected-word-count">
                  {t(
                    "option:writingPlayground.selectedWordCountLabel",
                    `${selectedWordCount} selected`,
                    { count: selectedWordCount }
                  )}
                </span>
              ) : null}
              {pendingRevisionCount > 0 ? (
                <span data-testid="writing-revision-pending-count">
                  {t(
                    "option:writingPlayground.pendingRevisionCountLabel",
                    `${pendingRevisionCount} pending`,
                    { count: pendingRevisionCount }
                  )}
                </span>
              ) : null}
              {generationTokenCount > 0 && (<span>{t("option:writingPlayground.generationTokensLabel", "{{count}} tokens", { count: generationTokenCount })}</span>)}
              {generationTokensPerSec > 0 && (<span>{t("option:writingPlayground.generationRateLabel", "{{rate}} tok/s", { rate: generationTokensPerSec >= 10 ? generationTokensPerSec.toFixed(1) : generationTokensPerSec.toFixed(2) })}</span>)}
              {isGenerating && generationElapsed > 0 && (<span>{generationElapsed}s</span>)}
              {feedback.moodEnabled && feedback.currentMood && (
                <Tag
                  color={MOOD_COLORS[feedback.currentMood]}
                  className="!text-xs !m-0"
                >
                  {feedback.currentMood}
                </Tag>
              )}
              <div className="flex-1" />
              {saveStatusLabel && (<span>{saveStatusLabel}</span>)}
              <span className="text-text-muted/60">{t("option:writingPlayground.shortcutsHint", "Ctrl+Enter to generate")}</span>
            </div>
          </div>
          </div>
        </WritingPlaygroundShell>
      )}

      {shouldRenderWritingModalHost ? (
        <React.Suspense fallback={null}>
          <LazyWritingPlaygroundModalHost
            t={t}
            settingsDisabled={settingsDisabled}
            supportsAdvancedCompat={supportsAdvancedCompat}
            extraBodyJsonModalOpen={extraBodyJsonModalOpen}
            setExtraBodyJsonModalOpen={setExtraBodyJsonModalOpen}
            extraBodyJsonError={extraBodyJsonError}
            setExtraBodyJsonError={setExtraBodyJsonError}
            extraBodyJsonDraft={extraBodyJsonDraft}
            setExtraBodyJsonDraft={setExtraBodyJsonDraft}
            applyExtraBodyJsonDraft={applyExtraBodyJsonDraft}
            contextPreviewModalOpen={contextPreviewModalOpen}
            setContextPreviewModalOpen={setContextPreviewModalOpen}
            handleCopyContextPreview={handleCopyContextPreview}
            handleExportContextPreview={handleExportContextPreview}
            contextPreviewJson={contextPreviewJson}
            templatesModalOpen={templatesModalOpen}
            setTemplatesModalOpen={setTemplatesModalOpen}
            templatesLoading={templatesLoading}
            templatesError={templatesError}
            templates={templates}
            editingTemplate={editingTemplate}
            templateImporting={templateImporting}
            templateRestoringDefaults={templateRestoringDefaults}
            handleTemplateNew={handleTemplateNew}
            handleTemplateDuplicate={handleTemplateDuplicate}
            templateDuplicateDisabled={templateDuplicateDisabled}
            templateFileInputRef={templateFileInputRef}
            templateExportDisabled={templateExportDisabled}
            exportTemplate={exportTemplate}
            templateRestoreDefaultsDisabled={templateRestoreDefaultsDisabled}
            handleTemplateRestoreDefaults={handleTemplateRestoreDefaults}
            templateForm={templateForm}
            templateFormDisabled={templateFormDisabled}
            updateTemplateForm={updateTemplateForm}
            handleTemplateSelect={handleTemplateSelect}
            templateSaveLoading={templateSaveLoading}
            templateSaveDisabled={templateSaveDisabled}
            handleTemplateSave={handleTemplateSave}
            templateDeleteDisabled={templateDeleteDisabled}
            deleteTemplateMutation={deleteTemplateMutation}
            confirmDeleteTemplate={confirmDeleteTemplate}
            handleTemplateImport={handleTemplateImport}
            themesModalOpen={themesModalOpen}
            setThemesModalOpen={setThemesModalOpen}
            themesLoading={themesLoading}
            themesError={themesError}
            themes={themes}
            editingTheme={editingTheme}
            themeImporting={themeImporting}
            themeRestoringDefaults={themeRestoringDefaults}
            handleThemeNew={handleThemeNew}
            handleThemeDuplicate={handleThemeDuplicate}
            themeDuplicateDisabled={themeDuplicateDisabled}
            themeFileInputRef={themeFileInputRef}
            themeExportDisabled={themeExportDisabled}
            exportTheme={exportTheme}
            themeRestoreDefaultsDisabled={themeRestoreDefaultsDisabled}
            handleThemeRestoreDefaults={handleThemeRestoreDefaults}
            themeForm={themeForm}
            themeFormDisabled={themeFormDisabled}
            updateThemeForm={updateThemeForm}
            handleThemeSelect={handleThemeSelect}
            themeSaveLoading={themeSaveLoading}
            themeSaveDisabled={themeSaveDisabled}
            handleThemeSave={handleThemeSave}
            themeDeleteDisabled={themeDeleteDisabled}
            deleteThemeMutation={deleteThemeMutation}
            confirmDeleteTheme={confirmDeleteTheme}
            handleThemeImport={handleThemeImport}
            createModalOpen={createModalOpen}
            setCreateModalOpen={setCreateModalOpen}
            createSessionMutation={createSessionMutation}
            canCreateSession={canCreateSession}
            newSessionName={newSessionName}
            setNewSessionName={setNewSessionName}
            renameModalOpen={renameModalOpen}
            setRenameModalOpen={setRenameModalOpen}
            renameTarget={renameTarget}
            renameSessionMutation={renameSessionMutation}
            canRenameSession={canRenameSession}
            renameSessionName={renameSessionName}
            setRenameSessionName={setRenameSessionName}
          />
        </React.Suspense>
      ) : null}
      <input ref={sessionFileInputRef} type="file" accept=".json,application/json" onChange={handleSessionImport} data-testid="writing-session-import" className="hidden" />
      <input ref={snapshotFileInputRef} type="file" accept=".json,application/json" onChange={handleSnapshotImport} data-testid="writing-snapshot-import" className="hidden" />
      <WritingAnalysisModalHost />
    </div>
  )
}
