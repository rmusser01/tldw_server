import { EMPTY_CHAT_MODELS, useChatModelCatalog } from "@/hooks/useChatModelCatalog"
import { normalizeSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useMessageOption } from "@/hooks/useMessageOption"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { getAllModelSettings } from "@/services/model-settings"
import { useStoreChatModelSettings, type ChatModelSettings } from "@/store/model"
import { useActorStore } from "@/store/actor"
import { useQuery } from "@tanstack/react-query"
import {
  Button,
  Drawer,
  Form,
  Modal,
  Skeleton,
  Tabs
} from "antd"
import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { shallow } from "zustand/shallow"
import { getOCRLanguage } from "@/services/ocr"
import { ocrLanguages } from "@/data/ocr-language"
import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { getProviderDisplayName } from "@/utils/provider-registry"
import type { ActorSettings, ActorTarget } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"
import {
  buildActorPrompt,
  buildActorSettingsFromForm,
  estimateActorTokens
} from "@/utils/actor"
import {
  getCanonicalModelKey,
  resolveModelSelectorSelection
} from "@/hooks/playground/modelSelectorUtils"
import type { Character } from "@/types/character"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import {
  ModelBasicsTab,
  ConversationTab,
  AdvancedParamsTab,
  ActorTab
} from "./tabs"
import { LlamaCppAdvancedControls } from "./LlamaCppAdvancedControls"
import { resolveSelectedSystemPromptContent } from "../system-prompt-utils"
import type { ConversationDraftStatus } from "./tabs/ConversationTab"
import { normalizeCurrentChatModelSettingValue } from "./current-chat-model-settings-values"

type Props = {
  open: boolean
  setOpen: (open: boolean) => void
  useDrawer?: boolean
  isOCREnabled?: boolean
  settingsScope?: string | null
}

type ModelConfigData = {
  temperature?: number
  topK?: number
  topP?: number
  keepAlive?: string
  numCtx?: number
  numGpu?: number
  numPredict?: number
  useMMap?: boolean
  minP?: number
  repeatLastN?: number
  repeatPenalty?: number
  useMlock?: boolean
  tfsZ?: number
  numKeep?: number
  numThread?: number
  llamaThinkingBudgetTokens?: number
  llamaGrammarMode?: "none" | "library" | "inline"
  llamaGrammarId?: string
  llamaGrammarInline?: string
  llamaGrammarOverride?: string
}

type ActorFormValues = {
  actorEnabled?: boolean
  actorNotes?: ActorSettings["notes"]
  actorNotesGmOnly?: ActorSettings["notesGmOnly"]
  actorChatPosition?: ActorSettings["chatPosition"]
  actorChatDepth?: ActorSettings["chatDepth"]
  actorChatRole?: ActorSettings["chatRole"]
  actorTemplateMode?: ActorSettings["templateMode"]
  actorAppendable?: ActorSettings["appendable"]
  [key: `actor_${string}`]: string | undefined
  [key: `actor_key_${string}`]: string | undefined
}

type CurrentChatModelFormValues = Partial<ChatModelSettings> & ActorFormValues

const CHAT_MODEL_SETTING_KEYS: ReadonlySet<keyof ChatModelSettings> = new Set([
  "systemPrompt",
  "thinking",
  "temperature",
  "topK",
  "topP",
  "keepAlive",
  "numCtx",
  "seed",
  "numGpu",
  "numPredict",
  "useMMap",
  "minP",
  "repeatLastN",
  "repeatPenalty",
  "useMlock",
  "tfsZ",
  "numKeep",
  "numThread",
  "reasoningEffort",
  "historyMessageLimit",
  "historyMessageOrder",
  "slashCommandInjectionMode",
  "apiProvider",
  "extraHeaders",
  "extraBody",
  "llamaThinkingBudgetTokens",
  "llamaGrammarMode",
  "llamaGrammarId",
  "llamaGrammarInline",
  "llamaGrammarOverride",
  "jsonMode"
])

const loadActorSettings = () => import("@/services/actor-settings")

const isChatModelSettingKey = (
  key: string
): key is keyof ChatModelSettings =>
  CHAT_MODEL_SETTING_KEYS.has(key as keyof ChatModelSettings)

export const CurrentChatModelSettings = ({
  open,
  setOpen,
  useDrawer,
  isOCREnabled,
  settingsScope
}: Props) => {
  const { t } = useTranslation("common")
  const [modal, modalContextHolder] = Modal.useModal()
  const [form] = Form.useForm<CurrentChatModelFormValues>()
  const {
    apiProvider, ocrLanguage, updateSetting, updateScopedSetting,
    setActiveSettingsScope, setOcrLanguage
  } = useStoreChatModelSettings(state => ({
    apiProvider: state.apiProvider,
    ocrLanguage: state.ocrLanguage,
    updateSetting: state.updateSetting,
    updateScopedSetting: state.updateScopedSetting,
    setActiveSettingsScope: state.setActiveSettingsScope,
    setOcrLanguage: state.setOcrLanguage
  }), shallow)
  const activeSettingsScope = useStoreChatModelSettings(state => state.activeSettingsScope)
  const editRevisionRef = React.useRef(0)
  const baselineRef = React.useRef<{
    values: CurrentChatModelFormValues
    actor: ActorSettings
    ocrLanguage?: string
  } | null>(null)
  const [stagedOcrLanguage, setStagedOcrLanguage] = React.useState(ocrLanguage)
  const {
    historyId,
    selectedSystemPrompt,
    uploadedFiles,
    removeUploadedFile,
    selectedModel,
    setSelectedModel,
    retrySelectedModelSave,
    serverChatId,
    serverChatVersion,
    serverChatAssistantKind,
    serverChatPersonaMemoryMode,
    serverChatTopic,
    setServerChatTopic,
    serverChatState,
    setServerChatState,
    setServerChatVersion
  } = useMessageOption()

  const [selectedCharacter, , selectedCharacterMeta] =
    useSelectedCharacter<Character | null>(null)
  const [selectedAssistant, , assistantMeta] = useSelectedAssistant(null)
  const accountIsCurrent = assistantMeta?.isCurrent
  const historySelection = useHistorySelectionContext()
  const selectionSignal = historySelection?.getSignal()
  const captureSelectionFence = historySelection?.fence
  const selectedCharacterId = selectedCharacter?.id ?? null
  const personaChatActive =
    serverChatAssistantKind === "persona" || selectedAssistant?.kind === "persona"
  const draftRevision = useStoreMessageOption((state) => state.chatDraftRevision)
  const renderedModelRevision = useStoreMessageOption(state => state.selectedModelRevision)
  const mountedRef = React.useRef(true)
  React.useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])
  const restoreRevision = usePlaygroundSessionStore(
    (state) => state.restoreRevision
  )
  const selectionRevision = usePlaygroundSessionStore(
    (state) => state.sourceSelectionRevision
  )
  const [readRevision, setReadRevision] = React.useState(0)
  const [saving, setSaving] = React.useState(false)
  const [saveError, setSaveError] = React.useState(false)
  const [modelSaveError, setModelSaveError] = React.useState<{ value: string | null; revision: number; panel: object } | null>(null)
  const [retryingModelSave, setRetryingModelSave] = React.useState(false)
  const modelRetryRef = React.useRef<object | null>(null)
  const saveAttemptRef = React.useRef<object | null>(null)
  const readSourceRef = React.useRef<(() => boolean) | null>(null)
  const loadedSourceRef = React.useRef<(() => boolean) | null>(null)
  const panelKey = JSON.stringify([
    open, historyId, serverChatId, selectedCharacterId, personaChatActive,
    draftRevision, restoreRevision, selectionRevision, readRevision, settingsScope
  ])
  const panel = { key: panelKey, selectionSignal, captureSelectionFence, accountIsCurrent }
  const panelRef = React.useRef(panel)
  if (Object.keys(panel).some(key => panel[key as keyof typeof panel] !== panelRef.current[key as keyof typeof panel]))
    panelRef.current = panel
  const renderedPanel = panelRef.current
  const viewKey = JSON.stringify([
    open,
    historyId,
    serverChatId,
    selectedCharacterId,
    personaChatActive,
    draftRevision,
    restoreRevision,
    selectionRevision,
    readRevision,
    selectedModel,
    renderedModelRevision,
    activeSettingsScope,
    settingsScope
  ])
  const viewRef = React.useRef({
    key: viewKey,
    selectionSignal,
    captureSelectionFence,
    accountIsCurrent,
    revision: 0
  })
  if (
    viewRef.current.key !== viewKey ||
    viewRef.current.selectionSignal !== selectionSignal ||
    viewRef.current.captureSelectionFence !== captureSelectionFence ||
    viewRef.current.accountIsCurrent !== accountIsCurrent
  ) {
    viewRef.current = {
      key: viewKey,
      selectionSignal,
      captureSelectionFence,
      accountIsCurrent,
      revision: viewRef.current.revision + 1
    }
    loadedSourceRef.current = null
    baselineRef.current = null
  }
  const renderedView = viewRef.current
  const conversationDraftRef = React.useRef({ view: renderedView, dirty: false, pending: false, revision: 0 })
  const [, refreshConversationDraft] = React.useReducer((value: number) => value + 1, 0)
  if (conversationDraftRef.current.view !== renderedView)
    conversationDraftRef.current = { view: renderedView, dirty: false, pending: false, revision: 0 }
  const onConversationDraftChange = useCallback((status: ConversationDraftStatus) => {
    if (viewRef.current !== renderedView) return
    const previous = conversationDraftRef.current
    if (previous.revision === status.revision && previous.dirty === status.dirty && previous.pending === status.pending) return
    if (previous.revision !== status.revision) editRevisionRef.current++
    conversationDraftRef.current = { view: renderedView, ...status }
    refreshConversationDraft()
  }, [renderedView])
  const conversationDirty = conversationDraftRef.current.dirty
  const conversationPending = conversationDraftRef.current.pending
  React.useEffect(() => {
    saveAttemptRef.current = null
    modelRetryRef.current = null
    setRetryingModelSave(false)
    setSaving(false)
    setSaveError(false)
  }, [renderedView])

  const { commitActorSettings, commitActorPreview } = useActorStore(state => ({
    commitActorSettings: state.setSettings,
    commitActorPreview: state.setPreviewAndTokens
  }), shallow)
  const [actorSettings, setActorSettingsState] = React.useState<ActorSettings | null>(null)
  const [{ prompt: actorPreview, tokens: actorTokenCount }, setScenePreview] = React.useState({ prompt: "", tokens: 0 })
  const loadedSource = loadedSourceRef.current
  const isEditorCurrent = useCallback(() => !saveAttemptRef.current && mountedRef.current &&
    viewRef.current === renderedView && Boolean(loadedSource?.()) &&
    useStoreMessageOption.getState().selectedModelRevision === renderedModelRevision,
    [loadedSource, renderedView, renderedModelRevision])
  const captureEditGuard = useCallback(() => {
    const revision = editRevisionRef.current
    return () => isEditorCurrent() && editRevisionRef.current === revision
  }, [isEditorCurrent])
  const setPreviewAndTokens = useCallback((prompt: string, tokens: number) => {
    if (isEditorCurrent()) setScenePreview({ prompt, tokens })
  }, [isEditorCurrent])
  const setActorSettings = useCallback((next: ActorSettings) => {
    if (!isEditorCurrent()) return
    editRevisionRef.current++
    setActorSettingsState(next)
  }, [isEditorCurrent])
  const isDirty = useCallback(() => {
    const baseline = baselineRef.current
    return conversationDraftRef.current.dirty || Boolean(baseline && (
      Object.entries(baseline.values).some(([key, value]) => JSON.stringify(form.getFieldValue(key as keyof CurrentChatModelFormValues)) !== JSON.stringify(value)) ||
      JSON.stringify(actorSettings) !== JSON.stringify(baseline.actor) ||
      stagedOcrLanguage !== baseline.ocrLanguage
    ))
  }, [actorSettings, form, stagedOcrLanguage])
  const requestClose = useCallback(() => {
    if (saveAttemptRef.current || saving || conversationDraftRef.current.pending || viewRef.current !== renderedView) return
    const revision = editRevisionRef.current
    const source = loadedSourceRef.current
    const close = () => {
      if (!saveAttemptRef.current && !conversationDraftRef.current.pending && mountedRef.current && viewRef.current === renderedView &&
        useStoreMessageOption.getState().selectedModelRevision === renderedModelRevision &&
        editRevisionRef.current === revision && (!source || source()))
        setOpen(false)
    }
    if (!isDirty()) {
      if (viewRef.current === renderedView) setOpen(false)
      return
    }
    if (!source?.()) return
    modal.confirm({
      title: t("modelSettings.discardTitle", "Discard unsaved settings?"),
      content: t("modelSettings.discardHelp", "Staged settings will be discarded. Model choices and conversation changes already applied remain in effect."),
      okText: t("discard", "Discard"),
      cancelText: t("keepEditing", "Keep editing"),
      onOk: close
    })
  }, [isDirty, modal, renderedView, renderedModelRevision, saving, setOpen, t])
  const [newAspectTarget, setNewAspectTarget] =
    React.useState<ActorTarget>("user")
  const [newAspectName, setNewAspectName] = React.useState<string>("")
  const actorPositionValue = Form.useWatch("actorChatPosition", form)
  const llamaThinkingBudgetTokens = Form.useWatch("llamaThinkingBudgetTokens", form)
  const llamaGrammarMode = Form.useWatch("llamaGrammarMode", form)
  const llamaGrammarId = Form.useWatch("llamaGrammarId", form)
  const llamaGrammarInline = Form.useWatch("llamaGrammarInline", form)
  const llamaGrammarOverride = Form.useWatch("llamaGrammarOverride", form)
  const llamaExtraBody = Form.useWatch("extraBody", form)

  const savePrompt = useCallback((value: string) => {
    if (!isEditorCurrent()) return
    editRevisionRef.current++
    form.setFieldValue("systemPrompt", value)
  }, [form, isEditorCurrent])

  const resetSystemPrompt = useCallback(async () => {
    const revision = editRevisionRef.current
    const source = loadedSourceRef.current
    if (viewRef.current !== renderedView || !source?.()) return
    const nextValue = await resolveSelectedSystemPromptContent(selectedSystemPrompt)
    if (viewRef.current !== renderedView || !source() || editRevisionRef.current !== revision) return
    savePrompt(nextValue)
  }, [renderedView, savePrompt, selectedSystemPrompt])

  const recomputeActorPreview = useCallback(() => {
    if (!isEditorCurrent()) return
    const values = form.getFieldsValue()
    const base = actorSettings ?? createDefaultActorSettings()

    const next: ActorSettings = buildActorSettingsFromForm(base, values)

    const preview = buildActorPrompt(next)
    setPreviewAndTokens(preview, estimateActorTokens(preview))
  }, [actorSettings, form, isEditorCurrent, setPreviewAndTokens])

  const timeoutRef = React.useRef<number | undefined>()

  const debouncedRecomputeActorPreview = React.useMemo(() => {
    return () => {
      if (!isEditorCurrent()) return
      const revision = editRevisionRef.current
      if (timeoutRef.current !== undefined) window.clearTimeout(timeoutRef.current)
      const timer = window.setTimeout(() => {
        if (timeoutRef.current !== timer || editRevisionRef.current !== revision || !isEditorCurrent()) return
        timeoutRef.current = undefined
        recomputeActorPreview()
      }, 150)
      timeoutRef.current = timer
    }
  }, [isEditorCurrent, recomputeActorPreview])

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current !== undefined) {
        window.clearTimeout(timeoutRef.current)
      }
    }
  }, [renderedView, actorSettings])

  React.useEffect(() => {
    if (!open) return
    if (import.meta?.env?.DEV) {
      console.count("CurrentChatModelSettings/recomputeActorPreview")
    }
    recomputeActorPreview()
  }, [actorSettings, open, recomputeActorPreview])

  const { data: composerModels = EMPTY_CHAT_MODELS, isLoading: modelsLoading } = useChatModelCatalog(open)

  const scopedApiProvider = useStoreChatModelSettings(state => state.activeSettingsScope ? state.scopedSettingsByModelKey?.[state.activeSettingsScope]?.apiProvider : undefined)
  const selectedSelection = useMemo(
    () => resolveModelSelectorSelection(composerModels, selectedModel, apiProvider, { scope: settingsScope?.trim() || activeSettingsScope, providerOverride: scopedApiProvider }),
    [apiProvider, composerModels, selectedModel, activeSettingsScope, scopedApiProvider, settingsScope]
  )

  const saveSettings = useCallback(
    async (values: CurrentChatModelFormValues) => {
      const isCurrent = loadedSourceRef.current
      if (
        !open ||
        viewRef.current !== renderedView ||
        !(accountIsCurrent?.() ?? true) ||
        (!personaChatActive && !isCurrent?.())
      )
        return false
      if (useStoreMessageOption.getState().selectedModelRevision !== renderedModelRevision) return false
      const editRevision = editRevisionRef.current
      const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
      const source = captureChatDraftSource(() => selectionIsCurrent() &&
        !selectionSignal?.aborted && (accountIsCurrent?.() ?? true))
      const canPublish = () => mountedRef.current && panelRef.current === renderedPanel &&
        source.isCurrent() && editRevisionRef.current === editRevision &&
        useStoreMessageOption.getState().selectedModelRevision === renderedModelRevision
      const canCommit = () => viewRef.current === renderedView && canPublish() && Boolean(isCurrent?.())
      const latestSettingsState = useStoreChatModelSettings.getState()
      const explicitSettingsScope =
        typeof settingsScope === "string" && settingsScope.trim()
          ? settingsScope.trim()
          : null
      const preferredScope =
        explicitSettingsScope || latestSettingsState.activeSettingsScope
      const targetSettingsScope = resolveModelSelectorSelection(
        composerModels,
        selectedModel,
        latestSettingsState.apiProvider,
        {
          scope: preferredScope,
          providerOverride: preferredScope
            ? latestSettingsState.scopedSettingsByModelKey[preferredScope]
                ?.apiProvider
            : undefined
        }
      ).settingsScope

      const base = actorSettings ?? createDefaultActorSettings()
      const next: ActorSettings = buildActorSettingsFromForm(base, values)

      if (!personaChatActive) {
        const { saveActorSettingsForChat } = await loadActorSettings()
        if (!canCommit()) return false
        const saved = await saveActorSettingsForChat({
          historyId,
          serverChatId,
          settings: next
        })
        if (!canCommit()) return false
        if (!saved) throw new Error("actor_settings_save_failed")
      }
      if (!canCommit()) return false
      if (targetSettingsScope) {
        setActiveSettingsScope(targetSettingsScope)
      }

      for (const key of Object.keys(values)) {
        if (!canPublish()) return false
        if (!isChatModelSettingKey(key)) continue
        const normalizedValue = normalizeCurrentChatModelSettingValue(
          key,
          values[key]
        )
        if (targetSettingsScope) {
          updateScopedSetting(targetSettingsScope, key, normalizedValue)
        } else updateSetting(key, normalizedValue)
      }

      if (!canPublish()) return false
      if (isOCREnabled && stagedOcrLanguage) setOcrLanguage(stagedOcrLanguage)
      if (!canPublish()) return false
      commitActorSettings(next)
      if (!canPublish()) return false
      const preview = buildActorPrompt(next)
      commitActorPreview(preview, estimateActorTokens(preview))
      if (!canPublish()) return false
      baselineRef.current = { values: structuredClone(form.getFieldsValue(true)), actor: next, ocrLanguage: stagedOcrLanguage }
      setOpen(false)
      return true
    },
    [
      actorSettings,
      accountIsCurrent,
      captureSelectionFence,
      selectionSignal,
      open,
      renderedView,
      renderedPanel,
      renderedModelRevision,
      historyId,
      personaChatActive,
      serverChatId,
      settingsScope,
      composerModels,
      selectedModel,
      setActiveSettingsScope,
      commitActorSettings,
      commitActorPreview,
      form,
      isOCREnabled,
      stagedOcrLanguage,
      setOcrLanguage,
      updateScopedSetting,
      updateSetting,
      setOpen
    ]
  )

  const handleModelChange = useCallback((selection: string | null | undefined) => {
    const value = normalizeSelectedModel(selection)
    if (saving || conversationDraftRef.current.pending || !isEditorCurrent()) return
    const revision = editRevisionRef.current
    const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
    const source = captureChatDraftSource(() => selectionIsCurrent() && (accountIsCurrent?.() ?? true))
    const canSelect = () => mountedRef.current && panelRef.current === renderedPanel &&
      editRevisionRef.current === revision && source.isCurrent() &&
      useStoreMessageOption.getState().selectedModelRevision === renderedModelRevision
    const change = () => {
      if (conversationDraftRef.current.pending || !isEditorCurrent() || !canSelect()) return
      const state = useStoreChatModelSettings.getState()
      const nextScope = resolveModelSelectorSelection(composerModels, value, state.apiProvider, {
        scope: state.activeSettingsScope,
        providerOverride: state.activeSettingsScope ? state.scopedSettingsByModelKey[state.activeSettingsScope]?.apiProvider : undefined
      }).settingsScope
      state.setActiveSettingsScope(nextScope)
      if (!canSelect()) return
      setModelSaveError(null)
      let write: unknown
      try { write = setSelectedModel(value) }
      catch (error) { write = Promise.reject(error) }
      const writtenRevision = renderedModelRevision + 1
      void Promise.resolve(write).catch(() => {
        const current = useStoreMessageOption.getState()
        if (mountedRef.current && panelRef.current === renderedPanel && source.isCurrent() &&
          current.selectedModelRevision === writtenRevision && current.selectedModel === value)
          setModelSaveError({ value, revision: writtenRevision, panel: renderedPanel })
      })
    }
    if (!isDirty()) return change()
    modal.confirm({
      title: t("modelSettings.switchTitle", "Discard staged settings and switch model?"),
      content: t("modelSettings.switchHelp", "Switching applies immediately and reloads model defaults and the saved scene. Your unsaved form edits will be discarded."),
      okText: t("modelSettings.switchAction", "Discard and switch"),
      cancelText: t("keepEditing", "Keep editing"),
      onOk: change
    })
  }, [accountIsCurrent, captureSelectionFence, composerModels, isDirty, isEditorCurrent, modal, renderedPanel, renderedModelRevision, saving, setSelectedModel, t])

  const retryModelSave = useCallback(async () => {
    const receipt = modelSaveError
    if (!receipt || modelRetryRef.current || !isEditorCurrent()) return
    const source = loadedSourceRef.current
    const isCurrent = () => mountedRef.current && panelRef.current === receipt.panel &&
      Boolean(source?.()) && useStoreMessageOption.getState().selectedModelRevision === receipt.revision &&
      normalizeSelectedModel(useStoreMessageOption.getState().selectedModel) === receipt.value
    if (!isCurrent()) return
    const attempt = {}
    modelRetryRef.current = attempt
    setRetryingModelSave(true)
    try {
      await retrySelectedModelSave()
      if (isCurrent()) setModelSaveError(current => current === receipt ? null : current)
    } catch {
      // The existing exact device receipt remains available for another Retry.
    } finally {
      if (modelRetryRef.current === attempt) {
        modelRetryRef.current = null
        if (mountedRef.current) setRetryingModelSave(false)
      }
    }
  }, [isEditorCurrent, modelSaveError, retrySelectedModelSave])

  const handleLlamaControlChange = useCallback(
    (key: keyof ChatModelSettings, value: number | string | undefined) => {
      if (!isEditorCurrent()) return
      editRevisionRef.current++
      form.setFieldValue(key, value)
    },
    [form, isEditorCurrent]
  )

  const buildBaseValues = useCallback((data?: ModelConfigData | null, promptFallback?: string) => {
    const effective = useStoreChatModelSettings.getState().getEffectiveSettings(selectedSelection.settingsScope)
    const values = Object.fromEntries([...CHAT_MODEL_SETTING_KEYS].map(key => [key, effective[key]])) as Partial<ChatModelSettings>
    if (data) for (const key of CHAT_MODEL_SETTING_KEYS) {
      if (values[key] === undefined && key in data)
        Object.assign(values, { [key]: data[key as keyof ModelConfigData] })
    }
    return { ...values, systemPrompt: effective.systemPrompt ?? promptFallback ?? "" }
  }, [selectedSelection.settingsScope])

  const { isLoading, isFetching, isError } = useQuery({
    queryKey: ["fetchModelConfig2", viewKey, renderedView.revision],
    queryFn: async ({ signal, client }) => {
      const sourceView = renderedView
      const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
      const { isCurrent } = captureChatDraftSource(
        () =>
          !signal.aborted &&
          selectionIsCurrent() &&
          (accountIsCurrent?.() ?? true) &&
          viewRef.current === sourceView &&
          usePlaygroundSessionStore.getState().restoreRevision ===
            restoreRevision &&
          usePlaygroundSessionStore.getState().sourceSelectionRevision ===
            selectionRevision
      )
      if (!isCurrent()) throw new Error("stale_selection")
      readSourceRef.current = isCurrent
      loadedSourceRef.current = null
      if (import.meta?.env?.DEV) {
        console.count("CurrentChatModelSettings/fetchModelConfig")
      }
      // Keep the existing model-config cache independent of fresh scene reads.
      const modelConfigKey = ["fetchModelConfig2", true, selectedCharacterId]
      const cachedModelConfig =
        client.getQueryData<ModelConfigData>(modelConfigKey)
      const data = cachedModelConfig ?? (await getAllModelSettings())

      const ocrLang = await getOCRLanguage()

      let tempSystemPrompt = ""

      if (selectedSystemPrompt) {
        tempSystemPrompt =
          await resolveSelectedSystemPromptContent(selectedSystemPrompt)
      }

      const baseValues = buildBaseValues(
        cachedModelConfig ? undefined : data,
        tempSystemPrompt
      )

      let actor: ActorSettings | null = null
      if (!personaChatActive) {
        const { getActorSettingsForChatWithCharacterFallback } =
          await loadActorSettings()
        if (!isCurrent()) throw new Error("stale_selection")
        actor = await getActorSettingsForChatWithCharacterFallback({
          historyId,
          serverChatId,
          characterId: selectedCharacterId
        })
      }
      if (!isCurrent()) throw new Error("stale_selection")
      if (!actor) actor = createDefaultActorSettings()
      loadedSourceRef.current = isCurrent
      if (!cachedModelConfig) client.setQueryData(modelConfigKey, data)
      const nextOcrLanguage = ocrLang || ocrLanguage
      setStagedOcrLanguage(nextOcrLanguage)
      setActorSettingsState(actor)
      commitActorSettings(actor)
      if (!isCurrent()) throw new Error("stale_selection")

      const actorFields: Record<string, any> = {
        actorEnabled: actor.isEnabled,
        actorNotes: actor.notes,
        actorNotesGmOnly: actor.notesGmOnly ?? false,
        actorChatPosition: actor.chatPosition,
        actorChatDepth: actor.chatDepth,
        actorChatRole: actor.chatRole,
        actorTemplateMode: actor.templateMode ?? "merge",
        actorAppendable: actor.appendable ?? false
      }
      for (const aspect of actor.aspects || []) {
        actorFields[`actor_${aspect.id}`] = aspect.value
        actorFields[`actor_key_${aspect.id}`] = aspect.key
      }

      form.resetFields()
      if (!isCurrent()) throw new Error("stale_selection")
      form.setFieldsValue({ ...baseValues, ...actorFields })
      if (!isCurrent()) throw new Error("stale_selection")
      baselineRef.current = { values: structuredClone(form.getFieldsValue(true)), actor, ocrLanguage: nextOcrLanguage }

      const preview = buildActorPrompt(actor)
      setScenePreview({ prompt: preview, tokens: estimateActorTokens(preview) })
      commitActorPreview(preview, estimateActorTokens(preview))
      if (!isCurrent()) throw new Error("stale_selection")
      return data
    },
    enabled: open && !selectedCharacterMeta.isLoading,
    refetchOnMount: "always",
    refetchOnWindowFocus: false
  })

  const modelOptions = useMemo(() => {
    type GroupOption = {
      label: React.ReactNode
      options: Array<{
        label: React.ReactNode
        value: string
        searchLabel: string
        disabled?: boolean
      }>
    }
    const models = composerModels
    const groups = new Map<string, GroupOption>()

    for (const m of models) {
      const rawProvider = m.details?.provider ?? m.provider
      const providerKey = String(rawProvider || "other").toLowerCase()
      const providerLabel = getProviderDisplayName(rawProvider)
      const modelLabel = m.nickname || m.model
      const details = m.details
      const caps = Array.isArray(details?.capabilities)
        ? (details.capabilities ?? [])
        : []
      const hasVision = caps.includes("vision")
      const hasTools = caps.includes("tools")
      const hasFast = caps.includes("fast")
      const providerIconKey = rawProvider ?? "default"

      const optionDisplay = `${providerLabel} - ${modelLabel}`
      const optionLabel = (
        <div className="flex items-center gap-2" data-title={`${providerLabel} - ${modelLabel}`}>
          <ProviderIcons provider={providerIconKey} className="h-4 w-4" />
          <div className="flex flex-col min-w-0">
            <span className="truncate">{optionDisplay}</span>
            {(hasVision || hasTools || hasFast) && (
              <div className="mt-0.5 flex flex-wrap gap-1 text-[10px]">
                {hasVision && (
                  <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-primary">
                    Vision
                  </span>
                )}
                {hasTools && (
                  <span className="rounded-full bg-accent/10 px-1.5 py-0.5 text-accent">
                    Tools
                  </span>
                )}
                {hasFast && (
                  <span className="rounded-full bg-success/10 px-1.5 py-0.5 text-success">
                    Fast
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )

      if (!groups.has(providerKey)) {
        groups.set(providerKey, {
          label: (
            <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-text-subtle">
              <ProviderIcons provider={providerIconKey} className="h-3 w-3" />
              <span>{providerLabel}</span>
            </div>
          ),
          options: []
        })
      }
      const group = groups.get(providerKey)!
      group.options.push({
        label: optionLabel,
        value: getCanonicalModelKey(m),
        searchLabel: optionDisplay.toLowerCase()
      })
    }

    const groupedOptions: GroupOption[] = Array.from(groups.values())

    if (selectedModel) {
      const hasSelected = groupedOptions.some((group) =>
        group.options.some((option) => option.value === selectedSelection.key)
      )

      if (!hasSelected) {
        const providerLabel = selectedSelection.provider
          ? getProviderDisplayName(selectedSelection.provider)
          : t("playground:composer.chooseProvider", "Choose provider")
        const displayText = `${providerLabel} - ${selectedSelection.modelId}`
        groupedOptions.push({
          label: <span className="truncate">{providerLabel}</span>,
          options: [
            {
              label: <span className="truncate">{displayText}</span>,
              value: selectedSelection.key || selectedModel,
              searchLabel: displayText.toLowerCase(),
              disabled: !selectedSelection.provider
            }
          ]
        })
      }
    }

    return groupedOptions
  }, [composerModels, selectedModel, selectedSelection.key, selectedSelection.modelId, selectedSelection.provider, t])

  const providerOptions = useMemo(() => {
    const models = composerModels
    const providers = new Map<string, string>()
    for (const model of models) {
      const rawProvider = model.details?.provider ?? model.provider
      if (!rawProvider) continue
      const key = String(rawProvider)
      if (!providers.has(key)) {
        providers.set(key, getProviderDisplayName(rawProvider))
      }
    }
    return Array.from(providers.entries()).map(([value, label]) => ({
      value,
      label
    }))
  }, [composerModels])

  const tabItems = useMemo(
    () =>
      [
        {
          key: "model",
          label: t("modelSettings.tabs.model", "Model"),
          children: (
            <ModelBasicsTab
              form={form}
              selectedModel={selectedSelection.key}
              onModelChange={handleModelChange}
              modelOptions={modelOptions}
              modelsLoading={modelsLoading}
              isOCREnabled={isOCREnabled}
              ocrLanguage={stagedOcrLanguage}
              ocrLanguages={ocrLanguages}
              onOcrLanguageChange={(value) => {
                editRevisionRef.current++
                setStagedOcrLanguage(value)
              }}
            />
          )
        },
        {
          key: "conversation",
          label: t("modelSettings.tabs.conversation", "Conversation"),
          children: (
            <ConversationTab
              key={renderedView.revision}
              isCurrent={isEditorCurrent}
              onDraftChange={onConversationDraftChange}
              serverChatVersion={serverChatVersion}
              useDrawer={useDrawer}
              historyId={historyId}
              selectedSystemPrompt={selectedSystemPrompt}
              onSystemPromptChange={savePrompt}
              onResetSystemPrompt={resetSystemPrompt}
              uploadedFiles={uploadedFiles}
              onRemoveFile={removeUploadedFile}
              serverChatId={serverChatId}
              serverChatAssistantKind={serverChatAssistantKind}
              serverChatPersonaMemoryMode={serverChatPersonaMemoryMode}
              serverChatState={serverChatState}
              onStateChange={(state) => setServerChatState(state)}
              serverChatTopic={serverChatTopic}
              onTopicChange={setServerChatTopic}
              onVersionChange={setServerChatVersion}
            />
          )
        },
        {
          key: "advanced",
          label: t("modelSettings.tabs.advanced", "Advanced"),
          children: (
            <AdvancedParamsTab
              form={form}
              providerOptions={providerOptions}
            />
          )
        },
        !personaChatActive
          ? {
              key: "actor",
              label: t("modelSettings.tabs.actor", "Scene Director"),
              children: (
                <ActorTab
                  form={form}
                  actorSettings={actorSettings}
                  setActorSettings={setActorSettings}
                  isCurrent={isEditorCurrent}
                  captureEditGuard={captureEditGuard}
                  actorPreview={actorPreview}
                  actorTokenCount={actorTokenCount}
                  onRecompute={recomputeActorPreview}
                  newAspectTarget={newAspectTarget}
                  setNewAspectTarget={setNewAspectTarget}
                  newAspectName={newAspectName}
                  setNewAspectName={setNewAspectName}
                  actorPositionValue={actorPositionValue}
                />
              )
            }
          : null
      ].filter(Boolean),
    [
      t,
      form,
      selectedSelection.key,
      handleModelChange,
      modelOptions,
      modelsLoading,
      isOCREnabled,
      stagedOcrLanguage,
      useDrawer,
      selectedSystemPrompt,
      savePrompt,
      resetSystemPrompt,
      uploadedFiles,
      removeUploadedFile,
      historyId,
      serverChatId,
      serverChatVersion,
      serverChatAssistantKind,
      serverChatPersonaMemoryMode,
      renderedView.revision,
      onConversationDraftChange,
      serverChatState,
      setServerChatState,
      serverChatTopic,
      setServerChatTopic,
      setServerChatVersion,
      providerOptions,
      actorSettings,
      setActorSettings,
      isEditorCurrent,
      captureEditGuard,
      actorPreview,
      actorTokenCount,
      personaChatActive,
      recomputeActorPreview,
      newAspectTarget,
      newAspectName,
      actorPositionValue
    ]
  )

  const sceneUnavailable = isError || !(accountIsCurrent?.() ?? true)
  const sceneReady = !isFetching && Boolean(loadedSourceRef.current?.())
  const renderBody = () => {
    return (
      <>
        {modalContextHolder}
        <p className="mb-3 text-xs text-text-subtle">
          {t("modelSettings.scopeHelp", "Model and Advanced parameters are defaults for the selected model. Save applies staged defaults and scene edits to future turns. API/model choices apply immediately; conversation controls save separately to this chat.")}
        </p>
        {sceneUnavailable ? (
          <div className="space-y-2">
            <p role="alert">
              {t(
                "playground:composer.sceneLoadError",
                "Scene settings could not be loaded. Existing chat setup was left unchanged."
              )}
            </p>
            {readSourceRef.current?.() ? (
              <Button
                onClick={() => {
                  if (
                    viewRef.current === renderedView &&
                    readSourceRef.current?.()
                  )
                    setReadRevision((value) => value + 1)
                }}>
                {t("common:retry", "Retry")}
              </Button>
            ) : (
              <p>
                {t(
                  "playground:composer.sceneSourceUnavailable",
                  "Close this panel and check your account verification and selected chat before reopening scene settings."
                )}
              </p>
            )}
          </div>
        ) : null}
        {modelSaveError?.revision === renderedModelRevision && modelSaveError.panel === renderedPanel ? (
          <div className="space-y-2">
            <p role="alert">
              {modelSaveError.value === null
                ? t("modelSettings.modelDeviceClearError", "Model selection is cleared, but it could not be saved on this device. Retry model save to keep it cleared after reopening.")
                : t("modelSettings.modelDeviceSaveError", "Model choice is active, but it could not be saved on this device. Retry model save to keep this choice after reopening.")}
            </p>
            <Button loading={retryingModelSave} disabled={saving || !sceneReady} onClick={() => void retryModelSave()}>
              {t("modelSettings.retryModelSave", "Retry model save")}
            </Button>
          </div>
        ) : null}
        {saveError ? (
          <p role="alert">
            {t(
              "playground:composer.sceneSaveError",
              "Scene settings could not be saved. Your edits are still here; retry Save."
            )}
          </p>
        ) : null}
        {!isLoading && !sceneUnavailable && sceneReady ? (
          <Form
            form={form}
            // A remounted dialog may reuse cached config without rerunning its query.
            initialValues={buildBaseValues()}
            layout="vertical"
            disabled={saving || conversationPending}
            onFinish={async (values) => {
              if (
                saveAttemptRef.current ||
                conversationDraftRef.current.dirty || conversationDraftRef.current.pending ||
                !sceneReady ||
                viewRef.current !== renderedView
              )
                return
              const attempt = {}
              const isCurrent = loadedSourceRef.current
              saveAttemptRef.current = attempt
              setSaving(true)
              setSaveError(false)
              try {
                await saveSettings({
                  ...form.getFieldsValue(true),
                  ...values
                })
              } catch {
                if (
                  viewRef.current === renderedView &&
                  (personaChatActive || isCurrent?.())
                )
                  setSaveError(true)
              } finally {
                if (saveAttemptRef.current === attempt) {
                  saveAttemptRef.current = null
                  if (viewRef.current === renderedView) setSaving(false)
                }
              }
            }}
            onValuesChange={(changedValues) => {
              editRevisionRef.current++
              const keys = Object.keys(changedValues || {})
              const shouldUpdate = keys.some(
                (k) =>
                  k === "actorEnabled" ||
                  k === "actorNotes" ||
                  k === "actorNotesGmOnly" ||
                  k === "actorChatPosition" ||
                  k === "actorChatDepth" ||
                  k === "actorChatRole" ||
                  k === "actorAppendable" ||
                  k.startsWith("actor_")
              )
              if (shouldUpdate) {
                debouncedRecomputeActorPreview()
              }
            }}>
            <fieldset
              className="min-w-0 border-0 p-0"
              disabled={saving || conversationPending || !sceneReady}>
              <Tabs
                defaultActiveKey="model"
                destroyOnHidden={false}
                items={tabItems}
                className="settings-tabs"
              />
              <div className="mt-4">
                <LlamaCppAdvancedControls
                  selectedModel={selectedModel}
                  thinkingBudget={llamaThinkingBudgetTokens}
                  grammarMode={llamaGrammarMode}
                  grammarId={llamaGrammarId}
                  grammarInline={llamaGrammarInline}
                  grammarOverride={llamaGrammarOverride}
                  extraBody={llamaExtraBody}
                  onChange={handleLlamaControlChange}
                />
              </div>
              <div className="sticky bottom-0 z-20 mt-4 border-t border-border bg-surface py-3">
                {conversationDirty && <p className="mb-2 text-xs text-text-muted">{t("modelSettings.conversationPendingHelp", "Apply or cancel conversation changes before saving model settings.")}</p>}
                <Button htmlType="button" disabled={saving || conversationPending} onClick={() => {
                  if (!conversationDraftRef.current.pending && isEditorCurrent()) setOpen(false)
                }}>
                  {t("cancel", "Cancel")}
                </Button>
                <button
                  type="submit"
                  aria-busy={saving}
                  disabled={saving || conversationDirty || conversationPending || !sceneReady}
                  className="inline-flex w-full items-center justify-center rounded-md border border-transparent bg-primary px-3 py-2 min-h-[40px] text-sm font-medium text-surface shadow-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:var(--color-focus)] disabled:opacity-50">
                  {saving ? t("saving", "Saving...") : t("save", "Save")}
                </button>
              </div>
            </fieldset>
          </Form>
        ) : !sceneUnavailable ? (
          <Skeleton active />
        ) : null}
      </>
    )
  }

  if (useDrawer) {
    return (
      <Drawer
        placement="right"
        open={open}
        onClose={requestClose}
        size={500}
        title={t("currentChatModelSettings")}>
        {renderBody()}
      </Drawer>
    )
  }

  return (
    <Modal
      title={t("currentChatModelSettings")}
      open={open}
      onOk={requestClose}
      onCancel={requestClose}
      style={{ top: "1rem", paddingBottom: 0 }}
      styles={{
        container: { maxHeight: "calc(100dvh - 2rem)", display: "flex", flexDirection: "column" },
        header: { flexShrink: 0 },
        body: { minHeight: 0, overflowY: "auto" }
      }}
      footer={null}>
      {renderBody()}
    </Modal>
  )
}
