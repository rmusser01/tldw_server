import { EMPTY_CHAT_MODELS, useChatModelCatalog } from "@/hooks/useChatModelCatalog"
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
  const [form] = Form.useForm<CurrentChatModelFormValues>()
  const {
    temperature,
    topK,
    topP,
    keepAlive,
    numCtx,
    seed,
    numGpu,
    numPredict,
    useMMap,
    minP,
    repeatLastN,
    repeatPenalty,
    useMlock,
    tfsZ,
    numKeep,
    numThread,
    reasoningEffort,
    thinking,
    historyMessageLimit,
    historyMessageOrder,
    slashCommandInjectionMode,
    apiProvider,
    extraHeaders,
    extraBody,
    llamaThinkingBudgetTokens: storeLlamaThinkingBudgetTokens,
    llamaGrammarMode: storeLlamaGrammarMode,
    llamaGrammarId: storeLlamaGrammarId,
    llamaGrammarInline: storeLlamaGrammarInline,
    llamaGrammarOverride: storeLlamaGrammarOverride,
    jsonMode,
    systemPrompt,
    ocrLanguage,
    updateSetting,
    updateScopedSetting,
    setActiveSettingsScope,
    setOcrLanguage
  } = useStoreChatModelSettings(
    (state) => ({
      temperature: state.temperature,
      topK: state.topK,
      topP: state.topP,
      keepAlive: state.keepAlive,
      numCtx: state.numCtx,
      seed: state.seed,
      numGpu: state.numGpu,
      numPredict: state.numPredict,
      useMMap: state.useMMap,
      minP: state.minP,
      repeatLastN: state.repeatLastN,
      repeatPenalty: state.repeatPenalty,
      useMlock: state.useMlock,
      tfsZ: state.tfsZ,
      numKeep: state.numKeep,
      numThread: state.numThread,
      reasoningEffort: state.reasoningEffort,
      thinking: state.thinking,
      historyMessageLimit: state.historyMessageLimit,
      historyMessageOrder: state.historyMessageOrder,
      slashCommandInjectionMode: state.slashCommandInjectionMode,
      apiProvider: state.apiProvider,
      extraHeaders: state.extraHeaders,
      extraBody: state.extraBody,
      llamaThinkingBudgetTokens: state.llamaThinkingBudgetTokens,
      llamaGrammarMode: state.llamaGrammarMode,
      llamaGrammarId: state.llamaGrammarId,
      llamaGrammarInline: state.llamaGrammarInline,
      llamaGrammarOverride: state.llamaGrammarOverride,
      jsonMode: state.jsonMode,
      systemPrompt: state.systemPrompt,
      ocrLanguage: state.ocrLanguage,
      updateSetting: state.updateSetting,
      updateScopedSetting: state.updateScopedSetting,
      setActiveSettingsScope: state.setActiveSettingsScope,
      setOcrLanguage: state.setOcrLanguage
    }),
    shallow
  )
  const {
    historyId,
    selectedSystemPrompt,
    uploadedFiles,
    removeUploadedFile,
    selectedModel,
    setSelectedModel,
    serverChatId,
    serverChatAssistantKind,
    serverChatPersonaMemoryMode,
    serverChatTopic,
    setServerChatTopic,
    serverChatState,
    setServerChatState,
    setServerChatVersion,
    setServerChatPersonaMemoryMode
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
  const restoreRevision = usePlaygroundSessionStore(
    (state) => state.restoreRevision
  )
  const selectionRevision = usePlaygroundSessionStore(
    (state) => state.sourceSelectionRevision
  )
  const [readRevision, setReadRevision] = React.useState(0)
  const [saving, setSaving] = React.useState(false)
  const [saveError, setSaveError] = React.useState(false)
  const saveAttemptRef = React.useRef<object | null>(null)
  const readSourceRef = React.useRef<(() => boolean) | null>(null)
  const loadedSourceRef = React.useRef<(() => boolean) | null>(null)
  const viewKey = JSON.stringify([
    open,
    historyId,
    serverChatId,
    selectedCharacterId,
    personaChatActive,
    draftRevision,
    restoreRevision,
    selectionRevision,
    readRevision
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
  }
  const renderedView = viewRef.current
  React.useEffect(() => {
    saveAttemptRef.current = null
    setSaving(false)
    setSaveError(false)
  }, [renderedView])

  const {
    settings: actorSettings,
    setSettings: setActorSettings,
    preview: actorPreview,
    tokenCount: actorTokenCount,
    setPreviewAndTokens
  } = useActorStore(
    (state) => ({
      settings: state.settings,
      setSettings: state.setSettings,
      preview: state.preview,
      tokenCount: state.tokenCount,
      setPreviewAndTokens: state.setPreviewAndTokens
    }),
    shallow
  )
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

  const savePrompt = useCallback(
    (value: string) => {
      updateSetting("systemPrompt", value)
    },
    [updateSetting]
  )

  const resetSystemPrompt = useCallback(async () => {
    const nextValue = await resolveSelectedSystemPromptContent(
      selectedSystemPrompt
    )
    form.setFieldValue("systemPrompt", nextValue)
    savePrompt(nextValue)
  }, [form, savePrompt, selectedSystemPrompt])

  const recomputeActorPreview = useCallback(() => {
    const values = form.getFieldsValue()
    const base = actorSettings ?? createDefaultActorSettings()

    const next: ActorSettings = buildActorSettingsFromForm(base, values)

    const preview = buildActorPrompt(next)
    setPreviewAndTokens(preview, estimateActorTokens(preview))
  }, [actorSettings, form, setPreviewAndTokens])

  const timeoutRef = React.useRef<number | undefined>()

  const debouncedRecomputeActorPreview = React.useMemo(() => {
    return () => {
      if (timeoutRef.current !== undefined) {
        window.clearTimeout(timeoutRef.current)
      }
      timeoutRef.current = window.setTimeout(() => {
        recomputeActorPreview()
      }, 150)
    }
  }, [recomputeActorPreview])

  React.useEffect(() => {
    return () => {
      if (timeoutRef.current !== undefined) {
        window.clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  React.useEffect(() => {
    if (!open) return
    if (import.meta?.env?.DEV) {
      console.count("CurrentChatModelSettings/recomputeActorPreview")
    }
    recomputeActorPreview()
  }, [actorSettings, open, recomputeActorPreview])

  const { data: composerModels = EMPTY_CHAT_MODELS, isLoading: modelsLoading } = useChatModelCatalog(open)

  const activeSettingsScope = useStoreChatModelSettings(state => state.activeSettingsScope)
  const scopedApiProvider = useStoreChatModelSettings(state => state.activeSettingsScope ? state.scopedSettingsByModelKey?.[state.activeSettingsScope]?.apiProvider : undefined)
  const selectedSelection = useMemo(
    () => resolveModelSelectorSelection(composerModels, selectedModel, apiProvider, { scope: activeSettingsScope, providerOverride: scopedApiProvider }),
    [apiProvider, composerModels, selectedModel, activeSettingsScope, scopedApiProvider]
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

      if (targetSettingsScope) {
        setActiveSettingsScope(targetSettingsScope)
      }

      Object.keys(values).forEach((key) => {
        if (!isChatModelSettingKey(key)) return
        const normalizedValue = normalizeCurrentChatModelSettingValue(
          key,
          values[key]
        )
        if (targetSettingsScope) {
          updateScopedSetting(targetSettingsScope, key, normalizedValue)
          return
        }
        updateSetting(key, normalizedValue)
      })

      const base = actorSettings ?? createDefaultActorSettings()
      const next: ActorSettings = buildActorSettingsFromForm(base, values)

      if (!personaChatActive) {
        const { saveActorSettingsForChat } = await loadActorSettings()
        if (!isCurrent?.()) return false
        const saved = await saveActorSettingsForChat({
          historyId,
          serverChatId,
          settings: next
        })
        if (!isCurrent()) return false
        if (!saved) throw new Error("actor_settings_save_failed")
      }
      setActorSettings(next)
      return true
    },
    [
      actorSettings,
      accountIsCurrent,
      open,
      renderedView,
      historyId,
      personaChatActive,
      serverChatId,
      settingsScope,
      composerModels,
      selectedModel,
      setActiveSettingsScope,
      setActorSettings,
      updateScopedSetting,
      updateSetting
    ]
  )

  const handleModelChange = useCallback((value: string) => {
    const state = useStoreChatModelSettings.getState()
    const nextScope = resolveModelSelectorSelection(composerModels, value, state.apiProvider, {
      scope: state.activeSettingsScope,
      providerOverride: state.activeSettingsScope ? state.scopedSettingsByModelKey[state.activeSettingsScope]?.apiProvider : undefined
    }).settingsScope
    if (nextScope !== state.activeSettingsScope) {
      state.setActiveSettingsScope(nextScope)
      const nextSettings = useStoreChatModelSettings.getState().getEffectiveSettings(nextScope)
      const nextValues: Partial<ChatModelSettings> = Object.fromEntries([...CHAT_MODEL_SETTING_KEYS].map(key => [key, nextSettings[key]]))
      form.setFieldsValue(nextValues)
    }
    setSelectedModel(value)
  }, [composerModels, form, setSelectedModel])

  const handleLlamaControlChange = useCallback(
    (key: keyof ChatModelSettings, value: number | string | undefined) => {
      form.setFieldValue(key, value)
    },
    [form]
  )

  const buildBaseValues = useCallback(
    (data?: ModelConfigData | null, promptFallback?: string) => ({
      temperature: temperature ?? data?.temperature,
      topK: topK ?? data?.topK,
      topP: topP ?? data?.topP,
      keepAlive: keepAlive ?? data?.keepAlive,
      numCtx: numCtx ?? data?.numCtx,
      seed,
      numGpu: numGpu ?? data?.numGpu,
      numPredict: numPredict ?? data?.numPredict,
      systemPrompt: systemPrompt ?? promptFallback ?? "",
      useMMap: useMMap ?? data?.useMMap,
      minP: minP ?? data?.minP,
      repeatLastN: repeatLastN ?? data?.repeatLastN,
      repeatPenalty: repeatPenalty ?? data?.repeatPenalty,
      useMlock: useMlock ?? data?.useMlock,
      tfsZ: tfsZ ?? data?.tfsZ,
      numKeep: numKeep ?? data?.numKeep,
      numThread: numThread ?? data?.numThread,
      reasoningEffort,
      thinking,
      historyMessageLimit,
      historyMessageOrder,
      slashCommandInjectionMode,
      apiProvider,
      extraHeaders,
      extraBody,
      llamaThinkingBudgetTokens:
        storeLlamaThinkingBudgetTokens ?? data?.llamaThinkingBudgetTokens,
      llamaGrammarMode: storeLlamaGrammarMode ?? data?.llamaGrammarMode,
      llamaGrammarId: storeLlamaGrammarId ?? data?.llamaGrammarId,
      llamaGrammarInline: storeLlamaGrammarInline ?? data?.llamaGrammarInline,
      llamaGrammarOverride:
        storeLlamaGrammarOverride ?? data?.llamaGrammarOverride,
      jsonMode
    }),
    [
      temperature,
      topK,
      topP,
      keepAlive,
      numCtx,
      seed,
      numGpu,
      numPredict,
      systemPrompt,
      useMMap,
      minP,
      repeatLastN,
      repeatPenalty,
      useMlock,
      tfsZ,
      numKeep,
      numThread,
      reasoningEffort,
      thinking,
      historyMessageLimit,
      historyMessageOrder,
      slashCommandInjectionMode,
      apiProvider,
      extraHeaders,
      extraBody,
      storeLlamaThinkingBudgetTokens,
      storeLlamaGrammarMode,
      storeLlamaGrammarId,
      storeLlamaGrammarInline,
      storeLlamaGrammarOverride,
      jsonMode
    ]
  )

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
      if (isOCREnabled && ocrLang) setOcrLanguage(ocrLang)
      setActorSettings(actor)

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

      form.setFieldsValue({
        ...baseValues,
        ...actorFields
      })

      const preview = buildActorPrompt(actor)
      setPreviewAndTokens(preview, estimateActorTokens(preview))
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
              ocrLanguage={ocrLanguage}
              ocrLanguages={ocrLanguages}
              onOcrLanguageChange={(value) => setOcrLanguage(value)}
            />
          )
        },
        {
          key: "conversation",
          label: t("modelSettings.tabs.conversation", "Conversation"),
          children: (
            <ConversationTab
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
              onPersonaMemoryModeChange={setServerChatPersonaMemoryMode}
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
      selectedModel,
      selectedSelection.key,
      handleModelChange,
      modelOptions,
      modelsLoading,
      isOCREnabled,
      ocrLanguage,
      setOcrLanguage,
      useDrawer,
      selectedSystemPrompt,
      savePrompt,
      resetSystemPrompt,
      uploadedFiles,
      removeUploadedFile,
      serverChatId,
      serverChatState,
      setServerChatState,
      serverChatTopic,
      setServerChatTopic,
      setServerChatVersion,
      providerOptions,
      actorSettings,
      setActorSettings,
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
  const sceneReady =
    personaChatActive || (!isFetching && Boolean(loadedSourceRef.current?.()))
  const renderBody = () => {
    return (
      <>
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
            disabled={saving}
            onFinish={async (values) => {
              if (
                saveAttemptRef.current ||
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
                const saved = await saveSettings({
                  ...form.getFieldsValue(true),
                  ...values
                })
                if (
                  saved &&
                  viewRef.current === renderedView &&
                  (personaChatActive || isCurrent?.())
                )
                  setOpen(false)
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
              disabled={saving || !sceneReady}>
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
              <div className="mt-4 border-t border-border pt-4">
                <button
                  type="submit"
                  aria-busy={saving}
                  disabled={saving || !sceneReady}
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
        onClose={() => setOpen(false)}
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
      onOk={() => setOpen(false)}
      onCancel={() => setOpen(false)}
      footer={null}>
      {renderBody()}
    </Modal>
  )
}
