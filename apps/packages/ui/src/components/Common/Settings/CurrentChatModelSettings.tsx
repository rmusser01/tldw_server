import { useMessageOption } from "@/hooks/useMessageOption"
import { getAllModelSettings } from "@/services/model-settings"
import { useStoreChatModelSettings, type ChatModelSettings } from "@/store/model"
import { useActorStore } from "@/store/actor"
import { useQuery } from "@tanstack/react-query"
import {
  Drawer,
  Form,
  Modal,
  Skeleton,
  Tabs
} from "antd"
import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { shallow } from "zustand/shallow"
import { SaveButton } from "../SaveButton"
import { getOCRLanguage } from "@/services/ocr"
import { ocrLanguages } from "@/data/ocr-language"
import { fetchChatModels } from "@/services/tldw-server"
import { ProviderIcons } from "@/components/Common/ProviderIcon"
import { getProviderDisplayName } from "@/utils/provider-registry"
import type { ActorSettings, ActorTarget } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"
import {
  buildActorPrompt,
  buildActorSettingsFromForm,
  estimateActorTokens
} from "@/utils/actor"
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider"
import {
  getCanonicalModelKey,
  getModelId,
  getModelProvider
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

type ModelDetails = {
  provider?: string
  capabilities?: string[]
}

type ChatModel = {
  model: string
  nickname?: string
  provider?: string
  details?: ModelDetails
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
  const [selectedAssistant] = useSelectedAssistant(null)
  const selectedCharacterId = selectedCharacter?.id ?? null
  const personaChatActive =
    serverChatAssistantKind === "persona" || selectedAssistant?.kind === "persona"

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

  const { data: composerModels = [], isLoading: modelsLoading } = useQuery<
    ChatModel[]
  >({
    queryKey: ["playground:chatModels", open],
    queryFn: async () => {
      try {
        return await fetchChatModels({ returnEmpty: true })
      } catch (error) {
        console.error("Failed to fetch chat models:", error)
        throw error
      }
    },
    enabled: open,
    retry: 2
  })

  const selectedModelSettingsScope = useMemo(() => {
    if (!selectedModel) return null

    const selected = selectedModel.trim()
    if (!selected) return null

    const selectedLower = selected.toLowerCase()
    const models = Array.isArray(composerModels) ? composerModels : []
    const providerSelection = parseProviderQualifiedModelSelection(selected)

    const canonicalMatch = models.find(
      (model) => getCanonicalModelKey(model).toLowerCase() === selectedLower
    )
    if (canonicalMatch) {
      return getCanonicalModelKey(canonicalMatch)
    }

    const selectedModelId = providerSelection.modelId
    const selectedProviderHint =
      providerSelection.provider ||
      (typeof apiProvider === "string" && apiProvider.trim()
        ? apiProvider.trim().toLowerCase()
        : null)

    const providerMatch = selectedProviderHint
      ? models.find(
          (model) =>
            getModelId(model) === selectedModelId &&
            getModelProvider(model) === selectedProviderHint
        )
      : null
    if (providerMatch) {
      return getCanonicalModelKey(providerMatch)
    }

    const modelIdMatch = models.find((model) => {
      const modelId = getModelId(model)
      return (
        modelId === selectedModelId ||
        String(model.model || "") === selectedModelId
      )
    })
    if (modelIdMatch) {
      return getCanonicalModelKey(modelIdMatch)
    }

    return getCanonicalModelKey(selectedProviderHint || "custom", selectedModelId)
  }, [apiProvider, composerModels, selectedModel])

  const saveSettings = useCallback(
    (values: CurrentChatModelFormValues) => {
      const latestSettingsState = useStoreChatModelSettings.getState()
      const explicitSettingsScope =
        typeof settingsScope === "string" && settingsScope.trim()
          ? settingsScope.trim()
          : null
      const targetSettingsScope =
        explicitSettingsScope ||
        latestSettingsState.activeSettingsScope ||
        selectedModelSettingsScope

      if (targetSettingsScope) {
        setActiveSettingsScope(targetSettingsScope)
      }

      Object.keys(values).forEach((key) => {
        if (!isChatModelSettingKey(key)) return
        if (targetSettingsScope) {
          updateScopedSetting(targetSettingsScope, key, values[key])
          return
        }
        updateSetting(key, values[key])
      })

      const base = actorSettings ?? createDefaultActorSettings()
      const next: ActorSettings = buildActorSettingsFromForm(base, values)

      setActorSettings(next)
      if (!personaChatActive) {
        void loadActorSettings().then(({ saveActorSettingsForChat }) =>
          saveActorSettingsForChat({
            historyId,
            serverChatId,
            settings: next
          })
        )
      }
    },
    [
      actorSettings,
      historyId,
      personaChatActive,
      serverChatId,
      settingsScope,
      selectedModelSettingsScope,
      setActiveSettingsScope,
      setActorSettings,
      updateScopedSetting,
      updateSetting
    ]
  )

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

  const { isLoading } = useQuery({
    queryKey: ["fetchModelConfig2", open, selectedCharacterId],
    queryFn: async () => {
      if (import.meta?.env?.DEV) {
        console.count("CurrentChatModelSettings/fetchModelConfig")
      }
      const data = await getAllModelSettings()

      const ocrLang = await getOCRLanguage()

      if (isOCREnabled && ocrLang) {
        setOcrLanguage(ocrLang)
      }
      let tempSystemPrompt = ""

      if (selectedSystemPrompt) {
        tempSystemPrompt = await resolveSelectedSystemPromptContent(
          selectedSystemPrompt
        )
      }

      const baseValues = buildBaseValues(data, tempSystemPrompt)

      let actor = actorSettings
      if (!actor && !personaChatActive) {
        const { getActorSettingsForChatWithCharacterFallback } =
          await loadActorSettings()
        actor = await getActorSettingsForChatWithCharacterFallback({
          historyId,
          serverChatId,
          characterId: selectedCharacterId
        })
      }
      if (!actor) {
        actor = createDefaultActorSettings()
      }
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
    enabled: open && !selectedCharacterMeta.isLoading && !personaChatActive,
    refetchOnMount: false,
    refetchOnWindowFocus: false
  })

  const modelOptions = useMemo(() => {
    type GroupOption = {
      label: React.ReactNode
      options: Array<{
        label: React.ReactNode
        value: string
        searchLabel: string
      }>
    }
    const models = composerModels
    if (!models.length) {
      if (selectedModel) {
        const displayText = `Custom - ${selectedModel}`
        const fallbackGroup: GroupOption = {
          label: <span className="truncate">Custom</span>,
          options: [
            {
              label: <span className="truncate">{displayText}</span>,
              value: selectedModel,
              searchLabel: displayText.toLowerCase()
            }
          ]
        }
        return [fallbackGroup]
      }
      return []
    }

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
        value: m.model,
        searchLabel: optionDisplay.toLowerCase()
      })
    }

    const groupedOptions: GroupOption[] = Array.from(groups.values())

    if (selectedModel) {
      const hasSelected = groupedOptions.some((group) =>
        group.options.some((option) => option.value === selectedModel)
      )

      if (!hasSelected) {
        const displayText = `Custom - ${selectedModel}`
        groupedOptions.push({
          label: <span className="truncate">Custom</span>,
          options: [
            {
              label: <span className="truncate">{displayText}</span>,
              value: selectedModel,
              searchLabel: displayText.toLowerCase()
            }
          ]
        })
      }
    }

    return groupedOptions
  }, [composerModels, selectedModel])

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
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
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
      setSelectedModel,
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

  const renderBody = () => {
    return (
      <>
        {!isLoading ? (
          <Form
            form={form}
            layout="vertical"
            onFinish={(values) => {
              saveSettings(values)
              setOpen(false)
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
              <SaveButton
                className="w-full text-center inline-flex items-center justify-center"
                btnType="submit"
              />
            </div>
          </Form>
        ) : (
          <Skeleton active />
        )}
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
