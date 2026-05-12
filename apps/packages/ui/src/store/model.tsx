import { createWithEqualityFn } from "zustand/traditional"
import {
  mergeGlobalAndScopedSettings,
  normalizeModelSettingsScope,
  stripUndefinedScopedSettings
} from "./model-settings-scope"

/**
 * Chat model settings - state values only (no actions)
 */
export type ChatModelSettings = {
  // Inference parameters
  f16KV?: boolean
  frequencyPenalty?: number
  keepAlive?: string
  logitsAll?: boolean
  mirostat?: number
  mirostatEta?: number
  mirostatTau?: number
  numBatch?: number
  numCtx?: number
  numGpu?: number
  numGqa?: number
  numKeep?: number
  numPredict?: number
  numThread?: number
  penalizeNewline?: boolean
  presencePenalty?: number
  repeatLastN?: number
  repeatPenalty?: number
  ropeFrequencyBase?: number
  ropeFrequencyScale?: number
  temperature?: number
  tfsZ?: number
  topK?: number
  topP?: number
  typicalP?: number
  useMLock?: boolean
  useMMap?: boolean
  useMlock?: boolean
  vocabOnly?: boolean
  seed?: number
  minP?: number

  // System configuration
  systemPrompt?: string
  reasoningEffort?: string
  thinking?: boolean
  ocrLanguage?: string

  // History & injection settings
  historyMessageLimit?: number
  historyMessageOrder?: string
  slashCommandInjectionMode?: string

  // API configuration
  apiProvider?: string
  extraHeaders?: string
  extraBody?: string
  llamaThinkingBudgetTokens?: number
  llamaGrammarMode?: "none" | "library" | "inline"
  llamaGrammarId?: string
  llamaGrammarInline?: string
  llamaGrammarOverride?: string

  // Response format
  jsonMode?: boolean
}

/**
 * Store type combining settings with actions
 */
type ChatModelSettingsStore = ChatModelSettings & {
  activeSettingsScope?: string
  globalSettings: ChatModelSettings
  scopedSettingsByModelKey: Record<string, Partial<ChatModelSettings>>

  setActiveSettingsScope: (scopeKey?: string | null) => void
  updateScopedSetting: <K extends keyof ChatModelSettings>(
    scopeKey: string,
    key: K,
    value: ChatModelSettings[K]
  ) => void
  getEffectiveSettings: (scopeKey?: string | null) => ChatModelSettings

  // Generic typed update method (replaces setX)
  updateSetting: <K extends keyof ChatModelSettings>(
    key: K,
    value: ChatModelSettings[K]
  ) => void
  updateSettings: (updates: Partial<ChatModelSettings>) => void
  reset: () => void

  // Individual setters (for backwards compatibility)
  setF16KV: (value: boolean) => void
  setFrequencyPenalty: (value: number) => void
  setKeepAlive: (value: string) => void
  setLogitsAll: (value: boolean) => void
  setMirostat: (value: number) => void
  setMirostatEta: (value: number) => void
  setMirostatTau: (value: number) => void
  setNumBatch: (value: number) => void
  setNumCtx: (value: number) => void
  setNumGpu: (value: number) => void
  setNumGqa: (value: number) => void
  setNumKeep: (value: number) => void
  setNumPredict: (value: number | undefined) => void
  setNumThread: (value: number) => void
  setPenalizeNewline: (value: boolean) => void
  setPresencePenalty: (value: number) => void
  setRepeatLastN: (value: number) => void
  setRepeatPenalty: (value: number) => void
  setRopeFrequencyBase: (value: number) => void
  setRopeFrequencyScale: (value: number) => void
  setTemperature: (value: number) => void
  setTfsZ: (value: number) => void
  setTopK: (value: number) => void
  setTopP: (value: number) => void
  setTypicalP: (value: number) => void
  setUseMLock: (value: boolean) => void
  setUseMMap: (value: boolean) => void
  setUseMlock: (value: boolean) => void
  setVocabOnly: (value: boolean) => void
  setSeed: (value: number | undefined) => void
  setMinP: (value: number) => void
  setSystemPrompt: (value: string) => void
  setReasoningEffort: (value: string) => void
  setThinking: (value: boolean) => void
  setOcrLanguage: (value: string) => void
  setHistoryMessageLimit: (value: number) => void
  setHistoryMessageOrder: (value: string) => void
  setSlashCommandInjectionMode: (value: string) => void
  setApiProvider: (value: string) => void
  setExtraHeaders: (value: string) => void
  setExtraBody: (value: string) => void
  setLlamaThinkingBudgetTokens: (value: number | undefined) => void
  setLlamaGrammarMode: (
    value: "none" | "library" | "inline" | undefined
  ) => void
  setLlamaGrammarId: (value: string | undefined) => void
  setLlamaGrammarInline: (value: string | undefined) => void
  setLlamaGrammarOverride: (value: string | undefined) => void
  setJsonMode: (value: boolean | undefined) => void
}

const INITIAL_STATE: ChatModelSettings = {
  f16KV: undefined,
  frequencyPenalty: undefined,
  keepAlive: undefined,
  logitsAll: undefined,
  mirostat: undefined,
  mirostatEta: undefined,
  mirostatTau: undefined,
  numBatch: undefined,
  numCtx: undefined,
  numGpu: undefined,
  numGqa: undefined,
  numKeep: undefined,
  numPredict: undefined,
  numThread: undefined,
  penalizeNewline: undefined,
  presencePenalty: undefined,
  repeatLastN: undefined,
  repeatPenalty: undefined,
  ropeFrequencyBase: undefined,
  ropeFrequencyScale: undefined,
  temperature: undefined,
  tfsZ: undefined,
  topK: undefined,
  topP: undefined,
  typicalP: undefined,
  useMLock: undefined,
  useMMap: undefined,
  useMlock: undefined,
  vocabOnly: undefined,
  seed: undefined,
  minP: undefined,
  systemPrompt: undefined,
  reasoningEffort: undefined,
  thinking: undefined,
  ocrLanguage: undefined,
  historyMessageLimit: undefined,
  historyMessageOrder: undefined,
  slashCommandInjectionMode: undefined,
  apiProvider: undefined,
  extraHeaders: undefined,
  extraBody: undefined,
  llamaThinkingBudgetTokens: undefined,
  llamaGrammarMode: undefined,
  llamaGrammarId: undefined,
  llamaGrammarInline: undefined,
  llamaGrammarOverride: undefined,
  jsonMode: undefined
}

const EMPTY_GLOBAL_SETTINGS: ChatModelSettings = {}

const trimScopeKey = (scopeKey?: string | null) => {
  if (typeof scopeKey !== "string") return undefined
  const trimmed = scopeKey.trim()
  if (trimmed.length === 0) return undefined

  const separatorIndex = trimmed.indexOf(":")
  if (separatorIndex <= 0 || separatorIndex === trimmed.length - 1) {
    return trimmed.toLowerCase()
  }

  return (
    normalizeModelSettingsScope(
      trimmed.slice(0, separatorIndex),
      trimmed.slice(separatorIndex + 1)
    ) ?? trimmed.toLowerCase()
  )
}

const withScopedSettings = (
  current: Record<string, Partial<ChatModelSettings>>,
  scopeKey: string,
  settings: Partial<ChatModelSettings>
) => {
  const next = { ...current }
  if (Object.keys(settings).length === 0) {
    delete next[scopeKey]
  } else {
    next[scopeKey] = settings
  }
  return next
}

const getEffectiveSettingsState = (
  globalSettings: ChatModelSettings,
  scopedSettingsByModelKey: Record<string, Partial<ChatModelSettings>>,
  activeSettingsScope?: string
): ChatModelSettings => ({
  ...INITIAL_STATE,
  ...mergeGlobalAndScopedSettings(
    globalSettings,
    activeSettingsScope
      ? scopedSettingsByModelKey[activeSettingsScope]
      : undefined
  )
})

const applyActiveSettingsUpdate = (
  state: ChatModelSettingsStore,
  updates: Partial<ChatModelSettings>
) => {
  const activeScope = trimScopeKey(state.activeSettingsScope)

  if (activeScope) {
    const scopedSettings = stripUndefinedScopedSettings({
      ...(state.scopedSettingsByModelKey[activeScope] || {}),
      ...updates
    })
    const scopedSettingsByModelKey = withScopedSettings(
      state.scopedSettingsByModelKey,
      activeScope,
      scopedSettings
    )

    return {
      scopedSettingsByModelKey,
      ...getEffectiveSettingsState(
        state.globalSettings,
        scopedSettingsByModelKey,
        activeScope
      )
    }
  }

  const globalSettings = stripUndefinedScopedSettings({
    ...state.globalSettings,
    ...updates
  })

  return {
    globalSettings,
    ...getEffectiveSettingsState(
      globalSettings,
      state.scopedSettingsByModelKey,
      undefined
    )
  }
}

export const useStoreChatModelSettings = createWithEqualityFn<ChatModelSettingsStore>(
  (set, get) => ({
    ...INITIAL_STATE,
    activeSettingsScope: undefined,
    globalSettings: EMPTY_GLOBAL_SETTINGS,
    scopedSettingsByModelKey: {},

    setActiveSettingsScope: (scopeKey) =>
      set((state) => {
        const activeSettingsScope = trimScopeKey(scopeKey)
        if (state.activeSettingsScope === activeSettingsScope) return state

        return {
          activeSettingsScope,
          ...getEffectiveSettingsState(
            state.globalSettings,
            state.scopedSettingsByModelKey,
            activeSettingsScope
          )
        }
      }),
    updateScopedSetting: (scopeKey, key, value) =>
      set((state) => {
        const normalizedScopeKey = trimScopeKey(scopeKey)
        if (!normalizedScopeKey) return state

        const scopedSettings = stripUndefinedScopedSettings({
          ...(state.scopedSettingsByModelKey[normalizedScopeKey] || {}),
          [key]: value
        })
        const scopedSettingsByModelKey = withScopedSettings(
          state.scopedSettingsByModelKey,
          normalizedScopeKey,
          scopedSettings
        )

        return {
          scopedSettingsByModelKey,
          ...(state.activeSettingsScope === normalizedScopeKey
            ? getEffectiveSettingsState(
                state.globalSettings,
                scopedSettingsByModelKey,
                normalizedScopeKey
              )
            : {})
        }
      }),
    getEffectiveSettings: (scopeKey) => {
      const state = get()
      return getEffectiveSettingsState(
        state.globalSettings,
        state.scopedSettingsByModelKey,
        trimScopeKey(scopeKey) || state.activeSettingsScope
      )
    },

    // Generic typed update methods
    updateSetting: (key, value) =>
      set((state) => applyActiveSettingsUpdate(state, { [key]: value })),
    updateSettings: (updates) =>
      set((state) => applyActiveSettingsUpdate(state, updates)),
    reset: () =>
      set({
        ...INITIAL_STATE,
        activeSettingsScope: undefined,
        globalSettings: EMPTY_GLOBAL_SETTINGS,
        scopedSettingsByModelKey: {}
      }),

    // Individual setters
    setF16KV: (value) => set((state) => applyActiveSettingsUpdate(state, { f16KV: value })),
    setFrequencyPenalty: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { frequencyPenalty: value })
      ),
    setKeepAlive: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { keepAlive: value })),
    setLogitsAll: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { logitsAll: value })),
    setMirostat: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { mirostat: value })),
    setMirostatEta: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { mirostatEta: value })),
    setMirostatTau: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { mirostatTau: value })),
    setNumBatch: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numBatch: value })),
    setNumCtx: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numCtx: value })),
    setNumGpu: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numGpu: value })),
    setNumGqa: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numGqa: value })),
    setNumKeep: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numKeep: value })),
    setNumPredict: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numPredict: value })),
    setNumThread: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { numThread: value })),
    setPenalizeNewline: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { penalizeNewline: value })
      ),
    setPresencePenalty: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { presencePenalty: value })
      ),
    setRepeatLastN: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { repeatLastN: value })),
    setRepeatPenalty: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { repeatPenalty: value })
      ),
    setRopeFrequencyBase: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { ropeFrequencyBase: value })
      ),
    setRopeFrequencyScale: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { ropeFrequencyScale: value })
      ),
    setTemperature: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { temperature: value })),
    setTfsZ: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { tfsZ: value })),
    setTopK: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { topK: value })),
    setTopP: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { topP: value })),
    setTypicalP: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { typicalP: value })),
    setUseMLock: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { useMLock: value })),
    setUseMMap: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { useMMap: value })),
    setUseMlock: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { useMlock: value })),
    setVocabOnly: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { vocabOnly: value })),
    setSeed: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { seed: value })),
    setMinP: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { minP: value })),
    setSystemPrompt: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { systemPrompt: value })),
    setReasoningEffort: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { reasoningEffort: value })
      ),
    setThinking: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { thinking: value })),
    setOcrLanguage: (value) =>
      set((state) =>
        state.ocrLanguage === value
          ? state
          : applyActiveSettingsUpdate(state, { ocrLanguage: value })
      ),
    setHistoryMessageLimit: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { historyMessageLimit: value })
      ),
    setHistoryMessageOrder: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { historyMessageOrder: value })
      ),
    setSlashCommandInjectionMode: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { slashCommandInjectionMode: value })
      ),
    setApiProvider: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { apiProvider: value })),
    setExtraHeaders: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { extraHeaders: value })),
    setExtraBody: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { extraBody: value })),
    setLlamaThinkingBudgetTokens: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { llamaThinkingBudgetTokens: value })
      ),
    setLlamaGrammarMode: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { llamaGrammarMode: value })
      ),
    setLlamaGrammarId: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { llamaGrammarId: value })
      ),
    setLlamaGrammarInline: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { llamaGrammarInline: value })
      ),
    setLlamaGrammarOverride: (value) =>
      set((state) =>
        applyActiveSettingsUpdate(state, { llamaGrammarOverride: value })
      ),
    setJsonMode: (value) =>
      set((state) => applyActiveSettingsUpdate(state, { jsonMode: value }))
  })
)

// Expose for Playwright tests and debugging (development only)
if (typeof window !== "undefined" && import.meta?.env?.DEV) {
  ;(window as any).__tldw_useStoreChatModelSettings = useStoreChatModelSettings
}
