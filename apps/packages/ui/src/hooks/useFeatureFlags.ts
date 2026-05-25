import { useCallback, useMemo } from "react"
import { useStorage } from "@plasmohq/storage/hook"
import {
  FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS,
  FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
  createRolloutSubjectId,
  isFlagEnabledForRollout,
  resolveRolloutPercentageFromCandidates
} from "@/utils/feature-rollout"

/**
 * Feature flags for gradual UX redesign rollout.
 * Most flags default to true (new UX); specific flags may default off.
 */

// Flag keys
export const FEATURE_FLAGS = {
  /** Redesigned chat interface with sidebar and streaming */
  NEW_CHAT: "ff_newChat",
  /** Redesigned settings pages layout */
  NEW_SETTINGS: "ff_newSettings",
  /** Cmd+K command palette for quick navigation */
  COMMAND_PALETTE: "ff_commandPalette",
  /** Compact message bubbles in chat */
  COMPACT_MESSAGES: "ff_compactMessages",
  /** Collapsible sidebar in chat view */
  CHAT_SIDEBAR: "ff_chatSidebar",
  /** Side-by-side model comparison in chat */
  COMPARE_MODE: "ff_compareMode",
  /** Streaming responses in knowledge QA */
  KNOWLEDGE_QA_STREAMING: "ff_knowledgeQaStreaming",
  /** Side-by-side comparison in knowledge QA */
  KNOWLEDGE_QA_COMPARISON: "ff_knowledgeQaComparison",
  /** Branching conversation trees in knowledge QA */
  KNOWLEDGE_QA_BRANCHING: "ff_knowledgeQaBranching",
  /** Navigation panel in media viewer */
  MEDIA_NAVIGATION_PANEL: "ff_mediaNavigationPanel",
  /** Rich content rendering in media viewer */
  MEDIA_RICH_RENDERING: "ff_mediaRichRendering",
  /** Display mode selector in media analysis */
  MEDIA_ANALYSIS_DISPLAY_MODE_SELECTOR: "ff_mediaAnalysisDisplayModeSelector",
  /** Use generated fallback as default in media navigation */
  MEDIA_NAVIGATION_GENERATED_FALLBACK_DEFAULT:
    "ff_mediaNavigationGeneratedFallbackDefault",
  /** Provenance tracking in Research Workspace */
  RESEARCH_WORKSPACE_PROVENANCE_V1: "research_workspace_provenance_v1",
  /** Status guardrails in Research Workspace */
  RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1:
    "research_workspace_status_guardrails_v1"
} as const

export type FeatureFlagKey = (typeof FEATURE_FLAGS)[keyof typeof FEATURE_FLAGS]

const FEATURE_FLAG_DEFAULTS: Partial<Record<FeatureFlagKey, boolean>> = {
  [FEATURE_FLAGS.COMPARE_MODE]: false,
  [FEATURE_FLAGS.KNOWLEDGE_QA_STREAMING]: true,
  [FEATURE_FLAGS.KNOWLEDGE_QA_COMPARISON]: false,
  [FEATURE_FLAGS.KNOWLEDGE_QA_BRANCHING]: true,
  [FEATURE_FLAGS.MEDIA_NAVIGATION_PANEL]: true,
  [FEATURE_FLAGS.MEDIA_RICH_RENDERING]: true,
  [FEATURE_FLAGS.MEDIA_ANALYSIS_DISPLAY_MODE_SELECTOR]: true,
  [FEATURE_FLAGS.MEDIA_NAVIGATION_GENERATED_FALLBACK_DEFAULT]: true,
  [FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1]: true,
  [FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1]: true
}

type ResearchWorkspaceRolloutFlag =
  | typeof FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1
  | typeof FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1

const RESEARCH_WORKSPACE_ROLLOUT_DEFAULT_PERCENTAGE = 100

const RESEARCH_WORKSPACE_ROLLOUT_CONFIG: Record<
  ResearchWorkspaceRolloutFlag,
  {
    storageKey: string
    viteEnvKey: string
    nextEnvKey: string
  }
> = {
  [FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1]: {
    storageKey:
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_workspace_provenance_v1,
    viteEnvKey: "VITE_RESEARCH_WORKSPACE_PROVENANCE_V1_ROLLOUT_PERCENTAGE",
    nextEnvKey: "NEXT_PUBLIC_RESEARCH_WORKSPACE_PROVENANCE_V1_ROLLOUT_PERCENTAGE"
  },
  [FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1]: {
    storageKey:
      FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
        .research_workspace_status_guardrails_v1,
    viteEnvKey: "VITE_RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1_ROLLOUT_PERCENTAGE",
    nextEnvKey:
      "NEXT_PUBLIC_RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1_ROLLOUT_PERCENTAGE"
  }
}

export const RESEARCH_WORKSPACE_ROLLOUT_PERCENTAGE_STORAGE_KEYS = {
  [FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1]:
    FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS.research_workspace_provenance_v1,
  [FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1]:
    FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS
      .research_workspace_status_guardrails_v1
} as const

const isResearchWorkspaceRolloutFlag = (
  flag: FeatureFlagKey
): flag is ResearchWorkspaceRolloutFlag =>
  flag === FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1 ||
  flag === FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1

const readLocalStorageValue = (key: string): string | null => {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

const writeLocalStorageValue = (key: string, value: string): void => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(key, value)
  } catch {
    // Ignore storage write failures; rollout will fall back to in-memory defaults.
  }
}

const readRolloutWindowOverrides = (): Record<string, unknown> | null => {
  if (typeof window === "undefined") return null
  const overrideValue = (
    window as Window & { __TLDW_RESEARCH_WORKSPACE_ROLLOUT__?: unknown }
  ).__TLDW_RESEARCH_WORKSPACE_ROLLOUT__
  if (!overrideValue || typeof overrideValue !== "object") return null
  return overrideValue as Record<string, unknown>
}

const resolveRolloutSubjectId = (): string => {
  const overridePayload = readRolloutWindowOverrides()
  const overrideSubject =
    overridePayload?.subjectId ?? overridePayload?.subject_id
  if (typeof overrideSubject === "string" && overrideSubject.trim().length > 0) {
    return overrideSubject.trim()
  }

  if (typeof window === "undefined") {
    return "server-rollout-subject"
  }

  const persistedSubjectId = readLocalStorageValue(
    FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY
  )
  if (typeof persistedSubjectId === "string" && persistedSubjectId.trim().length) {
    return persistedSubjectId.trim()
  }

  const createdSubjectId = createRolloutSubjectId()
  writeLocalStorageValue(
    FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY,
    createdSubjectId
  )
  return createdSubjectId
}

const resolveRolloutPercentageForFlag = (
  flag: ResearchWorkspaceRolloutFlag
): number => {
  const config = RESEARCH_WORKSPACE_ROLLOUT_CONFIG[flag]
  const overridePayload = readRolloutWindowOverrides()
  const overrideValue = overridePayload?.[flag]
  const storageValue = readLocalStorageValue(config.storageKey)

  const viteEnv = (
    import.meta as unknown as { env?: Record<string, unknown> }
  ).env
  const viteValue = viteEnv?.[config.viteEnvKey]
  const nextEnvValue = typeof process !== "undefined"
    ? (process as { env?: Record<string, string | undefined> }).env?.[
      config.nextEnvKey
    ]
    : undefined

  return resolveRolloutPercentageFromCandidates(
    [overrideValue, storageValue, viteValue, nextEnvValue],
    RESEARCH_WORKSPACE_ROLLOUT_DEFAULT_PERCENTAGE
  )
}

const isFeatureFlagEnabledByRollout = (flag: FeatureFlagKey): boolean => {
  if (!isResearchWorkspaceRolloutFlag(flag)) return true

  const rolloutPercentage = resolveRolloutPercentageForFlag(flag)
  const subjectId = resolveRolloutSubjectId()
  return isFlagEnabledForRollout({
    flagKey: flag,
    subjectId,
    rolloutPercentage
  })
}

/**
 * Hook to check if a feature flag is enabled.
 * @param flag - The feature flag key
 * @returns [isEnabled, setEnabled] tuple
 */
export function useFeatureFlag(flag: FeatureFlagKey) {
  const [persistedEnabled, setPersistedEnabled] = useStorage(
    flag,
    FEATURE_FLAG_DEFAULTS[flag] ?? true
  )
  const rolloutEnabled = useMemo(() => isFeatureFlagEnabledByRollout(flag), [flag])
  return [Boolean(persistedEnabled) && rolloutEnabled, setPersistedEnabled] as const
}

/**
 * Hook to get all feature flags at once.
 * Useful for settings page or debugging.
 */
export function useAllFeatureFlags() {
  const [newChat, setNewChat] = useStorage(FEATURE_FLAGS.NEW_CHAT, true)
  const [newSettings, setNewSettings] = useStorage(
    FEATURE_FLAGS.NEW_SETTINGS,
    true
  )
  const [commandPalette, setCommandPalette] = useStorage(
    FEATURE_FLAGS.COMMAND_PALETTE,
    true
  )
  const [compactMessages, setCompactMessages] = useStorage(
    FEATURE_FLAGS.COMPACT_MESSAGES,
    true
  )
  const [chatSidebar, setChatSidebar] = useStorage(
    FEATURE_FLAGS.CHAT_SIDEBAR,
    true
  )
  const [compareMode, setCompareMode] = useStorage(
    FEATURE_FLAGS.COMPARE_MODE,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.COMPARE_MODE] ?? true
  )
  const [knowledgeQaStreaming, setKnowledgeQaStreaming] = useStorage(
    FEATURE_FLAGS.KNOWLEDGE_QA_STREAMING,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.KNOWLEDGE_QA_STREAMING] ?? true
  )
  const [knowledgeQaComparison, setKnowledgeQaComparison] = useStorage(
    FEATURE_FLAGS.KNOWLEDGE_QA_COMPARISON,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.KNOWLEDGE_QA_COMPARISON] ?? true
  )
  const [knowledgeQaBranching, setKnowledgeQaBranching] = useStorage(
    FEATURE_FLAGS.KNOWLEDGE_QA_BRANCHING,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.KNOWLEDGE_QA_BRANCHING] ?? true
  )
  const [mediaNavigationPanel, setMediaNavigationPanel] = useStorage(
    FEATURE_FLAGS.MEDIA_NAVIGATION_PANEL,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.MEDIA_NAVIGATION_PANEL] ?? true
  )
  const [mediaRichRendering, setMediaRichRendering] = useStorage(
    FEATURE_FLAGS.MEDIA_RICH_RENDERING,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.MEDIA_RICH_RENDERING] ?? true
  )
  const [mediaAnalysisDisplayModeSelector, setMediaAnalysisDisplayModeSelector] =
    useStorage(
      FEATURE_FLAGS.MEDIA_ANALYSIS_DISPLAY_MODE_SELECTOR,
      FEATURE_FLAG_DEFAULTS[
        FEATURE_FLAGS.MEDIA_ANALYSIS_DISPLAY_MODE_SELECTOR
      ] ?? true
    )
  const [
    mediaNavigationGeneratedFallbackDefault,
    setMediaNavigationGeneratedFallbackDefault
  ] = useStorage(
    FEATURE_FLAGS.MEDIA_NAVIGATION_GENERATED_FALLBACK_DEFAULT,
    FEATURE_FLAG_DEFAULTS[
      FEATURE_FLAGS.MEDIA_NAVIGATION_GENERATED_FALLBACK_DEFAULT
    ] ?? true
  )
  const [researchWorkspaceProvenanceV1, setResearchWorkspaceProvenanceV1] = useStorage(
    FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1,
    FEATURE_FLAG_DEFAULTS[FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1] ?? true
  )
  const [researchWorkspaceStatusGuardrailsV1, setResearchWorkspaceStatusGuardrailsV1] =
    useStorage(
      FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1,
      FEATURE_FLAG_DEFAULTS[
        FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1
      ] ?? true
    )

  return {
    flags: {
      newChat,
      newSettings,
      commandPalette,
      compactMessages,
      chatSidebar,
      compareMode,
      knowledgeQaStreaming,
      knowledgeQaComparison,
      knowledgeQaBranching,
      mediaNavigationPanel,
      mediaRichRendering,
      mediaAnalysisDisplayModeSelector,
      mediaNavigationGeneratedFallbackDefault,
      researchWorkspaceProvenanceV1,
      researchWorkspaceStatusGuardrailsV1
    },
    setters: {
      setNewChat,
      setNewSettings,
      setCommandPalette,
      setCompactMessages,
      setChatSidebar,
      setCompareMode,
      setKnowledgeQaStreaming,
      setKnowledgeQaComparison,
      setKnowledgeQaBranching,
      setMediaNavigationPanel,
      setMediaRichRendering,
      setMediaAnalysisDisplayModeSelector,
      setMediaNavigationGeneratedFallbackDefault,
      setResearchWorkspaceProvenanceV1,
      setResearchWorkspaceStatusGuardrailsV1
    },
    // Enable all new UX features
    enableAll: useCallback(() => {
      setNewChat(true)
      setNewSettings(true)
      setCommandPalette(true)
      setCompactMessages(true)
      setChatSidebar(true)
      setCompareMode(true)
      setKnowledgeQaStreaming(true)
      setKnowledgeQaComparison(true)
      setKnowledgeQaBranching(true)
      setMediaNavigationPanel(true)
      setMediaRichRendering(true)
      setMediaAnalysisDisplayModeSelector(true)
      setMediaNavigationGeneratedFallbackDefault(true)
      setResearchWorkspaceProvenanceV1(true)
      setResearchWorkspaceStatusGuardrailsV1(true)
    }, [
      setNewChat,
      setNewSettings,
      setCommandPalette,
      setCompactMessages,
      setChatSidebar,
      setCompareMode,
      setKnowledgeQaStreaming,
      setKnowledgeQaComparison,
      setKnowledgeQaBranching,
      setMediaNavigationPanel,
      setMediaRichRendering,
      setMediaAnalysisDisplayModeSelector,
      setMediaNavigationGeneratedFallbackDefault,
      setResearchWorkspaceProvenanceV1,
      setResearchWorkspaceStatusGuardrailsV1
    ]),
    // Disable all new UX features (revert to old)
    disableAll: useCallback(() => {
      setNewChat(false)
      setNewSettings(false)
      setCommandPalette(false)
      setCompactMessages(false)
      setChatSidebar(false)
      setCompareMode(false)
      setKnowledgeQaStreaming(false)
      setKnowledgeQaComparison(false)
      setKnowledgeQaBranching(false)
      setMediaNavigationPanel(false)
      setMediaRichRendering(false)
      setMediaAnalysisDisplayModeSelector(false)
      setMediaNavigationGeneratedFallbackDefault(false)
      setResearchWorkspaceProvenanceV1(false)
      setResearchWorkspaceStatusGuardrailsV1(false)
    }, [
      setNewChat,
      setNewSettings,
      setCommandPalette,
      setCompactMessages,
      setChatSidebar,
      setCompareMode,
      setKnowledgeQaStreaming,
      setKnowledgeQaComparison,
      setKnowledgeQaBranching,
      setMediaNavigationPanel,
      setMediaRichRendering,
      setMediaAnalysisDisplayModeSelector,
      setMediaNavigationGeneratedFallbackDefault,
      setResearchWorkspaceProvenanceV1,
      setResearchWorkspaceStatusGuardrailsV1
    ])
  }
}

/**
 * Convenience hooks for specific features
 */
export function useNewChat() {
  return useFeatureFlag(FEATURE_FLAGS.NEW_CHAT)
}

export function useNewSettings() {
  return useFeatureFlag(FEATURE_FLAGS.NEW_SETTINGS)
}

export function useCommandPalette() {
  return useFeatureFlag(FEATURE_FLAGS.COMMAND_PALETTE)
}

export function useCompactMessages() {
  return useFeatureFlag(FEATURE_FLAGS.COMPACT_MESSAGES)
}

export function useChatSidebar() {
  return useFeatureFlag(FEATURE_FLAGS.CHAT_SIDEBAR)
}

export function useMediaNavigationPanel() {
  return useFeatureFlag(FEATURE_FLAGS.MEDIA_NAVIGATION_PANEL)
}

export function useKnowledgeQaStreaming() {
  return useFeatureFlag(FEATURE_FLAGS.KNOWLEDGE_QA_STREAMING)
}

export function useKnowledgeQaComparison() {
  return useFeatureFlag(FEATURE_FLAGS.KNOWLEDGE_QA_COMPARISON)
}

export function useKnowledgeQaBranching() {
  return useFeatureFlag(FEATURE_FLAGS.KNOWLEDGE_QA_BRANCHING)
}

export function useMediaRichRendering() {
  return useFeatureFlag(FEATURE_FLAGS.MEDIA_RICH_RENDERING)
}

export function useMediaAnalysisDisplayModeSelector() {
  return useFeatureFlag(FEATURE_FLAGS.MEDIA_ANALYSIS_DISPLAY_MODE_SELECTOR)
}

export function useMediaNavigationGeneratedFallbackDefault() {
  return useFeatureFlag(FEATURE_FLAGS.MEDIA_NAVIGATION_GENERATED_FALLBACK_DEFAULT)
}

export function useResearchWorkspaceProvenance() {
  return useFeatureFlag(FEATURE_FLAGS.RESEARCH_WORKSPACE_PROVENANCE_V1)
}

export function useResearchWorkspaceStatusGuardrails() {
  return useFeatureFlag(FEATURE_FLAGS.RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1)
}
