import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react"
import type {
  WizardStep,
  WizardQueueItem,
  IngestPreset,
  PresetConfig,
  WizardProcessingState,
  WizardResultItem,
  ItemProgress,
  ConferenceBatchMetadata,
} from "./types"
import {
  DEFAULT_PRESETS,
  DEFAULT_PRESET,
  mergePresetConfig,
  configMatchesPreset,
  type PresetMap,
} from "./presets"
import type {
  FirstSourceQuickIngestKind,
  QuickIngestOpenDetail,
} from "@/utils/quick-ingest-open"

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export type IngestWizardState = {
  currentStep: WizardStep
  /** Highest step the user has reached (for backward navigation guard). */
  highestStep: WizardStep
  queueItems: WizardQueueItem[]
  selectedPreset: IngestPreset
  customBasePreset: Exclude<IngestPreset, "custom">
  presetConfig: PresetConfig
  customOptions: Partial<PresetConfig>
  playlistPreflightSeed: QuickIngestOpenDetail | null
  firstSourceAddMode: FirstSourceQuickIngestKind | null
  conferenceBatchMetadata: ConferenceBatchMetadata | null
  processingState: WizardProcessingState
  results: WizardResultItem[]
  isMinimized: boolean
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type Action =
  | { type: "GO_TO_STEP"; step: WizardStep }
  | { type: "GO_NEXT" }
  | { type: "GO_BACK" }
  | { type: "SET_QUEUE_ITEMS"; items: WizardQueueItem[] }
  | { type: "SET_PRESET"; preset: IngestPreset }
  | { type: "SET_CUSTOM_OPTIONS"; options: Partial<PresetConfig> }
  | { type: "SET_PLAYLIST_PREFLIGHT_SEED"; seed: QuickIngestOpenDetail | null }
  | { type: "SET_CONFERENCE_BATCH_METADATA"; metadata: ConferenceBatchMetadata | null }
  | { type: "START_PROCESSING" }
  | { type: "CANCEL_PROCESSING" }
  | { type: "CANCEL_ITEM"; id: string }
  | { type: "UPDATE_ITEM_PROGRESS"; progress: ItemProgress }
  | { type: "UPDATE_PROCESSING_STATE"; state: Partial<WizardProcessingState> }
  | { type: "SET_RESULTS"; results: WizardResultItem[] }
  | { type: "SKIP_TO_PROCESSING" }
  | { type: "MINIMIZE" }
  | { type: "RESTORE" }
  | { type: "RESET" }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mergeCustomOptions = (
  current: Partial<PresetConfig>,
  incoming: Partial<PresetConfig>
): Partial<PresetConfig> => {
  const next: Partial<PresetConfig> = { ...current }

  if (incoming.common) {
    next.common = {
      ...(current.common ?? {}),
      ...incoming.common,
    }
  }

  if (incoming.typeDefaults) {
    next.typeDefaults = {
      audio: {
        ...(current.typeDefaults?.audio ?? {}),
        ...(incoming.typeDefaults.audio ?? {}),
      },
      document: {
        ...(current.typeDefaults?.document ?? {}),
        ...(incoming.typeDefaults.document ?? {}),
      },
      video: {
        ...(current.typeDefaults?.video ?? {}),
        ...(incoming.typeDefaults.video ?? {}),
      },
    }
  }

  if (incoming.advancedValues) {
    next.advancedValues = {
      ...(current.advancedValues ?? {}),
      ...incoming.advancedValues,
    }
  }

  if (Object.prototype.hasOwnProperty.call(incoming, "storeRemote")) {
    next.storeRemote = incoming.storeRemote
  }

  if (Object.prototype.hasOwnProperty.call(incoming, "reviewBeforeStorage")) {
    next.reviewBeforeStorage = incoming.reviewBeforeStorage
  }

  return next
}

const mergeCurrentPresetConfig = (
  current: PresetConfig,
  incoming: Partial<PresetConfig>
): PresetConfig => {
  const next = mergePresetConfig(current, incoming)

  if (incoming.advancedValues) {
    const advancedValues = { ...(current.advancedValues ?? {}) }
    for (const [key, value] of Object.entries(incoming.advancedValues)) {
      if (value === undefined) {
        delete advancedValues[key]
      } else {
        advancedValues[key] = value
      }
    }
    next.advancedValues = advancedValues
  }

  return next
}

const buildInitialProgress = (items: WizardQueueItem[]): ItemProgress[] =>
  items
    .filter(
      (item) =>
        item.validation.valid && item.conferenceOverride?.selected !== false
    )
    .map((item) => ({
      id: item.id,
      status: "queued" as const,
      progressPercent: 0,
      currentStage: "",
      estimatedRemaining: 0,
    }))

const findMatchingPreset = (
  config: PresetConfig,
  presetMap: PresetMap
): Exclude<IngestPreset, "custom"> | null => {
  for (const preset of ["quick", "standard", "deep"] as const) {
    if (configMatchesPreset(config, preset, presetMap)) {
      return preset
    }
  }
  return null
}

const INITIAL_PROCESSING_STATE: WizardProcessingState = {
  status: "idle",
  perItemProgress: [],
  elapsed: 0,
  estimatedRemaining: 0,
}

const createInitialState = (
  presetMap: PresetMap = DEFAULT_PRESETS
): IngestWizardState => ({
  currentStep: 1,
  highestStep: 1,
  queueItems: [],
  selectedPreset: DEFAULT_PRESET,
  customBasePreset: DEFAULT_PRESET,
  presetConfig: presetMap[DEFAULT_PRESET],
  customOptions: {},
  playlistPreflightSeed: null,
  firstSourceAddMode: null,
  conferenceBatchMetadata: null,
  processingState: { ...INITIAL_PROCESSING_STATE },
  results: [],
  isMinimized: false,
})

const createInitialStateFromSeed = (
  seed: Partial<IngestWizardState> | undefined,
  presetMap: PresetMap
): IngestWizardState => {
  const base = createInitialState(presetMap)
  if (!seed) return base

  const selectedPreset = seed.selectedPreset ?? base.selectedPreset
  const presetConfig =
    seed.presetConfig ??
    (selectedPreset === "custom"
      ? base.presetConfig
      : presetMap[selectedPreset])

  return {
    ...base,
    ...seed,
    queueItems: seed.queueItems ?? base.queueItems,
    selectedPreset,
    customBasePreset: seed.customBasePreset ?? base.customBasePreset,
    presetConfig,
    customOptions: seed.customOptions ?? base.customOptions,
    playlistPreflightSeed:
      seed.playlistPreflightSeed ?? base.playlistPreflightSeed,
    firstSourceAddMode: seed.firstSourceAddMode ?? base.firstSourceAddMode,
    conferenceBatchMetadata:
      seed.conferenceBatchMetadata ?? base.conferenceBatchMetadata,
    processingState: seed.processingState
      ? {
          ...INITIAL_PROCESSING_STATE,
          ...seed.processingState,
          perItemProgress:
            seed.processingState.perItemProgress ??
            INITIAL_PROCESSING_STATE.perItemProgress,
        }
      : { ...INITIAL_PROCESSING_STATE },
    results: seed.results ?? base.results,
    isMinimized: seed.isMinimized ?? base.isMinimized,
  }
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

const clampStep = (step: number): WizardStep =>
  Math.max(1, Math.min(5, step)) as WizardStep

const reducer = (
  state: IngestWizardState,
  action: Action,
  presetMap: PresetMap
): IngestWizardState => {
  switch (action.type) {
    case "GO_TO_STEP": {
      // Can only go backward (to a step <= highestStep)
      const target = clampStep(action.step)
      if (target > state.highestStep) return state
      return { ...state, currentStep: target }
    }

    case "GO_NEXT": {
      const next = clampStep(state.currentStep + 1)
      if (next === state.currentStep) return state // already at max
      const newHighest = Math.max(state.highestStep, next) as WizardStep
      return { ...state, currentStep: next, highestStep: newHighest }
    }

    case "GO_BACK": {
      const prev = clampStep(state.currentStep - 1)
      if (prev === state.currentStep) return state // already at min
      return { ...state, currentStep: prev }
    }

    case "SET_QUEUE_ITEMS":
      return { ...state, queueItems: action.items }

    case "SET_PRESET": {
      if (action.preset === "custom") {
        return {
          ...state,
          selectedPreset: "custom",
          customBasePreset:
            state.selectedPreset === "custom"
              ? state.customBasePreset
              : state.selectedPreset,
        }
      }

      return {
        ...state,
        selectedPreset: action.preset,
        customBasePreset: action.preset,
        customOptions: {},
        presetConfig: presetMap[action.preset],
      }
    }

    case "SET_CUSTOM_OPTIONS": {
      const customOptions = mergeCustomOptions(state.customOptions, action.options)
      const basePreset =
        state.selectedPreset === "custom"
          ? state.customBasePreset
          : state.selectedPreset
      const presetConfig = mergeCurrentPresetConfig(
        state.presetConfig,
        action.options
      )
      const matchedPreset = findMatchingPreset(presetConfig, presetMap)

      if (matchedPreset) {
        return {
          ...state,
          selectedPreset: matchedPreset,
          customBasePreset: matchedPreset,
          customOptions: {},
          presetConfig: presetMap[matchedPreset],
        }
      }

      return {
        ...state,
        selectedPreset: "custom",
        customBasePreset: basePreset,
        customOptions,
        presetConfig,
      }
    }

    case "SET_CONFERENCE_BATCH_METADATA":
      return { ...state, conferenceBatchMetadata: action.metadata }

    case "SET_PLAYLIST_PREFLIGHT_SEED":
      return { ...state, playlistPreflightSeed: action.seed }

    case "START_PROCESSING": {
      const perItemProgress = buildInitialProgress(state.queueItems)
      if (perItemProgress.length === 0) return state
      return {
        ...state,
        processingState: {
          status: "running",
          perItemProgress,
          elapsed: 0,
          estimatedRemaining: 0,
        },
        results: [],
      }
    }

    case "CANCEL_PROCESSING":
      return {
        ...state,
        processingState: {
          ...state.processingState,
          status: "cancelled",
          perItemProgress: state.processingState.perItemProgress.map((p) =>
            p.status === "queued" || p.status === "uploading" || p.status === "processing" || p.status === "analyzing" || p.status === "storing"
              ? { ...p, status: "cancelled" as const }
              : p
          ),
        },
      }

    case "CANCEL_ITEM":
      return {
        ...state,
        processingState: {
          ...state.processingState,
          perItemProgress: state.processingState.perItemProgress.map((p) =>
            p.id === action.id &&
            p.status !== "complete" &&
            p.status !== "failed" &&
            p.status !== "cancelled"
              ? { ...p, status: "cancelled" as const }
              : p
          ),
        },
      }

    case "UPDATE_ITEM_PROGRESS":
      const existingProgressItem = state.processingState.perItemProgress.find(
        (p) => p.id === action.progress.id
      )
      if (!existingProgressItem) return state
      if (
        existingProgressItem.status === action.progress.status &&
        existingProgressItem.progressPercent === action.progress.progressPercent &&
        existingProgressItem.currentStage === action.progress.currentStage &&
        existingProgressItem.estimatedRemaining === action.progress.estimatedRemaining &&
        existingProgressItem.error === action.progress.error
      ) {
        return state
      }
      return {
        ...state,
        processingState: {
          ...state.processingState,
          perItemProgress: state.processingState.perItemProgress.map((p) =>
            p.id === action.progress.id ? action.progress : p
          ),
        },
      }

    case "UPDATE_PROCESSING_STATE":
      if (
        Object.entries(action.state).every(
          ([key, value]) =>
            state.processingState[key as keyof WizardProcessingState] === value
        )
      ) {
        return state
      }
      return {
        ...state,
        processingState: { ...state.processingState, ...action.state },
      }

    case "SET_RESULTS":
      if (state.results === action.results) return state
      return { ...state, results: action.results }

    case "SKIP_TO_PROCESSING": {
      // Quick Mode: skip Steps 2-3, jump directly to Step 4 with default preset
      const perItemProgress = buildInitialProgress(state.queueItems)
      if (perItemProgress.length === 0) return state
      return {
        ...state,
        currentStep: 4 as WizardStep,
        highestStep: 4 as WizardStep,
        processingState: {
          status: "running",
          perItemProgress,
          elapsed: 0,
          estimatedRemaining: 0,
        },
        results: [],
      }
    }

    case "MINIMIZE":
      return { ...state, isMinimized: true }

    case "RESTORE":
      return { ...state, isMinimized: false }

    case "RESET":
      return createInitialState(presetMap)

    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// Context value type
// ---------------------------------------------------------------------------

type IngestWizardContextValue = {
  state: IngestWizardState
  // Navigation
  goToStep: (step: WizardStep) => void
  goNext: () => void
  goBack: () => void
  // Queue
  setQueueItems: (items: WizardQueueItem[]) => void
  // Presets & options
  setPreset: (preset: IngestPreset) => void
  setCustomOptions: (options: Partial<PresetConfig>) => void
  setPlaylistPreflightSeed: (seed: QuickIngestOpenDetail | null) => void
  setConferenceBatchMetadata: (metadata: ConferenceBatchMetadata | null) => void
  // Processing
  startProcessing: () => void
  skipToProcessing: () => void
  cancelProcessing: () => void
  cancelItem: (id: string) => void
  updateItemProgress: (progress: ItemProgress) => void
  updateProcessingState: (state: Partial<WizardProcessingState>) => void
  setResults: (results: WizardResultItem[]) => void
  // Minimize / restore
  minimize: () => void
  restore: () => void
  // Reset
  reset: () => void
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const IngestWizardContext = createContext<IngestWizardContextValue | null>(null)

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

type IngestWizardProviderProps = {
  children: React.ReactNode
  initialState?: Partial<IngestWizardState>
  onStateChange?: (state: IngestWizardState) => void
  presetMap?: PresetMap
}

export const IngestWizardProvider: React.FC<IngestWizardProviderProps> = ({
  children,
  initialState,
  onStateChange,
  presetMap = DEFAULT_PRESETS,
}) => {
  const reducerWithPresetMap = useCallback(
    (state: IngestWizardState, action: Action) =>
      reducer(state, action, presetMap),
    [presetMap]
  )
  const [state, dispatch] = useReducer(
    reducerWithPresetMap,
    initialState,
    (seed) => createInitialStateFromSeed(seed, presetMap)
  )

  useEffect(() => {
    onStateChange?.(state)
  }, [onStateChange, state])

  const goToStep = useCallback(
    (step: WizardStep) => dispatch({ type: "GO_TO_STEP", step }),
    []
  )
  const goNext = useCallback(() => dispatch({ type: "GO_NEXT" }), [])
  const goBack = useCallback(() => dispatch({ type: "GO_BACK" }), [])
  const setQueueItems = useCallback(
    (items: WizardQueueItem[]) => dispatch({ type: "SET_QUEUE_ITEMS", items }),
    []
  )
  const setPreset = useCallback(
    (preset: IngestPreset) => dispatch({ type: "SET_PRESET", preset }),
    []
  )
  const setCustomOptions = useCallback(
    (options: Partial<PresetConfig>) =>
      dispatch({ type: "SET_CUSTOM_OPTIONS", options }),
    []
  )
  const setPlaylistPreflightSeed = useCallback(
    (seed: QuickIngestOpenDetail | null) =>
      dispatch({ type: "SET_PLAYLIST_PREFLIGHT_SEED", seed }),
    []
  )
  const setConferenceBatchMetadata = useCallback(
    (metadata: ConferenceBatchMetadata | null) =>
      dispatch({ type: "SET_CONFERENCE_BATCH_METADATA", metadata }),
    []
  )
  const startProcessing = useCallback(
    () => dispatch({ type: "START_PROCESSING" }),
    []
  )
  const skipToProcessing = useCallback(
    () => dispatch({ type: "SKIP_TO_PROCESSING" }),
    []
  )
  const cancelProcessing = useCallback(
    () => dispatch({ type: "CANCEL_PROCESSING" }),
    []
  )
  const cancelItem = useCallback(
    (id: string) => dispatch({ type: "CANCEL_ITEM", id }),
    []
  )
  const updateItemProgress = useCallback(
    (progress: ItemProgress) =>
      dispatch({ type: "UPDATE_ITEM_PROGRESS", progress }),
    []
  )
  const updateProcessingState = useCallback(
    (ps: Partial<WizardProcessingState>) =>
      dispatch({ type: "UPDATE_PROCESSING_STATE", state: ps }),
    []
  )
  const setResults = useCallback(
    (results: WizardResultItem[]) => dispatch({ type: "SET_RESULTS", results }),
    []
  )
  const minimize = useCallback(() => dispatch({ type: "MINIMIZE" }), [])
  const restore = useCallback(() => dispatch({ type: "RESTORE" }), [])
  const reset = useCallback(() => dispatch({ type: "RESET" }), [])

  const value = useMemo<IngestWizardContextValue>(
    () => ({
      state,
      goToStep,
      goNext,
      goBack,
      setQueueItems,
      setPreset,
      setCustomOptions,
      setPlaylistPreflightSeed,
      setConferenceBatchMetadata,
      startProcessing,
      skipToProcessing,
      cancelProcessing,
      cancelItem,
      updateItemProgress,
      updateProcessingState,
      setResults,
      minimize,
      restore,
      reset,
    }),
    [
      state,
      goToStep,
      goNext,
      goBack,
      setQueueItems,
      setPreset,
      setCustomOptions,
      setPlaylistPreflightSeed,
      setConferenceBatchMetadata,
      startProcessing,
      skipToProcessing,
      cancelProcessing,
      cancelItem,
      updateItemProgress,
      updateProcessingState,
      setResults,
      minimize,
      restore,
      reset,
    ]
  )

  return (
    <IngestWizardContext.Provider value={value}>
      {children}
    </IngestWizardContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Access the ingest wizard context. Must be used within an IngestWizardProvider.
 */
export const useIngestWizard = (): IngestWizardContextValue => {
  const ctx = useContext(IngestWizardContext)
  if (!ctx) {
    throw new Error("useIngestWizard must be used within an IngestWizardProvider")
  }
  return ctx
}

export default IngestWizardContext
