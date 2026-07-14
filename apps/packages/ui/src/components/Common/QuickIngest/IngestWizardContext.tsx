import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react"
import {
  playlistHasMaterializationCues,
  type WizardStep,
  type WizardQueueItem,
  type IngestPreset,
  type PresetConfig,
  type WizardProcessingState,
  type WizardResultItem,
  type ItemProgress,
  type ConferenceBatchMetadata,
  type WizardDuplicatePolicy,
} from "./types"
import type {
  PlaylistIngestRunCreateRequest,
  PlaylistMetadataPatch,
  PlaylistReviewRequiredRecoveryItem,
  PlaylistReviewOverride,
  PlaylistRunInput,
} from "@/services/tldw/playlist-ingest"
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
  pendingRunRequest: PlaylistIngestRunCreateRequest | null
  processingBlock: WizardProcessingBlock | null
  isMinimized: boolean
}

export type WizardProcessingBlock = {
  code: "materialization_expired" | "review_required" | "invalid_run_request"
  occurrenceIds: string[]
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

type Action =
  | { type: "GO_TO_STEP"; step: WizardStep }
  | { type: "GO_NEXT" }
  | { type: "GO_BACK" }
  | { type: "SET_QUEUE_ITEMS"; items: WizardQueueItem[] }
  | {
      type: "UPDATE_QUEUE_ITEMS"
      updater: (items: WizardQueueItem[]) => WizardQueueItem[]
    }
  | { type: "SET_PRESET"; preset: IngestPreset }
  | { type: "SET_CUSTOM_OPTIONS"; options: Partial<PresetConfig> }
  | { type: "SET_PLAYLIST_PREFLIGHT_SEED"; seed: QuickIngestOpenDetail | null }
  | {
      type: "SET_CONFERENCE_BATCH_METADATA"
      metadata: ConferenceBatchMetadata | null
    }
  | { type: "START_PROCESSING" }
  | { type: "CANCEL_PROCESSING" }
  | { type: "CANCEL_ITEM"; id: string }
  | { type: "REQUEST_ITEM_CANCELLATION"; id: string }
  | { type: "UPDATE_ITEM_PROGRESS"; progress: ItemProgress }
  | { type: "UPDATE_PROCESSING_STATE"; state: Partial<WizardProcessingState> }
  | { type: "SET_RESULTS"; results: WizardResultItem[] }
  | {
      type: "APPLY_PLAYLIST_REVIEW_REQUIRED"
      items: PlaylistReviewRequiredRecoveryItem[]
    }
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
        item.validation.valid &&
        item.conferenceOverride?.selected !== false &&
        item.playlistReview?.selected !== false
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

export const playlistItemIsCurrentDuplicate = (item: WizardQueueItem): boolean =>
  item.playlist?.duplicateStatus === "duplicate_existing" ||
  item.playlist?.duplicateStatus === "duplicate_in_batch" ||
  item.playlistReview?.duplicateEvidence?.kind === "library" ||
  item.playlistReview?.duplicateEvidence?.kind === "in_run"

const ALL_DUPLICATE_POLICIES: readonly WizardDuplicatePolicy[] = [
  "skip",
  "include_existing",
  "update_metadata_only",
  "overwrite",
]
const INITIAL_IN_RUN_DUPLICATE_POLICIES: readonly WizardDuplicatePolicy[] = [
  "skip",
  "overwrite",
]

export const getPlaylistAllowedDuplicatePolicies = (
  item: WizardQueueItem
): readonly WizardDuplicatePolicy[] => {
  if (item.playlistReview?.allowedDuplicatePolicies) {
    return item.playlistReview.allowedDuplicatePolicies
  }
  const isInRunDuplicate =
    item.playlist?.duplicateStatus === "duplicate_in_batch" ||
    item.playlistReview?.duplicateEvidence?.kind === "in_run"
  return isInRunDuplicate ? INITIAL_IN_RUN_DUPLICATE_POLICIES : ALL_DUPLICATE_POLICIES
}

const MAX_PLAYLIST_METADATA_TEXT_LENGTH = 500
const MAX_PLAYLIST_KEYWORDS = 100
const MAX_PLAYLIST_KEYWORD_LENGTH = 128
const MAX_PLAYLIST_RUN_IDENTITY_LENGTH = 255
const MAX_PLAYLIST_RUN_URL_LENGTH = 8192
const MAX_PLAYLIST_RUN_DISPLAY_TEXT_LENGTH = 2000
const MAX_PLAYLIST_RUN_FILE_SIZE = 10 * 1024 ** 4
export const MAX_PLAYLIST_RUN_INPUTS = 500

type PlaylistMetadataPatchBuild = {
  patch: PlaylistMetadataPatch | undefined
  invalid: boolean
}

const buildExplicitMetadataPatch = (item: WizardQueueItem): PlaylistMetadataPatchBuild => {
  const review = item.playlistReview
  if (!review?.metadataPatch || !review.editedFields?.length) {
    return { patch: undefined, invalid: false }
  }
  const edited = new Set(review.editedFields)
  const patch: PlaylistMetadataPatch = {}
  let invalid = false
  if (edited.has("title")) {
    const title = review.metadataPatch.title?.trim() ?? ""
    if (!title || title.length > MAX_PLAYLIST_METADATA_TEXT_LENGTH) invalid = true
    else patch.title = title
  }
  if (edited.has("author")) {
    const author = review.metadataPatch.author?.trim() ?? ""
    if (!author || author.length > MAX_PLAYLIST_METADATA_TEXT_LENGTH) invalid = true
    else patch.author = author
  }
  if (edited.has("keywordsAdd")) {
    const rawKeywords = review.metadataPatch.keywordsAdd ?? []
    const keywords: string[] = []
    const seenKeywords = new Set<string>()
    if (rawKeywords.length === 0 || rawKeywords.length > MAX_PLAYLIST_KEYWORDS) {
      invalid = true
    }
    for (const rawKeyword of rawKeywords) {
      const keyword = rawKeyword.trim()
      if (!keyword || keyword.length > MAX_PLAYLIST_KEYWORD_LENGTH) {
        invalid = true
        continue
      }
      const dedupeKey = keyword.toLocaleLowerCase()
      if (seenKeywords.has(dedupeKey)) continue
      seenKeywords.add(dedupeKey)
      keywords.push(keyword)
    }
    if (keywords.length > MAX_PLAYLIST_KEYWORDS) invalid = true
    if (keywords.length > 0) patch.keywordsAdd = keywords
  }
  return {
    patch: Object.keys(patch).length > 0 ? patch : undefined,
    invalid,
  }
}

export const playlistItemHasValidExplicitMetadataPatch = (item: WizardQueueItem): boolean => {
  const patchBuild = buildExplicitMetadataPatch(item)
  return !patchBuild.invalid && Boolean(patchBuild.patch)
}

const getDirectUrlDisplayTitle = (item: WizardQueueItem): string | null => {
  const title =
    item.sourceRef?.kind === "direct_url"
      ? item.playlist?.title ?? item.fileName ?? null
      : item.playlist?.title ?? null
  const normalized = title?.trim() ?? ""
  return normalized || null
}

const queueItemToRunInput = (item: WizardQueueItem): PlaylistRunInput => {
  const sourceRef = item.sourceRef
  if (sourceRef?.kind === "materialized_playlist_item") {
    return {
      inputKind: "materialized_playlist_item",
      occurrenceId: sourceRef.occurrenceId,
      materializationId: sourceRef.materializationId,
    }
  }
  if (sourceRef?.kind === "direct_url") {
    return {
      inputKind: "direct_url",
      occurrenceId: sourceRef.occurrenceId,
      url: sourceRef.url,
      displayMetadata: { title: getDirectUrlDisplayTitle(item) },
    }
  }
  if (sourceRef?.kind === "file_stub") {
    return {
      inputKind: "file_stub",
      occurrenceId: sourceRef.occurrenceId,
      name: item.fileName || item.file?.name || item.id,
      contentType: item.mimeType || item.file?.type || undefined,
      sizeBytes: item.file?.size ?? item.fileSize,
      displayMetadata: { title: item.fileName || item.file?.name || item.id },
    }
  }
  if (item.url) {
    return {
      inputKind: "direct_url",
      occurrenceId: item.id,
      url: item.url,
      displayMetadata: { title: getDirectUrlDisplayTitle(item) },
    }
  }
  return {
    inputKind: "file_stub",
    occurrenceId: item.id,
    name: item.fileName || item.file?.name || item.id,
    contentType: item.mimeType || item.file?.type || undefined,
    sizeBytes: item.file?.size ?? item.fileSize,
    displayMetadata: { title: item.fileName || item.file?.name || item.id },
  }
}

const isCanonicalRunIdentity = (value: string): boolean =>
  Boolean(value.trim()) &&
  value.trim() === value &&
  value.length <= MAX_PLAYLIST_RUN_IDENTITY_LENGTH

const queueItemHasValidRunInput = (item: WizardQueueItem): boolean => {
  if (!isCanonicalRunIdentity(item.id)) return false
  const sourceRef = item.sourceRef
  if (
    sourceRef &&
    (sourceRef.occurrenceId !== item.id || !isCanonicalRunIdentity(sourceRef.occurrenceId))
  ) {
    return false
  }
  if (sourceRef?.kind === "materialized_playlist_item") {
    return isCanonicalRunIdentity(sourceRef.materializationId)
  }
  const url = sourceRef?.kind === "direct_url" ? sourceRef.url : sourceRef ? undefined : item.url
  if (url !== undefined) {
    if (!url.trim() || url.length > MAX_PLAYLIST_RUN_URL_LENGTH) return false
    const displayTitle = getDirectUrlDisplayTitle(item)
    return displayTitle === null || displayTitle.length <= MAX_PLAYLIST_RUN_DISPLAY_TEXT_LENGTH
  }
  const name = item.fileName || item.file?.name || item.id
  const contentType = item.mimeType || item.file?.type
  const sizeBytes = item.file?.size ?? item.fileSize
  return (
    Boolean(name.trim()) &&
    name.length <= MAX_PLAYLIST_RUN_IDENTITY_LENGTH &&
    (contentType === undefined || contentType.length <= MAX_PLAYLIST_RUN_IDENTITY_LENGTH) &&
    Number.isSafeInteger(sizeBytes) &&
    sizeBytes >= 0 &&
    sizeBytes <= MAX_PLAYLIST_RUN_FILE_SIZE
  )
}

export const mergePlaylistReviewRequired = (
  queueItems: WizardQueueItem[],
  recoveryItems: PlaylistReviewRequiredRecoveryItem[]
): WizardQueueItem[] => {
  const recoveryByOccurrence = new Map(
    recoveryItems.map((item) => [item.occurrenceId, item] as const)
  )
  return queueItems.map((item) => {
    const occurrenceId = item.sourceRef?.occurrenceId || item.id
    const recovery = recoveryByOccurrence.get(occurrenceId)
    if (!recovery) return item
    const currentPolicy = item.playlistReview?.duplicatePolicy
    const duplicatePolicy =
      recovery.reason !== "duplicate_target_changed" &&
      currentPolicy &&
      recovery.allowedActions.includes(currentPolicy)
        ? currentPolicy
        : undefined
    const duplicateStatus =
      recovery.evidence.kind === "library"
        ? "duplicate_existing"
        : recovery.evidence.kind === "in_run"
          ? "duplicate_in_batch"
          : "new"
    return {
      ...item,
      playlist: {
        ...(item.playlist || {}),
        duplicateStatus,
      },
      playlistReview: {
        selected: item.playlistReview?.selected ?? true,
        ...(item.playlistReview || {}),
        duplicatePolicy,
        duplicateEvidence: { ...recovery.evidence },
        allowedDuplicatePolicies: [...recovery.allowedActions],
        reviewReason: recovery.reason,
      },
    }
  })
}

export const buildPlaylistIngestRunRequest = (
  items: WizardQueueItem[],
  nowMs = Date.now()
): {
  request: PlaylistIngestRunCreateRequest | null
  block: WizardProcessingBlock | null
} => {
  const selectedItems = items.filter(
    (item) =>
      item.validation.valid &&
      item.conferenceOverride?.selected !== false &&
      item.playlistReview?.selected !== false
  )
  if (selectedItems.length === 0) {
    return {
      request: null,
      block: { code: "invalid_run_request", occurrenceIds: [] },
    }
  }
  if (selectedItems.length > MAX_PLAYLIST_RUN_INPUTS) {
    return {
      request: null,
      block: {
        code: "invalid_run_request",
        occurrenceIds: selectedItems
          .slice(MAX_PLAYLIST_RUN_INPUTS)
          .map((item) => item.sourceRef?.occurrenceId ?? item.id),
      },
    }
  }

  const invalidAuthorityOccurrenceIds = selectedItems.flatMap((item) => {
    const expectsMaterializedAuthority =
      item.sourceRef?.kind === "materialized_playlist_item" ||
      playlistHasMaterializationCues(item.playlist)
    if (expectsMaterializedAuthority && item.sourceRef?.kind !== "materialized_playlist_item") {
      return [item.id]
    }
    return queueItemHasValidRunInput(item) ? [] : [item.id]
  })
  const occurrenceIdCounts = new Map<string, number>()
  for (const item of selectedItems) {
    const occurrenceId = item.sourceRef?.occurrenceId ?? item.id
    occurrenceIdCounts.set(occurrenceId, (occurrenceIdCounts.get(occurrenceId) ?? 0) + 1)
  }
  const duplicateOccurrenceIds = [...occurrenceIdCounts.entries()].flatMap(
    ([occurrenceId, count]) => (count > 1 ? [occurrenceId] : [])
  )
  const invalidOccurrenceIds = Array.from(
    new Set([...invalidAuthorityOccurrenceIds, ...duplicateOccurrenceIds])
  )
  if (invalidOccurrenceIds.length > 0) {
    return {
      request: null,
      block: {
        code: "invalid_run_request",
        occurrenceIds: invalidOccurrenceIds,
      },
    }
  }

  const expiredOccurrenceIds = selectedItems.flatMap((item) => {
    if (item.sourceRef?.kind !== "materialized_playlist_item") return []
    const expiresAt = Date.parse(item.playlist?.materializationExpiresAt ?? "")
    return !Number.isFinite(expiresAt) || expiresAt <= nowMs ? [item.id] : []
  })
  if (expiredOccurrenceIds.length > 0) {
    return {
      request: null,
      block: {
        code: "materialization_expired",
        occurrenceIds: expiredOccurrenceIds,
      },
    }
  }

  const reviewRequiredIds = selectedItems.flatMap((item) => {
    if (!playlistItemIsCurrentDuplicate(item)) return []
    const patchBuild = buildExplicitMetadataPatch(item)
    const patch = patchBuild.patch
    const policy = item.playlistReview?.duplicatePolicy
    if (patchBuild.invalid) return [item.id]
    if (playlistItemIsCurrentDuplicate(item) && !policy) return [item.id]
    if (policy && !getPlaylistAllowedDuplicatePolicies(item).includes(policy)) return [item.id]
    if (patch && !policy) return [item.id]
    if (policy === "update_metadata_only" && !patch) return [item.id]
    if (patch && (policy === "skip" || policy === "include_existing")) {
      return [item.id]
    }
    return []
  })
  if (reviewRequiredIds.length > 0) {
    return {
      request: null,
      block: { code: "review_required", occurrenceIds: reviewRequiredIds },
    }
  }

  const reviewOverrides: Record<string, PlaylistReviewOverride> = {}
  for (const item of selectedItems) {
    if (!playlistItemIsCurrentDuplicate(item)) continue
    const policy = item.playlistReview?.duplicatePolicy
    if (!policy) continue
    const patch = buildExplicitMetadataPatch(item).patch
    const evidence = item.playlistReview?.duplicateEvidence
    reviewOverrides[item.id] = {
      duplicatePolicy: policy,
      ...(patch ? { metadataPatch: patch } : {}),
      ...(evidence?.existingMediaId ? { existingMediaId: evidence.existingMediaId } : {}),
      ...(evidence?.duplicateOfOccurrenceId
        ? { duplicateOfOccurrenceId: evidence.duplicateOfOccurrenceId }
        : {}),
    }
  }

  return {
    request: {
      inputs: selectedItems.map(queueItemToRunInput),
      ...(Object.keys(reviewOverrides).length > 0 ? { reviewOverrides } : {}),
    },
    block: null,
  }
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
  pendingRunRequest: null,
  processingBlock: null,
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
    pendingRunRequest: seed.pendingRunRequest ?? base.pendingRunRequest,
    processingBlock: seed.processingBlock ?? base.processingBlock,
    isMinimized: seed.isMinimized ?? base.isMinimized,
  }
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

const clampStep = (step: number): WizardStep =>
  Math.max(1, Math.min(5, step)) as WizardStep

export const applyPlaylistReviewRequiredState = (
  state: IngestWizardState,
  items: PlaylistReviewRequiredRecoveryItem[]
): IngestWizardState => {
  const queueItems = mergePlaylistReviewRequired(state.queueItems, items)
  const { block } = buildPlaylistIngestRunRequest(queueItems)
  return {
    ...state,
    currentStep: 3,
    highestStep: Math.max(state.highestStep, 3) as WizardStep,
    queueItems,
    pendingRunRequest: null,
    processingBlock: block,
    processingState: { ...INITIAL_PROCESSING_STATE },
    results: [],
  }
}

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
      return {
        ...state,
        queueItems: action.items,
        pendingRunRequest: null,
        processingBlock: null,
      }

    case "UPDATE_QUEUE_ITEMS":
      return {
        ...state,
        queueItems: action.updater(state.queueItems),
        pendingRunRequest: null,
        processingBlock: null,
      }

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
      const { request, block } = buildPlaylistIngestRunRequest(state.queueItems)
      if (!request) {
        return {
          ...state,
          currentStep: 3,
          highestStep: Math.max(state.highestStep, 3) as WizardStep,
          pendingRunRequest: null,
          processingBlock: block,
        }
      }
      const perItemProgress = buildInitialProgress(state.queueItems)
      if (perItemProgress.length === 0) return state
      return {
        ...state,
        pendingRunRequest: request,
        processingBlock: null,
        currentStep: 4 as WizardStep,
        highestStep: Math.max(state.highestStep, 4) as WizardStep,
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

    case "CANCEL_ITEM": {
      const remainingQueueItems = state.queueItems.filter(
        (item) => item.id !== action.id
      )
      const remainingRunInputs = state.pendingRunRequest?.inputs.filter(
        (input) => input.occurrenceId !== action.id
      )
      const remainingReviewOverrides = state.pendingRunRequest?.reviewOverrides
        ? Object.fromEntries(
            Object.entries(state.pendingRunRequest.reviewOverrides).filter(
              ([occurrenceId]) => occurrenceId !== action.id
            )
          )
        : undefined
      return {
        ...state,
        queueItems: remainingQueueItems,
        pendingRunRequest: state.pendingRunRequest
          ? {
              ...state.pendingRunRequest,
              inputs: remainingRunInputs || [],
              ...(remainingReviewOverrides &&
              Object.keys(remainingReviewOverrides).length > 0
                ? { reviewOverrides: remainingReviewOverrides }
                : { reviewOverrides: undefined }),
            }
          : null,
        processingState: {
          ...state.processingState,
          perItemProgress: state.processingState.perItemProgress.filter(
            (progress) => progress.id !== action.id
          ),
        },
      }
    }

    case "REQUEST_ITEM_CANCELLATION":
      return {
        ...state,
        processingState: {
          ...state.processingState,
          perItemProgress: state.processingState.perItemProgress.map((p) =>
            p.id === action.id && p.lifecycleState !== "terminal"
              ? {
                  ...p,
                  lifecycleState: "cancellation_requested" as const,
                  currentStage: "",
                }
              : p
          ),
        },
      }

    case "UPDATE_ITEM_PROGRESS": {
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

    case "APPLY_PLAYLIST_REVIEW_REQUIRED":
      return applyPlaylistReviewRequiredState(state, action.items)

    case "SKIP_TO_PROCESSING": {
      // Quick Mode: skip Steps 2-3, jump directly to Step 4 with default preset
      const { request, block } = buildPlaylistIngestRunRequest(state.queueItems)
      if (!request) {
        return {
          ...state,
          currentStep: 3,
          highestStep: Math.max(state.highestStep, 3) as WizardStep,
          pendingRunRequest: null,
          processingBlock: block,
        }
      }
      const perItemProgress = buildInitialProgress(state.queueItems)
      if (perItemProgress.length === 0) return state
      return {
        ...state,
        pendingRunRequest: request,
        processingBlock: null,
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
  updateQueueItems: (updater: (items: WizardQueueItem[]) => WizardQueueItem[]) => void
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
  checkStatus: (id: string) => void
  reconnect: () => void
  updateItemProgress: (progress: ItemProgress) => void
  updateProcessingState: (state: Partial<WizardProcessingState>) => void
  setResults: (results: WizardResultItem[]) => void
  applyPlaylistReviewRequired: (items: PlaylistReviewRequiredRecoveryItem[]) => void
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
  onCancelProcessing?: () => boolean
  onCancelItem?: (id: string) => boolean
  onCheckStatus?: (id: string) => void
  onReconnect?: () => void
}

export const IngestWizardProvider: React.FC<IngestWizardProviderProps> = ({
  children,
  initialState,
  onStateChange,
  presetMap = DEFAULT_PRESETS,
  onCancelProcessing,
  onCancelItem,
  onCheckStatus,
  onReconnect,
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

  const goToStep = useCallback((step: WizardStep) => dispatch({ type: "GO_TO_STEP", step }), [])
  const goNext = useCallback(() => dispatch({ type: "GO_NEXT" }), [])
  const goBack = useCallback(() => dispatch({ type: "GO_BACK" }), [])
  const setQueueItems = useCallback(
    (items: WizardQueueItem[]) => dispatch({ type: "SET_QUEUE_ITEMS", items }),
    []
  )
  const updateQueueItems = useCallback(
    (updater: (items: WizardQueueItem[]) => WizardQueueItem[]) =>
      dispatch({ type: "UPDATE_QUEUE_ITEMS", updater }),
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
  const startProcessing = useCallback(() => dispatch({ type: "START_PROCESSING" }), [])
  const skipToProcessing = useCallback(() => dispatch({ type: "SKIP_TO_PROCESSING" }), [])
  const cancelProcessing = useCallback(() => {
    if (onCancelProcessing?.()) return
    dispatch({ type: "CANCEL_PROCESSING" })
  }, [onCancelProcessing])
  const cancelItem = useCallback((id: string) => {
    if (onCancelItem?.(id)) {
      dispatch({ type: "REQUEST_ITEM_CANCELLATION", id })
      return
    }
    dispatch({ type: "CANCEL_ITEM", id })
  }, [onCancelItem])
  const checkStatus = useCallback((id: string) => onCheckStatus?.(id), [onCheckStatus])
  const reconnect = useCallback(() => onReconnect?.(), [onReconnect])
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
  const applyPlaylistReviewRequired = useCallback(
    (items: PlaylistReviewRequiredRecoveryItem[]) =>
      dispatch({ type: "APPLY_PLAYLIST_REVIEW_REQUIRED", items }),
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
      updateQueueItems,
      setPreset,
      setCustomOptions,
      setPlaylistPreflightSeed,
      setConferenceBatchMetadata,
      startProcessing,
      skipToProcessing,
      cancelProcessing,
      cancelItem,
      checkStatus,
      reconnect,
      updateItemProgress,
      updateProcessingState,
      setResults,
      applyPlaylistReviewRequired,
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
      updateQueueItems,
      setPreset,
      setCustomOptions,
      setPlaylistPreflightSeed,
      setConferenceBatchMetadata,
      startProcessing,
      skipToProcessing,
      cancelProcessing,
      cancelItem,
      checkStatus,
      reconnect,
      updateItemProgress,
      updateProcessingState,
      setResults,
      applyPlaylistReviewRequired,
      minimize,
      restore,
      reset,
    ]
  )

  return <IngestWizardContext.Provider value={value}>{children}</IngestWizardContext.Provider>
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
