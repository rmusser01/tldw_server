import React from "react"
import { useQuery } from "@tanstack/react-query"
import {
  getImageBackendConfigs,
  normalizeImageBackendConfig,
  resolveImageBackendConfig
} from "@/services/image-generation"
import {
  resolveImageGenerationEventSyncMode,
  normalizeImageGenerationEventSyncMode,
  normalizeImageGenerationEventSyncPolicy,
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
  type ImageGenerationEventSyncPolicy,
  type ImageGenerationEventSyncMode,
  type ImageGenerationRefineMetadata,
  type ImageGenerationPromptMode,
  type ImageGenerationRequestSnapshot
} from "@/utils/image-generation-chat"
import {
  buildImagePromptRefineMessages,
  extractImagePromptRefineCandidate
} from "@/utils/image-prompt-refinement"
import {
  createImagePromptDraftFromStrategy,
  deriveImagePromptRawContext,
  getImagePromptStrategies,
  type WeightedImagePromptContextEntry
} from "@/utils/image-prompt-strategies"
import {
  computeResponseDiffPreview,
  type CompareResponseDiff
} from "../compare-response-diff"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  loadServicePromptSnapshot,
  type ServicePromptSnapshot
} from "@/services/service-prompts"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { isChatSubmitSuccess, normalizeChatSubmitResult, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import { parseJsonObject } from "./utils"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundImageGenDeps {
  imageBackendDefaultTrimmed: string
  imageBackendOptions: Array<{ value: string; label: string; provider?: string }>
  imageEventSyncChatMode: ImageGenerationEventSyncMode
  imageEventSyncGlobalDefault: ImageGenerationEventSyncMode
  updateChatSettings: (patch: Record<string, unknown>) => void
  setImageEventSyncGlobalDefault: (value: ImageGenerationEventSyncMode) => void
  messages: Array<{ isBot?: boolean; message?: string; moodLabel?: string }>
  selectedCharacterName: string | null
  selectedModel: string | null
  currentApiProvider: string | undefined
  formMessage: string
  sendMessage: (payload: Record<string, any>) => Promise<ChatSubmitResult | void>
  captureCompletionGuard?: (forFocus?: boolean) => () => boolean
  textAreaFocus: () => void
  notificationApi: {
    error: (opts: { message: string; description?: string }) => void
    success?: (opts: { message: string; description?: string }) => void
  }
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
  setToolsPopoverOpen: (open: boolean) => void
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePlaygroundImageGen(deps: UsePlaygroundImageGenDeps) {
  const {
    imageBackendDefaultTrimmed,
    imageBackendOptions,
    imageEventSyncChatMode,
    imageEventSyncGlobalDefault,
    updateChatSettings,
    setImageEventSyncGlobalDefault,
    messages,
    selectedCharacterName,
    selectedModel,
    currentApiProvider,
    formMessage,
    sendMessage,
    captureCompletionGuard,
    textAreaFocus,
    notificationApi,
    t,
    setToolsPopoverOpen
  } = deps

  // ---- state ----
  const [imageGenerateModalOpen, setImageGenerateModalOpen] = React.useState(false)
  const [imageGenerateBackend, setImageGenerateBackend] = React.useState("")
  const [imageGeneratePrompt, setImageGeneratePrompt] = React.useState("")
  const [imageGeneratePromptMode, setImageGeneratePromptMode] =
    React.useState<ImageGenerationPromptMode>("scene")
  const [imageGenerateFormat, setImageGenerateFormat] = React.useState<
    "png" | "jpg" | "webp"
  >("png")
  const [imageGenerateNegativePrompt, setImageGenerateNegativePrompt] =
    React.useState("")
  const [imageGenerateWidth, setImageGenerateWidth] = React.useState<
    number | undefined
  >(undefined)
  const [imageGenerateHeight, setImageGenerateHeight] = React.useState<
    number | undefined
  >(undefined)
  const [imageGenerateSteps, setImageGenerateSteps] = React.useState<
    number | undefined
  >(undefined)
  const [imageGenerateCfgScale, setImageGenerateCfgScale] = React.useState<
    number | undefined
  >(undefined)
  const [imageGenerateSeed, setImageGenerateSeed] = React.useState<
    number | undefined
  >(undefined)
  const [imageGenerateSampler, setImageGenerateSampler] = React.useState("")
  const [imageGenerateModel, setImageGenerateModel] = React.useState("")
  const [imageGenerateExtraParams, setImageGenerateExtraParams] =
    React.useState("")
  const [imageGenerateReferenceFileId, setImageGenerateReferenceFileId] =
    React.useState<number | undefined>(undefined)
  const [imageGenerateSyncPolicy, setImageGenerateSyncPolicy] =
    React.useState<ImageGenerationEventSyncPolicy>("inherit")
  const [imagePromptContextBreakdown, setImagePromptContextBreakdown] =
    React.useState<WeightedImagePromptContextEntry[]>([])
  const [imagePromptRefineSubmitting, setImagePromptRefineSubmitting] =
    React.useState(false)
  const [imagePromptRefineBaseline, setImagePromptRefineBaseline] =
    React.useState("")
  const [imagePromptRefineCandidate, setImagePromptRefineCandidate] =
    React.useState("")
  const [imagePromptRefineModel, setImagePromptRefineModel] = React.useState<
    string | null
  >(null)
  const [imagePromptRefineLatencyMs, setImagePromptRefineLatencyMs] =
    React.useState<number | null>(null)
  const [imagePromptRefineDiff, setImagePromptRefineDiff] =
    React.useState<CompareResponseDiff | null>(null)
  const [imageGenerateRefineMetadata, setImageGenerateRefineMetadata] =
    React.useState<ImageGenerationRefineMetadata | undefined>(undefined)
  const [imageGenerateSubmitting, setImageGenerateSubmitting] =
    React.useState(false)

  const imageOperationRef = React.useRef<{ isCurrent: () => boolean } | null>(null)
  const refinementSourceRef = React.useRef<{ isCurrent: () => boolean; hasCandidate: boolean } | null>(null)
  const modalRevision = React.useRef(0)
  const settingsEditRevision = React.useRef(0)
  const settingsRequestRevision = React.useRef(0)
  const configurationSetters = React.useMemo(() => {
    const edited = <T,>(setter: React.Dispatch<React.SetStateAction<T>>) => (value: React.SetStateAction<T>) => {
      settingsEditRevision.current++
      setter(value)
    }
    return {
      setImageGenerateBackend: edited(setImageGenerateBackend),
      setImageGenerateFormat: edited(setImageGenerateFormat),
      setImageGenerateNegativePrompt: edited(setImageGenerateNegativePrompt),
      setImageGenerateWidth: edited(setImageGenerateWidth),
      setImageGenerateHeight: edited(setImageGenerateHeight),
      setImageGenerateSteps: edited(setImageGenerateSteps),
      setImageGenerateCfgScale: edited(setImageGenerateCfgScale),
      setImageGenerateSeed: edited(setImageGenerateSeed),
      setImageGenerateSampler: edited(setImageGenerateSampler),
      setImageGenerateModel: edited(setImageGenerateModel),
      setImageGenerateExtraParams: edited(setImageGenerateExtraParams)
    }
  }, [])
  React.useEffect(() => () => { imageOperationRef.current = null; refinementSourceRef.current = null; modalRevision.current++ }, [])

  // ---- derived ----
  const imagePromptStrategies = React.useMemo(() => getImagePromptStrategies(), [])
  const imageGenerationCharacterMood = React.useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const candidate = messages[i] as any
      if (candidate?.isBot && typeof candidate?.moodLabel === "string") {
        return candidate.moodLabel
      }
    }
    return null
  }, [messages])

  const imageGenerateBackendOptions = React.useMemo(() => {
    return imageBackendOptions.filter((option) => option.value.trim().length > 0)
  }, [imageBackendOptions])

  const imageGenerateBusy = imageGenerateSubmitting || imagePromptRefineSubmitting
  const referenceImageCandidatesQuery = useQuery({
    queryKey: ["reference-image-candidates"],
    queryFn: async () => (await tldwClient.listReferenceImageCandidates()).items,
    enabled: imageGenerateModalOpen,
    staleTime: 60_000
  })
  const referenceImageCandidates = referenceImageCandidatesQuery.data ?? []
  const referenceImageCandidatesLoading = referenceImageCandidatesQuery.isLoading

  const imageEventSyncBaselineMode = React.useMemo(
    () =>
      resolveImageGenerationEventSyncMode({
        requestPolicy: "inherit",
        chatMode: imageEventSyncChatMode,
        globalMode: normalizeImageGenerationEventSyncMode(
          imageEventSyncGlobalDefault,
          "off"
        )
      }),
    [imageEventSyncChatMode, imageEventSyncGlobalDefault]
  )

  const imageGenerateResolvedSyncMode = React.useMemo(
    () =>
      resolveImageGenerationEventSyncMode({
        requestPolicy: imageGenerateSyncPolicy,
        chatMode: imageEventSyncChatMode,
        globalMode: normalizeImageGenerationEventSyncMode(
          imageEventSyncGlobalDefault,
          "off"
        )
      }),
    [imageEventSyncChatMode, imageEventSyncGlobalDefault, imageGenerateSyncPolicy]
  )

  // ---- callbacks ----
  const clearImagePromptRefineCandidate = React.useCallback(() => {
    if (refinementSourceRef.current) refinementSourceRef.current.hasCandidate = false
    setImagePromptRefineBaseline("")
    setImagePromptRefineCandidate("")
    setImagePromptRefineModel(null)
    setImagePromptRefineLatencyMs(null)
    setImagePromptRefineDiff(null)
  }, [])

  const clearImagePromptRefineState = React.useCallback(() => {
    refinementSourceRef.current = null
    clearImagePromptRefineCandidate()
    setImageGenerateRefineMetadata(undefined)
  }, [clearImagePromptRefineCandidate])
  const renderedRefinementSource = refinementSourceRef.current
  React.useLayoutEffect(() => {
    const active = imageOperationRef.current
    if (active && !active.isCurrent()) {
      imageOperationRef.current = null
      setImageGenerateSubmitting(false)
      setImagePromptRefineSubmitting(false)
    }
    const refinement = refinementSourceRef.current
    if (refinement && !refinement.isCurrent()) clearImagePromptRefineState()
  }, [captureCompletionGuard, imageGenerateSubmitting, imagePromptRefineSubmitting,
    renderedRefinementSource, clearImagePromptRefineState])

  const hydrateImageGenerateSettings = React.useCallback(
    async (backend: string) => {
      const sourceIsCurrent = captureCompletionGuard?.() ?? (() => true)
      if (!backend || !sourceIsCurrent()) return
      const revision = modalRevision.current
      const editRevision = settingsEditRevision.current
      const requestRevision = ++settingsRequestRevision.current
      const isCurrent = () => sourceIsCurrent() && modalRevision.current === revision &&
        settingsEditRevision.current === editRevision && settingsRequestRevision.current === requestRevision
      try {
        const configs = await getImageBackendConfigs({ throwOnError: true })
        if (!isCurrent()) return
        const config = normalizeImageBackendConfig(resolveImageBackendConfig(backend, configs))
        const extraParams = config.extraParams == null
          ? ""
          : typeof config.extraParams === "string"
            ? config.extraParams
            : JSON.stringify(config.extraParams, null, 2)
        setImageGenerateFormat(config.format || "png")
        setImageGenerateNegativePrompt(config.negativePrompt || "")
        setImageGenerateWidth(config.width)
        setImageGenerateHeight(config.height)
        setImageGenerateSteps(config.steps)
        setImageGenerateCfgScale(config.cfgScale)
        setImageGenerateSeed(config.seed)
        setImageGenerateSampler(config.sampler || "")
        setImageGenerateModel(config.model || "")
        setImageGenerateExtraParams(extraParams)
      } catch (error) {
        if (!isCurrent()) return
        notificationApi.error({
          message: t("playground:imageGeneration.settingsLoadFailedTitle", "Image settings unavailable"),
          description: error instanceof Error ? error.message : t(
            "playground:imageGeneration.settingsLoadFailedBody",
            "Could not load saved image settings. Your current settings were kept."
          )
        })
      }
    },
    [captureCompletionGuard, notificationApi, t]
  )

  const openImageGenerateModal = React.useCallback(() => {
    modalRevision.current++
    imageOperationRef.current = null
    setImageGenerateSubmitting(false)
    setImagePromptRefineSubmitting(false)
    setToolsPopoverOpen(false)
    setImagePromptContextBreakdown([])
    clearImagePromptRefineState()
    setImageGenerateReferenceFileId(undefined)
    setImageGenerateSyncPolicy("inherit")
    const defaultBackend =
      imageBackendDefaultTrimmed ||
      imageGenerateBackendOptions[0]?.value ||
      ""
    setImageGenerateBackend(defaultBackend)
    if (!imageGeneratePrompt.trim()) {
      const draftFromComposer = String(formMessage || "").trim()
      if (draftFromComposer) {
        setImageGeneratePrompt(draftFromComposer)
      }
    }
    if (defaultBackend) {
      void hydrateImageGenerateSettings(defaultBackend)
    }
    setImageGenerateModalOpen(true)
  }, [
    clearImagePromptRefineState,
    formMessage,
    hydrateImageGenerateSettings,
    imageBackendDefaultTrimmed,
    imageGenerateBackendOptions,
    imageGeneratePrompt,
    setImageGenerateReferenceFileId,
    setToolsPopoverOpen
  ])

  const closeImageGenerateModal = React.useCallback(() => {
    modalRevision.current++
    imageOperationRef.current = null
    setImageGenerateSubmitting(false)
    setImagePromptRefineSubmitting(false)
    clearImagePromptRefineState()
    setImageGenerateReferenceFileId(undefined)
    setImageGenerateModalOpen(false)
  }, [clearImagePromptRefineState])

  const handleCreateImagePromptDraft = React.useCallback(() => {
    const rawContext = deriveImagePromptRawContext({
      messages: messages as Array<{ isBot?: boolean; message?: string }>,
      characterName: selectedCharacterName,
      moodLabel: imageGenerationCharacterMood,
      userIntent: formMessage || imageGeneratePrompt
    })
    const draftResult = createImagePromptDraftFromStrategy({
      strategyId: imageGeneratePromptMode,
      rawContext
    })
    setImageGeneratePrompt(draftResult.prompt)
    setImagePromptContextBreakdown(draftResult.weightedContext.entries.slice(0, 4))
    clearImagePromptRefineState()
  }, [
    clearImagePromptRefineState,
    formMessage,
    imageGeneratePromptMode,
    imageGenerationCharacterMood,
    imageGeneratePrompt,
    messages,
    selectedCharacterName
  ])

  const handleRefineImagePromptDraft = React.useCallback(async () => {
    const sourceIsCurrent = captureCompletionGuard?.() ?? (() => true)
    if (!sourceIsCurrent() || imageOperationRef.current) return
    const revision = modalRevision.current
    const prompt = imageGeneratePrompt.trim()
    if (!prompt) {
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:imageGeneration.refinePromptRequired",
          "Add or create a prompt before refining."
        )
      })
      return
    }

    const normalizedModel = String(selectedModel || "")
      .replace(/^tldw:/, "")
      .trim()
    if (!normalizedModel) {
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:imageGeneration.refineModelRequired",
          "Select a chat model before using Refine with LLM."
        )
      })
      return
    }

    const strategyLabel =
      imagePromptStrategies.find((entry) => entry.id === imageGeneratePromptMode)
        ?.label || imageGeneratePromptMode
    const contextEntries =
      imagePromptContextBreakdown.length > 0
        ? imagePromptContextBreakdown
        : createImagePromptDraftFromStrategy({
            strategyId: imageGeneratePromptMode,
            rawContext: deriveImagePromptRawContext({
              messages: messages as Array<{ isBot?: boolean; message?: string }>,
              characterName: selectedCharacterName,
              moodLabel: imageGenerationCharacterMood,
              userIntent: formMessage || imageGeneratePrompt
            })
          }).weightedContext.entries.slice(0, 4)

    const attempt = { isCurrent: () => sourceIsCurrent() && modalRevision.current === revision }
    imageOperationRef.current = attempt
    setImagePromptRefineSubmitting(true)
    setImageGenerateRefineMetadata(undefined)
    let servicePromptSnapshot: ServicePromptSnapshot | null = null
    const isCurrent = () => imageOperationRef.current === attempt && attempt.isCurrent() &&
      !servicePromptSnapshot?.scopeInvalidatedSignal.aborted
    const assertCurrent = () => {
      if (!isCurrent()) throw createServicePromptScopeChangedError()
    }
    try {
      assertCurrent()
      const startedAt =
        typeof performance !== "undefined" ? performance.now() : Date.now()
      await tldwClient.initialize().catch(() => null)
      assertCurrent()
      servicePromptSnapshot = await loadServicePromptSnapshot([
        "image.prompt.refinement"
      ])
      assertCurrent()
      const refinementPrompt =
        servicePromptSnapshot.definitions["image.prompt.refinement"]
      if (!refinementPrompt) {
        throw new Error("Image prompt refinement settings are unavailable.")
      }
      const provider = await resolveApiProviderForModel({
        modelId: normalizedModel,
        explicitProvider: currentApiProvider
      })
      assertCurrent()
      const completionResponse = await tldwClient.createChatCompletion(
        {
          model: normalizedModel,
          api_provider: provider || undefined,
          temperature: 0.1,
          max_tokens: 320,
          messages: buildImagePromptRefineMessages({
            originalPrompt: prompt,
            strategyLabel,
            backend: imageGenerateBackend,
            contextEntries,
            systemSemantics: refinementPrompt.parts.system_semantics,
            rewriteSemantics: refinementPrompt.parts.rewrite_semantics
          })
        },
        {
          signal: servicePromptSnapshot.scopeSignal,
          requestScope: servicePromptSnapshot.requestScope
        }
      )
      assertCurrent()
      const completionPayload = await completionResponse.json().catch(() => null)
      assertCurrent()
      const candidate = extractImagePromptRefineCandidate(completionPayload)
      if (!candidate) {
        throw new Error(
          t(
            "playground:imageGeneration.refineEmpty",
            "Refiner returned an empty prompt. Try again."
          )
        )
      }
      const elapsedRaw =
        typeof performance !== "undefined" ? performance.now() : Date.now()
      const latencyMs = Math.max(1, Math.round(elapsedRaw - startedAt))
      const diff = computeResponseDiffPreview({
        baseline: prompt,
        candidate,
        maxHighlights: 4
      })
      refinementSourceRef.current = { isCurrent: attempt.isCurrent, hasCandidate: true }
      setImagePromptRefineBaseline(prompt)
      setImagePromptRefineCandidate(candidate)
      setImagePromptRefineModel(normalizedModel)
      setImagePromptRefineLatencyMs(latencyMs)
      setImagePromptRefineDiff(diff)
    } catch (error: any) {
      if (!isCurrent()) return
      notificationApi.error({
        message: t(
          "playground:imageGeneration.refineFailedTitle",
          "Prompt refinement failed"
        ),
        description:
          error?.message ||
          t(
            "playground:imageGeneration.refineFailedBody",
            "Could not refine the image prompt."
          )
      })
    } finally {
      servicePromptSnapshot?.release()
      if (imageOperationRef.current === attempt) {
        imageOperationRef.current = null
        setImagePromptRefineSubmitting(false)
      }
    }
  }, [
    captureCompletionGuard,
    currentApiProvider,
    formMessage,
    imageGenerateBackend,
    imageGeneratePrompt,
    imageGeneratePromptMode,
    imageGenerationCharacterMood,
    imagePromptContextBreakdown,
    imagePromptStrategies,
    messages,
    notificationApi,
    selectedCharacterName,
    selectedModel,
    t
  ])

  const applyRefinedImagePromptCandidate = React.useCallback(() => {
    if (imageOperationRef.current || !renderedRefinementSource?.hasCandidate ||
      refinementSourceRef.current !== renderedRefinementSource || !renderedRefinementSource.isCurrent()) return
    const candidate = imagePromptRefineCandidate.trim()
    if (!candidate) return

    if (imagePromptRefineModel && imagePromptRefineLatencyMs != null) {
      const diffStats = imagePromptRefineDiff
        ? {
            baselineSegments: imagePromptRefineDiff.baselineSegments,
            candidateSegments: imagePromptRefineDiff.candidateSegments,
            sharedSegments: imagePromptRefineDiff.sharedSegments,
            overlapRatio: imagePromptRefineDiff.overlapRatio,
            addedCount: imagePromptRefineDiff.addedHighlights.length,
            removedCount: imagePromptRefineDiff.removedHighlights.length
          }
        : {
            baselineSegments: 0,
            candidateSegments: 0,
            sharedSegments: 0,
            overlapRatio: 0,
            addedCount: 0,
            removedCount: 0
          }
      setImageGenerateRefineMetadata({
        model: imagePromptRefineModel,
        latencyMs: imagePromptRefineLatencyMs,
        diffStats
      })
    }

    setImageGeneratePrompt(candidate)
    clearImagePromptRefineCandidate()
  }, [
    renderedRefinementSource,
    clearImagePromptRefineCandidate,
    imagePromptRefineCandidate,
    imagePromptRefineDiff,
    imagePromptRefineLatencyMs,
    imagePromptRefineModel
  ])

  const rejectRefinedImagePromptCandidate = React.useCallback(() => {
    if (imageOperationRef.current || !renderedRefinementSource?.hasCandidate ||
      refinementSourceRef.current !== renderedRefinementSource || !renderedRefinementSource.isCurrent()) return
    clearImagePromptRefineState()
  }, [renderedRefinementSource, clearImagePromptRefineState])

  const submitImageGenerateModal = React.useCallback(async () => {
    const sourceIsCurrent = captureCompletionGuard?.() ?? (() => true)
    const canFocus = captureCompletionGuard?.(true) ?? (() => true)
    if (!sourceIsCurrent() || imageOperationRef.current) return
    const revision = modalRevision.current

    const prompt = imageGeneratePrompt.trim()
    const backend = imageGenerateBackend.trim()
    if (!backend) {
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:imageGeneration.modalBackendRequired",
          "Select an image backend before generating."
        )
      })
      return
    }
    if (!prompt) {
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:imageGeneration.modalPromptRequired",
          "Image prompt is required."
        )
      })
      return
    }
    const parsedExtraParams = parseJsonObject(imageGenerateExtraParams)
    if (imageGenerateExtraParams.trim().length > 0 && !parsedExtraParams) {
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "playground:imageGeneration.modalExtraParamsInvalid",
          "Extra params must be valid JSON object."
        )
      })
      return
    }

    const request: Partial<ImageGenerationRequestSnapshot> = {
      prompt,
      backend,
      format: imageGenerateFormat,
      negativePrompt: imageGenerateNegativePrompt.trim() || undefined,
      referenceFileId:
        typeof imageGenerateReferenceFileId === "number" &&
        Number.isFinite(imageGenerateReferenceFileId)
          ? imageGenerateReferenceFileId
          : undefined,
      width:
        typeof imageGenerateWidth === "number" && Number.isFinite(imageGenerateWidth)
          ? imageGenerateWidth
          : undefined,
      height:
        typeof imageGenerateHeight === "number" && Number.isFinite(imageGenerateHeight)
          ? imageGenerateHeight
          : undefined,
      steps:
        typeof imageGenerateSteps === "number" && Number.isFinite(imageGenerateSteps)
          ? imageGenerateSteps
          : undefined,
      cfgScale:
        typeof imageGenerateCfgScale === "number" &&
        Number.isFinite(imageGenerateCfgScale)
          ? imageGenerateCfgScale
          : undefined,
      seed:
        typeof imageGenerateSeed === "number" && Number.isFinite(imageGenerateSeed)
          ? imageGenerateSeed
          : undefined,
      sampler: imageGenerateSampler.trim() || undefined,
      model: imageGenerateModel.trim() || undefined,
      extraParams: parsedExtraParams
    }

    const attempt = { isCurrent: () => sourceIsCurrent() && modalRevision.current === revision }
    imageOperationRef.current = attempt
    setImageGenerateSubmitting(true)
    try {
      const result = await sendMessage({
        message: prompt,
        image: "",
        docs: [],
        imageBackendOverride: backend,
        userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
        assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        imageGenerationRequest: request,
        imageGenerationRefine: imageGenerateRefineMetadata,
        imageGenerationPromptMode: imageGeneratePromptMode,
        imageGenerationSource: "generate-modal",
        imageEventSyncPolicy: imageGenerateSyncPolicy
      })
      if (imageOperationRef.current !== attempt || !attempt.isCurrent() || !isChatSubmitSuccess(normalizeChatSubmitResult(result))) return
      const focusIsCurrent = canFocus()
      closeImageGenerateModal()
      if (focusIsCurrent) textAreaFocus()
    } catch (error: any) {
      if (imageOperationRef.current !== attempt || !attempt.isCurrent()) return
      notificationApi.error({
        message: t("error", { defaultValue: "Error" }),
        description:
          error?.message ||
          t(
            "playground:imageGeneration.generateFailed",
            "Image generation request failed."
          )
      })
    } finally {
      if (imageOperationRef.current === attempt) {
        imageOperationRef.current = null
        setImageGenerateSubmitting(false)
      }
    }
  }, [
    imageGenerateBackend,
    imageGenerateCfgScale,
    imageGenerateExtraParams,
    imageGenerateFormat,
    imageGenerateHeight,
    imageGenerateModel,
    imageGenerateNegativePrompt,
    imageGeneratePromptMode,
    imageGenerateSyncPolicy,
    imageGeneratePrompt,
    imageGenerateReferenceFileId,
    imageGenerateSampler,
    imageGenerateSeed,
    imageGenerateSteps,
    imageGenerateWidth,
    imageGenerateRefineMetadata,
    closeImageGenerateModal,
    captureCompletionGuard,
    notificationApi,
    sendMessage,
    t,
    textAreaFocus
  ])

  return {
    ...configurationSetters,
    // state
    imageGenerateModalOpen,
    setImageGenerateModalOpen,
    imageGenerateBackend,
    imageGeneratePrompt,
    setImageGeneratePrompt,
    imageGeneratePromptMode,
    setImageGeneratePromptMode,
    imageGenerateFormat,
    imageGenerateNegativePrompt,
    imageGenerateWidth,
    imageGenerateHeight,
    imageGenerateSteps,
    imageGenerateCfgScale,
    imageGenerateSeed,
    imageGenerateSampler,
    imageGenerateModel,
    imageGenerateExtraParams,
    imageGenerateReferenceFileId,
    setImageGenerateReferenceFileId,
    imageGenerateSyncPolicy,
    setImageGenerateSyncPolicy,
    referenceImageCandidates,
    referenceImageCandidatesLoading,
    imagePromptContextBreakdown,
    imagePromptRefineSubmitting,
    imagePromptRefineBaseline,
    imagePromptRefineCandidate,
    imagePromptRefineModel,
    imagePromptRefineLatencyMs,
    imagePromptRefineDiff,
    imageGenerateRefineMetadata,
    imageGenerateSubmitting,
    // derived
    imagePromptStrategies,
    imageGenerationCharacterMood,
    imageGenerateBackendOptions,
    imageGenerateBusy,
    imageEventSyncBaselineMode,
    imageGenerateResolvedSyncMode,
    // callbacks
    clearImagePromptRefineState,
    closeImageGenerateModal,
    hydrateImageGenerateSettings,
    openImageGenerateModal,
    handleCreateImagePromptDraft,
    handleRefineImagePromptDraft,
    applyRefinedImagePromptCandidate,
    rejectRefinedImagePromptCandidate,
    submitImageGenerateModal,
    // re-exports for sync policy UI
    normalizeImageGenerationEventSyncMode,
    normalizeImageGenerationEventSyncPolicy
  }
}

export type UsePlaygroundImageGenReturn = ReturnType<typeof usePlaygroundImageGen>
