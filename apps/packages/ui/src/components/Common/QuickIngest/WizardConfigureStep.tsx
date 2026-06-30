import React from "react"
import { Button, Input, Segmented, Select, Switch, Tooltip, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { ArrowLeft, ArrowRight, ChevronRight } from "lucide-react"

import { useIngestWizard } from "./IngestWizardContext"
import { PresetSelector } from "./PresetSelector"
import type { CommonOptions, DetectedMediaType, TypeDefaults } from "./types"
import { SUPPORTED_LANGUAGES } from "@/utils/supported-languages"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const DRAFT_STORAGE_CAP_BYTES = 5 * 1024 * 1024
const CUSTOM_AUDIO_LANGUAGE_SENTINEL = "__custom__"

const CHUNKING_MODE_OPTIONS = [
  { label: "Auto", value: "auto" },
  { label: "Manual", value: "manual" },
]

const AUTO_CHUNKING_GOAL_OPTIONS = [
  { label: "Balanced", value: "balanced" },
  { label: "Search/Q&A", value: "qa_search" },
  { label: "Reading/Summary", value: "navigation_summary" },
]

const CHUNK_METHOD_OPTIONS = [
  { label: "Semantic", value: "semantic" },
  { label: "Tokens", value: "tokens" },
  { label: "Paragraphs", value: "paragraphs" },
  { label: "Sentences", value: "sentences" },
  { label: "Words", value: "words" },
]

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

const nextTypeDefaults = (
  previous: TypeDefaults,
  nextValue: React.SetStateAction<TypeDefaults | null>
): TypeDefaults => {
  const resolved =
    typeof nextValue === "function"
      ? nextValue(previous)
      : nextValue

  return resolved ?? {}
}

type WizardConfigureStepProps = {
  isStepVisible?: boolean
}

export const WizardConfigureStep: React.FC<WizardConfigureStepProps> = ({
  isStepVisible = true,
}) => {
  const { t } = useTranslation(["option"])
  const { state, setPreset, setCustomOptions, goNext, goBack } = useIngestWizard()
  const { queueItems, selectedPreset, presetConfig } = state

  const qi = React.useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  const detectedTypes = React.useMemo(() => {
    const types = new Set<DetectedMediaType>()
    for (const item of queueItems) {
      types.add(item.detectedType)
    }
    return types
  }, [queueItems])

  const hasAudioItems = detectedTypes.has("audio")
  const hasVideoItems = detectedTypes.has("video")
  const hasTranscriptionItems = hasAudioItems || hasVideoItems
  const [transcriptionModels, setTranscriptionModels] = React.useState<string[]>([])
  const [transcriptionModelsLoading, setTranscriptionModelsLoading] =
    React.useState(false)
  const [showAdvanced, setShowAdvanced] = React.useState(false)
  const chunkingMode =
    presetConfig.common.chunking_mode === "manual" ? "manual" : "auto"
  const autoChunkingGoal =
    presetConfig.common.auto_chunking_goal === "qa_search" ||
    presetConfig.common.auto_chunking_goal === "navigation_summary"
      ? presetConfig.common.auto_chunking_goal
      : "balanced"
  const manualChunkMethod =
    typeof presetConfig.advancedValues?.chunk_method === "string"
      ? presetConfig.advancedValues.chunk_method
      : "semantic"
  const manualChunkSize =
    presetConfig.advancedValues?.chunk_size == null
      ? ""
      : String(presetConfig.advancedValues.chunk_size)
  const manualChunkOverlap =
    presetConfig.advancedValues?.chunk_overlap == null
      ? ""
      : String(presetConfig.advancedValues.chunk_overlap)

  const hasDocumentItems =
    detectedTypes.has("document") ||
    detectedTypes.has("pdf") ||
    detectedTypes.has("ebook") ||
    detectedTypes.has("image")

  const rawTranscriptionModel = presetConfig.advancedValues?.transcription_model
  const normalizedTranscriptionModel =
    typeof rawTranscriptionModel === "string" ? rawTranscriptionModel.trim() : ""
  const shouldLoadTranscriptionModels =
    hasTranscriptionItems && isStepVisible

  React.useEffect(() => {
    if (!shouldLoadTranscriptionModels) {
      setTranscriptionModels([])
      setTranscriptionModelsLoading(false)
      return
    }

    let cancelled = false
    const loadModels = async () => {
      setTranscriptionModelsLoading(true)
      try {
        const result = await tldwClient.getTranscriptionModels({
          timeoutMs: 10_000
        })
        const raw = Array.isArray(result?.all_models) ? result.all_models : []
        const seen = new Set<string>()
        const uniqueModels: string[] = []
        for (const entry of raw) {
          const model = String(entry).trim()
          if (!model || seen.has(model)) {
            continue
          }
          seen.add(model)
          uniqueModels.push(model)
        }
        if (!cancelled) {
          setTranscriptionModels(uniqueModels)
        }
      } catch {
        // ignore model catalog fetch errors in the quick ingest flow
      } finally {
        if (!cancelled) {
          setTranscriptionModelsLoading(false)
        }
      }
    }

    void loadModels()

    return () => {
      cancelled = true
      setTranscriptionModelsLoading(false)
    }
  }, [shouldLoadTranscriptionModels])

  const transcriptionModelOptions = React.useMemo(() => {
    const uniqueModels = new Map<string, string>()
    for (const model of transcriptionModels) {
      const value = String(model).trim()
      if (!value || uniqueModels.has(value)) {
        continue
      }
      uniqueModels.set(value, value)
    }

    if (normalizedTranscriptionModel && !uniqueModels.has(normalizedTranscriptionModel)) {
      uniqueModels.set(normalizedTranscriptionModel, normalizedTranscriptionModel)
    }

    return [...uniqueModels.entries()].map(([value, label]) => ({
      value,
      label,
    }))
  }, [normalizedTranscriptionModel, transcriptionModels])

  const setCommon = React.useCallback(
    (nextValue: React.SetStateAction<CommonOptions>) => {
      const resolved =
        typeof nextValue === "function"
          ? nextValue(presetConfig.common)
          : nextValue
      setCustomOptions({ common: resolved })
    },
    [presetConfig.common, setCustomOptions]
  )

  const setTypeDefaults = React.useCallback(
    (nextValue: React.SetStateAction<TypeDefaults | null>) => {
      setCustomOptions({
        typeDefaults: nextTypeDefaults(presetConfig.typeDefaults ?? {}, nextValue),
      })
    },
    [presetConfig.typeDefaults, setCustomOptions]
  )

  const normalizedAudioLanguage =
    presetConfig.typeDefaults.audio?.language?.trim() || ""

  const supportedLanguageValues = React.useMemo(
    () => new Set(SUPPORTED_LANGUAGES.map((option) => option.value)),
    []
  )

  const isKnownLanguage =
    normalizedAudioLanguage !== "" &&
    supportedLanguageValues.has(normalizedAudioLanguage)

  const [audioLanguageMode, setAudioLanguageMode] = React.useState<
    "empty" | "standard" | "custom"
  >(() => {
    if (normalizedAudioLanguage === "") return "empty"
    return isKnownLanguage ? "standard" : "custom"
  })

  const savedAudioLanguageRef = React.useRef(normalizedAudioLanguage)

  const [customAudioLanguage, setCustomAudioLanguage] = React.useState(
    isKnownLanguage ? "" : normalizedAudioLanguage
  )

  React.useEffect(() => {
    if (savedAudioLanguageRef.current === normalizedAudioLanguage) {
      return
    }

    if (normalizedAudioLanguage === "") {
      setAudioLanguageMode("empty")
    } else if (isKnownLanguage) {
      setAudioLanguageMode("standard")
    } else {
      setAudioLanguageMode("custom")
      setCustomAudioLanguage(normalizedAudioLanguage)
    }

    savedAudioLanguageRef.current = normalizedAudioLanguage
  }, [isKnownLanguage, normalizedAudioLanguage])

  React.useEffect(() => {
    if (audioLanguageMode === "custom") {
      setCustomAudioLanguage(normalizedAudioLanguage)
    }
  }, [audioLanguageMode, normalizedAudioLanguage])

  const shouldShowCustomAudioLanguageInput =
    audioLanguageMode === "custom"

  const audioLanguageSelectValue = React.useMemo(() => {
    if (audioLanguageMode === "empty") {
      return undefined
    }
    if (audioLanguageMode === "custom") {
      return CUSTOM_AUDIO_LANGUAGE_SENTINEL
    }
    return normalizedAudioLanguage
  }, [audioLanguageMode, normalizedAudioLanguage])

  const handleAnalysisToggle = React.useCallback(
    (checked: boolean) => {
      setCommon((current) => ({ ...current, perform_analysis: checked }))
    },
    [setCommon]
  )

  const handleChunkingToggle = React.useCallback(
    (checked: boolean) => {
      setCommon((current) => ({
        ...current,
        perform_chunking: checked,
        ...(checked
          ? {
              chunking_mode: "auto",
              auto_chunking_goal: current.auto_chunking_goal ?? "balanced",
              auto_chunking_use_llm: current.auto_chunking_use_llm ?? false,
            }
          : {}),
      }))
    },
    [setCommon]
  )

  const handleChunkingModeChange = React.useCallback(
    (nextMode: string | number) => {
      const mode = nextMode === "manual" ? "manual" : "auto"
      setCommon((current) => ({
        ...current,
        chunking_mode: mode,
        auto_chunking_goal: current.auto_chunking_goal ?? "balanced",
        auto_chunking_use_llm: current.auto_chunking_use_llm ?? false,
      }))
    },
    [setCommon]
  )

  const handleAutoChunkingGoalChange = React.useCallback(
    (nextGoal: string) => {
      const goal =
        nextGoal === "qa_search" || nextGoal === "navigation_summary"
          ? nextGoal
          : "balanced"
      setCommon((current) => ({
        ...current,
        auto_chunking_goal: goal,
      }))
    },
    [setCommon]
  )

  const handleAutoChunkingUseLlmChange = React.useCallback(
    (checked: boolean) => {
      setCommon((current) => ({
        ...current,
        auto_chunking_use_llm: checked,
      }))
    },
    [setCommon]
  )

  const handleOverwriteToggle = React.useCallback(
    (checked: boolean) => {
      setCommon((current) => ({ ...current, overwrite_existing: checked }))
    },
    [setCommon]
  )

  const handleAudioLanguageOptionChange = React.useCallback(
    (nextValue: string) => {
      if (nextValue === CUSTOM_AUDIO_LANGUAGE_SENTINEL) {
        setAudioLanguageMode("custom")
        return
      }

      setAudioLanguageMode("standard")

      setTypeDefaults((previous) => ({
        ...(previous ?? {}),
        audio: {
          ...(previous?.audio ?? {}),
          language: nextValue,
        },
      }))
    },
    [setTypeDefaults]
  )

  const handleAudioLanguageClear = React.useCallback(() => {
    setAudioLanguageMode("empty")
    setCustomAudioLanguage("")
    setTypeDefaults((previous) => ({
      ...(previous ?? {}),
      audio: {
        ...(previous?.audio ?? {}),
        language: undefined,
      },
    }))
  }, [setTypeDefaults])

  const handleCustomAudioLanguageChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const nextLanguage = event.target.value.trim()
      setCustomAudioLanguage(nextLanguage)
      setTypeDefaults((previous) => ({
        ...(previous ?? {}),
        audio: {
          ...(previous?.audio ?? {}),
          language: nextLanguage || undefined,
        },
      }))
    },
    [setTypeDefaults]
  )

  const handleAudioDiarizeChange = React.useCallback(
    (checked: boolean) => {
      setTypeDefaults((previous) => ({
        ...(previous ?? {}),
        audio: {
          ...(previous?.audio ?? {}),
          diarize: checked,
        },
      }))
    },
    [setTypeDefaults]
  )

  const handleDocumentOcrChange = React.useCallback(
    (checked: boolean) => {
      setTypeDefaults((previous) => ({
        ...(previous ?? {}),
        document: {
          ...(previous?.document ?? {}),
          ocr: checked,
        },
      }))
    },
    [setTypeDefaults]
  )

  const handleVideoCaptionsChange = React.useCallback(
    (checked: boolean) => {
      setTypeDefaults((previous) => ({
        ...(previous ?? {}),
        video: {
          ...(previous?.video ?? {}),
          captions: checked,
        },
      }))
    },
    [setTypeDefaults]
  )

  const handleStoreRemoteChange = React.useCallback(
    (checked: boolean) => {
      setCustomOptions({ storeRemote: checked })
    },
    [setCustomOptions]
  )

  const handleReviewBeforeStorageChange = React.useCallback(
    (checked: boolean) => {
      setCustomOptions({
        reviewBeforeStorage: checked,
        ...(checked ? { storeRemote: true } : {}),
      })
    },
    [setCustomOptions]
  )

  const handleTranscriptionModelChange = React.useCallback(
    (nextValue?: string) => {
      const nextModel = typeof nextValue === "string" ? nextValue.trim() : ""
      setCustomOptions({
        advancedValues: {
          transcription_model: nextModel || undefined,
        },
      })
    },
    [setCustomOptions]
  )

  const handleTranscriptionModelClear = React.useCallback(() => {
    setCustomOptions({
      advancedValues: {
        transcription_model: undefined,
      },
    })
  }, [setCustomOptions])

  const setAdvancedValue = React.useCallback(
    (name: string, value: unknown) => {
      setCustomOptions({
        advancedValues: {
          [name]: value,
        },
      })
    },
    [setCustomOptions]
  )

  const handleManualChunkMethodChange = React.useCallback(
    (nextMethod: string) => {
      setAdvancedValue("chunk_method", nextMethod || undefined)
    },
    [setAdvancedValue]
  )

  const handleManualChunkNumberChange = React.useCallback(
    (name: "chunk_size" | "chunk_overlap") =>
      (event: React.ChangeEvent<HTMLInputElement>) => {
        const rawValue = event.target.value.trim()
        if (!rawValue) {
          setAdvancedValue(name, undefined)
          return
        }
        const numericValue = Number.parseInt(rawValue, 10)
        const clampedValue =
          name === "chunk_size"
            ? Math.max(1, numericValue)
            : Math.max(0, numericValue)
        setAdvancedValue(
          name,
          Number.isFinite(numericValue) ? clampedValue : undefined
        )
      },
    [setAdvancedValue]
  )

  const storageLabel = presetConfig.reviewBeforeStorage
    ? qi("reviewModeStorageLabel", "Review drafts locally")
    : presetConfig.storeRemote
      ? qi("storageServerLabel", "Server")
      : qi("storageLocalLabel", "Local only")

  React.useEffect(() => {
    if (presetConfig.reviewBeforeStorage && !presetConfig.storeRemote) {
      setCustomOptions({ storeRemote: true })
    }
  }, [presetConfig.reviewBeforeStorage, presetConfig.storeRemote, setCustomOptions])

  return (
    <div className="space-y-5 py-3">
      <PresetSelector
        qi={qi}
        value={selectedPreset}
        onChange={setPreset}
        queueItems={queueItems}
      />

      <div className="rounded-md border border-border bg-surface p-4">
        <div className="space-y-4">
          <div>
            <Typography.Title level={5} className="!mb-1">
              {t("quickIngest.commonOptions") || "Ingestion options"}
            </Typography.Title>
            <Typography.Text type="secondary" className="text-xs text-text-subtle">
              {qi(
                "defaultsForNewItems",
                "Defaults apply to items added after this point."
              )}
            </Typography.Text>
          </div>

          <div className="flex flex-wrap gap-4">
            <Tooltip
              title={qi(
                "analysisTooltip",
                "Generate AI summary and analysis of content"
              )}
            >
              <div className="flex flex-col">
                <label className="flex items-center gap-2 text-sm text-text">
                  <span>{qi("analysisLabel", "Analysis")}</span>
                  <Switch
                    aria-label="Ingestion options – analysis"
                    checked={presetConfig.common.perform_analysis}
                    onChange={handleAnalysisToggle}
                  />
                </label>
                <p className="text-xs text-text-muted">
                  {qi("configure.analysisHint", "Generate AI summary and key findings")}
                </p>
              </div>
            </Tooltip>

            <Tooltip
              title={qi(
                "chunkingTooltip",
                "Split content into chunks for RAG retrieval"
              )}
            >
              <div className="flex flex-col">
                <label className="flex items-center gap-2 text-sm text-text">
                  <span>{qi("chunkingLabel", "Chunking")}</span>
                  <Switch
                    aria-label="Ingestion options – chunking"
                    checked={presetConfig.common.perform_chunking}
                    onChange={handleChunkingToggle}
                  />
                </label>
                <p className="text-xs text-text-muted">
                  {qi("configure.chunkingHint", "Split into searchable sections for Knowledge QA")}
                </p>
              </div>
            </Tooltip>

            <Tooltip
              title={qi(
                "overwriteTooltip",
                "Replace existing content if URL was previously ingested"
              )}
            >
              <div className="flex flex-col">
                <label className="flex items-center gap-2 text-sm text-text">
                  <span>{qi("overwriteLabel", "Overwrite existing")}</span>
                  <Switch
                    aria-label="Ingestion options – overwrite existing"
                    checked={presetConfig.common.overwrite_existing}
                    onChange={handleOverwriteToggle}
                  />
                </label>
                <p className="text-xs text-text-muted">
                  {qi("configure.overwriteHint", "Replace content if previously ingested")}
                </p>
              </div>
            </Tooltip>
          </div>

          {presetConfig.common.perform_chunking && (
            <div className="rounded-md border border-border bg-surface2 p-3">
              <div className="flex flex-col gap-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <Typography.Text strong>
                    {qi("chunkingModeLabel", "Chunking mode")}
                  </Typography.Text>
                  <Segmented
                    aria-label={qi("chunkingModeLabel", "Chunking mode")}
                    options={CHUNKING_MODE_OPTIONS}
                    value={chunkingMode}
                    onChange={handleChunkingModeChange}
                  />
                </div>

                {chunkingMode === "auto" ? (
                  <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
                    <label className="flex flex-col gap-1 text-sm text-text">
                      <span>{qi("autoChunkingGoalLabel", "Auto chunking goal")}</span>
                      <Select
                        aria-label={qi("autoChunkingGoalLabel", "Auto chunking goal")}
                        value={autoChunkingGoal}
                        onChange={handleAutoChunkingGoalChange}
                        options={AUTO_CHUNKING_GOAL_OPTIONS}
                      />
                    </label>
                    <label className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2 text-sm text-text">
                      <span>
                        {qi(
                          "autoChunkingUseLlmLabel",
                          "Use AI to improve chunk boundaries"
                        )}
                      </span>
                      <Switch
                        aria-label={qi(
                          "autoChunkingUseLlmLabel",
                          "Use AI to improve chunk boundaries"
                        )}
                        checked={presetConfig.common.auto_chunking_use_llm === true}
                        onChange={handleAutoChunkingUseLlmChange}
                      />
                    </label>
                  </div>
                ) : (
                  <div className="grid gap-3 md:grid-cols-3">
                    <label className="flex flex-col gap-1 text-sm text-text">
                      <span>{qi("chunkMethodLabel", "Chunk method")}</span>
                      <Select
                        aria-label={qi("chunkMethodLabel", "Chunk method")}
                        value={manualChunkMethod}
                        onChange={handleManualChunkMethodChange}
                        options={CHUNK_METHOD_OPTIONS}
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-sm text-text">
                      <span>{qi("chunkSizeLabel", "Chunk size")}</span>
                      <Input
                        aria-label={qi("chunkSizeLabel", "Chunk size")}
                        type="number"
                        min={1}
                        value={manualChunkSize}
                        onChange={handleManualChunkNumberChange("chunk_size")}
                      />
                    </label>
                    <label className="flex flex-col gap-1 text-sm text-text">
                      <span>{qi("chunkOverlapLabel", "Chunk overlap")}</span>
                      <Input
                        aria-label={qi("chunkOverlapLabel", "Chunk overlap")}
                        type="number"
                        min={0}
                        value={manualChunkOverlap}
                        onChange={handleManualChunkNumberChange("chunk_overlap")}
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Advanced options toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(v => !v)}
            className="mt-3 flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium text-text-muted hover:text-text hover:bg-surface2 transition-colors"
            aria-expanded={showAdvanced}
          >
            <ChevronRight
              className={`h-3.5 w-3.5 transition-transform duration-150 ${showAdvanced ? "rotate-90" : ""}`}
              aria-hidden="true"
            />
            {showAdvanced
              ? qi("configure.hideAdvanced", "Hide advanced options")
              : qi("configure.showAdvanced", "Advanced options")}
          </button>

          {showAdvanced && (
            <div className="mt-3 space-y-4">
              <div className={`space-y-2 ${!hasTranscriptionItems ? "opacity-50" : ""}`}>
                <Typography.Title level={5} className="!mb-1">
                  {t("quickIngest.audioOptions") || "Audio options"}
                  {!hasTranscriptionItems && (
                    <span className="ml-2 text-xs font-normal text-text-muted">
                      {qi("audioOptionsDisabled", "(add audio or video to enable)")}
                    </span>
                  )}
                </Typography.Title>
                <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
                  <div className="space-y-2">
                    <Select
                      aria-label="Audio language"
                      title="Audio language"
                      placeholder={qi(
                        "audioLanguagePlaceholder",
                        "Select language"
                      )}
                      value={audioLanguageSelectValue}
                      allowClear
                      onClear={handleAudioLanguageClear}
                      onChange={handleAudioLanguageOptionChange}
                      options={[
                        ...SUPPORTED_LANGUAGES,
                        {
                          value: CUSTOM_AUDIO_LANGUAGE_SENTINEL,
                          label: qi("audioLanguageCustomLabel", "Custom"),
                        },
                      ]}
                      disabled={!hasTranscriptionItems}
                    />
                    {shouldShowCustomAudioLanguageInput && (
                      <Input
                        aria-label="Custom audio language"
                        title="Custom audio language"
                        placeholder={qi(
                          "audioCustomLanguagePlaceholder",
                          "Custom language (e.g., en-US)"
                        )}
                        value={customAudioLanguage}
                        onChange={handleCustomAudioLanguageChange}
                        disabled={!hasTranscriptionItems}
                      />
                    )}
                  </div>
                  <label className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2 text-sm text-text">
                    <span>{qi("audioDiarizationLabel", "Diarization")}</span>
                    <Switch
                      aria-label="Audio diarization toggle"
                      checked={presetConfig.typeDefaults.audio?.diarize ?? false}
                      onChange={handleAudioDiarizeChange}
                      disabled={!hasTranscriptionItems}
                    />
                  </label>
                </div>
                <Select
                  aria-label={qi("transcriptionModelLabel", "Transcription model")}
                  title={qi("transcriptionModelLabel", "Transcription model")}
                  placeholder={qi("transcriptionModelPlaceholder", "Select model")}
                  value={normalizedTranscriptionModel || undefined}
                  allowClear
                  showSearch
                  loading={transcriptionModelsLoading}
                  popupMatchSelectWidth={false}
                  styles={{
                    popup: {
                      root: {
                        width: "max-content",
                        maxWidth: "min(90vw, 960px)",
                      },
                    },
                  }}
                  onChange={handleTranscriptionModelChange}
                  onClear={handleTranscriptionModelClear}
                  options={transcriptionModelOptions}
                  disabled={!hasTranscriptionItems}
                />
              </div>

              <div className={`space-y-2 ${!hasDocumentItems ? "opacity-50" : ""}`}>
                <Typography.Title level={5} className="!mb-1">
                  {t("quickIngest.documentOptions") || "Document options"}
                  {!hasDocumentItems && (
                    <span className="ml-2 text-xs font-normal text-text-muted">
                      {qi("documentOptionsDisabled", "(add document to enable)")}
                    </span>
                  )}
                </Typography.Title>
                <label className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2 text-sm text-text">
                  <span>{qi("ocrLabel", "OCR")}</span>
                  <Switch
                    aria-label="OCR toggle"
                    title="OCR toggle"
                    checked={presetConfig.typeDefaults.document?.ocr ?? false}
                    onChange={handleDocumentOcrChange}
                    disabled={!hasDocumentItems}
                  />
                </label>
              </div>

              <div className={`space-y-2 ${!hasVideoItems ? "opacity-50" : ""}`}>
                <Typography.Title level={5} className="!mb-1">
                  {t("quickIngest.videoOptions") || "Video options"}
                  {!hasVideoItems && (
                    <span className="ml-2 text-xs font-normal text-text-muted">
                      {qi("videoOptionsDisabled", "(add video to enable)")}
                    </span>
                  )}
                </Typography.Title>
                <label className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2 text-sm text-text">
                  <span>{qi("captionsLabel", "Captions")}</span>
                  <Switch
                    aria-label="Captions toggle"
                    title="Captions toggle"
                    checked={presetConfig.typeDefaults.video?.captions ?? false}
                    onChange={handleVideoCaptionsChange}
                    disabled={!hasVideoItems}
                  />
                </label>
              </div>

              <div className="rounded-md border border-border bg-surface2 p-3">
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <Typography.Text strong>
                        {t(
                          "quickIngest.storageHeading",
                          "Where ingest results are stored"
                        )}
                      </Typography.Text>
                      <div className="mt-2 space-y-1 text-xs text-text-muted">
                        <div className="flex items-start gap-2">
                          <span className="mt-[2px]">•</span>
                          <span>
                            {t(
                              "quickIngest.storageServerDescription",
                              "Stored on your tldw server (recommended for RAG and shared workspaces)."
                            )}
                          </span>
                        </div>
                        <div className="flex items-start gap-2">
                          <span className="mt-[2px]">•</span>
                          <span>
                            {t(
                              "quickIngest.storageLocalDescription",
                              "Kept in this browser only; no data written to your server."
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                    <label className="flex items-center gap-2 text-sm text-text">
                      <Switch
                        aria-label={
                          presetConfig.storeRemote
                            ? t(
                                "quickIngest.storeRemoteAria",
                                "Store ingest results on your tldw server"
                              )
                            : t(
                                "quickIngest.processOnlyAria",
                                "Process ingest results locally only"
                              )
                        }
                        checked={presetConfig.storeRemote}
                        disabled={presetConfig.reviewBeforeStorage}
                        onChange={handleStoreRemoteChange}
                      />
                      <span>{storageLabel}</span>
                    </label>
                  </div>

                  <div className="border-t border-border pt-3 text-xs text-text-muted">
                    <div className="flex items-start justify-between gap-3">
                      <label className="flex items-center gap-2 text-sm text-text">
                        <Switch
                          aria-label={qi(
                            "reviewBeforeStorage",
                            "Review before saving"
                          )}
                          checked={presetConfig.reviewBeforeStorage}
                          onChange={handleReviewBeforeStorageChange}
                        />
                        <span>{qi("reviewBeforeStorage", "Review before saving")}</span>
                      </label>
                      {presetConfig.reviewBeforeStorage ? (
                        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                          {qi("reviewEnabled", "Review mode")}
                        </span>
                      ) : null}
                    </div>
                    <div className="mt-2 flex items-start gap-2">
                      <span className="mt-[2px]">•</span>
                      <span>
                        {qi(
                          "reviewBeforeStorageHint",
                          "Process now, then edit drafts locally before committing to your server."
                        )}
                      </span>
                    </div>
                    <div className="mt-1 flex items-start gap-2">
                      <span className="mt-[2px]">•</span>
                      <span>
                        {qi("reviewStorageCap", "Local drafts are capped at {{cap}}.", {
                          cap: formatBytes(DRAFT_STORAGE_CAP_BYTES),
                        })}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between pt-2">
        <Button onClick={goBack}>
          <ArrowLeft className="mr-1 h-4 w-4" />
          {qi("wizard.back", "Back")}
        </Button>
        <Button type="primary" onClick={goNext}>
          {qi("wizard.next", "Next")}
          <ArrowRight className="ml-1 h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

export default WizardConfigureStep
