import React from "react"
import {
  Button,
  Input,
  Progress,
  Segmented,
  Select,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography
} from "antd"
import type { TFunction } from "i18next"
import { AlertTriangle } from "lucide-react"
import { useQuery } from "@tanstack/react-query"

import {
  listChunkingTemplates,
  type ChunkingTemplateListResponse
} from "@/services/chunking"

type CommonOptions = {
  perform_analysis: boolean
  perform_chunking: boolean
  overwrite_existing: boolean
  chunking_mode?: "auto" | "manual"
  auto_chunking_goal?: "balanced" | "qa_search" | "navigation_summary"
  auto_chunking_use_llm?: boolean
}

const CHUNKING_MODE_OPTIONS = [
  { label: "Auto", value: "auto" },
  { label: "Manual", value: "manual" },
]

const AUTO_CHUNKING_GOAL_OPTIONS = [
  { label: "Balanced", value: "balanced" },
  { label: "Search/Q&A", value: "qa_search" },
  { label: "Reading/Summary", value: "navigation_summary" },
]

type TypeDefaults = {
  audio?: { language?: string; diarize?: boolean }
  document?: { ocr?: boolean }
  video?: { captions?: boolean }
}

type ProgressMeta = {
  total: number
  done: number
  pct: number
  elapsedLabel?: string | null
  state?: "running" | "failed" | "complete" | "cancelled" | "ready"
  error?: string | null
}

type IngestOptionsPanelProps = {
  qi: (
    key: string,
    defaultValue: string,
    options?: Record<string, unknown>
  ) => string
  t: TFunction
  hasAudioItems: boolean
  hasDocumentItems: boolean
  hasVideoItems: boolean
  running: boolean
  ingestBlocked: boolean
  common: CommonOptions
  setCommon: React.Dispatch<React.SetStateAction<CommonOptions>>
  normalizedTypeDefaults: TypeDefaults
  setTypeDefaults: React.Dispatch<React.SetStateAction<TypeDefaults | null>>
  transcriptionModelOptions: string[]
  transcriptionModelsLoading: boolean
  transcriptionModelValue?: string
  onTranscriptionModelChange: (value?: string) => void
  ragEmbeddingLabel?: string | null
  openModelSettings: () => void
  storeRemote: boolean
  setStoreRemote: (value: boolean) => void
  reviewBeforeStorage: boolean
  handleReviewToggle: (value: boolean) => void
  storageLabel: string
  storageHintSeen: boolean
  setStorageHintSeen: (value: boolean) => void
  draftStorageCapLabel: string
  doneCount: number
  totalCount: number
  plannedCount: number
  progressMeta: ProgressMeta
  lastRunError?: string | null
  run: () => void
  hasMissingFiles: boolean
  missingFileCount: number
  ingestConnectionStatus:
    | "online"
    | "offline"
    | "unconfigured"
    | "unknown"
  checkOnce?: () => Promise<void> | void
  onClose: () => void
  // Chunking template props (optional for backwards compatibility)
  chunkingTemplateName?: string
  setChunkingTemplateName?: (value: string | undefined) => void
  autoApplyTemplate?: boolean
  setAutoApplyTemplate?: (value: boolean) => void
}

export const IngestOptionsPanel: React.FC<IngestOptionsPanelProps> = ({
  qi,
  t,
  hasAudioItems,
  hasDocumentItems,
  hasVideoItems,
  running,
  ingestBlocked,
  common,
  setCommon,
  normalizedTypeDefaults,
  setTypeDefaults,
  transcriptionModelOptions,
  transcriptionModelsLoading,
  transcriptionModelValue,
  onTranscriptionModelChange,
  ragEmbeddingLabel,
  openModelSettings,
  storeRemote,
  setStoreRemote,
  reviewBeforeStorage,
  handleReviewToggle,
  storageLabel,
  storageHintSeen,
  setStorageHintSeen,
  draftStorageCapLabel,
  doneCount,
  totalCount,
  plannedCount,
  progressMeta,
  lastRunError,
  run,
  hasMissingFiles,
  missingFileCount,
  ingestConnectionStatus,
  checkOnce,
  onClose,
  chunkingTemplateName,
  setChunkingTemplateName,
  autoApplyTemplate,
  setAutoApplyTemplate
}) => {
  const done = doneCount || 0
  const total = totalCount || 0
  const progressState =
    progressMeta?.state ||
    (running ? "running" : lastRunError ? "failed" : "ready")
  const getIngestBlockedLabel = () => {
    if (ingestConnectionStatus === "unconfigured") {
      return t(
        "quickIngest.unavailableUnconfigured",
        "Ingest unavailable \u2014 server not configured"
      )
    }
    if (ingestConnectionStatus === "unknown") {
      return t("quickIngest.checkingTitle", "Checking server connection\u2026")
    }
    return t(
      "quickIngest.unavailableOffline",
      "Ingest unavailable \u2014 not connected"
    )
  }
  const ingestBlockedLabel = getIngestBlockedLabel()
  const getPrimaryActionLabel = () => {
    if (ingestBlocked) return ingestBlockedLabel
    if (reviewBeforeStorage) return qi("reviewRunLabel", "Review")
    if (storeRemote) return t("quickIngest.ingest", "Ingest")
    return t("quickIngest.process", "Process")
  }
  const primaryActionLabel = getPrimaryActionLabel()
  const isIngestDisabled =
    running || plannedCount === 0 || ingestBlocked || hasMissingFiles
  const handleAnalysisToggle = React.useCallback(
    (value: boolean) => {
      setCommon((current) => ({ ...current, perform_analysis: value }))
    },
    [setCommon]
  )
  const handleChunkingToggle = React.useCallback(
    (value: boolean) => {
      setCommon((current) => ({
        ...current,
        perform_chunking: value,
        ...(value
          ? {
              chunking_mode: "auto",
              auto_chunking_goal: current.auto_chunking_goal ?? "balanced",
              auto_chunking_use_llm: current.auto_chunking_use_llm ?? false
            }
          : {})
      }))
    },
    [setCommon]
  )
  const chunkingMode = common.chunking_mode === "manual" ? "manual" : "auto"
  const autoChunkingGoal =
    common.auto_chunking_goal === "qa_search" ||
    common.auto_chunking_goal === "navigation_summary"
      ? common.auto_chunking_goal
      : "balanced"
  const handleChunkingModeChange = React.useCallback(
    (value: string | number) => {
      const mode = value === "manual" ? "manual" : "auto"
      setCommon((current) => ({
        ...current,
        chunking_mode: mode,
        auto_chunking_goal: current.auto_chunking_goal ?? "balanced",
        auto_chunking_use_llm: current.auto_chunking_use_llm ?? false
      }))
    },
    [setCommon]
  )
  const handleAutoChunkingGoalChange = React.useCallback(
    (value: string) => {
      const goal =
        value === "qa_search" || value === "navigation_summary"
          ? value
          : "balanced"
      setCommon((current) => ({ ...current, auto_chunking_goal: goal }))
    },
    [setCommon]
  )
  const handleAutoChunkingUseLlmChange = React.useCallback(
    (value: boolean) => {
      setCommon((current) => ({ ...current, auto_chunking_use_llm: value }))
    },
    [setCommon]
  )
  const handleOverwriteToggle = React.useCallback(
    (value: boolean) => {
      setCommon((current) => ({ ...current, overwrite_existing: value }))
    },
    [setCommon]
  )
  const handleAudioLanguageChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const nextValue = event.target.value
      const normalizedValue = nextValue === "" ? undefined : nextValue
      setTypeDefaults((prev) => {
        const nextAudio = { ...(prev?.audio || {}) }
        if (normalizedValue === undefined || normalizedValue === null) {
          delete nextAudio.language
        } else {
          nextAudio.language = normalizedValue
        }
        return { ...(prev || {}), audio: nextAudio }
      })
    },
    [setTypeDefaults]
  )
  const handleAudioDiarizeChange = React.useCallback(
    (value: boolean) => {
      setTypeDefaults((prev) => ({
        ...(prev || {}),
        audio: { ...(prev?.audio || {}), diarize: Boolean(value) }
      }))
    },
    [setTypeDefaults]
  )
  const handleDocumentOcrChange = React.useCallback(
    (value: boolean) => {
      setTypeDefaults((prev) => ({
        ...(prev || {}),
        document: { ...(prev?.document || {}), ocr: Boolean(value) }
      }))
    },
    [setTypeDefaults]
  )
  const handleVideoCaptionsChange = React.useCallback(
    (value: boolean) => {
      setTypeDefaults((prev) => ({
        ...(prev || {}),
        video: { ...(prev?.video || {}), captions: Boolean(value) }
      }))
    },
    [setTypeDefaults]
  )
  const handleStorageDocsClick = React.useCallback(() => {
    const defaultUrl = "https://github.com/rmusser01/tldw_browser_assistant"
    const allowedHostnames = new Set([
      "github.com",
      "docs.tldw.io",
      "readthedocs.io"
    ])
    const docsUrl =
      t("quickIngest.storageDocsUrl", defaultUrl) || defaultUrl
    let validatedUrl = defaultUrl
    if (docsUrl.startsWith("/")) {
      const resolvedUrl = new URL(docsUrl, window.location.origin)
      if (resolvedUrl.protocol === "http:" || resolvedUrl.protocol === "https:") {
        validatedUrl = resolvedUrl.toString()
      }
    } else {
      try {
        const parsedUrl = new URL(docsUrl)
        if (
          (parsedUrl.protocol === "http:" ||
            parsedUrl.protocol === "https:") &&
          allowedHostnames.has(parsedUrl.hostname)
        ) {
          validatedUrl = parsedUrl.toString()
        }
      } catch {
        // Invalid URL, fallback to defaultUrl
      }
    }
    try {
      window.open(validatedUrl, "_blank", "noopener,noreferrer")
    } catch (error) {
      console.warn("Failed to open docs link:", error)
    }
    setStorageHintSeen(true)
  }, [setStorageHintSeen, t])
  const handleCheckOnce = React.useCallback(async () => {
    try {
      await checkOnce?.()
    } catch {
      // ignore check errors; footer is informational
    }
  }, [checkOnce])
  const transcriptionOptions = React.useMemo(() => {
    const options = transcriptionModelOptions.map((value) => ({
      value,
      label: value
    }))
    if (
      transcriptionModelValue &&
      !transcriptionModelOptions.includes(transcriptionModelValue)
    ) {
      options.unshift({
        value: transcriptionModelValue,
        label: transcriptionModelValue
      })
    }
    return options
  }, [transcriptionModelOptions, transcriptionModelValue])
  const hasTranscriptionItems = hasAudioItems || hasVideoItems

  // Fetch chunking templates when chunking is enabled
  const {
    data: templateList,
    isLoading: templatesLoading
  } = useQuery<ChunkingTemplateListResponse>({
    queryKey: ["chunking-templates", "ingest-options"],
    queryFn: () =>
      listChunkingTemplates({ includeBuiltin: true, includeCustom: true }),
    staleTime: 60 * 1000,
    enabled: common.perform_chunking && !!setChunkingTemplateName
  })

  // Group templates into built-in and custom
  const templateOptions = React.useMemo(() => {
    if (!templateList?.templates) return []
    const builtinTemplates = templateList.templates.filter((t) => t.is_builtin)
    const customTemplates = templateList.templates.filter((t) => !t.is_builtin)

    const options: { label: string; options: { value: string; label: string; title?: string }[] }[] = []

    if (builtinTemplates.length > 0) {
      options.push({
        label: qi("chunkingTemplates.builtinGroup", "Built-in"),
        options: builtinTemplates.map((t) => ({
          value: t.name,
          label: t.name,
          title: t.description || undefined
        }))
      })
    }

    if (customTemplates.length > 0) {
      options.push({
        label: qi("chunkingTemplates.customGroup", "Custom"),
        options: customTemplates.map((t) => ({
          value: t.name,
          label: t.name,
          title: t.description || undefined
        }))
      })
    }

    return options
  }, [templateList, qi])

  const handleChunkingTemplateChange = React.useCallback(
    (value: string | undefined) => {
      setChunkingTemplateName?.(value)
    },
    [setChunkingTemplateName]
  )

  const handleAutoApplyChange = React.useCallback(
    (checked: boolean) => {
      setAutoApplyTemplate?.(checked)
      // Clear selected template when enabling auto-apply
      if (checked) {
        setChunkingTemplateName?.(undefined)
      }
    },
    [setAutoApplyTemplate, setChunkingTemplateName]
  )
  return (
    <div className="rounded-md border border-border bg-surface p-3 space-y-3">
      <Typography.Title level={5} className="!mb-2">
        {t("quickIngest.commonOptions") || "Ingestion options"}
      </Typography.Title>
      {(hasAudioItems || hasDocumentItems || hasVideoItems) && (
        <Typography.Text type="secondary" className="text-xs text-text-subtle">
          {qi(
            "defaultsForNewItems",
            "Defaults apply to items added after this point."
          )}
        </Typography.Text>
      )}
      <Space wrap size="middle" align="center">
        <Tooltip
          title={qi("analysisTooltip", "Generate AI summary and analysis of content")}
        >
          <Space align="center">
            <span>{qi("analysisLabel", "Analysis")}</span>
            <Switch
              aria-label="Ingestion options \u2013 analysis"
              checked={common.perform_analysis}
              onChange={handleAnalysisToggle}
              disabled={running}
            />
          </Space>
        </Tooltip>
        <Tooltip
          title={qi("chunkingTooltip", "Split content into chunks for RAG retrieval")}
        >
          <Space align="center">
            <span>{qi("chunkingLabel", "Chunking")}</span>
            <Switch
              aria-label="Ingestion options \u2013 chunking"
              checked={common.perform_chunking}
              onChange={handleChunkingToggle}
              disabled={running}
            />
          </Space>
        </Tooltip>
        <Tooltip
          title={qi("overwriteTooltip", "Replace existing content if URL was previously ingested")}
        >
          <Space align="center">
            <span>{qi("overwriteLabel", "Overwrite existing")}</span>
            <Switch
              aria-label="Ingestion options \u2013 overwrite existing"
              checked={common.overwrite_existing}
              onChange={handleOverwriteToggle}
              disabled={running}
            />
          </Space>
        </Tooltip>
      </Space>

      {common.perform_chunking && (
        <div className="mt-2 pt-2 border-t border-border space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-sm font-medium">
              {qi("chunkingModeLabel", "Chunking mode")}
            </span>
            <Segmented
              aria-label={qi("chunkingModeLabel", "Chunking mode")}
              options={CHUNKING_MODE_OPTIONS}
              value={chunkingMode}
              onChange={handleChunkingModeChange}
              disabled={running}
            />
          </div>
          {chunkingMode === "auto" ? (
            <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
              <label className="flex flex-col gap-1 text-sm text-text">
                <span>{qi("autoChunkingGoalLabel", "Auto chunking goal")}</span>
                <Select
                  aria-label={qi("autoChunkingGoalLabel", "Auto chunking goal")}
                  value={autoChunkingGoal}
                  onChange={handleAutoChunkingGoalChange}
                  options={AUTO_CHUNKING_GOAL_OPTIONS}
                  disabled={running}
                />
              </label>
              <Tooltip
                title={qi(
                  "autoChunkingUseLlmTooltip",
                  "Allow configured AI providers to refine chunk boundaries when available."
                )}
              >
                <Space align="center" size="small">
                  <Switch
                    checked={common.auto_chunking_use_llm === true}
                    onChange={handleAutoChunkingUseLlmChange}
                    disabled={running}
                    aria-label={qi(
                      "autoChunkingUseLlmLabel",
                      "Use AI to improve chunk boundaries"
                    )}
                  />
                  <span className="text-sm">
                    {qi(
                      "autoChunkingUseLlmLabel",
                      "Use AI to improve chunk boundaries"
                    )}
                  </span>
                </Space>
              </Tooltip>
            </div>
          ) : null}
        </div>
      )}

      {/* Chunking template selector - visible for manual chunking only */}
      {common.perform_chunking && chunkingMode === "manual" && setChunkingTemplateName && (
        <div className="mt-2 pt-2 border-t border-border space-y-2">
          <div className="flex items-center gap-2">
            <span className="min-w-32 text-sm">
              {qi("chunkingTemplates.label", "Chunking template")}
            </span>
            <Select
              className="flex-1 min-w-40"
              allowClear
              showSearch
              loading={templatesLoading}
              value={chunkingTemplateName || undefined}
              placeholder={qi(
                "chunkingTemplates.placeholder",
                "Select template (optional)"
              )}
              aria-label={qi("chunkingTemplates.label", "Chunking template")}
              onChange={handleChunkingTemplateChange}
              options={templateOptions}
              disabled={running || autoApplyTemplate}
              filterOption={(input, option) =>
                (option?.label ?? "")
                  .toString()
                  .toLowerCase()
                  .includes(input.toLowerCase())
              }
            />
          </div>
          {setAutoApplyTemplate && (
            <div className="flex items-center gap-2">
              <Tooltip
                title={qi(
                  "chunkingTemplates.autoApplyTooltip",
                  "Automatically select the best matching template based on content type and patterns"
                )}
              >
                <Space align="center" size="small">
                  <Switch
                    checked={autoApplyTemplate}
                    onChange={handleAutoApplyChange}
                    disabled={running}
                    aria-label={qi(
                      "chunkingTemplates.autoApplyLabel",
                      "Auto-detect template"
                    )}
                  />
                  <span className="text-sm">
                    {qi("chunkingTemplates.autoApplyLabel", "Auto-detect template")}
                  </span>
                </Space>
              </Tooltip>
            </div>
          )}
          <Typography.Text type="secondary" className="text-xs block">
            {autoApplyTemplate
              ? qi(
                  "chunkingTemplates.autoApplyHint",
                  "Template will be selected automatically based on content type."
                )
              : qi(
                  "chunkingTemplates.hint",
                  "Apply a saved template's chunking settings. Leave empty to use defaults."
                )}
          </Typography.Text>
        </div>
      )}

      {ragEmbeddingLabel && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-text-subtle">
          <span>
            {t(
              "quickIngest.ragEmbeddingHintInline",
              "Uses {{label}} for RAG search.",
              { label: ragEmbeddingLabel }
            )}
          </span>
          <button
            type="button"
            onClick={openModelSettings}
            className="text-primary underline underline-offset-2"
          >
            {t("option:header.modelSettings", "Model settings")}
          </button>
        </div>
      )}

      <div className={`space-y-1 ${!hasTranscriptionItems ? 'opacity-50' : ''}`}>
        <Typography.Title level={5} className="!mb-1">
          {t("quickIngest.audioOptions") || "Audio options"}
          {!hasTranscriptionItems && (
            <span className="ml-2 text-xs font-normal text-text-muted">
              {qi("audioOptionsDisabled", "(add audio to enable)")}
            </span>
          )}
        </Typography.Title>
        <Space className="w-full">
          <Input
            placeholder={t("quickIngest.audioLanguage") || "Language (e.g., en)"}
            value={normalizedTypeDefaults.audio?.language || ""}
            onChange={handleAudioLanguageChange}
            disabled={running || !hasTranscriptionItems}
            aria-label="Audio language"
            title="Audio language"
          />
          <Select
            className="min-w-40"
            value={normalizedTypeDefaults.audio?.diarize ?? false}
            onChange={handleAudioDiarizeChange}
            aria-label="Audio diarization toggle"
            title="Audio diarization toggle"
            options={[
              {
                label: qi("audioDiarizationOff", "Diarization: Off"),
                value: false
              },
              {
                label: qi("audioDiarizationOn", "Diarization: On"),
                value: true
              }
            ]}
            disabled={running || !hasTranscriptionItems}
          />
        </Space>
        <div className="flex items-center gap-2">
          <span className="min-w-40 text-sm">
            {qi("transcriptionModelLabel", "Transcription model")}
          </span>
          <Select
            className="min-w-60"
            allowClear
            showSearch
            loading={transcriptionModelsLoading}
            value={transcriptionModelValue}
            placeholder={qi("transcriptionModelPlaceholder", "Select model")}
            aria-label={qi("transcriptionModelLabel", "Transcription model")}
            onChange={(value) => onTranscriptionModelChange(value ?? undefined)}
            options={transcriptionOptions}
            disabled={running || !hasTranscriptionItems}
          />
        </div>
        {hasTranscriptionItems && (
          <>
            <Typography.Text type="secondary" className="text-xs">
              {t("quickIngest.audioDiarizationHelp") ||
                "Turn on to separate speakers in transcripts; applies to new audio items added after this point."}
            </Typography.Text>
            <Typography.Text
              className="text-[11px] text-text-subtle block"
              title={qi(
                "audioSettingsTitle",
                "These audio settings apply to new audio items added after this point."
              )}
            >
              {qi(
                "audioSettingsHint",
                "These settings apply to new audio items added after this point."
              )}
            </Typography.Text>
          </>
        )}
      </div>

      <div className={`space-y-1 ${!hasDocumentItems ? 'opacity-50' : ''}`}>
        <Typography.Title level={5} className="!mb-1">
          {t("quickIngest.documentOptions") || "Document options"}
          {!hasDocumentItems && (
            <span className="ml-2 text-xs font-normal text-text-muted">
              {qi("documentOptionsDisabled", "(add document to enable)")}
            </span>
          )}
        </Typography.Title>
        <Select
          className="min-w-40"
          value={normalizedTypeDefaults.document?.ocr ?? true}
          onChange={handleDocumentOcrChange}
          aria-label="OCR toggle"
          title="OCR toggle"
          options={[
            { label: qi("ocrOff", "OCR: Off"), value: false },
            { label: qi("ocrOn", "OCR: On"), value: true }
          ]}
          disabled={running || !hasDocumentItems}
        />
        {hasDocumentItems && (
          <>
            <Typography.Text type="secondary" className="text-xs">
              {t("quickIngest.ocrHelp") ||
                "OCR helps extract text from scanned PDFs or images; applies to new document/PDF items added after this point."}
            </Typography.Text>
            <Typography.Text
              className="text-[11px] text-text-subtle block"
              title={qi(
                "documentSettingsTitle",
                "These document settings apply to new document/PDF items added after this point."
              )}
            >
              {qi(
                "documentSettingsHint",
                "Applies to new document/PDF items added after this point."
              )}
            </Typography.Text>
          </>
        )}
      </div>

      <div className={`space-y-1 ${!hasVideoItems ? 'opacity-50' : ''}`}>
        <Typography.Title level={5} className="!mb-1">
          {t("quickIngest.videoOptions") || "Video options"}
          {!hasVideoItems && (
            <span className="ml-2 text-xs font-normal text-text-muted">
              {qi("videoOptionsDisabled", "(add video to enable)")}
            </span>
          )}
        </Typography.Title>
        <Select
          className="min-w-40"
          value={normalizedTypeDefaults.video?.captions ?? false}
          onChange={handleVideoCaptionsChange}
          aria-label="Captions toggle"
          title="Captions toggle"
          options={[
            { label: qi("captionsOff", "Captions: Off"), value: false },
            { label: qi("captionsOn", "Captions: On"), value: true }
          ]}
          disabled={running || !hasVideoItems}
        />
        {hasVideoItems && (
          <>
            <Typography.Text type="secondary" className="text-xs">
              {t("quickIngest.captionsHelp") ||
                "Include timestamps/captions for new video items added after this point; helpful for search and summaries."}
            </Typography.Text>
            <Typography.Text
              className="text-[11px] text-text-subtle block"
              title={qi(
                "videoSettingsTitle",
                "These video settings apply to new video items added after this point."
              )}
            >
              {qi(
                "videoSettingsHint",
                "Applies to new video items added after this point."
              )}
            </Typography.Text>
          </>
        )}
      </div>

      <div className="rounded-md border border-border bg-surface2 p-3">
        <div className="flex flex-col gap-2">
          <div className="sr-only" aria-live="polite" role="status">
            {progressState === "running" && total > 0
              ? t(
                  "quickIngest.progress",
                  "Processing {{done}} / {{total}} items\u2026",
                  {
                    done,
                    total
                  }
                )
              : progressState === "failed" && total > 0
                ? qi(
                    "progressFailed",
                    "Last run failed after {{done}} / {{total}} items.",
                    {
                      done,
                      total
                    }
                  )
              : qi("itemsReadySr", "{{count}} item(s) ready", {
                  count: plannedCount || 0
                })}
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between text-sm text-text">
            <div className="flex-1">
              <div className="rounded-md border border-border bg-surface2 p-3">
                <div className="flex flex-col gap-2">
                  <div className="flex items-start justify-between gap-2">
                    <Typography.Text strong>
                      {t(
                        "quickIngest.storageHeading",
                        "Where ingest results are stored"
                      )}
                    </Typography.Text>
                    <Space align="center" size="small">
                      <Tooltip
                        title={
                          reviewBeforeStorage
                            ? qi(
                                "reviewModeStorageDisabled",
                                "Storage location is locked when review mode is enabled"
                              )
                            : qi(
                                "storageToggleTooltip",
                                "Store on server for RAG search, or process locally for one-time use"
                              )
                        }
                      >
                        <span>
                          <Switch
                            aria-label={
                              storeRemote
                                ? t(
                                    "quickIngest.storeRemoteAria",
                                    "Store ingest results on your tldw server"
                                  )
                                : t(
                                    "quickIngest.processOnlyAria",
                                    "Process ingest results locally only"
                                  )
                            }
                            checked={storeRemote}
                            onChange={setStoreRemote}
                            disabled={running || reviewBeforeStorage}
                          />
                        </span>
                      </Tooltip>
                      <Typography.Text>{storageLabel}</Typography.Text>
                    </Space>
                  </div>
                  <div className="mt-1 space-y-1 text-xs text-text-muted">
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
                    {!storageHintSeen && (
                      <div className="pt-1">
                        <button
                          type="button"
                          className="text-xs underline text-primary hover:text-primaryStrong"
                          onClick={handleStorageDocsClick}
                        >
                          {t(
                            "quickIngest.storageDocsLink",
                            "Learn more about ingest & storage"
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="mt-3 border-t border-border pt-3 text-xs text-text-muted">
                    <div className="flex items-start justify-between gap-2">
                      <Tooltip
                        title={qi(
                          "reviewToggleTooltip",
                          "Save drafts locally to edit before committing to server"
                        )}
                      >
                        <Space align="center" size="small">
                          <Switch
                            aria-label={qi(
                              "reviewBeforeStorage",
                              "Review before saving"
                            )}
                            checked={reviewBeforeStorage}
                            onChange={handleReviewToggle}
                            disabled={running}
                          />
                          <Typography.Text>
                            {qi("reviewBeforeStorage", "Review before saving")}
                          </Typography.Text>
                        </Space>
                      </Tooltip>
                      {reviewBeforeStorage ? (
                        <Tag color="blue">
                          {qi("reviewEnabled", "Review mode")}
                        </Tag>
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
                          cap: draftStorageCapLabel
                        })}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <span
              className="mt-2 text-xs text-text-subtle sm:mt-0"
              title={
                progressState === "running" && total > 0
                  ? qi("ingestProgressTitle", "Current ingest progress")
                  : progressState === "failed" && total > 0
                    ? qi("ingestFailedTitle", "Last ingest run failed")
                  : qi("itemsReadyTitle", "Items ready to ingest")
              }
            >
              {progressState === "running" && total > 0
                ? t(
                    "quickIngest.progress",
                    "Processing {{done}} / {{total}} items\u2026",
                    {
                      done,
                      total
                    }
                  )
                : progressState === "failed" && total > 0
                  ? qi(
                      "progressFailed",
                      "Last run failed after {{done}} / {{total}} items.",
                      {
                        done,
                        total
                      }
                    )
                : qi("itemsReady", "{{count}} item(s) ready", {
                    count: plannedCount || 0
                  })}
            </span>
          </div>
          {progressState === "failed" && lastRunError ? (
            <div className="text-xs text-danger">{lastRunError}</div>
          ) : null}
        </div>
        <div className="flex justify-end gap-2 mt-2">
          <Button
            type="primary"
            loading={running}
            onClick={run}
            disabled={isIngestDisabled}
            aria-label={primaryActionLabel}
            title={primaryActionLabel}
          >
            {primaryActionLabel}
          </Button>
          <Button
            onClick={onClose}
            disabled={running}
            aria-label={qi("closeQuickIngest", "Close quick ingest")}
            title={qi("closeQuickIngest", "Close quick ingest")}
          >
            {t("quickIngest.cancel") || "Cancel"}
          </Button>
        </div>
        {hasMissingFiles && (
          <div className="mt-1 flex items-center gap-2 text-xs text-warn">
            <AlertTriangle className="h-4 w-4" />
            <span>
              {qi("missingFilesBlock", "Reattach {{count}} local file(s) to run ingest.", {
                count: missingFileCount
              })}
            </span>
          </div>
        )}
        {ingestBlocked && (
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-warn">
            <span>
              {ingestConnectionStatus === "unconfigured"
                ? t(
                    "quickIngest.unconfiguredFooter",
                    "Server not configured. Configure your server URL and API key under Settings \u2192 tldw server to run ingest."
                    )
                : ingestConnectionStatus === "unknown"
                  ? t(
                      "quickIngest.checkingDescription",
                      "Checking your tldw server before running ingest. Inputs are disabled until we confirm the connection."
                    )
                : t(
                    "quickIngest.offlineFooter",
                    "Cannot reach your server. Connect to run ingest."
                  )}
            </span>
            {ingestConnectionStatus === "offline" && checkOnce ? (
              <Button
                size="small"
                onClick={handleCheckOnce}
              >
                {qi("retryConnection", "Retry connection")}
              </Button>
            ) : null}
          </div>
        )}
        {progressMeta.total > 0 && (
          <div className="mt-2">
            <Progress percent={progressMeta.pct} showInfo={false} size="small" />
            <div className="flex justify-between text-xs text-text-muted mt-1">
              <span>
                {qi("processedCount", "{{done}}/{{total}} processed", {
                  done: progressMeta.done,
                  total: progressMeta.total
                })}
              </span>
              {progressMeta.elapsedLabel ? (
                <span>
                  {qi("elapsedLabel", "Elapsed {{time}}", {
                    time: progressMeta.elapsedLabel
                  })}
                </span>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default IngestOptionsPanel
