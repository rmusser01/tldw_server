import React, { useCallback, useEffect, useRef, useState } from "react"
import { Button, Card, Input, Select, Skeleton, Tag, Tooltip, Typography } from "antd"
import { Ban, Copy, CopyPlus, RotateCcw, Save } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useComparisonTranscribe } from "@/hooks/useComparisonTranscribe"
import type { ComparisonResult } from "@/hooks/useComparisonTranscribe"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import type { SttModelOption } from "@/components/Option/Audio/audio-readiness"
import {
  formatByteSize,
  formatClientLatency,
  formatCreatedAt
} from "@/components/Option/Audio/comparison-provenance"

const { Text } = Typography

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ComparisonPanelProps {
  blob: Blob | null
  availableModels: string[]
  availableModelOptions?: SttModelOption[]
  selectedModels?: string[]
  onSelectedModelsChange?: (models: string[]) => void
  sttOptions: Record<string, any>
  onSaveToNotes: (text: string, model: string) => void
  onComparisonComplete?: (results: ComparisonResult[]) => void
}

// ---------------------------------------------------------------------------
// ResultCard sub-component
// ---------------------------------------------------------------------------

interface ResultCardProps {
  result: ComparisonResult
  onCopy: (text: string) => void
  onRetry: (target: string) => void
  onSave: (text: string, model: string) => void
  onDuplicate: (target: string) => void
  onToggleDisabled: (target: string, disabled: boolean) => void
}

const ResultCard: React.FC<ResultCardProps> = ({
  result,
  onCopy,
  onRetry,
  onSave,
  onDuplicate,
  onToggleDisabled,
}) => {
  const { t } = useTranslation("playground")
  const [editedText, setEditedText] = useState(result.text)

  useEffect(() => {
    setEditedText(result.text)
  }, [result.text])

  const isPending = result.status === "pending" || result.status === "running"
  const resultTarget = result.id || result.model
  const metadata = result.metadata
  const config = result.config
  const wordCount =
    metadata?.wordCount ??
    result.wordCount ??
    editedText.split(/\s+/).filter(Boolean).length
  const provenanceTags = [
    metadata?.createdAt ? formatCreatedAt(metadata.createdAt) : undefined,
    metadata?.audioSourceLabel,
    formatByteSize(metadata?.audioSizeBytes),
    formatClientLatency(metadata?.clientLatencyMs ?? result.latencyMs),
    metadata?.language || config?.language
      ? `Language ${metadata?.language || config?.language}`
      : undefined,
    config?.task ? `Task ${config.task}` : undefined,
    config?.responseFormat ? `Format ${config.responseFormat}` : undefined,
    config?.timestampGranularities?.length
      ? `Timestamps ${config.timestampGranularities.join(", ")}`
      : undefined,
    config?.segmentationEnabled ? "Segmentation on" : undefined,
    config?.diarizationRequested ? "Diarization requested" : undefined,
    metadata?.durationSeconds != null
      ? `Duration ${metadata.durationSeconds.toFixed(1)}s`
      : undefined,
    metadata?.segmentCount != null
      ? `${metadata.segmentCount} ${
          metadata.segmentCount === 1 ? "segment" : "segments"
        }`
      : undefined,
    `${wordCount} ${t("stt.comparison.words", "words")}`,
    result.disabled ? "Disabled for Run All" : undefined
  ].filter((tag): tag is string => Boolean(tag))

  return (
    <Card
      size="small"
      title={<Text strong>{result.model}</Text>}
      role="region"
      aria-label={`Transcription result from ${result.model}`}
    >
      {isPending && (
        <Skeleton active paragraph={{ rows: 3 }} />
      )}

      {result.status === "error" && (
        <div className="space-y-2">
          <Text type="danger">
            {result.error || t("stt.comparison.unknownError", "Transcription failed")}
          </Text>
          {result.errorRecovery && (
            <Text type="secondary" className="block text-xs">
              {result.errorRecovery}
            </Text>
          )}
          <div className="flex flex-wrap gap-2">
            {result.errorSettingsHref && (
              <Button size="small" type="link" href={result.errorSettingsHref}>
                Open Settings
              </Button>
            )}
            <Button
              size="small"
              icon={<RotateCcw className="h-3.5 w-3.5" />}
              onClick={() => onRetry(resultTarget)}
            >
              {t("stt.comparison.retry", "Retry")}
            </Button>
          </div>
        </div>
      )}

      {result.status === "done" && (
        <div className="space-y-2">
          <Input.TextArea
            value={editedText}
            onChange={(e) => setEditedText(e.target.value)}
            autoSize={{ minRows: 3, maxRows: 8 }}
            aria-live="polite"
            aria-label={`Transcript from ${result.model}`}
          />
          <div className="flex flex-wrap items-center gap-2">
            {provenanceTags.map((tag) => (
              <Tag key={tag} bordered>
                {tag}
              </Tag>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Tooltip title={t("stt.comparison.copyTooltip", "Copy to clipboard")}>
              <Button
                size="small"
                icon={<Copy className="h-3.5 w-3.5" />}
                onClick={() => onCopy(editedText)}
              >
                {t("stt.comparison.copy", "Copy")}
              </Button>
            </Tooltip>
            <Tooltip title={t("stt.comparison.saveTooltip", "Save to Notes")}>
              <Button
                size="small"
                icon={<Save className="h-3.5 w-3.5" />}
                onClick={() => onSave(editedText, result.model)}
              >
                {t("stt.comparison.saveToNotes", "Save to Notes")}
              </Button>
            </Tooltip>
            <Tooltip
              title={t(
                "stt.comparison.duplicateTooltip",
                "Duplicate this row with the same transcription settings"
              )}
            >
              <Button
                size="small"
                icon={<CopyPlus className="h-3.5 w-3.5" />}
                onClick={() => onDuplicate(resultTarget)}
              >
                {t("stt.comparison.duplicate", "Duplicate")}
              </Button>
            </Tooltip>
            <Tooltip
              title={
                result.disabled
                  ? t("stt.comparison.enableTooltip", "Include this row in Run All")
                  : t("stt.comparison.disableTooltip", "Skip this row in Run All")
              }
            >
              <Button
                size="small"
                icon={<Ban className="h-3.5 w-3.5" />}
                onClick={() => onToggleDisabled(resultTarget, !result.disabled)}
              >
                {result.disabled
                  ? t("stt.comparison.enable", "Enable")
                  : t("stt.comparison.disable", "Disable")}
              </Button>
            </Tooltip>
          </div>
        </div>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// ComparisonPanel
// ---------------------------------------------------------------------------

export const ComparisonPanel: React.FC<ComparisonPanelProps> = ({
  blob,
  availableModels,
  availableModelOptions,
  selectedModels: selectedModelsProp,
  onSelectedModelsChange,
  sttOptions,
  onSaveToNotes,
  onComparisonComplete,
}) => {
  const { t } = useTranslation("playground")
  const notification = useAntdNotification()
  const [models, setModels] = useState<string[]>(selectedModelsProp ?? [])

  const {
    results,
    isRunning,
    transcribeAll,
    retryModel,
    duplicateResult,
    setResultDisabled
  } = useComparisonTranscribe()

  // Sync from prop when provided
  useEffect(() => {
    if (selectedModelsProp) {
      setModels(selectedModelsProp)
    }
  }, [selectedModelsProp])

  const handleModelsChange = useCallback(
    (nextModels: string[]) => {
      setModels(nextModels)
      onSelectedModelsChange?.(nextModels)
    },
    [onSelectedModelsChange]
  )

  const handleTranscribeAll = useCallback(async () => {
    if (!blob || models.length === 0) return
    await transcribeAll(blob, models, sttOptions)
  }, [blob, models, sttOptions, transcribeAll])

  // Notify parent when comparison run finishes
  const onCompleteRef = useRef(onComparisonComplete)
  onCompleteRef.current = onComparisonComplete
  const prevIsRunning = useRef(false)
  useEffect(() => {
    if (prevIsRunning.current && !isRunning && results.length > 0) {
      onCompleteRef.current?.(results)
    }
    prevIsRunning.current = isRunning
  }, [isRunning, results])

  const handleCopy = useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text)
        notification.success({
          message: t("stt.comparison.copied", "Copied to clipboard"),
        })
      } catch {
        notification.error({
          message: t("stt.comparison.copyFailed", "Failed to copy"),
        })
      }
    },
    [notification, t]
  )

  const handleRetry = useCallback(
    (model: string) => {
      if (!blob) return
      retryModel(blob, model, sttOptions)
    },
    [blob, sttOptions, retryModel]
  )

  const canTranscribe = !!blob && models.length > 0 && !isRunning
  const modelSelectOptions =
    availableModelOptions && availableModelOptions.length > 0
      ? availableModelOptions.map((model) => ({
          label: model.label,
          value: model.id,
          title: [
            model.description,
            `Availability: ${model.availability}`,
            model.sources.availability ? `Source: ${model.sources.availability}` : null
          ].filter(Boolean).join(" | ")
        }))
      : availableModels.map((m) => ({ label: m, value: m }))

  // Cmd/Ctrl+Enter keyboard shortcut to trigger Transcribe All
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault()
        if (!!blob && models.length > 0 && !isRunning) {
          handleTranscribeAll()
        }
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [blob, models, isRunning, handleTranscribeAll])

  return (
    <div className="space-y-3">
      {/* Model Selection Bar */}
      <div className="flex flex-wrap items-center gap-3">
        <Select
          mode="multiple"
          data-testid="stt-model-selector"
          placeholder={t(
            "stt.comparison.selectModels",
            "Select models to compare"
          )}
          value={models}
          onChange={handleModelsChange}
          style={{ minWidth: 280, flex: 1 }}
          options={modelSelectOptions}
        />
        <Tooltip
          title={
            !blob
              ? t("stt.comparison.noBlobHint", "Record or upload audio first")
              : models.length === 0
                ? t("stt.comparison.noModelsHint", "Select at least one model")
                : undefined
          }
        >
          <Button
            type="primary"
            disabled={!canTranscribe}
            loading={isRunning}
            onClick={handleTranscribeAll}
          >
            {t("stt.comparison.transcribeAll", "Transcribe All")}{" "}
            <kbd className="ml-1 text-xs opacity-60">{navigator.platform?.includes("Mac") ? "⌘" : "Ctrl+"}⏎</kbd>
          </Button>
        </Tooltip>
      </div>

      {/* Results Grid */}
      {results.length === 0 ? (
        <Text type="secondary">
          {t(
            "stt.comparison.emptyState",
            "Select models and record audio to compare transcription results."
          )}
        </Text>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {results.map((result) => (
            <ResultCard
              key={result.id || result.model}
              result={result}
              onCopy={handleCopy}
              onRetry={handleRetry}
              onSave={onSaveToNotes}
              onDuplicate={duplicateResult}
              onToggleDisabled={setResultDisabled}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default ComparisonPanel
