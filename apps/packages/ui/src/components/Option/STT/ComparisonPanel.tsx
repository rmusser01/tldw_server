import React, { useCallback, useEffect, useRef, useState } from "react"
import { Button, Card, Input, Select, Skeleton, Tag, Tooltip, Typography } from "antd"
import { Copy, RotateCcw, Save } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useComparisonTranscribe } from "@/hooks/useComparisonTranscribe"
import type { ComparisonResult } from "@/hooks/useComparisonTranscribe"
import { useAntdNotification } from "@/hooks/useAntdNotification"

const { Text } = Typography

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ComparisonPanelProps {
  blob: Blob | null
  availableModels: string[]
  selectedModels?: string[]
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
  onRetry: (model: string) => void
  onSave: (text: string, model: string) => void
}

const ResultCard: React.FC<ResultCardProps> = ({
  result,
  onCopy,
  onRetry,
  onSave,
}) => {
  const { t } = useTranslation("playground")
  const [editedText, setEditedText] = useState(result.text)

  useEffect(() => {
    setEditedText(result.text)
  }, [result.text])

  const isPending = result.status === "pending" || result.status === "running"

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
          <div>
            <Button
              size="small"
              icon={<RotateCcw className="h-3.5 w-3.5" />}
              onClick={() => onRetry(result.model)}
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
            {result.latencyMs != null && (
              <Tag bordered>
                {(result.latencyMs / 1000).toFixed(1)}s
              </Tag>
            )}
            <Tag bordered>
              {editedText.split(/\s+/).filter(Boolean).length}{" "}
              {t("stt.comparison.words", "words")}
            </Tag>
          </div>
          <div className="flex items-center gap-2">
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
  selectedModels: selectedModelsProp,
  sttOptions,
  onSaveToNotes,
  onComparisonComplete,
}) => {
  const { t } = useTranslation("playground")
  const notification = useAntdNotification()
  const [models, setModels] = useState<string[]>(selectedModelsProp ?? [])

  const { results, isRunning, transcribeAll, retryModel } =
    useComparisonTranscribe()

  // Sync from prop when provided
  useEffect(() => {
    if (selectedModelsProp) {
      setModels(selectedModelsProp)
    }
  }, [selectedModelsProp])

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
          onChange={setModels}
          style={{ minWidth: 280, flex: 1 }}
          options={availableModels.map((m) => ({ label: m, value: m }))}
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
              key={result.model}
              result={result}
              onCopy={handleCopy}
              onRetry={handleRetry}
              onSave={onSaveToNotes}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default ComparisonPanel
