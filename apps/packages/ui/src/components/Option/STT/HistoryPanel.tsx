import React, { useCallback, useState } from "react"
import { Button, Card, Collapse, Modal, Tag, Typography } from "antd"
import { Download, RefreshCcw, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import type {
  SttComparisonConfig,
  SttComparisonMetadata
} from "@/components/Option/Audio/comparison-provenance"
import {
  formatByteSize,
  formatClientLatency,
  formatCreatedAt
} from "@/components/Option/Audio/comparison-provenance"

const { Text } = Typography

/* ------------------------------------------------------------------ */
/*  Public types                                                       */
/* ------------------------------------------------------------------ */

export interface SttHistoryResult {
  model: string
  text: string
  latencyMs?: number
  wordCount?: number
  config?: SttComparisonConfig
  metadata?: SttComparisonMetadata
}

export interface SttHistoryEntry {
  id: string
  recordingId: string
  createdAt: string
  durationMs: number
  results: SttHistoryResult[]
}

/* ------------------------------------------------------------------ */
/*  Props                                                              */
/* ------------------------------------------------------------------ */

interface HistoryPanelProps {
  entries: SttHistoryEntry[]
  onRecompare: (entry: SttHistoryEntry) => void
  onExport: (entry: SttHistoryEntry) => void
  onDelete: (id: string) => void
  onClearAll: () => void
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const formatDuration = (ms: number): string => `${(ms / 1000).toFixed(1)}s`

const formatTimestamp = (iso: string): string => {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

const TRUNCATE_LENGTH = 120

const truncateText = (text: string): string =>
  text.length > TRUNCATE_LENGTH
    ? `${text.slice(0, TRUNCATE_LENGTH)}...`
    : text

const buildProvenanceTags = (result: SttHistoryResult): string[] => {
  const metadata = result.metadata
  const config = result.config
  const wordCount = metadata?.wordCount ?? result.wordCount
  return [
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
    wordCount != null ? `${wordCount} words` : undefined
  ].filter((tag): tag is string => Boolean(tag))
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export const HistoryPanel: React.FC<HistoryPanelProps> = ({
  entries,
  onRecompare,
  onExport,
  onDelete,
  onClearAll
}) => {
  const { t } = useTranslation()
  const [clearModalOpen, setClearModalOpen] = useState(false)

  const handleClearConfirm = useCallback(() => {
    onClearAll()
    setClearModalOpen(false)
  }, [onClearAll])

  const collapseItems = entries.map((entry) => ({
    key: entry.id,
    label: (
      <span>
        {formatTimestamp(entry.createdAt)}{" "}
        <Tag>{formatDuration(entry.durationMs)}</Tag>{" "}
        <Tag color="blue">
          {entry.results.length} models compared
        </Tag>
      </span>
    ),
    children: (
      <div>
        {entry.results.map((result) => (
          <div key={result.model} style={{ marginBottom: 8 }}>
            <Tag color="green">{result.model}</Tag>
            {buildProvenanceTags(result).map((tag) => (
              <Tag key={tag}>{tag}</Tag>
            ))}
            <div style={{ marginTop: 4 }}>
              <Text type="secondary">{truncateText(result.text)}</Text>
            </div>
          </div>
        ))}

        <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <Button
            size="small"
            icon={<RefreshCcw size={14} />}
            onClick={() => onRecompare(entry)}
          >
            {t("stt.history.recompare", "Re-compare")}
          </Button>
          <Button
            size="small"
            icon={<Download size={14} />}
            onClick={() => onExport(entry)}
          >
            {t("stt.history.export", "Export")}
          </Button>
          <Button
            size="small"
            danger
            icon={<Trash2 size={14} />}
            onClick={() => onDelete(entry.id)}
          >
            {t("stt.history.delete", "Delete")}
          </Button>
        </div>
      </div>
    )
  }))

  const extra =
    entries.length > 0 ? (
      <Button
        size="small"
        danger
        onClick={(e) => {
          e.stopPropagation()
          setClearModalOpen(true)
        }}
      >
        {t("stt.history.clearAll", "Clear all")}
      </Button>
    ) : undefined

  return (
    <>
      <Card
        title={t("stt.history.title", "Recording History")}
        extra={extra}
        size="small"
      >
        {entries.length === 0 ? (
          <Text type="secondary">
            {t(
              "stt.history.empty",
              "Start a recording to see transcripts here."
            )}
          </Text>
        ) : (
          <Collapse items={collapseItems} />
        )}
      </Card>

      <Modal
        title={t("stt.history.clearConfirmTitle", "Confirm Clear")}
        open={clearModalOpen}
        onOk={handleClearConfirm}
        onCancel={() => setClearModalOpen(false)}
        okText={t("stt.history.clearConfirmOk", "Delete")}
        okButtonProps={{ danger: true }}
      >
        <Text>
          {t(
            "stt.history.clearConfirmBody",
            `Delete ${entries.length} recordings? Cannot be undone.`
          )}
        </Text>
      </Modal>
    </>
  )
}
