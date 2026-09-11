import React from "react"
import { Button, Empty, List, Modal, Skeleton, Tag, Typography } from "antd"
import { ClockCircleOutlined, EyeOutlined } from "@ant-design/icons"
import { useTranslation } from "react-i18next"

import { Badge as DesignSystemBadge } from "@/components/ui/primitives"
import type { OsceCitation, OsceRevealedAttempt } from "@/services/osce"
import type { SourceCitation } from "@/services/quizzes"
import { QuizMarkdown } from "../components/QuizMarkdown"
import { SourceCitations } from "../components/SourceCitations"
import {
  useCompletedOsceAttemptsQuery,
  useOsceAttemptQuery
} from "../hooks/useOsceQueries"

const toSourceCitations = (citations: OsceCitation[]): SourceCitation[] => citations.map((citation) => ({
  source_type: citation.source_type === "media" || citation.source_type === "note"
    ? citation.source_type
    : null,
  source_id: citation.source_id,
  label: citation.label,
  quote: citation.quote,
  media_id: citation.media_id,
  chunk_id: citation.chunk_id,
  timestamp_seconds: citation.timestamp_seconds,
  source_url: citation.source_url
}))

const formatElapsed = (seconds: number | null): string => {
  if (seconds == null) return "Not recorded"
  const hours = Math.floor(seconds / 3_600)
  const minutes = Math.floor((seconds % 3_600) / 60)
  const remainder = Math.floor(seconds % 60)
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`
}

const parsePositiveId = (value: string): number | null => {
  if (!value.trim()) return null
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null
}

export interface OsceResultsPanelProps {
  quizId?: number | null
}

export const OsceResultsPanel: React.FC<OsceResultsPanelProps> = ({ quizId = null }) => {
  const { t } = useTranslation(["option", "common"])
  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(10)
  const [quizFilter, setQuizFilter] = React.useState(quizId == null ? "" : String(quizId))
  const [stationFilter, setStationFilter] = React.useState("")
  const [selectedAttemptId, setSelectedAttemptId] = React.useState<number | null>(null)
  const normalizedQuizId = quizId ?? parsePositiveId(quizFilter)
  const normalizedStationId = parsePositiveId(stationFilter)
  const offset = (page - 1) * pageSize
  const completedQuery = useCompletedOsceAttemptsQuery({
    quiz_id: normalizedQuizId ?? undefined,
    station_id: normalizedStationId ?? undefined,
    limit: pageSize,
    offset
  }, { enabled: true })
  const detailQuery = useOsceAttemptQuery(selectedAttemptId, { enabled: selectedAttemptId != null })
  const attempts = completedQuery.data?.items ?? []
  const pageCount = completedQuery.data?.items.length ?? 0
  const total = completedQuery.data?.pagination.total ?? (
    offset + pageCount + (completedQuery.data?.has_more ? 1 : 0)
  )
  const selectedAttempt = detailQuery.data?.state === "completed"
    ? detailQuery.data as OsceRevealedAttempt
    : null
  const detailTitle = selectedAttempt
    ? `${selectedAttempt.station.title} practice details`
    : "Practice details"

  React.useEffect(() => {
    setPage(1)
  }, [normalizedQuizId, normalizedStationId])

  const renderDetail = () => (
    <Modal
      title={detailTitle}
      open={selectedAttemptId != null}
      onCancel={() => setSelectedAttemptId(null)}
      footer={(
        <Button onClick={() => setSelectedAttemptId(null)}>
          {t("common:close", { defaultValue: "Close" })}
        </Button>
      )}
      width={760}
      destroyOnHidden
    >
      {detailQuery.isLoading || detailQuery.isFetching ? (
        <Skeleton active paragraph={{ rows: 7 }} />
      ) : selectedAttempt ? (
        <div className="space-y-5">
          <div className="flex flex-wrap items-center gap-2">
            <DesignSystemBadge variant="info">
              {t("option:quiz.osceSelfMarked", { defaultValue: "Self-marked study practice" })}
            </DesignSystemBadge>
            <Tag icon={<ClockCircleOutlined />}>{formatElapsed(selectedAttempt.elapsed_seconds)}</Tag>
          </div>

          <section className="space-y-3" aria-label="Candidate context">
            <div>
              <Typography.Text strong>{t("option:quiz.osceTask", { defaultValue: "Task" })}</Typography.Text>
              <QuizMarkdown content={selectedAttempt.station.candidate_task} className="max-w-[72ch]" />
            </div>
            <div>
              <Typography.Text strong>{t("option:quiz.oscePatientContext", { defaultValue: "Patient context" })}</Typography.Text>
              <QuizMarkdown content={selectedAttempt.station.patient_context.text} className="max-w-[72ch]" />
            </div>
          </section>

          <section className="space-y-2" aria-label="Candidate notes">
            <Typography.Text strong>{t("option:quiz.oscePrivateNotes", { defaultValue: "Private practice notes" })}</Typography.Text>
            <QuizMarkdown content={selectedAttempt.notes || "No notes recorded."} className="max-w-[72ch]" />
          </section>

          <section className="space-y-3" aria-label="Checklist results">
            <Typography.Title level={5} className="!mb-0">Checklist</Typography.Title>
            {selectedAttempt.station.checklist_items.map((item) => (
              <div key={item.id} className="flex flex-col gap-1 border-b border-border pb-2 sm:flex-row sm:items-start sm:justify-between">
                <QuizMarkdown content={item.label} className="max-w-[62ch]" />
                <Tag className="shrink-0">
                  {selectedAttempt.checklist_selections[item.id] === "met" ? "Met" : "Not met"}
                </Tag>
              </div>
            ))}
          </section>

          <section className="space-y-3" aria-label="Rubric results">
            <Typography.Title level={5} className="!mb-0">Rubric</Typography.Title>
            {selectedAttempt.station.rubric_domains.map((domain) => {
              const selectedLevel = domain.levels.find(
                (level) => level.id === selectedAttempt.rubric_selections[domain.id]
              )
              return (
                <div key={domain.id} className="border-b border-border pb-2">
                  <Typography.Text strong className="block">{domain.label}</Typography.Text>
                  <Typography.Text className="block">{selectedLevel?.label ?? "Not selected"}</Typography.Text>
                  {selectedLevel?.description && (
                    <Typography.Text className="block text-sm text-text-muted">
                      {selectedLevel.description}
                    </Typography.Text>
                  )}
                </div>
              )
            })}
          </section>

          <section className="space-y-3" aria-label="Expected points">
            <Typography.Title level={5} className="!mb-0">Expected points</Typography.Title>
            {selectedAttempt.station.expected_key_points.map((point) => (
              <div key={point.id} className="border-b border-border pb-3 last:border-b-0">
                <QuizMarkdown content={point.text} className="max-w-[72ch]" />
                <SourceCitations citations={toSourceCitations(point.citations)} />
              </div>
            ))}
          </section>
        </div>
      ) : (
        <Empty description="Practice details are unavailable." />
      )}
    </Modal>
  )

  return (
    <section className="space-y-4" aria-label="OSCE practice results">
      {renderDetail()}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end" data-testid="osce-results-toolbar">
        <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm text-text-muted">
          {t("option:quiz.osceQuizFilter", { defaultValue: "Filter by quiz ID" })}
          <input
            type="number"
            min={1}
            inputMode="numeric"
            aria-label="Filter by quiz ID"
            value={quizFilter}
            disabled={quizId != null}
            onChange={(event) => setQuizFilter(event.target.value)}
            className="min-h-11 w-full rounded-md border border-border bg-surface2 px-3 text-text focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
          />
        </label>
        <label className="flex min-w-0 flex-1 flex-col gap-1 text-sm text-text-muted">
          {t("option:quiz.osceStationFilter", { defaultValue: "Filter by station ID" })}
          <input
            type="number"
            min={1}
            inputMode="numeric"
            aria-label="Filter by station ID"
            value={stationFilter}
            onChange={(event) => setStationFilter(event.target.value)}
            className="min-h-11 w-full rounded-md border border-border bg-surface2 px-3 text-text focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30"
          />
        </label>
        <Button
          className="min-h-11 whitespace-normal"
          onClick={() => {
            if (quizId == null) setQuizFilter("")
            setStationFilter("")
            setPage(1)
          }}
        >
          {t("common:reset", { defaultValue: "Reset filters" })}
        </Button>
      </div>

      <DesignSystemBadge variant="info">
        {t("option:quiz.osceSelfMarked", { defaultValue: "Self-marked study practice" })}
      </DesignSystemBadge>

      {completedQuery.isLoading ? (
        <div className="space-y-3" data-testid="osce-results-loading">
          <Skeleton active paragraph={{ rows: 2 }} />
          <Skeleton active paragraph={{ rows: 2 }} />
        </div>
      ) : (
        <List
          dataSource={attempts}
          locale={{ emptyText: "No completed OSCE practice matches these filters." }}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: total > pageSize,
            onChange: (nextPage, nextPageSize) => {
              setPage(nextPage)
              if (nextPageSize !== pageSize) {
                setPageSize(nextPageSize)
                setPage(1)
              }
            }
          }}
          renderItem={(attemptSummary) => (
            <List.Item>
              <div className="flex w-full flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="min-w-0 space-y-1">
                  <Typography.Text strong className="block break-words">
                    {attemptSummary.station_title}
                  </Typography.Text>
                  <Typography.Text className="block text-xs text-text-muted">
                    Quiz #{attemptSummary.quiz_id} · Station #{attemptSummary.station_id}
                  </Typography.Text>
                  <div className="flex flex-wrap gap-2">
                    <Tag>{attemptSummary.checklist_met_count ?? 0} of {attemptSummary.checklist_total ?? 0} met</Tag>
                    <Tag icon={<ClockCircleOutlined />}>{formatElapsed(attemptSummary.elapsed_seconds)}</Tag>
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-text-muted">
                    {attemptSummary.rubric_results.map((result) => (
                      <span key={result.domain_id}>{result.domain_label}: {result.level_label}</span>
                    ))}
                  </div>
                </div>
                <Button
                  icon={<EyeOutlined />}
                  className="min-h-11 shrink-0 whitespace-normal"
                  onClick={() => setSelectedAttemptId(attemptSummary.id)}
                >
                  {t("option:quiz.osceViewDetails", { defaultValue: "View practice details" })}
                </Button>
              </div>
            </List.Item>
          )}
        />
      )}
    </section>
  )
}

export default OsceResultsPanel
