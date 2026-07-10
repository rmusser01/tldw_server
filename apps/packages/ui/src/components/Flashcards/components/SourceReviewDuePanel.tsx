import { useAntdMessage } from "@/hooks/useAntdMessage"
import type {
  SourceReviewActivity,
  SourceReviewOccurrenceActionResponse
} from "@/services/flashcards"
import {
  type SourceReviewFlashcardsIntent,
  type SourceReviewHandoffPayload,
  buildSourceReviewFlashcardsIntent
} from "@/services/tldw/source-review-handoff"
import { Button, Skeleton, Tag, Typography } from "antd"
import {
  CalendarClock,
  Check,
  Play,
  Plus,
  RotateCcw,
  SkipForward
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import {
  useCompleteSourceReviewOccurrenceMutation,
  useDueSourceReviewOccurrencesQuery,
  useSkipSourceReviewOccurrenceMutation,
  useStartSourceReviewOccurrenceMutation
} from "../hooks/useSourceReviewQueries"
import { SourceReviewPlanDrawer } from "./SourceReviewPlanDrawer"

const { Text } = Typography
const COLLAPSED_DUE_COUNT = 3

type SourceReviewDuePanelProps = {
  isActive: boolean
  onSourceReviewGenerate: (intent: SourceReviewFlashcardsIntent) => void
  onSourceReviewQuiz: (payload: SourceReviewHandoffPayload) => void
}

const activityLabels: Record<SourceReviewActivity, string> = {
  reread: "Reread",
  quiz: "Quiz",
  flashcards: "Flashcards",
  cloze: "Fill in the blank"
}

const formatDueAt = (value: string): string => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(date)
}

const sourceSummaryText = (
  occurrence: SourceReviewOccurrenceActionResponse
): string | null => {
  const items = occurrence.source_summary ?? []
  if (items.length === 0) return null
  const labels = items
    .slice(0, 2)
    .map((item) => item.label?.trim() || item.source_id)
    .filter(Boolean)
    .join(", ")
  const remainder = items.length > 2 ? `, +${items.length - 2}` : ""
  return `${items.length} ${items.length === 1 ? "source" : "sources"}${
    labels ? ` · ${labels}${remainder}` : ""
  }`
}

const handoffPayload = (
  occurrence: SourceReviewOccurrenceActionResponse
): SourceReviewHandoffPayload | null => {
  const launch = occurrence.launch_state
  if (!launch) return null
  return {
    occurrence_id: occurrence.id,
    plan_id: occurrence.plan_id,
    plan_title: occurrence.plan_title,
    activity_type: occurrence.activity_type,
    source_bundle: launch.source_bundle
  }
}

export const SourceReviewDuePanel: React.FC<SourceReviewDuePanelProps> = ({
  isActive,
  onSourceReviewGenerate,
  onSourceReviewQuiz
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const dueQuery = useDueSourceReviewOccurrencesQuery({
    enabled: isActive,
    limit: 100
  })
  const startMutation = useStartSourceReviewOccurrenceMutation()
  const completeMutation = useCompleteSourceReviewOccurrenceMutation()
  const skipMutation = useSkipSourceReviewOccurrenceMutation()
  const [plannerOpen, setPlannerOpen] = React.useState(false)
  const [localOccurrences, setLocalOccurrences] = React.useState<
    Record<number, SourceReviewOccurrenceActionResponse>
  >({})
  const [hiddenIds, setHiddenIds] = React.useState<Set<number>>(() => new Set())
  const [expandedRereadId, setExpandedRereadId] = React.useState<number | null>(
    null
  )
  const [pendingAction, setPendingAction] = React.useState<{
    occurrenceId: number
    action: "start" | "complete" | "skip"
  } | null>(null)
  const [showAll, setShowAll] = React.useState(false)

  React.useEffect(() => {
    setLocalOccurrences({})
    setHiddenIds(new Set())
    setShowAll(false)
  }, [dueQuery.data?.now])

  const occurrences = (dueQuery.data?.items ?? [])
    .map((item) => localOccurrences[item.id] ?? item)
    .filter((item) => !hiddenIds.has(item.id))
  const visibleOccurrences = showAll
    ? occurrences
    : occurrences.slice(0, COLLAPSED_DUE_COUNT)
  const dueTotal = Math.max(
    occurrences.length,
    (dueQuery.data?.total ?? occurrences.length) - hiddenIds.size
  )
  const dueSummary = dueQuery.isError
    ? "Source reviews unavailable"
    : dueTotal > 0
      ? `${dueTotal} due now`
      : "No source reviews due"

  const launchOccurrence = React.useCallback(
    (occurrence: SourceReviewOccurrenceActionResponse) => {
      const payload = handoffPayload(occurrence)
      if (!payload) {
        message.error(
          t("option:flashcards.sourceReviewLaunchMissing", {
            defaultValue:
              "This review could not load its saved source snapshot."
          })
        )
        return
      }
      if (occurrence.activity_type === "reread") {
        setExpandedRereadId(occurrence.id)
        return
      }
      if (occurrence.activity_type === "quiz") {
        onSourceReviewQuiz(payload)
        return
      }
      onSourceReviewGenerate(buildSourceReviewFlashcardsIntent(payload))
    },
    [message, onSourceReviewGenerate, onSourceReviewQuiz, t]
  )

  const handleStart = React.useCallback(
    async (occurrence: SourceReviewOccurrenceActionResponse) => {
      setPendingAction({ occurrenceId: occurrence.id, action: "start" })
      try {
        const started = await startMutation.mutateAsync(occurrence.id)
        setLocalOccurrences((current) => ({
          ...current,
          [started.id]: started
        }))
        launchOccurrence(started)
      } catch {
        message.error(
          t("option:flashcards.sourceReviewStartFailed", {
            defaultValue: "Could not start this source review."
          })
        )
      } finally {
        setPendingAction(null)
      }
    },
    [launchOccurrence, message, startMutation, t]
  )

  const handleComplete = React.useCallback(
    async (occurrenceId: number) => {
      setPendingAction({ occurrenceId, action: "complete" })
      try {
        await completeMutation.mutateAsync(occurrenceId)
        setHiddenIds((current) => new Set(current).add(occurrenceId))
      } catch {
        message.error(
          t("option:flashcards.sourceReviewCompleteFailed", {
            defaultValue: "Could not complete this source review."
          })
        )
      } finally {
        setPendingAction(null)
      }
    },
    [completeMutation, message, t]
  )

  const handleSkip = React.useCallback(
    async (occurrenceId: number) => {
      setPendingAction({ occurrenceId, action: "skip" })
      try {
        await skipMutation.mutateAsync(occurrenceId)
        setHiddenIds((current) => new Set(current).add(occurrenceId))
      } catch {
        message.error(
          t("option:flashcards.sourceReviewSkipFailed", {
            defaultValue: "Could not skip this source review."
          })
        )
      } finally {
        setPendingAction(null)
      }
    },
    [message, skipMutation, t]
  )

  if (!isActive) return null

  return (
    <section
      aria-labelledby="source-review-due-heading"
      className="mb-3 border-y border-border bg-surface2/40 py-2"
      data-testid="source-review-due-panel">
      <div className="flex flex-wrap items-center justify-between gap-2 px-1">
        <div className="flex min-w-0 items-center gap-2">
          <CalendarClock
            className="size-4 shrink-0 text-primary"
            aria-hidden="true"
          />
          <div className="min-w-0">
            <Text strong id="source-review-due-heading" className="block">
              Source reviews
            </Text>
            <Text type="secondary" className="block text-xs" aria-live="polite">
              {dueSummary}
            </Text>
          </div>
        </div>
        <Button
          size="small"
          icon={<Plus className="size-4" />}
          onClick={() => setPlannerOpen(true)}>
          Schedule
        </Button>
      </div>

      {dueQuery.isLoading ? (
        <div className="px-1 pt-2">
          <Skeleton active paragraph={{ rows: 1 }} title={false} />
        </div>
      ) : dueQuery.isError ? (
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-border px-1 pt-2">
          <Text type="danger">Could not load due source reviews.</Text>
          <Button
            size="small"
            loading={dueQuery.isFetching}
            onClick={() => void dueQuery.refetch()}>
            Retry
          </Button>
        </div>
      ) : occurrences.length > 0 ? (
        <div className="mt-2 divide-y divide-border border-t border-border">
          {visibleOccurrences.map((occurrence) => {
            const isPending = occurrence.status === "pending"
            const isBusy = pendingAction?.occurrenceId === occurrence.id
            const sourceSummary = sourceSummaryText(occurrence)
            const sourcePreview = occurrence.source_summary
              ?.find((item) => item.excerpt_preview?.trim())
              ?.excerpt_preview?.trim()
            const rereadItems =
              expandedRereadId === occurrence.id
                ? occurrence.launch_state?.source_bundle.items ?? []
                : []
            return (
              <article key={occurrence.id} className="px-1 py-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Text strong className="truncate">
                        {occurrence.plan_title ||
                          `Review plan ${occurrence.plan_id}`}
                      </Text>
                      <Tag className="!m-0">
                        {activityLabels[occurrence.activity_type]}
                      </Tag>
                    </div>
                    <Text type="secondary" className="block text-xs">
                      Due {formatDueAt(occurrence.due_at)}
                    </Text>
                    {sourceSummary ? (
                      <Text
                        type="secondary"
                        className="block max-w-sm truncate text-xs">
                        {sourceSummary}
                      </Text>
                    ) : null}
                    {sourcePreview ? (
                      <Text
                        type="secondary"
                        className="block max-w-xl truncate text-xs italic">
                        {sourcePreview}
                      </Text>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap items-center gap-1">
                    <Button
                      size="small"
                      type="primary"
                      icon={
                        isPending ? (
                          <Play className="size-4" />
                        ) : (
                          <RotateCcw className="size-4" />
                        )
                      }
                      loading={isBusy && pendingAction?.action === "start"}
                      disabled={isBusy}
                      onClick={() => {
                        if (isPending) {
                          void handleStart(occurrence)
                        } else {
                          launchOccurrence(occurrence)
                        }
                      }}>
                      {isPending ? "Start" : "Resume"}
                    </Button>
                    {!isPending ? (
                      <Button
                        size="small"
                        icon={<Check className="size-4" />}
                        loading={isBusy && pendingAction?.action === "complete"}
                        disabled={isBusy}
                        onClick={() => void handleComplete(occurrence.id)}>
                        Complete
                      </Button>
                    ) : null}
                    <Button
                      size="small"
                      type="text"
                      icon={<SkipForward className="size-4" />}
                      loading={isBusy && pendingAction?.action === "skip"}
                      disabled={isBusy}
                      onClick={() => void handleSkip(occurrence.id)}>
                      Skip
                    </Button>
                  </div>
                </div>

                {rereadItems.length > 0 ? (
                  <div
                    role="region"
                    aria-label="Source snapshot"
                    className="mt-2 max-h-64 divide-y divide-border overflow-y-auto rounded border border-border bg-background">
                    {rereadItems.map((item, itemIndex) => (
                      <div
                        key={`${item.source_type}:${item.source_id}:${itemIndex}`}
                        className="p-2">
                        <Text strong className="block text-sm">
                          {item.label || item.source_title || item.source_id}
                        </Text>
                        {item.excerpt_text ? (
                          <Text className="mt-1 block whitespace-pre-wrap text-sm">
                            {item.excerpt_text}
                          </Text>
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : null}
              </article>
            )
          })}
          {occurrences.length > COLLAPSED_DUE_COUNT ? (
            <div className="flex justify-center px-1 py-2">
              <Button
                size="small"
                type="link"
                onClick={() => setShowAll((current) => !current)}>
                {showAll ? "Show fewer" : `Show all ${occurrences.length}`}
              </Button>
            </div>
          ) : null}
        </div>
      ) : null}

      {plannerOpen ? (
        <SourceReviewPlanDrawer
          open
          onClose={() => setPlannerOpen(false)}
          onCreated={() => setPlannerOpen(false)}
        />
      ) : null}
    </section>
  )
}

export default SourceReviewDuePanel
