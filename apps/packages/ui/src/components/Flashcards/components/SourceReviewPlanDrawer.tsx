import { useAntdMessage } from "@/hooks/useAntdMessage"
import type {
  SourceReviewActivity,
  SourceReviewOffsetUnit,
  SourceReviewPlanResponse,
  StudyPackSourceSelection,
  StudyPackSourceType
} from "@/services/flashcards"
import { Button, Drawer, Input, Select, Typography } from "antd"
import { Plus, Trash2 } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"

import { useCreateSourceReviewPlanMutation } from "../hooks/useSourceReviewQueries"

const { Text } = Typography
const EMPTY_SOURCE_ITEMS: StudyPackSourceSelection[] = []
const SOURCE_EXCERPT_LIMIT = 20_000
const SOURCE_LOCATOR_BYTE_LIMIT = 8 * 1024

type SourceReviewPlanDrawerProps = {
  open: boolean
  onClose: () => void
  initialSourceItems?: StudyPackSourceSelection[]
  onCreated?: (plan: SourceReviewPlanResponse) => void
}

type ScheduleDraft = {
  key: string
  offsetValue: string
  offsetUnit: SourceReviewOffsetUnit
  activityType: SourceReviewActivity
}

const PRESET_SCHEDULE: ReadonlyArray<
  Pick<ScheduleDraft, "offsetValue" | "offsetUnit" | "activityType">
> = [
  { offsetValue: "1", offsetUnit: "day", activityType: "reread" },
  { offsetValue: "3", offsetUnit: "day", activityType: "reread" },
  { offsetValue: "7", offsetUnit: "day", activityType: "reread" },
  { offsetValue: "14", offsetUnit: "day", activityType: "reread" },
  { offsetValue: "28", offsetUnit: "day", activityType: "reread" },
  { offsetValue: "3", offsetUnit: "month", activityType: "reread" },
  { offsetValue: "6", offsetUnit: "month", activityType: "reread" }
]

const makeSchedule = (): ScheduleDraft[] =>
  PRESET_SCHEDULE.map((row, index) => ({ ...row, key: `preset-${index}` }))

const localDate = (): string => {
  const now = new Date()
  const month = String(now.getMonth() + 1).padStart(2, "0")
  const day = String(now.getDate()).padStart(2, "0")
  return `${now.getFullYear()}-${month}-${day}`
}

const browserTimezone = (): string => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
  } catch {
    return "UTC"
  }
}

const isValidTimezone = (value: string): boolean => {
  if (!value.trim()) return false
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: value.trim() })
    return true
  } catch {
    return false
  }
}

const utf8Bytes = (value: string): number =>
  new TextEncoder().encode(value).length

const scheduleLabel = (row: ScheduleDraft): string => {
  const value = row.offsetValue || "0"
  if (row.offsetUnit === "month") {
    return `${value} ${value === "1" ? "month" : "months"}`
  }
  return `Day ${value}`
}

const scheduleErrors = (rows: ScheduleDraft[]): Array<string | null> => {
  const errors = rows.map((row) => {
    const value = Number(row.offsetValue)
    const maximum = row.offsetUnit === "month" ? 120 : 3650
    if (!Number.isInteger(value) || value < 1 || value > maximum) {
      return `Enter a whole number between 1 and ${maximum}.`
    }
    if (!["reread", "quiz", "flashcards", "cloze"].includes(row.activityType)) {
      return "Choose a review activity."
    }
    return null
  })

  const duplicateRows = new Map<string, number[]>()
  rows.forEach((row, index) => {
    if (errors[index]) return
    const key = `${Number(row.offsetValue)}:${row.offsetUnit}:${row.activityType}`
    duplicateRows.set(key, [...(duplicateRows.get(key) ?? []), index])
  })
  duplicateRows.forEach((indices) => {
    if (indices.length < 2) return
    indices.forEach((index) => {
      errors[index] = "Duplicate interval and activity."
    })
  })
  return errors
}

const apiErrorDetail = (error: unknown): string | null => {
  if (!error || typeof error !== "object" || !("response" in error)) return null
  const detail = (error as { response?: { data?: { detail?: unknown } } })
    .response?.data?.detail
  return typeof detail === "string" ? detail : null
}

export const SourceReviewPlanDrawer: React.FC<SourceReviewPlanDrawerProps> = ({
  open,
  onClose,
  initialSourceItems = EMPTY_SOURCE_ITEMS,
  onCreated
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const createMutation = useCreateSourceReviewPlanMutation()
  const nextRowRef = React.useRef(PRESET_SCHEDULE.length)

  const [title, setTitle] = React.useState("")
  const [startsOn, setStartsOn] = React.useState(localDate)
  const [timezone, setTimezone] = React.useState(browserTimezone)
  const [sourceItems, setSourceItems] = React.useState<
    StudyPackSourceSelection[]
  >([])
  const [sourceType, setSourceType] =
    React.useState<StudyPackSourceType>("note")
  const [sourceId, setSourceId] = React.useState("")
  const [sourceLabel, setSourceLabel] = React.useState("")
  const [sourceExcerpt, setSourceExcerpt] = React.useState("")
  const [sourceLocator, setSourceLocator] = React.useState("")
  const [schedule, setSchedule] = React.useState<ScheduleDraft[]>(makeSchedule)
  const [submitError, setSubmitError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!open) return
    setTitle("")
    setStartsOn(localDate())
    setTimezone(browserTimezone())
    setSourceItems(initialSourceItems.map((item) => ({ ...item })))
    setSourceType("note")
    setSourceId("")
    setSourceLabel("")
    setSourceExcerpt("")
    setSourceLocator("")
    setSchedule(makeSchedule())
    setSubmitError(null)
    nextRowRef.current = PRESET_SCHEDULE.length
  }, [initialSourceItems, open])

  const locatorResult = React.useMemo(() => {
    if (!sourceLocator.trim()) return { value: undefined, error: null }
    try {
      const parsed = JSON.parse(sourceLocator)
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
        return { value: undefined, error: "Locator must be a JSON object." }
      }
      const serialized = JSON.stringify(parsed)
      if (utf8Bytes(serialized) > SOURCE_LOCATOR_BYTE_LIMIT) {
        return {
          value: undefined,
          error: "Use 8 KiB or less for locator JSON."
        }
      }
      return { value: parsed as Record<string, unknown>, error: null }
    } catch {
      return { value: undefined, error: "Enter valid locator JSON." }
    }
  }, [sourceLocator])
  const rowErrors = React.useMemo(() => scheduleErrors(schedule), [schedule])
  const titleError =
    title.trim().length > 200 ? "Use 200 characters or fewer." : null
  const timezoneError = isValidTimezone(timezone)
    ? null
    : "Enter a valid IANA timezone."
  const sourceExcerptError =
    sourceExcerpt.length > SOURCE_EXCERPT_LIMIT
      ? "Use 20,000 characters or fewer."
      : null
  const sourceItemsValid = sourceItems.every(
    (item) =>
      (item.excerpt_text?.length ?? 0) <= SOURCE_EXCERPT_LIMIT &&
      (!item.locator ||
        utf8Bytes(JSON.stringify(item.locator)) <= SOURCE_LOCATOR_BYTE_LIMIT)
  )
  const canAddSource =
    sourceId.trim().length > 0 &&
    !sourceExcerptError &&
    !locatorResult.error &&
    sourceItems.length < 10
  const canCreate =
    title.trim().length > 0 &&
    !titleError &&
    startsOn.length > 0 &&
    !timezoneError &&
    sourceItems.length > 0 &&
    sourceItems.length <= 10 &&
    sourceItemsValid &&
    schedule.length > 0 &&
    schedule.length <= 24 &&
    rowErrors.every((error) => error === null) &&
    !createMutation.isPending

  const updateSchedule = React.useCallback(
    (key: string, patch: Partial<ScheduleDraft>) => {
      setSchedule((current) =>
        current.map((row) => (row.key === key ? { ...row, ...patch } : row))
      )
    },
    []
  )

  const handleAddSource = React.useCallback(() => {
    if (!canAddSource) return
    setSourceItems((current) => [
      ...current,
      {
        source_type: sourceType,
        source_id: sourceId.trim(),
        ...(sourceLabel.trim() ? { label: sourceLabel.trim() } : {}),
        ...(sourceExcerpt.trim() ? { excerpt_text: sourceExcerpt.trim() } : {}),
        ...(locatorResult.value ? { locator: locatorResult.value } : {})
      }
    ])
    setSourceId("")
    setSourceLabel("")
    setSourceExcerpt("")
    setSourceLocator("")
  }, [
    canAddSource,
    locatorResult.value,
    sourceExcerpt,
    sourceId,
    sourceLabel,
    sourceType
  ])

  const handleCreate = React.useCallback(async () => {
    if (!canCreate) return
    setSubmitError(null)
    try {
      const plan = await createMutation.mutateAsync({
        title: title.trim(),
        starts_on: startsOn,
        timezone: timezone.trim(),
        source_items: sourceItems,
        schedule: schedule.map((row) => ({
          offset_value: Number(row.offsetValue),
          offset_unit: row.offsetUnit,
          activity_type: row.activityType
        }))
      })
      message.success(
        t("option:flashcards.sourceReviewPlanCreated", {
          defaultValue: "Review plan created."
        })
      )
      onCreated?.(plan)
      onClose()
    } catch (error) {
      const detail =
        apiErrorDetail(error) ||
        t("option:flashcards.sourceReviewPlanCreateFailed", {
          defaultValue: "Could not create the review plan."
        })
      setSubmitError(detail)
      message.error(detail)
    }
  }, [
    canCreate,
    createMutation,
    message,
    onClose,
    onCreated,
    schedule,
    sourceItems,
    startsOn,
    t,
    timezone,
    title
  ])

  return (
    <Drawer
      open={open}
      onClose={onClose}
      destroyOnClose
      size="large"
      title={t("option:flashcards.sourceReviewPlanTitle", {
        defaultValue: "Schedule source review"
      })}
      footer={
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            disabled={!canCreate}
            loading={createMutation.isPending}
            onClick={() => void handleCreate()}>
            {t("option:flashcards.sourceReviewCreatePlan", {
              defaultValue: "Create plan"
            })}
          </Button>
        </div>
      }>
      <div className="space-y-5">
        {submitError ? (
          <div
            role="alert"
            className="rounded border border-danger/30 bg-danger/5 px-3 py-2">
            <Text type="danger">{submitError}</Text>
          </div>
        ) : null}
        <section
          className="grid gap-3 sm:grid-cols-2"
          aria-labelledby="source-review-plan-details">
          <h3 id="source-review-plan-details" className="sr-only">
            Plan details
          </h3>
          <label className="space-y-1 sm:col-span-2">
            <Text strong className="block">
              Plan title
            </Text>
            <Input
              aria-label="Plan title"
              value={title}
              status={titleError ? "error" : undefined}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Cardiac physiology review"
            />
            {titleError ? <Text type="danger">{titleError}</Text> : null}
          </label>
          <label className="space-y-1">
            <Text strong className="block">
              Start date
            </Text>
            <Input
              type="date"
              aria-label="Start date"
              value={startsOn}
              onChange={(event) => setStartsOn(event.target.value)}
            />
          </label>
          <label className="space-y-1">
            <Text strong className="block">
              Timezone
            </Text>
            <Input
              aria-label="Timezone"
              value={timezone}
              status={timezoneError ? "error" : undefined}
              onChange={(event) => setTimezone(event.target.value)}
            />
            {timezoneError ? <Text type="danger">{timezoneError}</Text> : null}
          </label>
        </section>

        <section
          aria-labelledby="source-review-sources"
          className="border-t border-border pt-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <Text strong id="source-review-sources" className="block">
                Sources
              </Text>
              <Text type="secondary" className="text-sm">
                The saved snapshot grounds every future review.
              </Text>
            </div>
            <Text type="secondary" className="shrink-0 whitespace-nowrap">
              {sourceItems.length}/10
            </Text>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <label className="space-y-1">
              <Text className="block text-sm">Source type</Text>
              <Select
                aria-label="Source type"
                className="w-full"
                value={sourceType}
                onChange={setSourceType}
                options={[
                  { value: "note", label: "Note" },
                  { value: "media", label: "Media" },
                  { value: "message", label: "Message" }
                ]}
              />
            </label>
            <label className="space-y-1">
              <Text className="block text-sm">Source ID</Text>
              <Input
                aria-label="Source ID"
                value={sourceId}
                onChange={(event) => setSourceId(event.target.value)}
              />
            </label>
            <label className="space-y-1 sm:col-span-2">
              <Text className="block text-sm">Source label</Text>
              <Input
                aria-label="Source label"
                value={sourceLabel}
                onChange={(event) => setSourceLabel(event.target.value)}
                placeholder="Optional label"
              />
            </label>
            <label className="space-y-1 sm:col-span-2">
              <Text className="block text-sm">Source excerpt</Text>
              <Input.TextArea
                aria-label="Source excerpt"
                rows={3}
                value={sourceExcerpt}
                status={sourceExcerptError ? "error" : undefined}
                onChange={(event) => setSourceExcerpt(event.target.value)}
                placeholder="Optional excerpt to preserve with the plan"
              />
              {sourceExcerptError ? (
                <Text type="danger">{sourceExcerptError}</Text>
              ) : null}
            </label>
            <label className="space-y-1 sm:col-span-2">
              <Text className="block text-sm">Source locator JSON</Text>
              <Input.TextArea
                aria-label="Source locator JSON"
                rows={2}
                value={sourceLocator}
                status={locatorResult.error ? "error" : undefined}
                onChange={(event) => setSourceLocator(event.target.value)}
                placeholder='Optional, for example {"page": 12}'
              />
              {locatorResult.error ? (
                <Text type="danger">{locatorResult.error}</Text>
              ) : null}
            </label>
          </div>
          <div className="mt-2 flex justify-end">
            <Button
              icon={<Plus className="size-4" />}
              disabled={!canAddSource}
              onClick={handleAddSource}>
              Add source
            </Button>
          </div>
          {sourceItems.length > 0 ? (
            <div className="mt-3 divide-y divide-border rounded border border-border">
              {sourceItems.map((item, index) => (
                <div
                  key={`${item.source_type}:${item.source_id}:${index}`}
                  className="flex items-start justify-between gap-3 p-2">
                  <div className="min-w-0">
                    <Text strong className="block truncate">
                      {item.label || item.source_title || item.source_id}
                    </Text>
                    <Text type="secondary" className="block text-xs">
                      {item.source_type} · {item.source_id}
                    </Text>
                  </div>
                  <Button
                    type="text"
                    icon={<Trash2 className="size-4" />}
                    aria-label={`Remove source ${index + 1}`}
                    onClick={() =>
                      setSourceItems((current) =>
                        current.filter((_, itemIndex) => itemIndex !== index)
                      )
                    }
                  />
                </div>
              ))}
            </div>
          ) : null}
        </section>

        <section
          aria-labelledby="source-review-schedule"
          className="border-t border-border pt-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <Text strong id="source-review-schedule" className="block">
                Review schedule
              </Text>
              <Text type="secondary" className="text-sm">
                Choose what happens at each interval.
              </Text>
            </div>
            <Button
              size="small"
              icon={<Plus className="size-4" />}
              disabled={schedule.length >= 24}
              onClick={() => {
                const index = nextRowRef.current++
                setSchedule((current) => [
                  ...current,
                  {
                    key: `custom-${index}`,
                    offsetValue: "1",
                    offsetUnit: "day",
                    activityType: "reread"
                  }
                ])
              }}>
              Add interval
            </Button>
          </div>
          <div className="space-y-2">
            {schedule.map((row, index) => (
              <div key={row.key} className="rounded border border-border p-2">
                <div className="grid items-end gap-2 sm:grid-cols-[88px_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.25fr)_32px]">
                  <Text strong className="self-center text-sm">
                    {scheduleLabel(row)}
                  </Text>
                  <label className="space-y-1">
                    <Text className="block text-xs">Offset</Text>
                    <Input
                      type="number"
                      min={1}
                      step={1}
                      aria-label={`Offset ${index + 1}`}
                      value={row.offsetValue}
                      status={rowErrors[index] ? "error" : undefined}
                      onChange={(event) =>
                        updateSchedule(row.key, {
                          offsetValue: event.target.value
                        })
                      }
                    />
                  </label>
                  <label className="space-y-1">
                    <Text className="block text-xs">Unit</Text>
                    <Select
                      aria-label={`Unit ${index + 1}`}
                      className="w-full"
                      value={row.offsetUnit}
                      onChange={(value) =>
                        updateSchedule(row.key, { offsetUnit: value })
                      }
                      options={[
                        { value: "day", label: "Days" },
                        { value: "month", label: "Months" }
                      ]}
                    />
                  </label>
                  <label className="space-y-1">
                    <Text className="block text-xs">Activity</Text>
                    <Select
                      aria-label={`Activity ${index + 1}`}
                      className="w-full"
                      value={row.activityType}
                      onChange={(value) =>
                        updateSchedule(row.key, { activityType: value })
                      }
                      options={[
                        { value: "reread", label: "Reread" },
                        { value: "flashcards", label: "Flashcards" },
                        { value: "cloze", label: "Fill in the blank" },
                        { value: "quiz", label: "Quiz" }
                      ]}
                    />
                  </label>
                  <Button
                    type="text"
                    icon={<Trash2 className="size-4" />}
                    aria-label={`Remove interval ${index + 1}`}
                    onClick={() =>
                      setSchedule((current) =>
                        current.filter((item) => item.key !== row.key)
                      )
                    }
                  />
                </div>
                {rowErrors[index] ? (
                  <Text
                    type="danger"
                    role="alert"
                    className="mt-1 block text-xs">
                    {rowErrors[index]}
                  </Text>
                ) : null}
              </div>
            ))}
          </div>
        </section>
      </div>
    </Drawer>
  )
}

export default SourceReviewPlanDrawer
