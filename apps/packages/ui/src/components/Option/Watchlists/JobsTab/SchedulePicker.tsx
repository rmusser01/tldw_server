import React, { useEffect, useMemo, useState } from "react"
import { Button, Input, InputNumber, Select, Switch } from "antd"
import { Clock } from "lucide-react"
import { useTranslation } from "react-i18next"
import { CronDisplay, WatchlistsHelpTooltip } from "../shared"
import { trackWatchlistsPreventionTelemetry } from "@/utils/watchlists-prevention-telemetry"
import {
  buildCronFromPreset,
  createDefaultPresetState,
  INTERVAL_HOURS_MAX,
  INTERVAL_HOURS_MIN,
  INTERVAL_MINUTES_MAX,
  INTERVAL_MINUTES_MIN,
  parsePresetFromCron,
  type PresetScheduleState,
  type ScheduleIntervalUnit,
  type SchedulePresetKey,
  type WeekdayToken
} from "./schedule-utils"
import { analyzeScheduleFrequency, MIN_SCHEDULE_INTERVAL_MINUTES } from "./schedule-frequency"

interface SchedulePickerProps {
  value: string | null | undefined
  onChange: (schedule: string | null) => void
  timezone?: string
  onTimezoneChange?: (tz: string) => void
}

const TIMEZONES = [
  { value: "UTC", label: "UTC" },
  { value: "America/New_York", label: "US Eastern" },
  { value: "America/Chicago", label: "US Central" },
  { value: "America/Denver", label: "US Mountain" },
  { value: "America/Los_Angeles", label: "US Pacific" },
  { value: "Europe/London", label: "London" },
  { value: "Europe/Paris", label: "Paris" },
  { value: "Europe/Berlin", label: "Berlin" },
  { value: "Asia/Tokyo", label: "Tokyo" },
  { value: "Asia/Shanghai", label: "Shanghai" },
  { value: "Asia/Singapore", label: "Singapore" },
  { value: "Australia/Sydney", label: "Sydney" }
]

const WEEKDAY_OPTIONS: Array<{ value: WeekdayToken; label: string }> = [
  { value: "MON", label: "Monday" },
  { value: "TUE", label: "Tuesday" },
  { value: "WED", label: "Wednesday" },
  { value: "THU", label: "Thursday" },
  { value: "FRI", label: "Friday" },
  { value: "SAT", label: "Saturday" },
  { value: "SUN", label: "Sunday" }
]

const INTERVAL_UNIT_OPTIONS: Array<{ value: ScheduleIntervalUnit; label: string }> = [
  { value: "hours", label: "Hours" },
  { value: "minutes", label: "Minutes" }
]

const CRON_FIELDS = 5
const CRON_TOKEN_PATTERN = /^[A-Z0-9*,/?-]+$/i
const CRON_EXAMPLES = [
  {
    id: "daily0900",
    labelKey: "watchlists:schedule.examples.daily0900",
    fallbackLabel: "Daily 09:00",
    expression: "0 9 * * *"
  },
  {
    id: "weekdays0800",
    labelKey: "watchlists:schedule.examples.weekdays0800",
    fallbackLabel: "Weekdays 08:00",
    expression: "0 8 * * MON-FRI"
  },
  {
    id: "every6hours",
    labelKey: "watchlists:schedule.examples.every6hours",
    fallbackLabel: "Every 6 hours",
    expression: "0 */6 * * *"
  }
]

type CronFormatValidationResult = "field_count" | "invalid_token" | null

const validateCronFormat = (expression: string): CronFormatValidationResult => {
  const tokens = expression.trim().split(/\s+/)
  if (tokens.length !== CRON_FIELDS) return "field_count"
  if (tokens.some((token) => !CRON_TOKEN_PATTERN.test(token))) return "invalid_token"
  return null
}

export const SchedulePicker: React.FC<SchedulePickerProps> = ({
  value,
  onChange,
  timezone = "UTC",
  onTimezoneChange
}) => {
  const { t } = useTranslation(["watchlists"])
  const [customCron, setCustomCron] = useState(value || "")
  const [advancedMode, setAdvancedMode] = useState(false)
  const [presetState, setPresetState] = useState<PresetScheduleState>(createDefaultPresetState())
  const [customValidationError, setCustomValidationError] = useState<string | null>(null)

  const getCronFormatError = (expression: string): string | null => {
    const validationResult = validateCronFormat(expression)
    if (validationResult === "field_count") {
      return t(
        "watchlists:schedule.cronFieldCountError",
        "Use exactly 5 cron fields: minute hour day-of-month month day-of-week."
      )
    }
    if (validationResult === "invalid_token") {
      return t(
        "watchlists:schedule.cronInvalidTokenError",
        "Cron tokens can only include letters, numbers, *, /, -, ?, and comma."
      )
    }
    return null
  }

  const customCronFormatError = !advancedMode
    ? null
    : (() => {
        const normalized = customCron.trim()
        if (!normalized) return null
        return getCronFormatError(normalized)
      })()

  useEffect(() => {
    setCustomCron(value || "")
    const parsed = parsePresetFromCron(value)
    if (parsed) {
      setPresetState(parsed)
      setAdvancedMode(false)
      return
    }
    if (!value) {
      setPresetState(createDefaultPresetState())
      setAdvancedMode(false)
      return
    }
    setAdvancedMode(true)
  }, [value])

  const presetOptions = useMemo(
    () =>
      [
        {
          value: "interval",
          label: t("watchlists:schedule.preset.interval.label", "Every N hours/minutes"),
          description: t(
            "watchlists:schedule.preset.interval.description",
            "Runs repeatedly at the interval you choose."
          )
        },
        {
          value: "daily",
          label: t("watchlists:schedule.preset.daily.label", "Daily"),
          description: t(
            "watchlists:schedule.preset.daily.description",
            "Runs every day at the selected time."
          )
        },
        {
          value: "weekdays",
          label: t("watchlists:schedule.preset.weekdays.label", "Weekdays"),
          description: t(
            "watchlists:schedule.preset.weekdays.description",
            "Runs Monday through Friday at the selected time."
          )
        },
        {
          value: "weekly",
          label: t("watchlists:schedule.preset.weekly.label", "Weekly"),
          description: t(
            "watchlists:schedule.preset.weekly.description",
            "Runs once a week on the selected weekday and time."
          )
        }
      ] as Array<{ value: SchedulePresetKey; label: string; description: string }>,
    [t]
  )

  const selectedPreset = presetOptions.find((item) => item.value === presetState.preset)

  const applyPreset = (nextState: PresetScheduleState) => {
    onChange(buildCronFromPreset(nextState))
  }

  const updatePresetState = (updater: (previous: PresetScheduleState) => PresetScheduleState) => {
    setPresetState((previous) => {
      const next = updater(previous)
      if (!advancedMode) {
        applyPreset(next)
      }
      return next
    })
  }

  const handlePresetChange = (preset: SchedulePresetKey) => {
    updatePresetState((previous) => ({ ...previous, preset }))
  }

  const handleIntervalValueChange = (value: number | null) => {
    updatePresetState((previous) => {
      const min =
        previous.intervalUnit === "minutes" ? INTERVAL_MINUTES_MIN : INTERVAL_HOURS_MIN
      const max =
        previous.intervalUnit === "minutes" ? INTERVAL_MINUTES_MAX : INTERVAL_HOURS_MAX
      const parsed = typeof value === "number" && value >= min ? Math.floor(value) : min
      return {
        ...previous,
        intervalValue: Math.min(max, parsed)
      }
    })
  }

  const handleIntervalUnitChange = (intervalUnit: ScheduleIntervalUnit) => {
    updatePresetState((previous) => {
      const min = intervalUnit === "minutes" ? INTERVAL_MINUTES_MIN : INTERVAL_HOURS_MIN
      const max = intervalUnit === "minutes" ? INTERVAL_MINUTES_MAX : INTERVAL_HOURS_MAX
      return {
        ...previous,
        intervalUnit,
        intervalValue: Math.min(max, Math.max(min, previous.intervalValue))
      }
    })
  }

  const handleMinuteChange = (value: number | null) => {
    updatePresetState((previous) => ({
      ...previous,
      minute: typeof value === "number" && value >= 0 ? Math.min(59, Math.floor(value)) : 0
    }))
  }

  const handleHourChange = (value: number | null) => {
    updatePresetState((previous) => ({
      ...previous,
      hour: typeof value === "number" && value >= 0 ? Math.min(23, Math.floor(value)) : 0
    }))
  }

  const handleWeekdayChange = (weekday: WeekdayToken) => {
    updatePresetState((previous) => ({ ...previous, weekday }))
  }

  const handleAdvancedToggle = (checked: boolean) => {
    setAdvancedMode(checked)
    if (!checked) {
      setCustomValidationError(null)
      applyPreset(presetState)
    } else {
      setCustomCron(value || buildCronFromPreset(presetState))
    }
  }

  const handleCustomApply = () => {
    const normalized = customCron.trim()
    if (!normalized) return
    const formatError = getCronFormatError(normalized)
    if (formatError) {
      setCustomValidationError(formatError)
      return
    }
    const frequency = analyzeScheduleFrequency(normalized, MIN_SCHEDULE_INTERVAL_MINUTES)
    if (frequency.tooFrequent) {
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_validation_blocked",
        surface: "schedule_picker",
        rule: "schedule_too_frequent",
        remediation: "increase_interval",
        minutes: MIN_SCHEDULE_INTERVAL_MINUTES
      })
      setCustomValidationError(
        t(
          "watchlists:schedule.tooFrequent",
          "Schedule is too frequent. Minimum interval is every {{minutes}} minutes.",
          { minutes: MIN_SCHEDULE_INTERVAL_MINUTES }
        )
      )
      return
    }
    setCustomValidationError(null)
    onChange(normalized)
  }

  const handleClear = () => {
    setCustomCron("")
    setCustomValidationError(null)
    setPresetState(createDefaultPresetState())
    setAdvancedMode(false)
    onChange(null)
  }

  return (
    <div className="space-y-4">
      {value && (
        <div className="flex items-center justify-between rounded-lg border border-primary/30 bg-primary/10 p-3">
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-primary" />
            <CronDisplay expression={value} showIcon={false} />
          </div>
          <Button size="small" onClick={handleClear}>
            {t("watchlists:schedule.clear", "Clear")}
          </Button>
        </div>
      )}

      <div className="rounded-lg border border-border p-3">
        <div className="mb-3 text-sm font-medium">
          {t("watchlists:schedule.presets", "Schedule presets")}
        </div>
        <Select
          value={presetState.preset}
          onChange={(value) => handlePresetChange(value as SchedulePresetKey)}
          options={presetOptions}
          className="w-full"
        />
        <div className="mt-2 text-xs text-text-muted">
          {selectedPreset?.description}
        </div>
        <div className="mt-1 text-xs text-text-muted">
          {t(
            "watchlists:schedule.beginnerHint",
            "Most users should use presets. Turn on cron only for uncommon timing."
          )}
        </div>

        <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-3">
          {presetState.preset === "interval" && (
            <>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:schedule.intervalValue", "Every")}
                </div>
                <InputNumber
                  aria-label={t("watchlists:schedule.intervalValueA11y", "Interval value")}
                  min={
                    presetState.intervalUnit === "minutes"
                      ? INTERVAL_MINUTES_MIN
                      : INTERVAL_HOURS_MIN
                  }
                  max={
                    presetState.intervalUnit === "minutes"
                      ? INTERVAL_MINUTES_MAX
                      : INTERVAL_HOURS_MAX
                  }
                  precision={0}
                  value={presetState.intervalValue}
                  onChange={handleIntervalValueChange}
                  className="w-full"
                />
              </div>
              <div>
                <div className="mb-1 text-xs text-text-muted">
                  {t("watchlists:schedule.intervalUnit", "Unit")}
                </div>
                <Select
                  aria-label={t("watchlists:schedule.intervalUnitA11y", "Interval unit")}
                  value={presetState.intervalUnit}
                  onChange={(value) => handleIntervalUnitChange(value as ScheduleIntervalUnit)}
                  options={INTERVAL_UNIT_OPTIONS.map((item) => ({
                    value: item.value,
                    label: t(`watchlists:schedule.intervalUnitOption.${item.value}`, item.label)
                  }))}
                  className="w-full"
                />
              </div>
            </>
          )}
          {(presetState.preset === "daily" ||
            presetState.preset === "weekdays" ||
            presetState.preset === "weekly") && (
            <div>
              <div className="mb-1 text-xs text-text-muted">
                {t("watchlists:schedule.hour", "Hour")}
              </div>
              <InputNumber
                aria-label={t("watchlists:schedule.hourA11y", "Hour")}
                min={0}
                max={23}
                precision={0}
                value={presetState.hour}
                onChange={handleHourChange}
                className="w-full"
              />
            </div>
          )}
          {(presetState.preset !== "interval" || presetState.intervalUnit === "hours") && (
            <div>
              <div className="mb-1 text-xs text-text-muted">
                {t("watchlists:schedule.minute", "Minute")}
              </div>
              <InputNumber
                aria-label={t("watchlists:schedule.minuteA11y", "Minute")}
                min={0}
                max={59}
                precision={0}
                value={presetState.minute}
                onChange={handleMinuteChange}
                className="w-full"
              />
            </div>
          )}
          {presetState.preset === "weekly" && (
            <div>
              <div className="mb-1 text-xs text-text-muted">
                {t("watchlists:schedule.weekday", "Weekday")}
              </div>
              <Select
                aria-label={t("watchlists:schedule.weekdayA11y", "Weekday")}
                value={presetState.weekday}
                onChange={(value) => handleWeekdayChange(value as WeekdayToken)}
                options={WEEKDAY_OPTIONS.map((item) => ({
                  value: item.value,
                  label: t(`watchlists:schedule.weekdayOption.${item.value}`, item.label)
                }))}
                className="w-full"
              />
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between rounded-md border border-border px-3 py-2">
        <div>
          <div className="flex items-center gap-1 text-sm font-medium">
            {t("watchlists:schedule.advancedLabel", "Use custom cron (advanced)")}
            <WatchlistsHelpTooltip topic="cron" />
          </div>
          <div className="text-xs text-text-muted">
            {t(
              "watchlists:schedule.advancedOptionalHint",
              "Most users should use presets. Turn on cron only for uncommon timing."
            )}
          </div>
        </div>
        <Switch checked={advancedMode} onChange={handleAdvancedToggle} />
      </div>

      {advancedMode && (
        <div>
          <div className="mb-2 text-sm font-medium">
            {t("watchlists:schedule.custom", "Custom schedule")}
          </div>
          <div className="flex gap-2">
            <Input
              placeholder={t(
                "watchlists:schedule.cronPlaceholder",
                "Cron expression (advanced), e.g., 0 9 * * MON"
              )}
              value={customCron}
              onChange={(event) => {
                setCustomCron(event.target.value)
                setCustomValidationError(null)
              }}
              className="flex-1"
              onPressEnter={handleCustomApply}
            />
            <Button
              type="primary"
              onClick={handleCustomApply}
              disabled={!customCron.trim() || Boolean(customCronFormatError)}
            >
              {t("watchlists:schedule.apply", "Apply")}
            </Button>
          </div>
          <div className="mt-2 text-xs text-text-muted">
            {t(
              "watchlists:schedule.cronFieldOrderHint",
              "Field order: minute hour day-of-month month day-of-week."
            )}
          </div>
          <div className="mt-2 text-xs text-text-muted">
            {t(
              "watchlists:schedule.cronBeginnerHint",
              "If cron is new, start with a quick example below and edit one field at a time."
            )}
          </div>
          <div className="mt-2 rounded-md border border-border bg-surface p-2">
            <div className="text-xs font-medium text-text-muted">
              {t("watchlists:schedule.examplesTitle", "Quick examples")}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {CRON_EXAMPLES.map((example) => (
                <Button
                  key={example.id}
                  size="small"
                  data-testid={`schedule-example-${example.id}`}
                  onClick={() => {
                    setCustomCron(example.expression)
                    setCustomValidationError(null)
                  }}
                >
                  {t(example.labelKey, example.fallbackLabel)}
                </Button>
              ))}
            </div>
            <div className="mt-2 text-xs text-text-muted">
              {t(
                "watchlists:schedule.examplesHint",
                "Choose one, then adjust values if you need a different cadence."
              )}
            </div>
          </div>
          {customValidationError ? (
            <div className="mt-2 text-xs text-danger">{customValidationError}</div>
          ) : null}
          {!customValidationError && customCronFormatError ? (
            <div className="mt-2 text-xs text-danger">{customCronFormatError}</div>
          ) : null}
          {customCron.trim().length > 0 && (
            <div className="mt-2 text-sm text-text-muted">
              <span className="font-medium">{t("watchlists:schedule.preview", "Preview")}:</span>{" "}
              <CronDisplay expression={customCron} showIcon={false} />
            </div>
          )}
        </div>
      )}

      {onTimezoneChange && (
        <div>
          <div className="mb-2 text-sm font-medium">
            {t("watchlists:schedule.timezone", "Timezone")}
          </div>
          <Select
            value={timezone}
            onChange={onTimezoneChange}
            className="w-56"
            options={TIMEZONES}
            showSearch
            optionFilterProp="label"
          />
        </div>
      )}

      <div className="text-xs text-text-muted">
        {advancedMode
          ? t(
              "watchlists:schedule.helpAdvanced",
              "Cron format: minute hour day-of-month month day-of-week. Example: 0 9 * * MON runs every Monday at 09:00."
            )
          : t(
              "watchlists:schedule.helpSimple",
              "Use a preset above for most schedules. Turn on Advanced cron only if you need uncommon timing patterns."
            )}
      </div>
    </div>
  )
}
