import React from "react"
import { Button, Checkbox, Input, Select, Tooltip, Typography } from "antd"
import { useTranslation } from "react-i18next"

import type { DeckSchedulerDraftState } from "../hooks/useDeckSchedulerDraft"
import type { SchedulerPresetId } from "../utils/scheduler-settings"
import { getSchedulerPresets } from "../utils/scheduler-settings"

const { Text } = Typography

type DeckSchedulerSettingsEditorProps = {
  schedulerDraft: DeckSchedulerDraftState
  advancedDefaultOpen?: boolean
}

type Sm2FieldConfig = {
  key: Exclude<keyof DeckSchedulerDraftState["draft"]["sm2_plus"], "enable_fuzz">
  label: string
  placeholder?: string
  testId: string
}

const SM2_ADVANCED_FIELDS: Sm2FieldConfig[] = [
  {
    key: "new_steps_minutes",
    label: "New steps (minutes)",
    placeholder: "1, 10",
    testId: "deck-scheduler-editor-field-new-steps"
  },
  {
    key: "relearn_steps_minutes",
    label: "Relearn steps (minutes)",
    placeholder: "10",
    testId: "deck-scheduler-editor-field-relearn-steps"
  },
  {
    key: "graduating_interval_days",
    label: "Graduating interval (days)",
    testId: "deck-scheduler-editor-field-graduating-interval"
  },
  {
    key: "easy_interval_days",
    label: "Easy interval (days)",
    testId: "deck-scheduler-editor-field-easy-interval"
  },
  {
    key: "easy_bonus",
    label: "Easy bonus",
    testId: "deck-scheduler-editor-field-easy-bonus"
  },
  {
    key: "interval_modifier",
    label: "Interval modifier",
    testId: "deck-scheduler-editor-field-interval-modifier"
  },
  {
    key: "max_interval_days",
    label: "Max interval (days)",
    testId: "deck-scheduler-editor-field-max-interval"
  },
  {
    key: "leech_threshold",
    label: "Leech threshold",
    testId: "deck-scheduler-editor-field-leech-threshold"
  }
]

/** Default tooltip strings for SM-2+ fields — used as i18n defaultValue fallbacks. */
const SM2_FIELD_TOOLTIP_DEFAULTS: Record<string, string> = {
  new_steps_minutes:
    "Minutes between first reviews of a new card. E.g. '1, 10' means you see it again after 1 min, then 10 min.",
  relearn_steps_minutes:
    "Minutes between re-reviews when you forget a card. E.g. '10' means one re-review after 10 min.",
  graduating_interval_days:
    "Days until the next review after a new card 'graduates' from learning. Higher = fewer reviews.",
  easy_interval_days:
    "Days until next review when you rate a new card 'Easy'. Should be higher than the graduating interval.",
  easy_bonus:
    "Multiplier applied when you rate 'Easy'. Values above 1.0 extend the interval further.",
  interval_modifier:
    "Global multiplier for all review intervals. Below 1.0 = more frequent reviews. Above 1.0 = less frequent.",
  max_interval_days:
    "The longest any card can wait between reviews, in days. Prevents cards from disappearing for too long.",
  leech_threshold:
    "Number of times you can forget a card before it's flagged as a 'leech' — a card that isn't sticking."
}

/** Default tooltip strings for FSRS fields — used as i18n defaultValue fallbacks. */
const FSRS_FIELD_TOOLTIP_DEFAULTS: Record<string, string> = {
  target_retention:
    "Your target recall rate (0.0–1.0). Higher values (e.g. 0.95) mean more frequent reviews but better memory. Default is 0.9 (90%).",
  maximum_interval_days:
    "The longest any card can wait between reviews, in days."
}

/** Default tooltip strings for scheduler presets — used as i18n defaultValue fallbacks. */
const PRESET_DESCRIPTION_DEFAULTS: Record<string, string> = {
  default: "Balanced settings suitable for most learners. Good starting point.",
  fast_acquisition: "Shorter intervals for rapid initial learning. More reviews per day.",
  conservative_review: "Longer intervals, gentler pace. Fewer daily reviews.",
  high_retention: "Higher target recall (95%). More reviews to keep cards fresh.",
  long_horizon: "Lower target recall (85%). Fewer reviews, spread over a longer time."
}

type FsrsFieldConfig = {
  key: Exclude<keyof DeckSchedulerDraftState["draft"]["fsrs"], "enable_fuzz">
  label: string
  placeholder?: string
  testId: string
}

const FSRS_FIELDS: FsrsFieldConfig[] = [
  {
    key: "target_retention",
    label: "Target retention",
    placeholder: "0.90",
    testId: "deck-scheduler-editor-field-target-retention"
  },
  {
    key: "maximum_interval_days",
    label: "Maximum interval (days)",
    placeholder: "36500",
    testId: "deck-scheduler-editor-field-maximum-interval"
  }
]

export const DeckSchedulerSettingsEditor: React.FC<DeckSchedulerSettingsEditorProps> = ({
  schedulerDraft,
  advancedDefaultOpen = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const [advancedOpen, setAdvancedOpen] = React.useState(advancedDefaultOpen)
  const schedulerType = schedulerDraft.draft.scheduler_type
  const presets = React.useMemo(() => getSchedulerPresets(schedulerType), [schedulerType])

  const renderFieldError = React.useCallback(
    (field: string) => {
      const error =
        schedulerType === "fsrs"
          ? schedulerDraft.errors.fsrs[field as keyof typeof schedulerDraft.errors.fsrs]
          : schedulerDraft.errors.sm2_plus[field as keyof typeof schedulerDraft.errors.sm2_plus]
      if (!error) return null
      return (
        <Text type="danger" className="text-xs">
          {error}
        </Text>
      )
    },
    [schedulerDraft.errors.fsrs, schedulerDraft.errors.sm2_plus, schedulerType]
  )

  const renderPresetButtons = () => (
    <div className="space-y-2">
      <Text strong>
        {t("option:flashcards.schedulerPresets", { defaultValue: "Presets" })}
      </Text>
      <div className="flex flex-wrap gap-2">
        {presets.map((preset) => (
          <Tooltip key={preset.id} title={t(`option:flashcards.schedulerPresetDesc.${preset.id}`, { defaultValue: PRESET_DESCRIPTION_DEFAULTS[preset.id] ?? "" })}>
            <Button
              size="small"
              onClick={() => schedulerDraft.applyPreset(preset.id as SchedulerPresetId)}
              data-testid={`deck-scheduler-editor-preset-${preset.id}`}
            >
              {preset.label}
            </Button>
          </Tooltip>
        ))}
        <Button
          size="small"
          onClick={schedulerDraft.resetToDefaults}
          data-testid="deck-scheduler-editor-reset"
        >
          {t("option:flashcards.schedulerResetAction", {
            defaultValue: "Reset to defaults"
          })}
        </Button>
      </div>
    </div>
  )

  return (
    <div className="space-y-4">
      <label className="flex flex-col gap-1">
        <Text strong>
          {t("option:flashcards.schedulerTypeLabel", {
            defaultValue: "Scheduler type"
          })}
        </Text>
        <Select
          value={schedulerType}
          onChange={(value) => schedulerDraft.updateSchedulerType(value)}
          options={[
            { value: "sm2_plus", label: "SM-2+" },
            { value: "fsrs", label: "FSRS" }
          ]}
          data-testid="deck-scheduler-editor-field-scheduler-type"
        />
      </label>

      {renderPresetButtons()}

      <div className="rounded border border-border bg-muted/20 p-3">
        <Text
          type={schedulerDraft.summary ? "secondary" : "danger"}
          data-testid="deck-scheduler-editor-summary"
        >
          {schedulerDraft.summary ??
            t("option:flashcards.schedulerDraftInvalid", {
              defaultValue: "Draft has validation errors."
            })}
        </Text>
      </div>

      {schedulerType === "sm2_plus" ? (
        <>
          <div>
            <Button
              type="text"
              size="small"
              onClick={() => setAdvancedOpen((current) => !current)}
              data-testid="deck-scheduler-editor-toggle-advanced"
            >
              {advancedOpen
                ? t("option:flashcards.hideAdvancedScheduler", {
                    defaultValue: "Hide advanced settings"
                  })
                : t("option:flashcards.showAdvancedScheduler", {
                    defaultValue: "Customize scheduler"
                  })}
            </Button>
          </div>

          {advancedOpen && (
            <div className="grid gap-4 md:grid-cols-2">
              {SM2_ADVANCED_FIELDS.map((field) => (
                <label key={field.key} className="flex flex-col gap-1">
                  <Tooltip trigger={["hover", "focus"]} title={t(`option:flashcards.schedulerTooltip.${field.key}`, { defaultValue: SM2_FIELD_TOOLTIP_DEFAULTS[field.key] })}>
                    <Text strong tabIndex={0} className="cursor-help underline decoration-dotted">{field.label}</Text>
                  </Tooltip>
                  <Input
                    value={schedulerDraft.draft.sm2_plus[field.key]}
                    onChange={(event) =>
                      schedulerDraft.updateSm2Field(field.key, event.target.value)
                    }
                    placeholder={field.placeholder}
                    data-testid={field.testId}
                  />
                  {renderFieldError(field.key)}
                </label>
              ))}

              <div className="md:col-span-2">
                <Checkbox
                  checked={schedulerDraft.draft.sm2_plus.enable_fuzz}
                  onChange={(event) =>
                    schedulerDraft.updateSm2Field("enable_fuzz", event.target.checked)
                  }
                  data-testid="deck-scheduler-editor-field-enable-fuzz"
                >
                  {t("option:flashcards.schedulerEnableFuzz", {
                    defaultValue: "Enable review fuzz"
                  })}
                </Checkbox>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {FSRS_FIELDS.map((field) => (
            <label key={field.key} className="flex flex-col gap-1">
              <Tooltip trigger={["hover", "focus"]} title={t(`option:flashcards.schedulerTooltip.${field.key}`, { defaultValue: FSRS_FIELD_TOOLTIP_DEFAULTS[field.key] })}>
                <Text strong tabIndex={0} className="cursor-help underline decoration-dotted">{field.label}</Text>
              </Tooltip>
              <Input
                value={schedulerDraft.draft.fsrs[field.key]}
                onChange={(event) =>
                  schedulerDraft.updateFsrsField(field.key, event.target.value)
                }
                placeholder={field.placeholder}
                data-testid={field.testId}
              />
              {renderFieldError(field.key)}
            </label>
          ))}

          <div className="md:col-span-2">
            <Checkbox
              checked={schedulerDraft.draft.fsrs.enable_fuzz}
              onChange={(event) =>
                schedulerDraft.updateFsrsField("enable_fuzz", event.target.checked)
              }
              data-testid="deck-scheduler-editor-field-fsrs-enable-fuzz"
            >
              {t("option:flashcards.schedulerEnableFuzz", {
                defaultValue: "Enable review fuzz"
              })}
            </Checkbox>
          </div>
        </div>
      )}
    </div>
  )
}

export default DeckSchedulerSettingsEditor
