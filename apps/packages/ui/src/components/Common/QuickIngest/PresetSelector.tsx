import React from "react"
import type { IngestPreset, WizardQueueItem } from "./types"
import { PRESET_META, DEFAULT_PRESET } from "./presets"

type PresetSelectorProps = {
  /**
   * Translation function for quick-ingest namespace.
   */
  qi: (key: string, defaultValue: string, options?: Record<string, unknown>) => string
  /**
   * Currently selected preset.
   */
  value: IngestPreset
  /**
   * Callback when preset changes.
   */
  onChange: (preset: IngestPreset) => void
  /**
   * Callback to reset to default preset.
   */
  onReset?: () => void
  /**
   * Whether the selector is disabled.
   */
  disabled?: boolean
  /**
   * Current queue (accepted for compatibility with existing wizard callers).
   */
  queueItems?: WizardQueueItem[]
}

/**
 * Card presets displayed in order. "custom" is not a card.
 */
const CARD_PRESETS: Exclude<IngestPreset, "custom">[] = ["quick", "standard", "deep"]

/**
 * Plain-language descriptions used as fallback when i18n key is empty.
 */
const CARD_DESCRIPTIONS: Record<Exclude<IngestPreset, "custom">, string> = {
  quick: "Basic transcript only",
  standard: "Transcript + analysis + chunking",
  deep: "Full analysis + diarization",
}

const PresetCard: React.FC<{
  preset: Exclude<IngestPreset, "custom">
  selected: boolean
  disabled: boolean
  qi: PresetSelectorProps["qi"]
  onClick: () => void
}> = ({ preset, selected, disabled, qi, onClick }) => {
  const meta = PRESET_META[preset]
  const isRecommended = preset === DEFAULT_PRESET

  const label = qi(
    meta.labelKey,
    preset.charAt(0).toUpperCase() + preset.slice(1)
  )
  const description = qi(meta.descriptionKey, CARD_DESCRIPTIONS[preset])

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={selected}
      aria-label={`${label} preset${isRecommended ? " (Recommended)" : ""}`}
      className={[
        "relative flex flex-col items-center gap-2 rounded-lg border-2 px-4 py-4 text-center",
        "transition-colors duration-150 cursor-pointer",
        "min-w-[140px] flex-1",
        selected
          ? "border-primary bg-primary/5 ring-2 ring-primary/30"
          : "border-border-default bg-bg-surface hover:border-primary/50",
        disabled ? "opacity-50 cursor-not-allowed" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {/* Icon */}
      <span className="text-2xl" aria-hidden="true">
        {meta.icon}
      </span>

      {/* Label */}
      <span className="text-sm font-semibold text-text-default">{label}</span>

      {/* Recommended badge */}
      {isRecommended && (
        <span className="absolute -top-2.5 right-2 rounded-full bg-primary px-2 py-0.5 text-[10px] font-medium text-white">
          {qi("preset.recommended", "Recommended")}
        </span>
      )}

      {/* Selected indicator */}
      {selected && (
        <span
          className="absolute top-2 left-2 h-2.5 w-2.5 rounded-full bg-primary"
          aria-hidden="true"
        />
      )}

      {/* Description */}
      <span className="text-xs leading-snug text-text-muted">{description}</span>
    </button>
  )
}

export const PresetSelector: React.FC<PresetSelectorProps> = ({
  qi,
  value,
  onChange,
  onReset,
  disabled = false,
}) => {
  const handleCardClick = React.useCallback(
    (preset: Exclude<IngestPreset, "custom">) => {
      if (!disabled) {
        onChange(preset)
      }
    },
    [onChange, disabled]
  )

  return (
    <div className="flex flex-col gap-3">
      {/* Card grid */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {CARD_PRESETS.map((preset) => (
          <PresetCard
            key={preset}
            preset={preset}
            selected={value === preset}
            disabled={disabled}
            qi={qi}
            onClick={() => handleCardClick(preset)}
          />
        ))}
      </div>

      {/* Preset helper line + reset */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <span className="text-xs text-text-muted">
          {qi(
            "preset.helper",
            "Presets are starting points. Adjust any settings below or in Advanced options to fit this run."
          )}
          {value === "custom"
            ? ` ${qi("preset.custom.active", "Using custom settings (no preset match).")}`
            : ""}
        </span>
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            disabled={disabled}
            className="text-xs text-primary hover:text-primaryStrong underline underline-offset-2 disabled:opacity-50"
          >
            {qi("preset.reset", "Reset to defaults")}
          </button>
        )}
      </div>
    </div>
  )
}

export default PresetSelector
