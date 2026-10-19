import { Slider, Tooltip } from "antd"
import { RotateCcw } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { classNames } from "@/libs/class-name"

/**
 * One sampling parameter in the side panel's Pro "Model Parameters" list
 * (XS-16).
 *
 * An unset value is not sent, so the provider's own default applies. The
 * field says "Default" and shows a muted slider until the user moves it, and
 * a set value can be reset back to the provider default.
 */
type ModelParamSliderFieldProps = {
  label: string
  tooltip: string
  value: number | undefined
  min: number
  max: number
  step: number
  /** Decimal places for the readout and slider tooltip. */
  precision: number
  /** Where the muted slider rests while the value is unset. */
  restingValue: number
  onChange: (value: number) => void
  onReset: () => void
}

export const ModelParamSliderField = ({
  label,
  tooltip,
  value,
  min,
  max,
  step,
  precision,
  restingValue,
  onChange,
  onReset
}: ModelParamSliderFieldProps) => {
  const { t } = useTranslation(["sidepanel"])
  const isSet = typeof value === "number" && Number.isFinite(value)
  const defaultLabel = t("sidepanel:modelParams.default", "Default")
  return (
    <div className="space-y-1" data-param-field>
      <div className="flex items-center justify-between gap-2">
        <Tooltip title={tooltip}>
          <label className="text-xs text-text-muted cursor-help">{label}</label>
        </Tooltip>
        <span className="flex items-center gap-1">
          <span
            className={classNames(
              "text-xs tabular-nums",
              isSet ? "text-text-subtle" : "text-text-muted italic"
            )}
          >
            {isSet ? value : defaultLabel}
          </span>
          {isSet && (
            <button
              type="button"
              onClick={onReset}
              aria-label={t(
                "sidepanel:modelParams.resetToDefault",
                "Reset {{label}} to default",
                { label }
              )}
              title={t(
                "sidepanel:modelParams.resetToDefault",
                "Reset {{label}} to default",
                { label }
              )}
              className="inline-flex h-5 w-5 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              <RotateCcw className="size-3" aria-hidden="true" />
            </button>
          )}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={isSet ? value : restingValue}
        onChange={(next) => onChange(next)}
        className={classNames("my-0", !isSet && "opacity-50")}
        aria-label={label}
        ariaValueTextFormatterForHandle={(next) =>
          isSet && typeof next === "number" ? next.toFixed(precision) : defaultLabel
        }
        tooltip={{
          formatter: (next) =>
            isSet && typeof next === "number" ? next.toFixed(precision) : defaultLabel
        }}
      />
    </div>
  )
}
