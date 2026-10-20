import { Slider, Tooltip } from "antd"
import { RotateCcw } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { classNames } from "@/libs/class-name"

/**
 * One sampling parameter in the side panel's Pro "Model Parameters" list
 * (XS-16).
 *
 * An unset value reads "Default (x)", where x is what the request sends for
 * it, and the muted slider rests at x until the user moves it. When the
 * request sends nothing for the parameter (no defaultValue) it reads
 * "Default". A value set for this model can be reset; a value that only comes
 * from the all-models settings reads "x (all models)" and has no reset here.
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
  /** Where the muted slider rests while the value is unset and nothing is sent. */
  restingValue: number
  /** What the request sends while the value is unset, if anything. */
  defaultValue?: number
  /** False when the value only comes from the all-models settings. */
  isSetHere?: boolean
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
  defaultValue,
  isSetHere = true,
  onChange,
  onReset
}: ModelParamSliderFieldProps) => {
  const { t } = useTranslation(["sidepanel"])
  const isSet = typeof value === "number" && Number.isFinite(value)
  const isInherited = isSet && !isSetHere
  const defaultLabel =
    typeof defaultValue === "number"
      ? t("sidepanel:modelParams.defaultWithValue", "Default ({{value}})", {
          value: defaultValue
        })
      : t("sidepanel:modelParams.default", "Default")
  const readout = !isSet
    ? defaultLabel
    : isInherited
      ? t("sidepanel:modelParams.allModelsValue", "{{value}} (all models)", {
          value
        })
      : String(value)
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
              isSet && !isInherited
                ? "text-text-subtle"
                : "text-text-muted italic"
            )}
            title={
              isInherited
                ? t(
                    "sidepanel:modelParams.allModelsHint",
                    "Set for all models. Move the slider to change it for this model only."
                  )
                : undefined
            }
          >
            {readout}
          </span>
          {isSet && !isInherited && (
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
        value={isSet ? value : defaultValue ?? restingValue}
        onChange={(next) => onChange(next)}
        className={classNames("my-0", (!isSet || isInherited) && "opacity-50")}
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
