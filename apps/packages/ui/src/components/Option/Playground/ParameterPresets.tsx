import React from "react"
import { Tooltip, Segmented } from "antd"
import { useTranslation } from "react-i18next"
import { useStoreChatModelSettings, type ChatModelSettings } from "@/store/model"
import { Sparkles, Target, Scale, Sliders } from "lucide-react"

export type PresetKey = "creative" | "balanced" | "precise" | "custom"

export type ParameterPreset = {
  key: PresetKey
  label: string
  description: string
  icon: React.ReactNode
  settings: Partial<ChatModelSettings>
}

const PRESETS: ParameterPreset[] = [
  {
    key: "creative",
    label: "Creative",
    description: "Higher temperature for creative writing, brainstorming, and varied responses",
    icon: <Sparkles className="h-3.5 w-3.5" />,
    settings: {
      temperature: 1.2,
      topP: 0.95,
      topK: 100,
      frequencyPenalty: 0.3,
      presencePenalty: 0.3,
      repeatPenalty: 1.05
    }
  },
  {
    key: "balanced",
    label: "Balanced",
    description: "Default settings for general-purpose conversations",
    icon: <Scale className="h-3.5 w-3.5" />,
    settings: {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      frequencyPenalty: 0,
      presencePenalty: 0,
      repeatPenalty: 1.0
    }
  },
  {
    key: "precise",
    label: "Precise",
    description: "Lower temperature for factual, deterministic, and consistent responses",
    icon: <Target className="h-3.5 w-3.5" />,
    settings: {
      temperature: 0.2,
      topP: 0.8,
      topK: 20,
      frequencyPenalty: 0,
      presencePenalty: 0,
      repeatPenalty: 1.0
    }
  },
  {
    key: "custom",
    label: "Custom",
    description: "Fine-tune parameters manually",
    icon: <Sliders className="h-3.5 w-3.5" />,
    settings: {}
  }
]

const PRESET_SETTING_LABELS: Record<string, string> = {
  temperature: "Temperature",
  topP: "Top P",
  topK: "Top K",
  frequencyPenalty: "Frequency penalty",
  presencePenalty: "Presence penalty",
  repeatPenalty: "Repeat penalty",
  numPredict: "Max tokens",
  jsonMode: "JSON mode"
}

const formatPresetSettingEntries = (settings: Partial<ChatModelSettings>) =>
  Object.entries(settings)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => ({
      key,
      label: PRESET_SETTING_LABELS[key] || key,
      value: String(value)
    }))

type Props = {
  compact?: boolean
  onChange?: (preset: PresetKey) => void
  className?: string
}

export function detectCurrentPreset(settings: ChatModelSettings): PresetKey {
  const { temperature, topP, topK } = settings

  // Check if settings match any preset (with some tolerance)
  const tolerance = 0.05

  for (const preset of PRESETS) {
    if (preset.key === "custom") continue

    const presetTemp = preset.settings.temperature
    const presetTopP = preset.settings.topP
    const presetTopK = preset.settings.topK

    if (
      presetTemp !== undefined &&
      temperature !== undefined &&
      Math.abs(temperature - presetTemp) <= tolerance &&
      (presetTopP === undefined ||
        topP === undefined ||
        Math.abs(topP - presetTopP) <= tolerance) &&
      (presetTopK === undefined || topK === undefined || topK === presetTopK)
    ) {
      return preset.key
    }
  }

  return "custom"
}

export const getPresetByKey = (key: PresetKey): ParameterPreset | undefined =>
  PRESETS.find((preset) => preset.key === key)

export const ParameterPresets: React.FC<Props> = ({
  compact = false,
  onChange,
  className
}) => {
  const { t } = useTranslation(["playground", "common"])
  const settings = useStoreChatModelSettings()
  const updateSettings = useStoreChatModelSettings((s) => s.updateSettings)

  const currentPreset = React.useMemo(
    () => detectCurrentPreset(settings),
    [settings]
  )

  const handlePresetChange = (key: string | number) => {
    const presetKey = key as PresetKey
    const preset = PRESETS.find((p) => p.key === presetKey)

    if (preset && preset.key !== "custom") {
      updateSettings(preset.settings)
    }

    onChange?.(presetKey)
  }

  const options = PRESETS.map((preset) => {
    const presetLabel = t(
      `playground:presets.${preset.key}.label`,
      preset.label
    )

    return {
      label: (
        <Tooltip
          title={
            <div className="text-xs">
              <div className="font-medium">
                {presetLabel}
              </div>
              <div className="mt-1 text-text-subtle">
                {t(
                  `playground:presets.${preset.key}.description`,
                  preset.description
                )}
              </div>
              {preset.key !== "custom" && (
                <div className="mt-2 space-y-0.5 text-[10px] text-text-muted">
                  {formatPresetSettingEntries(preset.settings).map((entry) => (
                    <div key={`${preset.key}-${entry.key}`}>
                      {entry.label}: {entry.value}
                    </div>
                  ))}
                </div>
              )}
            </div>
          }
          placement="bottom"
          mouseEnterDelay={0.3}>
          <div className="flex items-center gap-1.5">
            {preset.icon}
            <span className={compact ? "sr-only" : undefined}>
              {presetLabel}
            </span>
          </div>
        </Tooltip>
      ),
      title: presetLabel,
      value: preset.key
    }
  })

  return (
    <div className={className}>
      <Segmented
        aria-label={
          compact
            ? t("playground:presets.generationStyle", "Generation style")
            : undefined
        }
        options={options}
        value={currentPreset}
        onChange={handlePresetChange}
        size={compact ? "small" : "middle"}
        className="parameter-presets-segmented"
      />
    </div>
  )
}

export const ParameterPresetsDropdown: React.FC<Props> = ({
  onChange,
  className
}) => {
  const { t } = useTranslation(["playground", "common"])
  const settings = useStoreChatModelSettings()
  const updateSettings = useStoreChatModelSettings((s) => s.updateSettings)

  const currentPreset = React.useMemo(
    () => detectCurrentPreset(settings),
    [settings]
  )

  const handlePresetSelect = (presetKey: PresetKey) => {
    const preset = PRESETS.find((p) => p.key === presetKey)
    if (preset && preset.key !== "custom") {
      updateSettings(preset.settings)
    }
    onChange?.(presetKey)
  }

  const currentPresetData = PRESETS.find((p) => p.key === currentPreset)

  return (
    <div className={`flex flex-col gap-2 ${className || ""}`}>
      <div className="text-xs font-medium text-text-muted">
        {t("playground:presets.title", "Parameter Presets")}
      </div>
      <div className="grid grid-cols-2 gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset.key}
            type="button"
            onClick={() => handlePresetSelect(preset.key)}
            className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors ${
              currentPreset === preset.key
                ? "border-primary bg-primary/10 text-primary"
                : "border-border bg-surface hover:border-primary/50 hover:bg-surface-hover"
            }`}>
            <span
              className={
                currentPreset === preset.key ? "text-primary" : "text-text-muted"
              }>
              {preset.icon}
            </span>
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">
                {t(`playground:presets.${preset.key}.label`, preset.label)}
              </div>
              {preset.key !== "custom" && (
                <div className="text-[10px] text-text-subtle">
                  T:{preset.settings.temperature} P:{preset.settings.topP}
                </div>
              )}
            </div>
          </button>
        ))}
      </div>
      {currentPreset === "custom" && (
        <div className="text-[10px] text-text-subtle">
          {t(
            "playground:presets.custom.hint",
            "Adjust parameters in the settings panel"
          )}
        </div>
      )}
    </div>
  )
}

export { PRESETS }
