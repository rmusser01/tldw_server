import React from "react"
import { Volume2, Settings } from "lucide-react"
import { Button, Segmented, Tag, Tooltip } from "antd"
import type { TtsPresetKey } from "@/hooks/useTtsPlayground"

type Props = {
  provider: string
  model: string
  voice: string
  format: string
  speed: number
  presetValue: TtsPresetKey
  onPresetChange: (preset: TtsPresetKey) => void
  onLabelClick: (tab: "voice" | "output" | "advanced", field?: string) => void
  onGearClick: () => void
}

const PRESET_TOOLTIPS: Record<TtsPresetKey, string> = {
  fast: "Streaming on, mp3, punctuation split, 1.2x",
  balanced: "No streaming, mp3, punctuation split, 1.0x",
  quality: "No streaming, wav, paragraph split, 0.9x"
}

export function TtsProviderStrip({
  provider,
  model,
  voice,
  format,
  speed,
  presetValue,
  onPresetChange,
  onLabelClick,
  onGearClick
}: Props) {
  const isBrowser = provider === "browser"

  return (
    <div className="flex flex-wrap items-center gap-2" aria-live="polite">
      <Volume2 className="h-4 w-4 shrink-0" />

      {isBrowser ? (
        <>
          <Tag className="cursor-pointer">Browser local output</Tag>
          {voice && (
            <Tooltip title={`Voice: ${voice}`}>
              <Tag
                className="cursor-pointer"
                onClick={() => onLabelClick("voice", "voice")}
              >
                {voice}
              </Tag>
            </Tooltip>
          )}
        </>
      ) : (
        <>
          <Tooltip title={`Model: ${model}`}>
            <Tag
              className="cursor-pointer"
              onClick={() => onLabelClick("voice", "model")}
            >
              {model}
            </Tag>
          </Tooltip>

          <Tooltip title={`Voice: ${voice}`}>
            <Tag
              className="cursor-pointer"
              onClick={() => onLabelClick("voice", "voice")}
            >
              {voice}
            </Tag>
          </Tooltip>

          <Tooltip title={`Format: ${format}`}>
            <Tag
              className="cursor-pointer"
              onClick={() => onLabelClick("output", "format")}
            >
              {format}
            </Tag>
          </Tooltip>

          {speed !== 1 && (
            <Tooltip title={`Speed: ${speed}x`}>
              <Tag
                className="cursor-pointer"
                onClick={() => onLabelClick("output", "speed")}
              >
                {speed}x
              </Tag>
            </Tooltip>
          )}
        </>
      )}

      <div className="flex-1" />

      <Segmented
        size="small"
        value={presetValue}
        onChange={(val) => onPresetChange(val as TtsPresetKey)}
        options={(["fast", "balanced", "quality"] as TtsPresetKey[]).map(
          (key) => ({
            label: (
              <Tooltip title={PRESET_TOOLTIPS[key]}>
                <span>{key.charAt(0).toUpperCase() + key.slice(1)}</span>
              </Tooltip>
            ),
            value: key
          })
        )}
      />

      <Button
        type="text"
        size="small"
        icon={<Settings className="h-4 w-4" />}
        aria-label="Open configuration"
        onClick={onGearClick}
      />
    </div>
  )
}
