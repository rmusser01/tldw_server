import React, { useEffect, useRef } from "react"
import { Select, Slider, Switch } from "antd"
import { useTranslation } from "react-i18next"
import { VoicePreviewButton } from "@/components/Common/VoicePreviewButton"
import { TTS_PROVIDER_OPTIONS } from "@/services/tts-providers"

type Props = {
  provider: string
  model: string
  voice: string
  backend?: string
  allowFallback?: boolean
  onProviderChange: (value: string) => void
  onModelChange: (value: string) => void
  onVoiceChange: (value: string) => void
  modelOptions: { label: string; value: string }[]
  voiceOptions: { label: string; value: string }[]
  language?: string
  onLanguageChange?: (value: string) => void
  languageOptions?: { label: string; value: string }[]
  emotion?: string
  onEmotionChange?: (value: string) => void
  emotionIntensity?: number
  onEmotionIntensityChange?: (value: number) => void
  supportsEmotion?: boolean
  useVoiceRoles?: boolean
  onVoiceRolesChange?: (value: boolean) => void
  voiceRolesContent?: React.ReactNode
  focusField: string | null
  onFocusHandled: () => void
}

const showModel = (provider: string) => provider !== "browser"
const showEmotion = (provider: string) => provider === "tldw"
const showVoiceRoles = (provider: string) => provider === "tldw"

export const TtsVoiceTab: React.FC<Props> = (props) => {
  const { t } = useTranslation("playground")
  const modelRef = useRef<any>(null)
  const voiceRef = useRef<any>(null)

  useEffect(() => {
    if (!props.focusField) return
    const timer = setTimeout(() => {
      if (props.focusField === "model") modelRef.current?.focus?.()
      if (props.focusField === "voice") voiceRef.current?.focus?.()
      props.onFocusHandled()
    }, 100)
    return () => clearTimeout(timer)
  }, [props.focusField, props.onFocusHandled])

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm text-text mb-1 block">Provider</label>
        <Select
          className="w-full"
          value={props.provider}
          onChange={props.onProviderChange}
          options={TTS_PROVIDER_OPTIONS.map(({ label, value }) => ({ label, value }))}
        />
      </div>
      {showModel(props.provider) && (
        <div>
          <label className="text-sm text-text mb-1 block">Model</label>
          <Select
            ref={modelRef}
            className="w-full"
            value={props.model}
            onChange={props.onModelChange}
            options={props.modelOptions}
            showSearch
            optionFilterProp="label"
          />
        </div>
      )}
      <div>
        <label className="text-sm text-text mb-1 block">Voice</label>
        <Select
          ref={voiceRef}
          className="w-full"
          value={props.voice}
          onChange={props.onVoiceChange}
          options={props.voiceOptions}
          showSearch
          optionFilterProp="label"
        />
        {props.provider !== "browser" && (
          <div className="mt-1">
            <VoicePreviewButton
              model={props.model}
              voice={props.voice}
              provider={props.provider}
              backend={props.backend}
              allowFallback={props.allowFallback}
            />
          </div>
        )}
      </div>
      {props.languageOptions && props.languageOptions.length > 0 && (
        <div>
          <label className="text-sm text-text mb-1 block">Language</label>
          <Select
            className="w-full"
            value={props.language}
            onChange={props.onLanguageChange}
            options={props.languageOptions}
            allowClear
            placeholder="Auto"
          />
        </div>
      )}
      {showEmotion(props.provider) && props.supportsEmotion && (
        <>
          <div>
            <label className="text-sm text-text mb-1 block">Emotion preset</label>
            <Select
              className="w-full"
              value={props.emotion}
              onChange={props.onEmotionChange}
              allowClear
              placeholder="Default"
              options={[
                { label: "Neutral", value: "neutral" },
                { label: "Calm", value: "calm" },
                { label: "Energetic", value: "energetic" },
                { label: "Happy", value: "happy" },
                { label: "Sad", value: "sad" },
                { label: "Angry", value: "angry" }
              ]}
            />
          </div>
          <div>
            <label className="text-sm text-text mb-1 block">Emotion intensity</label>
            <Slider
              min={0.1}
              max={2}
              step={0.1}
              value={props.emotionIntensity ?? 1}
              onChange={props.onEmotionIntensityChange}
            />
          </div>
        </>
      )}
      {showVoiceRoles(props.provider) && props.onVoiceRolesChange && (
        <div className="rounded-md border border-border p-3 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-text">Multi-voice narration</div>
              <div className="text-xs text-text-subtle">
                Assign roles to 1–4 voices for multi-speaker output.
              </div>
            </div>
            <Switch
              size="small"
              checked={props.useVoiceRoles ?? false}
              onChange={props.onVoiceRolesChange}
            />
          </div>
          {props.useVoiceRoles && props.voiceRolesContent}
        </div>
      )}
    </div>
  )
}
