import React from "react"
import { Button, Switch, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { TTS_PRESETS } from "@/hooks/useTtsPlayground"

const { Text } = Typography

type Props = {
  useDraftEditor: boolean
  onDraftEditorChange: (value: boolean) => void
  useTtsJob: boolean
  onTtsJobChange: (value: boolean) => void
  ssmlEnabled: boolean
  onSsmlChange: (value: boolean) => void
  removeReasoning: boolean
  onRemoveReasoningChange: (value: boolean) => void
  allowFallback: boolean
  onAllowFallbackChange: (value: boolean) => void
  isTldw: boolean
  onOpenVoiceCloning?: () => void
  voiceCloningContent?: React.ReactNode
}

const TOGGLE_ITEMS = (props: Props) => [
  {
    label: "Draft editor",
    description: "Outline + transcript mode for longform content.",
    checked: props.useDraftEditor,
    onChange: props.onDraftEditorChange
  },
  {
    label: "Use TTS Job",
    description: "Server-side job queue for long content. Progress tracked live.",
    checked: props.useTtsJob,
    onChange: props.onTtsJobChange,
    hidden: !props.isTldw
  },
  {
    label: "Allow configured fallback",
    description: "Permit the server to try an administrator-configured backup backend.",
    checked: props.allowFallback,
    onChange: props.onAllowFallbackChange,
    hidden: !props.isTldw
  },
  {
    label: "Enable SSML",
    description: "Speech Synthesis Markup Language tags.",
    checked: props.ssmlEnabled,
    onChange: props.onSsmlChange
  },
  {
    label: "Remove <think> tags",
    description: "Strip reasoning blocks before speaking.",
    checked: props.removeReasoning,
    onChange: props.onRemoveReasoningChange
  }
]

export const TtsAdvancedTab: React.FC<Props> = (props) => {
  const { t } = useTranslation("playground")

  return (
    <div className="space-y-4">
      {TOGGLE_ITEMS(props)
        .filter((item) => !item.hidden)
        .map(({ label, description, checked, onChange }) => (
          <div key={label} className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm text-text">{label}</div>
              <div className="text-xs text-text-subtle">{description}</div>
            </div>
            <Switch checked={checked} onChange={onChange} />
          </div>
        ))}

      {props.isTldw && (
        <div className="border-t border-border pt-3">
          <div className="text-sm text-text mb-2">Voice Cloning</div>
          {props.voiceCloningContent ?? (
            <Button size="small" onClick={props.onOpenVoiceCloning}>
              Manage custom voices
            </Button>
          )}
        </div>
      )}

      <div className="border-t border-border pt-3">
        <div className="text-sm text-text mb-2">Preset Reference</div>
        <div className="space-y-1">
          {Object.entries(TTS_PRESETS).map(([key, preset]) => (
            <div key={key} className="flex items-baseline gap-2">
              <Text strong className="text-xs capitalize w-16">
                {key}:
              </Text>
              <Text className="text-xs text-text-subtle">
                {preset.streaming ? "stream" : "no stream"}, {preset.responseFormat},{" "}
                {preset.splitBy}, {preset.speed}x
              </Text>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
