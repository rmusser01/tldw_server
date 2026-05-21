import React from "react"
import { Button, Input, Space, Typography } from "antd"
import { Mic, Square } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"

const { Text } = Typography

interface VoiceTranscriptComposerProps {
  transcript: string
  isListening: boolean
  supported: boolean
  isSubmitting: boolean
  onTranscriptChange: (value: string) => void
  onStartListening: () => void
  onStopListening: () => void
  onCancel: () => void
  onSubmit: () => void
}

export const VoiceTranscriptComposer: React.FC<VoiceTranscriptComposerProps> = ({
  transcript,
  isListening,
  supported,
  isSubmitting,
  onTranscriptChange,
  onStartListening,
  onStopListening,
  onCancel,
  onSubmit
}) => {
  const { t } = useTranslation(["option", "common"])

  return (
    <div className="rounded border border-border bg-surface p-3">
      <Space orientation="vertical" size={12} className="w-full">
        <div>
          <Text strong>
            {t("option:flashcards.studyAssistantConfirmTranscript", {
              defaultValue: "Confirm transcript"
            })}
          </Text>
          <Text type="secondary" className="mt-1 block text-xs">
            {t("option:flashcards.studyAssistantTranscriptHelp", {
              defaultValue:
                "Edit the transcript before sending so the assistant checks exactly what you meant."
            })}
          </Text>
        </div>
        {!supported && (
          <Alert
            variant="info"
            title={t("option:flashcards.studyAssistantVoiceUnavailable", {
              defaultValue: "Voice transcript is unavailable in this browser."
            })}
          />
        )}
        <Input.TextArea
          aria-label={t("option:flashcards.studyAssistantTranscriptLabel", {
            defaultValue: "Transcript"
          })}
          value={transcript}
          onChange={(event) => onTranscriptChange(event.target.value)}
          rows={4}
          placeholder={t("option:flashcards.studyAssistantTranscriptPlaceholder", {
            defaultValue: "Say or type your explanation here."
          })}
        />
        <Space wrap>
          {supported && (
            <Button
              icon={isListening ? <Square className="size-4" /> : <Mic className="size-4" />}
              onClick={isListening ? onStopListening : onStartListening}
            >
              {isListening
                ? t("option:flashcards.studyAssistantStopListening", {
                    defaultValue: "Stop dictation"
                  })
                : t("option:flashcards.studyAssistantStartListening", {
                    defaultValue: "Start dictation"
                  })}
            </Button>
          )}
          <Button onClick={onCancel}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            loading={isSubmitting}
            disabled={!transcript.trim()}
            onClick={onSubmit}
          >
            {t("option:flashcards.studyAssistantSendFactCheck", {
              defaultValue: "Send fact-check"
            })}
          </Button>
        </Space>
      </Space>
    </div>
  )
}

export default VoiceTranscriptComposer
