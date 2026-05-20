import React, { useCallback, useEffect, useMemo, useRef } from "react"
import { Button, Card, Tooltip, Typography, Upload } from "antd"
import { Mic, Settings, Square, Trash2, Upload as UploadIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAudioRecorder } from "@/hooks/useAudioRecorder"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { classifyAudioError } from "@/components/Option/Audio/audio-error-classification"

export interface RecordingStripProps {
  onBlobReady: (blob: Blob, durationMs: number) => void
  onSettingsToggle?: () => void
  disabled?: boolean
  disabledReason?: string
  disabledStatusType?: "secondary" | "warning" | "danger"
}

const { Text } = Typography

/** Convert milliseconds to `mm:ss` display string. */
export function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
}

export const RecordingStrip: React.FC<RecordingStripProps> = ({
  onBlobReady,
  onSettingsToggle,
  disabled = false,
  disabledReason,
  disabledStatusType = "danger"
}) => {
  const { t } = useTranslation(["playground"])
  const notification = useAntdNotification()
  const recorder = useAudioRecorder()
  const audioUrlRef = useRef<string | null>(null)

  const isRecording = recorder.status === "recording"
  const hasBlob = recorder.blob != null && !isRecording
  const blocksNewCapture = disabled && !isRecording
  const recordingUnavailableText =
    disabledReason ||
    t("playground:stt.recordingUnavailable", "Recording is unavailable.")
  const sourceStatus = blocksNewCapture
    ? recordingUnavailableText
    : isRecording
      ? t("playground:stt.sourceRecording", "Audio source: microphone recording")
      : hasBlob
        ? t("playground:stt.sourceSelected", "Audio source: audio selected")
        : t("playground:stt.sourceEmpty", "Audio source: no audio selected")
  const recordTooltip = blocksNewCapture
    ? recordingUnavailableText
    : isRecording
      ? t("playground:stt.stopTooltip", "Stop recording")
      : t("playground:stt.startTooltip", "Start recording")

  // Notify parent when blob changes
  useEffect(() => {
    if (recorder.blob) {
      onBlobReady(recorder.blob, recorder.durationMs)
    }
  }, [recorder.blob, recorder.durationMs, onBlobReady])

  // Create/revoke object URL for playback
  const audioUrl = useMemo(() => {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
    if (recorder.blob) {
      const url = URL.createObjectURL(recorder.blob)
      audioUrlRef.current = url
      return url
    }
    return null
  }, [recorder.blob])

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
        audioUrlRef.current = null
      }
    }
  }, [])

  // Listen for stt-toggle-record custom event (Space shortcut from parent)
  useEffect(() => {
    const handler = (event: Event) => {
      if (blocksNewCapture) return
      event.preventDefault()
      if (isRecording) {
        recorder.stopRecording()
      } else {
        Promise.resolve(recorder.startRecording()).catch((err) => {
          const classified = classifyAudioError(err)
          notification.error({
            message:
              classified.title || t("playground:stt.micError", "Microphone error"),
            description: classified.recovery
          })
        })
      }
    }
    window.addEventListener("stt-toggle-record", handler)
    return () => {
      window.removeEventListener("stt-toggle-record", handler)
    }
  }, [blocksNewCapture, isRecording, recorder, notification, t])

  const handleRecordClick = useCallback(() => {
    if (blocksNewCapture) return
    if (isRecording) {
      recorder.stopRecording()
    } else {
      Promise.resolve(recorder.startRecording()).catch((err) => {
        const classified = classifyAudioError(err)
        notification.error({
          message:
            classified.title || t("playground:stt.micError", "Microphone error"),
          description: classified.recovery
        })
      })
    }
  }, [blocksNewCapture, isRecording, recorder, notification, t])

  const handleUpload = useCallback(
    (file: File) => {
      if (blocksNewCapture) return false
      recorder.loadBlob(file, 0)
      return false // prevent auto-upload
    },
    [blocksNewCapture, recorder]
  )

  return (
    <Card size="small">
      <div className="flex flex-wrap items-center gap-3">
        {/* Record / Stop button */}
        <Tooltip title={recordTooltip}>
          <Button
            type={isRecording ? "default" : "primary"}
            danger={isRecording}
            aria-label={
              isRecording
                ? t("playground:stt.stopRecording", "Stop recording (Space)")
                : t("playground:stt.startRecording", "Start recording (Space)")
            }
            disabled={blocksNewCapture}
            icon={
              isRecording ? (
                <Square className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )
            }
            onClick={handleRecordClick}
          />
        </Tooltip>

        {/* Duration timer */}
        <span
          className="font-mono text-sm tabular-nums"
          aria-live="polite"
        >
          {formatDuration(recorder.durationMs)}
        </span>

        <Text
          type={blocksNewCapture ? disabledStatusType : "secondary"}
          className="text-xs"
          role="status"
        >
          {sourceStatus}
        </Text>

        {/* Audio level indicator — visible only while recording */}
        {isRecording && (
          <div
            className="flex items-end gap-0.5"
            role="meter"
            aria-label={t(
              "playground:stt.audioLevel",
              "Audio level"
            )}
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={0.5}
          >
            {[1, 2, 3, 4, 5].map((i) => (
              <span
                key={i}
                className="inline-block w-1 rounded-sm bg-red-500 animate-pulse"
                style={{ height: `${6 + i * 3}px`, animationDelay: `${i * 80}ms` }}
              />
            ))}
          </div>
        )}

        {/* Playback — shown when blob exists and not recording */}
        {hasBlob && audioUrl && (
          <audio controls src={audioUrl} className="h-8 max-w-[200px]" />
        )}

        {/* Clear button */}
        {hasBlob && (
          <Tooltip title={t("playground:stt.clearRecording", "Clear recording")}>
            <Button
              type="text"
              icon={<Trash2 className="h-4 w-4" />}
              aria-label={t("playground:stt.clearRecording", "Clear recording")}
              onClick={recorder.clearRecording}
            />
          </Tooltip>
        )}

        {/* Upload file button */}
        <Upload
          accept="audio/*"
          showUploadList={false}
          disabled={blocksNewCapture}
          beforeUpload={handleUpload}
        >
          <Tooltip title={t("playground:stt.uploadAudio", "Upload audio file")}>
            <Button
              type="text"
              icon={<UploadIcon className="h-4 w-4" />}
              aria-label={t("playground:stt.uploadAudio", "Upload audio file")}
              disabled={blocksNewCapture}
            />
          </Tooltip>
        </Upload>

        {/* Settings gear — only if prop provided */}
        {onSettingsToggle && (
          <Tooltip title={t("playground:stt.toggleSettings", "Toggle settings")}>
            <Button
              type="text"
              icon={<Settings className="h-4 w-4" />}
              aria-label={t("playground:stt.toggleSettings", "Toggle settings")}
              onClick={onSettingsToggle}
            />
          </Tooltip>
        )}
      </div>
    </Card>
  )
}

export default RecordingStrip
