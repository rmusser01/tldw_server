import React from "react"
import { useTranslation } from "react-i18next"
import { Popover, Select, Slider, Tooltip, Spin, Progress } from "antd"
import {
  Volume2,
  Play,
  Pause,
  Square,
  Download
} from "lucide-react"
import { useDocumentTTS } from "@/hooks/document-workspace/useDocumentTTS"

interface TTSPanelProps {
  /** Text to speak when play is clicked without prior selection */
  defaultText?: string
}

/**
 * TTS playback controls panel.
 *
 * Features:
 * - Play/Pause/Stop controls
 * - Voice selection
 * - Speed control
 */
export const TTSPanel: React.FC<TTSPanelProps> = ({ defaultText }) => {
  const { t } = useTranslation(["option", "common"])
  const {
    state,
    voice,
    speed,
    volume,
    progress,
    audioUrl,
    voices,
    voicesLoading,
    speak,
    pause,
    resume,
    stop,
    setVoice,
    setSpeed,
    setVolume
  } = useDocumentTTS()

  const textToSpeak = defaultText || state.lastSpokenText
  const canPlay = !!(textToSpeak || state.currentText)

  const handlePlayPause = () => {
    if (state.isPlaying) {
      pause()
    } else if (state.isPaused) {
      resume()
    } else if (textToSpeak) {
      speak(textToSpeak)
    }
  }

  const speedMarks: Record<number, string> = {
    0.5: "0.5x",
    1: "1x",
    1.5: "1.5x",
    2: "2x"
  }

  const content = (
    <div className="w-64 space-y-4">
      {/* Playback controls */}
      <div className="flex items-center justify-center gap-2">
        <Tooltip title={state.isPlaying ? t("common:pause", "Pause") : t("common:play", "Play")}>
          <button
            onClick={handlePlayPause}
            disabled={state.isLoading || !canPlay}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-white hover:bg-primary/90 disabled:bg-muted disabled:cursor-not-allowed"
          >
            {state.isLoading ? (
              <Spin size="small" />
            ) : state.isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5 ml-0.5" />
            )}
          </button>
        </Tooltip>

        <Tooltip title={t("common:stop", "Stop")}>
          <button
            onClick={stop}
            disabled={!state.isPlaying && !state.isPaused}
            className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-hover disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Square className="h-4 w-4" />
          </button>
        </Tooltip>
      </div>

      {/* Current text preview */}
      {state.currentText ? (
        <div className="rounded border border-border bg-surface-hover p-2 text-xs text-text-secondary line-clamp-2">
          "{state.currentText}"
        </div>
      ) : !canPlay ? (
        <div className="rounded border border-border bg-surface-hover p-2 text-xs text-text-muted text-center">
          {t("option:documentWorkspace.ttsSelectHint", "Select text in the document to listen")}
        </div>
      ) : state.lastSpokenText ? (
        <div className="rounded border border-border bg-surface-hover p-2 text-xs text-text-secondary line-clamp-2">
          {t("option:documentWorkspace.ttsReplay", "Replay:")} "{state.lastSpokenText}"
        </div>
      ) : null}

      {/* Error message */}
      {state.error && (
        <div className="rounded border border-danger/30 bg-danger/10 p-2 text-xs text-danger">
          {state.error}
        </div>
      )}

      {/* Voice selection */}
      <div>
        <div className="mb-1.5 text-xs font-medium text-text-secondary">
          {t("option:documentWorkspace.voice", "Voice")}
        </div>
        <Select
          value={voice}
          onChange={setVoice}
          loading={voicesLoading}
          className="w-full"
          size="small"
          options={voices.map((v) => ({
            value: v.id,
            label: (
              <span>
                {v.name}
                <span className="ml-1 text-text-muted text-xs">({v.provider})</span>
              </span>
            )
          }))}
          placeholder={t("option:documentWorkspace.selectVoice", "Select voice")}
        />
      </div>

      {/* Speed control */}
      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-xs font-medium text-text-secondary">
            {t("option:documentWorkspace.speed", "Speed")}
          </span>
          <span className="text-xs text-text-muted">{speed}x</span>
        </div>
        <Slider
          min={0.5}
          max={2}
          step={0.1}
          value={speed}
          onChange={setSpeed}
          marks={speedMarks}
          tooltip={{ formatter: (v) => `${v}x` }}
        />
      </div>

      {/* Volume control */}
      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-xs font-medium text-text-secondary">
            {t("option:documentWorkspace.volume", "Volume")}
          </span>
          <span className="text-xs text-text-muted">{Math.round(volume * 100)}%</span>
        </div>
        <Slider
          min={0}
          max={100}
          step={1}
          value={Math.round(volume * 100)}
          onChange={(v) => setVolume(v / 100)}
          tooltip={{ formatter: (v) => `${v}%` }}
        />
      </div>

      {/* Playback progress */}
      {(state.isPlaying || state.isPaused) && (
        <div>
          <Progress
            percent={Math.round(progress)}
            size="small"
            showInfo={false}
            strokeColor="var(--color-primary)"
          />
          <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
            <span>{Math.round(progress)}%</span>
          </div>
        </div>
      )}

      {/* Download audio */}
      {audioUrl && (
        <div className="flex justify-center">
          <a
            href={audioUrl}
            download="tts-audio.mp3"
            className="flex items-center gap-1 text-xs text-primary hover:underline"
          >
            <Download className="h-3 w-3" />
            {t("option:documentWorkspace.downloadAudio", "Download audio")}
          </a>
        </div>
      )}
    </div>
  )

  const isActive = state.isPlaying || state.isPaused || state.isLoading

  return (
    <Popover
      content={content}
      title={t("option:documentWorkspace.textToSpeechTitle", "Read Aloud")}
      trigger="click"
      placement="bottomRight"
    >
      <Tooltip title={t("option:documentWorkspace.textToSpeech", "Read aloud")}>
        <button
          className={`rounded p-1.5 hover:bg-hover ${isActive ? "text-primary" : ""}`}
          aria-label={t("option:documentWorkspace.textToSpeech", "Read aloud")}
        >
          <Volume2 className="h-4 w-4" />
        </button>
      </Tooltip>
    </Popover>
  )
}

export default TTSPanel
