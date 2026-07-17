import React from "react"
import { Button, Drawer, Empty, Modal, Tag, Tooltip } from "antd"
import { Download, Trash2, Volume2, Square } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useLiveQuery } from "dexie-react-hooks"
import { db } from "@/db/dexie/schema"
import type { TtsClip, TtsClipSegment } from "@/db/dexie/types"
import { clearTtsClips, deleteTtsClip } from "@/db/dexie/tts-clips"
import { downloadBlob } from "@/utils/download-blob"
import { useAntdNotification } from "@/hooks/useAntdNotification"

const formatTimestamp = (value: number) => {
  const date = new Date(value)
  const now = new Date()
  const sameDay = date.toDateString() === now.toDateString()
  return date.toLocaleString([], {
    month: sameDay ? undefined : "short",
    day: sameDay ? undefined : "numeric",
    hour: "numeric",
    minute: "2-digit"
  })
}

const resolveAudioExtension = (type?: string) => {
  const normalized = (type || "").toLowerCase()
  if (normalized.includes("wav")) return "wav"
  if (normalized.includes("opus")) return "opus"
  if (normalized.includes("aac")) return "aac"
  if (normalized.includes("flac")) return "flac"
  if (normalized.includes("pcm") || normalized.includes("l16")) return "pcm"
  if (normalized.includes("ogg")) return "ogg"
  return "mp3"
}

const buildFilenameBase = (clip: TtsClip) => {
  const date = new Date(clip.createdAt)
  const stamp = date.toISOString().replace(/[:.]/g, "-")
  const raw = clip.textPreview || "tts-clip"
  const safe = raw
    .replace(/[/\\?%*:|"<>]/g, "-")
    .replace(/\s+/g, "_")
    .slice(0, 48)
  return `tts_${stamp}_${safe || "clip"}`
}

const canCombineSegments = (segments: TtsClipSegment[]) =>
  segments.length > 1 && segments.every((segment) => segment.mimeType === "audio/mpeg")

const buildCombinedBlob = (segments: TtsClipSegment[]) =>
  new Blob(segments.map((segment) => segment.blob), {
    type: segments[0]?.mimeType || "audio/mpeg"
  })

type TtsClipsDrawerProps = {
  open: boolean
  onClose: () => void
}

export const TtsClipsDrawer: React.FC<TtsClipsDrawerProps> = ({ open, onClose }) => {
  const { t } = useTranslation(["playground", "common"])
  const notification = useAntdNotification()
  const clips = useLiveQuery(
    () => db.ttsClips.orderBy("createdAt").reverse().toArray(),
    []
  )
  const [playingClipId, setPlayingClipId] = React.useState<string | null>(null)
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const abortRef = React.useRef<AbortController | null>(null)

  const stopPlayback = React.useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current = null
    }
    setPlayingClipId(null)
  }, [])

  React.useEffect(() => {
    if (!open) {
      stopPlayback()
    }
  }, [open, stopPlayback])

  React.useEffect(() => () => stopPlayback(), [stopPlayback])

  const playBlob = React.useCallback(
    (blob: Blob, playbackRate = 1, signal?: AbortSignal) =>
      new Promise<void>((resolve, reject) => {
        const url = URL.createObjectURL(blob)
        const audio = new Audio(url)
        audioRef.current = audio
        audio.playbackRate = playbackRate
        const finish = (err?: unknown) => {
          URL.revokeObjectURL(url)
          audio.onended = null
          audio.onerror = null
          if (signal) {
            signal.removeEventListener("abort", handleAbort)
          }
          if (err) reject(err)
          else resolve()
        }
        const handleAbort = () => {
          audio.pause()
          audio.currentTime = 0
          finish()
        }
        if (signal) {
          if (signal.aborted) {
            finish()
            return
          }
          signal.addEventListener("abort", handleAbort, { once: true })
        }
        audio.onended = () => finish()
        audio.onerror = () => finish(audio.error || new Error("Audio playback failed"))
        const playPromise = audio.play()
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch((err) => finish(err))
        }
      }),
    []
  )

  const handleTogglePlay = React.useCallback(
    async (clip: TtsClip) => {
      if (playingClipId === clip.id) {
        stopPlayback()
        return
      }

      const ordered = [...clip.segments].sort((a, b) => a.index - b.index)
      if (!ordered.length) return

      const controller = new AbortController()
      abortRef.current = controller
      setPlayingClipId(clip.id)
      try {
        for (const segment of ordered) {
          if (controller.signal.aborted) break
          await playBlob(segment.blob, clip.playbackSpeed || 1, controller.signal)
        }
      } catch (error) {
        notification.error({
          message: t("playground:tts.playErrorTitle", "Error"),
          description:
            error instanceof Error
              ? error.message
              : t(
                  "playground:tts.playErrorDescription",
                  "Something went wrong while trying to play the audio"
                )
        })
      } finally {
        setPlayingClipId(null)
      }
    },
    [notification, playBlob, playingClipId, stopPlayback, t]
  )

  const handleDownload = React.useCallback(
    (clip: TtsClip) => {
      if (!clip.segments.length) return
      const ordered = [...clip.segments].sort((a, b) => a.index - b.index)
      const extension = resolveAudioExtension(ordered[0]?.mimeType || clip.mimeType)
      const base = buildFilenameBase(clip)

      if (ordered.length === 1) {
        downloadBlob(ordered[0].blob, `${base}.${extension}`)
        return
      }

      if (canCombineSegments(ordered)) {
        const combined = buildCombinedBlob(ordered)
        downloadBlob(combined, `${base}.${extension}`)
        return
      }

      ordered.forEach((segment, index) => {
        downloadBlob(
          segment.blob,
          `${base}_part${String(index + 1).padStart(2, "0")}.${extension}`
        )
      })
    },
    []
  )

  const handleDelete = React.useCallback(
    (clip: TtsClip) => {
      Modal.confirm({
        title: t("common:confirmTitle", "Please confirm"),
        content: t("playground:ttsClips.deleteConfirm", "Delete this clip?"),
        okText: t("common:delete", "Delete"),
        cancelText: t("common:cancel", "Cancel"),
        okButtonProps: { danger: true },
        onOk: async () => {
          await deleteTtsClip(clip.id)
        }
      })
    },
    [t]
  )

  const handleClearAll = React.useCallback(() => {
    Modal.confirm({
      title: t("common:confirmTitle", "Please confirm"),
      content: t("playground:ttsClips.clearConfirm", "Clear all saved TTS clips?"),
      okText: t("common:delete", "Delete"),
      cancelText: t("common:cancel", "Cancel"),
      okButtonProps: { danger: true },
      onOk: async () => {
        await clearTtsClips()
      }
    })
  }, [t])

  const list = clips || []

  return (
    <Drawer
      open={open}
      onClose={() => {
        stopPlayback()
        onClose()
      }}
      title={t("playground:ttsClips.title", "TTS clips")}
      styles={{ wrapper: { width: 360 } }}
      className="tldw-tts-clips-drawer"
      extra={
        list.length ? (
          <Button size="small" danger onClick={handleClearAll}>
            {t("playground:ttsClips.clearAll", "Clear all")}
          </Button>
        ) : null
      }
    >
      {!list.length ? (
        <Empty
          description={t(
            "playground:ttsClips.empty",
            "Generate chat audio to save it here."
          )}
        />
      ) : (
        <div className="flex flex-col gap-3">
          {list.map((clip) => (
            <div
              key={clip.id}
              className="flex flex-col gap-2 rounded-xl border border-border bg-surface2/70 p-3"
            >
              <Tooltip title={clip.utterance || clip.textPreview}>
                <div className="text-sm text-text truncate">
                  {clip.textPreview || clip.utterance}
                </div>
              </Tooltip>
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
                <span>{formatTimestamp(clip.createdAt)}</span>
                <span>{t("playground:ttsClips.segmentsLabel", { count: clip.segments.length, defaultValue: "{{count}} segments" })}</span>
                {clip.provider && (
                  <Tag color="blue" className="!text-[10px]">
                    {clip.provider}
                  </Tag>
                )}
                {clip.requestedBackend && (
                  <Tag className="!text-[10px]">
                    {t("playground:ttsClips.requestedBackend", {
                      backend: clip.requestedBackend,
                      defaultValue: "Requested {{backend}}"
                    })}
                  </Tag>
                )}
                {clip.actualBackends?.length ? (
                  <Tag color="cyan" className="!text-[10px]">
                    {t("playground:ttsClips.actualBackend", {
                      backend: clip.actualBackends.join(", "),
                      defaultValue: "Actual {{backend}}"
                    })}
                  </Tag>
                ) : null}
                {clip.fallbackUsed ? (
                  <Tag color="orange" className="!text-[10px]">
                    {t(
                      "playground:ttsClips.fallbackUsed",
                      "Fallback used"
                    )}
                  </Tag>
                ) : null}
                {clip.voice && (
                  <Tag color="purple" className="!text-[10px]">
                    {clip.voice}
                  </Tag>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Tooltip
                  title={
                    playingClipId === clip.id
                      ? t("playground:tts.stop", "Stop")
                      : t("playground:tts.play", "Play")
                  }
                >
                  <button
                    type="button"
                    onClick={() => handleTogglePlay(clip)}
                    className="rounded-md border border-border bg-surface px-2 py-1 text-text-muted hover:bg-surface2 hover:text-text"
                  >
                    {playingClipId === clip.id ? (
                      <Square className="h-4 w-4 text-danger" />
                    ) : (
                      <Volume2 className="h-4 w-4" />
                    )}
                  </button>
                </Tooltip>
                <Tooltip
                  title={
                    clip.segments.length > 1
                      ? t(
                          "playground:ttsClips.downloadAll",
                          "Download all segments"
                        )
                      : t("playground:ttsClips.download", "Download")
                  }
                >
                  <button
                    type="button"
                    onClick={() => handleDownload(clip)}
                    className="rounded-md border border-border bg-surface px-2 py-1 text-text-muted hover:bg-surface2 hover:text-text"
                  >
                    <Download className="h-4 w-4" />
                  </button>
                </Tooltip>
                <Tooltip title={t("common:delete", "Delete") as string}>
                  <button
                    type="button"
                    onClick={() => handleDelete(clip)}
                    className="rounded-md border border-border bg-surface px-2 py-1 text-text-muted hover:bg-surface2 hover:text-text"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </Tooltip>
              </div>
            </div>
          ))}
        </div>
      )}
    </Drawer>
  )
}

export default TtsClipsDrawer
