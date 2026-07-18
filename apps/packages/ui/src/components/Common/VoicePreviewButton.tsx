import React, { useCallback, useEffect, useRef, useState } from "react"
import { Button, Tooltip } from "antd"
import { Play, Square, Loader2 } from "lucide-react"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const PREVIEW_TEXT = "Hello, this is a preview of the selected voice."

type VoicePreviewButtonProps = {
  model: string
  voice: string
  provider: string
  backend?: string
  allowFallback?: boolean
  className?: string
}

type PreviewState = "idle" | "loading" | "playing"

export function VoicePreviewButton({
  model,
  voice,
  provider,
  backend,
  allowFallback,
  className,
}: VoicePreviewButtonProps) {
  const [state, setState] = useState<PreviewState>("idle")
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const urlRef = useRef<string | null>(null)

  const cleanup = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current)
      urlRef.current = null
    }
  }, [])

  useEffect(() => {
    return cleanup
  }, [cleanup])

  const handleClick = useCallback(async () => {
    if (state === "playing") {
      cleanup()
      setState("idle")
      return
    }

    setState("loading")
    try {
      const result = await tldwClient.synthesizeSpeechDetailed(PREVIEW_TEXT, {
        model,
        voice,
        responseFormat: "mp3",
        backend,
        allowFallback
      })

      const blob = new Blob([result.buffer], { type: "audio/mpeg" })
      const url = URL.createObjectURL(blob)
      urlRef.current = url

      const audio = new Audio(url)
      audioRef.current = audio

      audio.onended = () => {
        cleanup()
        setState("idle")
      }

      await audio.play()
      setState("playing")
    } catch {
      cleanup()
      setState("idle")
    }
  }, [state, model, voice, backend, allowFallback, cleanup])

  const disabled = !voice || provider === "browser"

  const icon =
    state === "loading" ? (
      <Loader2 className="h-3.5 w-3.5 animate-spin" />
    ) : state === "playing" ? (
      <Square className="h-3.5 w-3.5" />
    ) : (
      <Play className="h-3.5 w-3.5" />
    )

  const label = state === "playing" ? "Stop" : "Preview"

  return (
    <Tooltip title="Preview voice">
      <Button
        size="small"
        type="text"
        disabled={disabled}
        onClick={handleClick}
        aria-label="Preview voice"
        className={className}
        icon={icon}
      >
        {label}
      </Button>
    </Tooltip>
  )
}
