import React from "react"

export type StreamingAudioMode = "stream" | "buffer"

type StreamingAudioState = {
  playing: boolean
  mode: StreamingAudioMode
  format: string | null
  error: string | null
}

const formatToMime = (format: string): string => {
  const normalized = String(format || "").trim().toLowerCase()
  switch (normalized) {
    case "wav":
      return "audio/wav"
    case "opus":
      return "audio/opus"
    case "aac":
      return "audio/aac"
    case "flac":
      return "audio/flac"
    case "ogg":
      return "audio/ogg"
    case "webm":
      return "audio/webm"
    case "ulaw":
      return "audio/basic"
    case "pcm":
      return "audio/L16"
    case "mp3":
    default:
      return "audio/mpeg"
  }
}

const canStreamMime = (mime: string): boolean => {
  if (typeof MediaSource === "undefined") return false
  try {
    return MediaSource.isTypeSupported(mime)
  } catch {
    return false
  }
}

const toUint8 = (chunk: ArrayBuffer | Uint8Array): Uint8Array<ArrayBuffer> => {
  if (chunk instanceof Uint8Array) {
    if (chunk.buffer instanceof ArrayBuffer) {
      return chunk as Uint8Array<ArrayBuffer>
    }
    const buffer = new ArrayBuffer(chunk.byteLength)
    new Uint8Array(buffer).set(chunk)
    return new Uint8Array(buffer)
  }
  return new Uint8Array(chunk)
}

export const useStreamingAudioPlayer = () => {
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const mediaSourceRef = React.useRef<MediaSource | null>(null)
  const sourceBufferRef = React.useRef<SourceBuffer | null>(null)
  const objectUrlRef = React.useRef<string | null>(null)
  const pendingChunksRef = React.useRef<Uint8Array<ArrayBuffer>[]>([])
  const allChunksRef = React.useRef<Uint8Array<ArrayBuffer>[]>([])
  const endOfStreamPendingRef = React.useRef(false)
  const streamFailedRef = React.useRef(false)
  const modeRef = React.useRef<StreamingAudioMode>("buffer")
  const formatRef = React.useRef<string | null>(null)

  const [state, setState] = React.useState<StreamingAudioState>({
    playing: false,
    mode: "buffer",
    format: null,
    error: null
  })

  const clearAudioElement = React.useCallback(() => {
    const audio = audioRef.current
    if (audio) {
      audio.onended = null
      audio.onerror = null
      try {
        audio.pause()
      } catch {}
    }
    audioRef.current = null
  }, [])

  const revokeObjectUrl = React.useCallback(() => {
    if (objectUrlRef.current) {
      try {
        URL.revokeObjectURL(objectUrlRef.current)
      } catch {}
      objectUrlRef.current = null
    }
  }, [])

  const cleanupMediaSource = React.useCallback(() => {
    const mediaSource = mediaSourceRef.current
    const sourceBuffer = sourceBufferRef.current
    if (sourceBuffer) {
      try {
        sourceBuffer.onupdateend = null
      } catch {}
    }
    if (mediaSource) {
      try {
        if (mediaSource.readyState === "open") {
          mediaSource.endOfStream()
        }
      } catch {}
    }
    sourceBufferRef.current = null
    mediaSourceRef.current = null
  }, [])

  const resetBuffers = React.useCallback(() => {
    pendingChunksRef.current = []
    allChunksRef.current = []
    endOfStreamPendingRef.current = false
    streamFailedRef.current = false
  }, [])

  const cleanup = React.useCallback(() => {
    cleanupMediaSource()
    clearAudioElement()
    revokeObjectUrl()
    resetBuffers()
    formatRef.current = null
    modeRef.current = "buffer"
    setState({ playing: false, mode: "buffer", format: null, error: null })
  }, [cleanupMediaSource, clearAudioElement, revokeObjectUrl, resetBuffers])

  const flushPending = React.useCallback(() => {
    const sourceBuffer = sourceBufferRef.current
    if (!sourceBuffer || sourceBuffer.updating) return
    const next = pendingChunksRef.current.shift()
    if (!next) {
      if (endOfStreamPendingRef.current && mediaSourceRef.current) {
        try {
          mediaSourceRef.current.endOfStream()
        } catch {}
        endOfStreamPendingRef.current = false
      }
      return
    }
    try {
      sourceBuffer.appendBuffer(next)
    } catch {
      streamFailedRef.current = true
    }
  }, [])

  const start = React.useCallback(
    (format: string, preferStream: boolean) => {
      cleanup()
      const normalizedFormat = String(format || "mp3").toLowerCase()
      const mime = formatToMime(normalizedFormat)
      const streaming = preferStream && canStreamMime(mime)

      formatRef.current = normalizedFormat
      modeRef.current = streaming ? "stream" : "buffer"
      setState((prev) => ({
        ...prev,
        mode: streaming ? "stream" : "buffer",
        format: normalizedFormat,
        error: null
      }))

      const audio = new Audio()
      audio.autoplay = true
      audio.onended = () => {
        if (audioRef.current !== audio) return
        setState((prev) => ({ ...prev, playing: false }))
      }
      audio.onerror = () => {
        if (audioRef.current !== audio) return
        streamFailedRef.current = true
        setState((prev) => ({
          ...prev,
          playing: false,
          error: "Audio playback failed"
        }))
      }
      audioRef.current = audio

      if (streaming) {
        const mediaSource = new MediaSource()
        mediaSourceRef.current = mediaSource
        mediaSource.addEventListener(
          "sourceopen",
          () => {
            if (audioRef.current !== audio || mediaSourceRef.current !== mediaSource) return
            try {
              const sourceBuffer = mediaSource.addSourceBuffer(mime)
              sourceBufferRef.current = sourceBuffer
              sourceBuffer.onupdateend = () => {
                if (audioRef.current !== audio || sourceBufferRef.current !== sourceBuffer) return
                flushPending()
              }
              flushPending()
            } catch {
              streamFailedRef.current = true
            }
          },
          { once: true }
        )
        const url = URL.createObjectURL(mediaSource)
        objectUrlRef.current = url
        audio.src = url
        const playPromise = audio.play()
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch(() => {
            if (audioRef.current !== audio) return
            streamFailedRef.current = true
            setState((prev) => ({
              ...prev,
              playing: false,
              error: "Audio playback blocked"
            }))
          })
        }
        setState((prev) => ({ ...prev, playing: true }))
      }
    },
    [cleanup, flushPending]
  )

  const append = React.useCallback(
    (chunk: ArrayBuffer | Uint8Array) => {
      const data = toUint8(chunk)
      allChunksRef.current.push(data)
      if (modeRef.current === "stream" && !streamFailedRef.current) {
        const sourceBuffer = sourceBufferRef.current
        if (!sourceBuffer || sourceBuffer.updating) {
          pendingChunksRef.current.push(data)
          return
        }
        try {
          sourceBuffer.appendBuffer(data)
        } catch {
          streamFailedRef.current = true
          pendingChunksRef.current.push(data)
        }
      } else {
        pendingChunksRef.current.push(data)
      }
    },
    []
  )

  const finish = React.useCallback(() => {
    if (modeRef.current === "stream" && !streamFailedRef.current) {
      endOfStreamPendingRef.current = true
      flushPending()
      return
    }

    const audio = audioRef.current
    const normalizedFormat = formatRef.current || "mp3"
    const mime = formatToMime(normalizedFormat)
    if (!audio) {
      const nextAudio = new Audio()
      nextAudio.autoplay = true
      nextAudio.onended = () => {
        if (audioRef.current !== nextAudio) return
        setState((prev) => ({ ...prev, playing: false }))
      }
      nextAudio.onerror = () => {
        if (audioRef.current !== nextAudio) return
        setState((prev) => ({
          ...prev,
          playing: false,
          error: "Audio playback failed"
        }))
      }
      audioRef.current = nextAudio
    }

    // Revoke any previously-held object URL (e.g. the MediaSource blob URL from
    // start()) before overwriting the ref with the buffered fallback blob URL,
    // otherwise the stream->buffer fallback path leaks one URL per failed stream.
    revokeObjectUrl()
    const blob = new Blob(allChunksRef.current, { type: mime })
    const url = URL.createObjectURL(blob)
    objectUrlRef.current = url
    const playbackAudio = audioRef.current!
    playbackAudio.src = url
    const playPromise = playbackAudio.play()
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        if (audioRef.current !== playbackAudio) return
        setState((prev) => ({
          ...prev,
          playing: false,
          error: "Audio playback blocked"
        }))
      })
    }
    setState((prev) => ({ ...prev, playing: true }))
  }, [flushPending, revokeObjectUrl])

  const getBufferedBlob = React.useCallback(() => {
    if (!allChunksRef.current.length) return null
    const normalizedFormat = formatRef.current || "mp3"
    const mime = formatToMime(normalizedFormat)
    return new Blob(allChunksRef.current, { type: mime })
  }, [])

  const stop = React.useCallback(() => {
    cleanup()
  }, [cleanup])

  React.useEffect(() => () => cleanup(), [cleanup])

  return {
    start,
    append,
    finish,
    stop,
    state,
    getBufferedBlob
  }
}
