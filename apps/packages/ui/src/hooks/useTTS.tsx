import { useEffect, useState } from "react"
import {
  getElevenLabsModel,
  getElevenLabsVoiceId,
  getOpenAITTSModel,
  getOpenAITTSVoice,
  getTldwTTSModel,
  getTldwTTSVoice,
  getVoice
} from "@/services/tts"
import { splitMessageContent } from "@/utils/tts"
import { resolveTtsProviderContext } from "@/services/tts-provider"
import { useAntdNotification } from "./useAntdNotification"
import { useTranslation } from "react-i18next"
import { isChromiumTarget } from "@/config/platform"
import { saveTtsClip } from "@/db/dexie/tts-clips"
import type { TtsClipSegment } from "@/db/dexie/types"

export type TtsClipMeta = {
  historyId?: string | null
  serverChatId?: string | null
  messageId?: string | null
  serverMessageId?: string | null
  role?: "user" | "assistant" | "system"
  source?: "chat" | "selection"
}

export interface VoiceOptions {
  utterance: string
  saveClip?: boolean
  clipMeta?: TtsClipMeta
}

export const useTTS = () => {
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(
    null
  )
  const notification = useAntdNotification()
  const { t } = useTranslation("playground")

  const speak = async ({ utterance, saveClip, clipMeta }: VoiceOptions) => {
    let debugMeta: { provider?: string; mimeType?: string; size?: number } | null =
      null
    try {
      const resolveArrayBuffer = async (value: unknown): Promise<ArrayBuffer | null> => {
        if (!value) return null
        if (value instanceof ArrayBuffer) return value
        if (typeof SharedArrayBuffer !== "undefined" && value instanceof SharedArrayBuffer) {
          return new Uint8Array(value).slice(0).buffer
        }
        if (ArrayBuffer.isView(value)) {
          const view = value as ArrayBufferView
          if (
            typeof SharedArrayBuffer !== "undefined" &&
            view.buffer instanceof SharedArrayBuffer
          ) {
            const copy = new Uint8Array(view.byteLength)
            copy.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength))
            return copy.buffer
          }
          if (view.buffer instanceof ArrayBuffer) {
            return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
          }
        }
        if (value instanceof Blob) {
          return await value.arrayBuffer()
        }
        const tag = Object.prototype.toString.call(value)
        if (tag === "[object ArrayBuffer]" && typeof (value as any).slice === "function") {
          return (value as any).slice(0)
        }
        if (Array.isArray(value) && value.every((entry) => typeof entry === "number")) {
          return new Uint8Array(value).buffer
        }
        if (typeof value === "object") {
          const record = value as Record<string, any>
          if (
            typeof record.byteLength === "number" &&
            typeof record.slice === "function"
          ) {
            try {
              const sliced = record.slice(0)
              if (
                typeof SharedArrayBuffer !== "undefined" &&
                sliced instanceof SharedArrayBuffer
              ) {
                return new Uint8Array(sliced).slice(0).buffer
              }
              return sliced
            } catch {
              // ignore and continue
            }
          }
          if (typeof record.arrayBuffer === "function") {
            return await record.arrayBuffer()
          }
          if (record.data !== undefined) {
            const nested = await resolveArrayBuffer(record.data)
            if (nested) return nested
          }
          if (record.buffer !== undefined) {
            const nested = await resolveArrayBuffer(record.buffer)
            if (nested) return nested
          }
        }
        return null
      }

      const resolveProviderMeta = async (providerName: string) => {
        switch (providerName) {
          case "tldw":
            return {
              model: await getTldwTTSModel(),
              voice: await getTldwTTSVoice()
            }
          case "openai":
            return {
              model: await getOpenAITTSModel(),
              voice: await getOpenAITTSVoice()
            }
          case "elevenlabs":
            return {
              model: await getElevenLabsModel(),
              voice: await getElevenLabsVoiceId()
            }
          default:
            return {}
        }
      }

      let context = await resolveTtsProviderContext(utterance)

      if (context.provider === "tldw") {
        const resolvedFormat = context.formatInfo?.resolved || "mp3"
        const mimeType = (() => {
          switch (resolvedFormat) {
            case "wav":
              return "audio/wav"
            case "opus":
              return "audio/opus"
            case "aac":
              return "audio/aac"
            case "flac":
              return "audio/flac"
            case "pcm":
              return "audio/L16; rate=24000; channels=1"
            case "mp3":
            default:
              return "audio/mpeg"
          }
        })()
        const probe = new Audio()
        const canPlay = probe.canPlayType(mimeType)
        if (!canPlay && resolvedFormat !== "mp3") {
          context = await resolveTtsProviderContext(utterance, {
            tldwResponseFormat: "mp3"
          })
        }
      }

      const {
        provider,
        utterance: processedUtterance,
        playbackSpeed,
        synthesize,
        supported
      } = context

      const shouldSaveClip = Boolean(saveClip && provider !== "browser")
      const clipId = shouldSaveClip ? crypto.randomUUID() : null
      const savedSegments: TtsClipSegment[] = []
      let clipFormat: string | undefined
      let clipMimeType: string | undefined
      const requestedBackend = context.cacheSettings?.backend || undefined
      const providerMeta = shouldSaveClip ? await resolveProviderMeta(provider) : {}

      if (!supported) {
        throw new Error(`Unsupported TTS provider: ${provider}`)
      }

      if (provider === "browser") {
        const voice = await getVoice()
        if (isChromiumTarget && typeof chrome !== "undefined" && chrome.tts) {
          chrome.tts.speak(processedUtterance, {
            voiceName: voice,
            rate: playbackSpeed,
            onEvent(event) {
              if (event.type === "start") {
                setIsSpeaking(true)
              } else if (event.type === "end") {
                setIsSpeaking(false)
              }
            }
          })
        } else if (typeof window !== "undefined" && window.speechSynthesis) {
          const synthesisUtterance = new SpeechSynthesisUtterance(processedUtterance)
          synthesisUtterance.rate = playbackSpeed
          synthesisUtterance.onstart = () => {
            setIsSpeaking(true)
          }
          synthesisUtterance.onend = () => {
            setIsSpeaking(false)
          }
          const voices = window.speechSynthesis.getVoices()
          const selectedVoice = voices.find((v) => v.name === voice)
          if (selectedVoice) {
            synthesisUtterance.voice = selectedVoice
          } else {
            window.speechSynthesis.onvoiceschanged = () => {
              const updatedVoices = window.speechSynthesis.getVoices()
              const newVoice = updatedVoices.find((v) => v.name === voice)
              if (newVoice) {
                synthesisUtterance.voice = newVoice
              }
            }
          }
          window.speechSynthesis.speak(synthesisUtterance)
        }
        return
      }

      if (!synthesize) {
        throw new Error(`Unsupported TTS provider: ${provider}`)
      }

      const synthesizeSegment = synthesize
      type AudioResult = Awaited<ReturnType<typeof synthesizeSegment>>
      const sentences = splitMessageContent(processedUtterance)
      let nextAudioData: AudioResult | null = null
      let nextAudioPromise: Promise<AudioResult> | null = null

      for (let i = 0; i < sentences.length; i++) {
        setIsSpeaking(true)

        let currentAudioData: AudioResult
        if (nextAudioData) {
          currentAudioData = nextAudioData
          nextAudioData = null
        } else {
          currentAudioData = await synthesizeSegment(sentences[i])
        }

        if (i < sentences.length - 1) {
          nextAudioPromise = synthesizeSegment(sentences[i + 1])
        }

        const resolvedBuffer = await resolveArrayBuffer(
          (currentAudioData as any)?.buffer ?? currentAudioData
        )
        if (!resolvedBuffer) {
          throw new Error("TTS returned an invalid audio buffer.")
        }
        if (resolvedBuffer.byteLength === 0) {
          throw new Error("TTS returned an empty audio buffer.")
        }

        debugMeta = {
          provider,
          mimeType: currentAudioData.mimeType || "audio/mpeg",
          size: resolvedBuffer.byteLength
        }
        const blob = new Blob([resolvedBuffer], {
          type: currentAudioData.mimeType || "audio/mpeg"
        })
        if (shouldSaveClip && clipId) {
          const formatValue =
            currentAudioData.format ||
            context.formatInfo?.resolved ||
            "mp3"
          const mimeValue = currentAudioData.mimeType || "audio/mpeg"
          if (!clipFormat) clipFormat = formatValue
          if (!clipMimeType) clipMimeType = mimeValue
          const hasGatewayMetadata = Boolean(
            requestedBackend ||
              currentAudioData.actualBackend ||
              currentAudioData.fallbackUsed
          )
          savedSegments.push({
            id: `${clipId}:${i}`,
            index: i,
            text: sentences[i],
            format: formatValue,
            mimeType: mimeValue,
            blob,
            sizeBytes: blob.size,
            ...(hasGatewayMetadata
              ? {
                  actualBackend: currentAudioData.actualBackend,
                  fallbackUsed: Boolean(currentAudioData.fallbackUsed)
                }
              : {})
          })
        }
        const url = URL.createObjectURL(blob)
        const audio = new Audio(url)
        audio.playbackRate = playbackSpeed

        const canPlay = audio.canPlayType(currentAudioData.mimeType)
        if (!canPlay) {
          URL.revokeObjectURL(url)
          throw new Error(
            `Your browser cannot play ${currentAudioData.mimeType}. Try MP3 or WAV.`
          )
        }

        setAudioElement(audio)

        const playAudio = () =>
          new Promise<void>((resolve, reject) => {
            let done = false
            const finish = (err?: unknown) => {
              if (done) return
              done = true
              audio.onended = null
              audio.onerror = null
              if (err) reject(err)
              else resolve()
            }
            audio.onended = () => finish()
            audio.onerror = () =>
              finish(
                audio.error ||
                  new Error("Audio playback failed to start.")
              )
            const playPromise = audio.play()
            if (playPromise && typeof playPromise.catch === "function") {
              playPromise.catch((err) => finish(err))
            }
          })

        try {
          await Promise.all([
            playAudio(),
            nextAudioPromise
              ?.then((data) => {
                nextAudioData = data
              })
              .catch(console.error) || Promise.resolve()
          ])
        } finally {
          URL.revokeObjectURL(url)
        }
      }

      if (shouldSaveClip && clipId && savedSegments.length > 0) {
        const textPreview = processedUtterance.replace(/\s+/g, " ").trim()
        const preview =
          textPreview.length > 160
            ? `${textPreview.slice(0, 157)}...`
            : textPreview
        const totalBytes = savedSegments.reduce(
          (sum, segment) => sum + segment.sizeBytes,
          0
        )
        const actualBackends = Array.from(
          new Set(
            savedSegments
              .map((segment) => segment.actualBackend)
              .filter((backend): backend is string => Boolean(backend))
          )
        )
        const fallbackUsed = savedSegments.some(
          (segment) => segment.fallbackUsed
        )
        const hasGatewayMetadata = Boolean(
          requestedBackend || actualBackends.length || fallbackUsed
        )
        try {
          await saveTtsClip({
            id: clipId,
            createdAt: Date.now(),
            provider,
            model: providerMeta.model ?? null,
            voice: providerMeta.voice ?? null,
            format: clipFormat || context.formatInfo?.resolved,
            mimeType: clipMimeType,
            playbackSpeed,
            ...(requestedBackend ? { requestedBackend } : {}),
            ...(actualBackends.length ? { actualBackends } : {}),
            ...(hasGatewayMetadata ? { fallbackUsed } : {}),
            utterance: processedUtterance,
            textPreview: preview,
            totalBytes,
            segments: savedSegments,
            source: clipMeta?.source,
            role: clipMeta?.role,
            historyId: clipMeta?.historyId ?? null,
            serverChatId: clipMeta?.serverChatId ?? null,
            messageId: clipMeta?.messageId ?? null,
            serverMessageId: clipMeta?.serverMessageId ?? null
          })
        } catch (error) {
          console.error("[tldw][tts] Failed to save clip", error)
        }
      }

      setIsSpeaking(false)
      setAudioElement(null)
    } catch (error) {
      setIsSpeaking(false)
      setAudioElement(null)
      // eslint-disable-next-line no-console
      console.error("[tldw][tts] Playback failed", error, debugMeta)
      notification.error({
        message: t("tts.playErrorTitle", "Error"),
        description:
          error instanceof Error
            ? error.message
            : t(
                "tts.playErrorDescription",
                "Something went wrong while trying to play the audio"
              )
      })
    }
  }

  const cancel = () => {
    if (audioElement) {
      audioElement.pause()
      audioElement.currentTime = 0
      setAudioElement(null)
      setIsSpeaking(false)
      return
    }

    if (
      isChromiumTarget &&
      typeof chrome !== "undefined" &&
      chrome.tts
    ) {
      chrome.tts.stop()
    } else if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    setIsSpeaking(false)
  }

  useEffect(() => {
    return () => {
      cancel()
    }
  }, [])

  return {
    speak,
    cancel,
    isSpeaking
  }
}
