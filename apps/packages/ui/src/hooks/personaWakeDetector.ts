export type WakeDetectorState =
  | "idle"
  | "starting"
  | "listening"
  | "detected"
  | "unavailable"
  | "error"

export type WakeDetectorKind = "browser_transcript"

export type WakeDetectedEvent = {
  canonicalPhrase: string
  transcript: string
  detectedAtMs: number
  detectorKind: WakeDetectorKind
}

export type WakeDetectorError = {
  message: string
  code?: string
}

export type WakeDetectorConfig = {
  phrases: string[]
  locale?: string
  onWake: (event: WakeDetectedEvent) => void
  onStateChange?: (state: WakeDetectorState) => void
  onError?: (error: WakeDetectorError) => void
}

export interface WakeDetector {
  isAvailable(): Promise<boolean>
  start(config: WakeDetectorConfig): Promise<void>
  stop(): Promise<void>
}

type SpeechRecognitionAlternativeLike = {
  transcript?: string
}

type SpeechRecognitionResultLike = ArrayLike<SpeechRecognitionAlternativeLike>

type SpeechRecognitionEventLike = {
  results?: ArrayLike<SpeechRecognitionResultLike>
}

type SpeechRecognitionErrorEventLike = {
  error?: string
  message?: string
}

type SpeechRecognitionLike = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike

type SpeechRecognitionWindow = Window & {
  SpeechRecognition?: SpeechRecognitionCtor
  webkitSpeechRecognition?: SpeechRecognitionCtor
}

const getSpeechRecognitionCtor = (): SpeechRecognitionCtor | null => {
  if (typeof window === "undefined") return null
  const scopedWindow = window as SpeechRecognitionWindow
  return scopedWindow.SpeechRecognition || scopedWindow.webkitSpeechRecognition || null
}

export const normalizeWakePhraseText = (value: unknown): string =>
  String(value || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim()

export const findCanonicalWakePhrase = (
  transcript: unknown,
  phrases: string[]
): string | null => {
  const normalizedTranscript = ` ${normalizeWakePhraseText(transcript)} `
  if (!normalizedTranscript.trim()) return null
  for (const phrase of phrases || []) {
    const normalizedPhrase = normalizeWakePhraseText(phrase)
    if (!normalizedPhrase) continue
    if (normalizedTranscript.includes(` ${normalizedPhrase} `)) {
      return String(phrase || "").trim()
    }
  }
  return null
}

const buildTranscript = (event: SpeechRecognitionEventLike): string =>
  Array.from(event.results || [])
    .map((result) => result?.[0]?.transcript || "")
    .join(" ")

const normalizeWakePhrases = (phrases: string[]): string[] =>
  (phrases || [])
    .map((phrase) => String(phrase || "").trim())
    .filter(Boolean)

const FATAL_SPEECH_RECOGNITION_ERRORS = new Set([
  "audio-capture",
  "bad-grammar",
  "language-not-supported",
  "not-allowed",
  "service-not-allowed"
])

export class BrowserTranscriptWakeDetector implements WakeDetector {
  private recognition: SpeechRecognitionLike | null = null
  private active = false

  private clearRecognitionHandlers(recognition: SpeechRecognitionLike | null): void {
    if (!recognition) return
    recognition.onresult = null
    recognition.onerror = null
    recognition.onend = null
  }

  async isAvailable(): Promise<boolean> {
    return Boolean(getSpeechRecognitionCtor())
  }

  async start(config: WakeDetectorConfig): Promise<void> {
    await this.stop()
    const Ctor = getSpeechRecognitionCtor()
    const phrases = normalizeWakePhrases(config.phrases)
    if (!Ctor || phrases.length === 0) {
      config.onStateChange?.("unavailable")
      return
    }

    config.onStateChange?.("starting")

    const recognition = new Ctor()
    this.recognition = recognition
    this.active = true
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = config.locale || "en-US"
    recognition.onresult = (event) => {
      if (!this.active || this.recognition !== recognition) return
      const transcript = buildTranscript(event)
      const canonicalPhrase = findCanonicalWakePhrase(transcript, phrases)
      if (!canonicalPhrase) return
      config.onStateChange?.("detected")
      config.onWake({
        canonicalPhrase,
        transcript,
        detectedAtMs: Date.now(),
        detectorKind: "browser_transcript"
      })
    }
    recognition.onerror = (event) => {
      if (!this.active || this.recognition !== recognition) return
      const code = String(event?.error || "")
      const fatal = FATAL_SPEECH_RECOGNITION_ERRORS.has(code)
      if (fatal) {
        this.active = false
        this.recognition = null
      }
      config.onStateChange?.("error")
      config.onError?.({
        code,
        message: String(event?.message || event?.error || "Wake detector error")
      })
      if (fatal) {
        this.clearRecognitionHandlers(recognition)
      }
    }
    recognition.onend = () => {
      if (this.active && this.recognition === recognition) {
        config.onStateChange?.("idle")
        try {
          recognition.start()
          config.onStateChange?.("listening")
        } catch (error) {
          this.active = false
          config.onStateChange?.("error")
          config.onError?.({
            code: "restart_failed",
            message:
              error instanceof Error
                ? error.message
                : "Wake detector could not restart."
          })
        }
      }
    }
    recognition.start()
    config.onStateChange?.("listening")
  }

  async stop(): Promise<void> {
    this.active = false
    const recognition = this.recognition
    this.recognition = null
    try {
      recognition?.stop()
    } catch {
      // SpeechRecognition implementations throw when stop races with end.
    } finally {
      this.clearRecognitionHandlers(recognition)
    }
  }
}
