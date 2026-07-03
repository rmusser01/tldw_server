import { useCallback, useEffect, useRef, useState } from "react"
import {
  createAudioCaptureSessionCoordinator,
  type AudioCaptureSessionCoordinator
} from "@/audio"

export type AudioRecorderStatus = "idle" | "recording"

export type AudioRecorderOptions = {
  deviceId?: string | null
}

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

const getAudioCaptureSessionCoordinator = (): AudioCaptureSessionCoordinator => {
  const globalState = globalThis as typeof globalThis & {
    [AUDIO_CAPTURE_COORDINATOR_KEY]?: AudioCaptureSessionCoordinator
  }
  if (!globalState[AUDIO_CAPTURE_COORDINATOR_KEY]) {
    globalState[AUDIO_CAPTURE_COORDINATOR_KEY] =
      createAudioCaptureSessionCoordinator()
  }
  return globalState[AUDIO_CAPTURE_COORDINATOR_KEY]
}

function buildAudioConstraints(
  deviceId?: string | null
): MediaStreamConstraints["audio"] {
  return deviceId ? { deviceId: { exact: deviceId } } : true
}

function createCaptureBusyError(activeOwner: string): Error {
  return new Error(`Audio capture is already active for ${activeOwner}`)
}

export interface AudioRecorderResult {
  status: AudioRecorderStatus
  blob: Blob | null
  durationMs: number
  startRecording: (options?: AudioRecorderOptions) => Promise<void>
  stopRecording: () => void
  clearRecording: () => void
  loadBlob: (blob: Blob, durationMs: number) => void
}

const TIMER_INTERVAL_MS = 200

export function useAudioRecorder(): AudioRecorderResult {
  const [status, setStatus] = useState<AudioRecorderStatus>("idle")
  const [blob, setBlob] = useState<Blob | null>(null)
  const [durationMs, setDurationMs] = useState(0)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startTimeRef = useRef<number>(0)
  const captureOwnerRef = useRef(false)
  const startingRef = useRef(false)

  const stopTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const releaseCaptureOwner = useCallback(() => {
    if (!captureOwnerRef.current) return
    captureOwnerRef.current = false
    getAudioCaptureSessionCoordinator().release("speech_playground")
  }, [])

  const reserveCaptureOwner = useCallback(() => {
    if (captureOwnerRef.current) return
    const coordinator = getAudioCaptureSessionCoordinator()
    const activeOwner = coordinator.getActiveOwner()
    if (activeOwner !== null) {
      throw createCaptureBusyError(activeOwner)
    }
    coordinator.claim("speech_playground")
    captureOwnerRef.current = true
  }, [])

  const stopMediaTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    releaseCaptureOwner()
  }, [releaseCaptureOwner])

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop()
    }
  }, [])

  const startRecording = useCallback(async (options: AudioRecorderOptions = {}) => {
    // Synchronous re-entry guard: a rapid double-start (or a start while
    // already recording) must not run a second getUserMedia and orphan the
    // first stream. Refs are checked/set synchronously before the first await.
    if (startingRef.current || recorderRef.current) return
    reserveCaptureOwner()
    startingRef.current = true
    chunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: buildAudioConstraints(options.deviceId)
      })
      streamRef.current = stream

      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      recorder.onerror = () => {
        // A MediaRecorder error may fire without a matching onstop (device
        // unplugged, hardware glitch). Mirror onstop's teardown so the mic
        // track is stopped and the capture owner released on every path.
        stopTimer()
        stopMediaTracks()
        setStatus("idle")
        recorderRef.current = null
      }

      recorder.onstop = () => {
        const recordedBlob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm"
        })
        const finalDuration = Date.now() - startTimeRef.current
        setBlob(recordedBlob)
        setDurationMs(finalDuration)
        stopTimer()
        stopMediaTracks()
        setStatus("idle")
        recorderRef.current = null
      }

      startTimeRef.current = Date.now()
      recorder.start()
      setStatus("recording")

      timerRef.current = setInterval(() => {
        setDurationMs(Date.now() - startTimeRef.current)
      }, TIMER_INTERVAL_MS)
    } catch (error) {
      stopTimer()
      stopMediaTracks()
      throw error
    } finally {
      startingRef.current = false
    }
  }, [reserveCaptureOwner, stopTimer, stopMediaTracks])

  const clearRecording = useCallback(() => {
    setBlob(null)
    setDurationMs(0)
  }, [])

  const loadBlob = useCallback((externalBlob: Blob, externalDurationMs: number) => {
    setBlob(externalBlob)
    setDurationMs(externalDurationMs)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTimer()
      if (recorderRef.current && recorderRef.current.state === "recording") {
        recorderRef.current.stop()
      }
      stopMediaTracks()
    }
  }, [stopTimer, stopMediaTracks])

  return {
    status,
    blob,
    durationMs,
    startRecording,
    stopRecording,
    clearRecording,
    loadBlob
  }
}
