import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createAudioCaptureSessionCoordinator,
  type AudioCaptureSessionCoordinator,
} from "@/audio"

export type MicCaptureOptions = {
  deviceId?: string | null
}

export type MicStreamFormat = "pcm16" | "float32"

export type MicStreamOwner =
  | "live_voice"
  | "voice_chat"
  | "push_to_talk"
  | "dictation"
  | "captions"

export type MicStreamOptions = {
  format?: MicStreamFormat
  owner?: MicStreamOwner
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

function floatTo16BitPCM(float32Array: Float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2)
  const view = new DataView(buffer)
  let offset = 0
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return buffer
}

function floatTo32BitPCM(float32Array: Float32Array) {
  const buffer = new ArrayBuffer(
    float32Array.length * Float32Array.BYTES_PER_ELEMENT
  )
  const copy = new Float32Array(buffer)
  copy.set(float32Array)
  return buffer
}

export function useMicStream(
  onChunk: (pcmChunk: ArrayBuffer) => void,
  options: MicStreamOptions = {}
) {
  const streamFormat = options.format ?? "pcm16"
  const captureOwner = options.owner ?? "live_voice"
  const [active, setActive] = useState(false)
  const activeRef = useRef(false)
  const startingRef = useRef(false)
  const captureOwnerRef = useRef<MicStreamOwner | null>(null)
  const onChunkRef = useRef(onChunk)
  const startIdRef = useRef(0)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)

  useEffect(() => {
    onChunkRef.current = onChunk
  }, [onChunk])

  const releaseCaptureOwner = useCallback(() => {
    const claimedOwner = captureOwnerRef.current
    if (!claimedOwner) return
    captureOwnerRef.current = null
    getAudioCaptureSessionCoordinator().release(claimedOwner)
  }, [])

  const reserveCaptureOwner = useCallback(() => {
    if (captureOwnerRef.current) return
    const coordinator = getAudioCaptureSessionCoordinator()
    const activeOwner = coordinator.getActiveOwner()
    if (activeOwner !== null) {
      throw createCaptureBusyError(activeOwner)
    }
    coordinator.claim(captureOwner)
    captureOwnerRef.current = captureOwner
  }, [captureOwner])

  const stop = useCallback(() => {
    startIdRef.current += 1
    try { processorRef.current?.disconnect() } catch {}
    try { sourceRef.current?.disconnect() } catch {}
    try { ctxRef.current?.close() } catch {}
    mediaStreamRef.current?.getTracks().forEach(t => t.stop())
    processorRef.current = null
    sourceRef.current = null
    mediaStreamRef.current = null
    ctxRef.current = null
    startingRef.current = false
    if (activeRef.current) {
      activeRef.current = false
      setActive(false)
    }
    releaseCaptureOwner()
  }, [releaseCaptureOwner])

  const start = useCallback(async (options: MicCaptureOptions = {}) => {
    if (activeRef.current || startingRef.current) return
    reserveCaptureOwner()
    startingRef.current = true
    const startId = startIdRef.current + 1
    startIdRef.current = startId
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: buildAudioConstraints(options.deviceId)
      })
      if (startIdRef.current !== startId) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      mediaStreamRef.current = stream
      const ctx = new AudioContext({ sampleRate: 16000 })
      ctxRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      sourceRef.current = source
      const processor = ctx.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor
      source.connect(processor)
      processor.connect(ctx.destination)
      processor.onaudioprocess = (e) => {
        if (startIdRef.current !== startId) return
        const input = e.inputBuffer.getChannelData(0)
        const pcm =
          streamFormat === "float32"
            ? floatTo32BitPCM(input)
            : floatTo16BitPCM(input)
        try {
          onChunkRef.current(pcm)
        } catch {
          // ignore handler errors to keep the audio pipeline alive
        }
      }
      activeRef.current = true
      setActive(true)
    } catch (err) {
      if (startIdRef.current === startId) stop()
      throw err
    } finally {
      if (startIdRef.current === startId) startingRef.current = false
    }
  }, [reserveCaptureOwner, stop, streamFormat])

  useEffect(() => {
    activeRef.current = active
  }, [active])

  useEffect(() => () => stop(), [stop])

  return { start, stop, active }
}
