import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { usePersonaLiveVoiceController } from "../usePersonaLiveVoiceController"

const mocks = vi.hoisted(() => ({
  micStart: vi.fn(async () => {}),
  micStop: vi.fn(),
  audioStart: vi.fn(),
  audioAppend: vi.fn(),
  audioFinish: vi.fn(),
  audioStop: vi.fn(),
  audioState: { playing: false, error: null as string | null },
  chunk: null as null | ((chunk: ArrayBuffer) => void)
}))
vi.mock("@/hooks/useMicStream", () => ({
  useMicStream: (chunk: typeof mocks.chunk) => {
    mocks.chunk = chunk
    return { start: mocks.micStart, stop: mocks.micStop, active: false }
  }
}))
vi.mock("@/hooks/useStreamingAudioPlayer", () => ({
  useStreamingAudioPlayer: () => ({
    start: mocks.audioStart,
    append: mocks.audioAppend,
    finish: mocks.audioFinish,
    stop: mocks.audioStop,
    state: mocks.audioState
  })
}))
vi.mock("@/hooks/useAudioSourceCatalog", () => ({
  useAudioSourceCatalog: () => ({ devices: [], isSettled: true })
}))
vi.mock("@/hooks/useAudioSourcePreferences", () => ({
  useAudioSourcePreferences: () => ({
    preference: { sourceKind: "default_mic", deviceId: null },
    isLoading: false
  })
}))

const defaults = {
  sttLanguage: "en",
  sttModel: "whisper",
  ttsProvider: "kokoro",
  ttsVoice: "af_heart",
  confirmationMode: "destructive_only" as const,
  wakeBehavior: "one_shot" as const,
  voiceChatTriggerPhrases: [],
  autoResume: true,
  bargeIn: false,
  autoCommitEnabled: true,
  vadThreshold: 0.5,
  minSilenceMs: 250,
  turnStopSecs: 0.2,
  minUtteranceSecs: 0.4
}
const setup = (overrides: Partial<typeof defaults> = {}) => {
  const ws = {
    readyState: WebSocket.OPEN,
    send: vi.fn()
  } as unknown as WebSocket
  const hook = renderHook(
    ({ sessionId }) =>
      usePersonaLiveVoiceController({
        ws,
        connected: true,
        sessionId,
        personaId: "persona",
        resolvedDefaults: { ...defaults, ...overrides },
        canUseServerStt: true
      }),
    { initialProps: { sessionId: "session" } }
  )
  const sent = () =>
    vi.mocked(ws.send).mock.calls.map(([data]) => JSON.parse(String(data)))
  const prepare = () =>
    sent().findLast((payload) => payload.type === "voice_prepare")
  const start = async () => {
    let pending!: Promise<void>
    await act(async () => {
      pending = hook.result.current.startListening()
      await Promise.resolve()
    })
    return { pending }
  }
  const ready = async (pending: Promise<void>) => {
    await act(async () => {
      hook.result.current.handlePayload({
        ...prepare(),
        event: "voice_readiness",
        ready: true
      })
      await pending
    })
  }
  return { ...hook, ws, sent, prepare, start, ready }
}

describe("Persona voice readiness and ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.micStart.mockImplementation(async () => {})
    mocks.audioState.playing = false
    mocks.audioState.error = null
  })
  afterEach(() => vi.useRealTimers())

  it("replaces revised transcript snapshots without deleting intentional repeats", async () => {
    const h = setup({ autoCommitEnabled: false })
    const { pending } = await h.start()
    await h.ready(pending)
    for (const transcript of ["blue boat", "blue notebook is ready", "blue notebook is ready", "blue notebook is ready ready"]) {
      act(() => h.result.current.handlePayload({
        ...h.prepare(), event: "partial_transcript", transcript, text_delta: "legacy delta"
      }))
    }
    expect(h.result.current.heardText).toBe("blue notebook is ready ready")
    act(() => h.result.current.handlePayload({ ...h.prepare(), event: "partial_transcript", transcript: "", text_delta: "" }))
    expect(h.result.current.heardText).toBe("")
    act(() => h.result.current.handlePayload({ ...h.prepare(), event: "partial_transcript", transcript: "blue notebook is ready ready" }))
    act(() => h.result.current.sendCurrentTranscriptNow())
    expect(h.sent().findLast((message) => message.type === "voice_commit")?.transcript).toBe("blue notebook is ready ready")
  })

  it("stops rejected audio capture and requires explicit retry after throttling", async () => {
    const h = setup()
    const { pending } = await h.start()
    await h.ready(pending)
    const notice = { ...h.prepare(), event: "notice", reason_code: "AUDIO_RATE_LIMITED", message: "Audio chunk rate limit exceeded" }
    act(() => h.result.current.handlePayload({ ...notice, client_message_id: "stale" }))
    expect(h.result.current.state).toBe("listening")
    mocks.micStop.mockClear()
    act(() => h.result.current.handlePayload(notice))
    expect(mocks.micStop).toHaveBeenCalledOnce()
    expect(h.result.current.state).toBe("error")
    expect(h.result.current.warning).toMatch(/wait.*minute.*start/i)
    const beforeLateChunk = h.sent().length
    act(() => mocks.chunk?.(new ArrayBuffer(8192)))
    expect(h.sent()).toHaveLength(beforeLateChunk)
    const retry = await h.start()
    await h.ready(retry.pending)
    expect(h.result.current.state).toBe("listening")
  })

  it("waits for matching server readiness before requesting microphone capture", async () => {
    const h = setup()
    const { pending } = await h.start()
    expect(h.prepare()).toEqual(
      expect.objectContaining({
        session_id: "session",
        client_message_id: expect.any(String)
      })
    )
    expect(mocks.micStart).not.toHaveBeenCalled()
    expect(h.result.current.isPreparing).toBe(true)
    act(() =>
      h.result.current.handlePayload({
        ...h.prepare(),
        event: "voice_readiness",
        session_id: "other",
        ready: true
      })
    )
    expect(mocks.micStart).not.toHaveBeenCalled()
    await h.ready(pending)
    expect(mocks.micStart).toHaveBeenCalledOnce()
    act(() => mocks.chunk?.(new ArrayBuffer(4)))
    expect(h.sent().at(-1)).toMatchObject({
      type: "audio_chunk",
      client_message_id: h.prepare().client_message_id
    })
  })

  it("shows actionable preparation failure without microphone access", async () => {
    const h = setup()
    const { pending } = await h.start()
    await act(async () => {
      h.result.current.handlePayload({
        ...h.prepare(),
        event: "voice_readiness",
        ready: false,
        reason_code: "VOICE_STT_UNAVAILABLE",
        message: "Install the selected Whisper model in Audio settings."
      })
      await pending
    })
    expect(mocks.micStart).not.toHaveBeenCalled()
    expect(h.result.current.warning).toContain("Whisper")
    expect(h.result.current.state).toBe("error")
  })

  it("bounds a missing readiness response and permits an explicit retry", async () => {
    vi.useFakeTimers()
    const h = setup()
    const { pending } = await h.start()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
      await pending
    })
    expect(mocks.micStart).not.toHaveBeenCalled()
    expect(h.result.current.warning).toMatch(/timed out/i)
    const retry = await h.start()
    await h.ready(retry.pending)
    expect(mocks.micStart).toHaveBeenCalledOnce()
  })

  it("Stop cancels pending preparation and rejects its late readiness", async () => {
    const h = setup()
    const { pending } = await h.start()
    const old = h.prepare()
    act(() => h.result.current.resetTurn())
    await act(async () => {
      await pending
    })
    act(() =>
      h.result.current.handlePayload({
        ...old,
        event: "voice_readiness",
        ready: true
      })
    )
    expect(mocks.micStart).not.toHaveBeenCalled()
    expect(h.sent()).toContainEqual(
      expect.objectContaining({
        type: "voice_stop",
        client_message_id: old.client_message_id
      })
    )
    expect(h.result.current.isListening).toBe(false)
  })

  it("only a matching TTS header admits one binary frame; Stop rejects late output", async () => {
    const h = setup()
    const { pending } = await h.start()
    await h.ready(pending)
    const owner = h.prepare()
    act(() => h.result.current.handleBinaryPayload(new ArrayBuffer(4)))
    expect(mocks.audioAppend).not.toHaveBeenCalled()
    act(() => {
      h.result.current.handlePayload({
        ...owner,
        event: "tts_audio",
        chunk_index: 0,
        chunk_count: 2
      })
      h.result.current.handleBinaryPayload(new ArrayBuffer(4))
      h.result.current.handleBinaryPayload(new ArrayBuffer(4))
    })
    expect(mocks.audioAppend).toHaveBeenCalledOnce()
    act(() => h.result.current.resetTurn())
    act(() => {
      h.result.current.handlePayload({
        ...owner,
        event: "tts_audio",
        chunk_index: 0,
        chunk_count: 1
      })
      h.result.current.handleBinaryPayload(new ArrayBuffer(4))
      h.result.current.handlePayload({
        ...owner,
        event: "partial_transcript",
        text_delta: "late"
      })
      h.result.current.handlePayload({
        ...owner,
        event: "assistant_delta",
        text_delta: "late"
      })
    })
    expect(mocks.audioStart).toHaveBeenCalledOnce()
    expect(mocks.audioAppend).toHaveBeenCalledOnce()
    expect(h.result.current.heardText).toBe("")
    expect(h.result.current.state).toBe("idle")
    expect(mocks.micStart).toHaveBeenCalledOnce()
  })

  it("a session change cancels a pending microphone start and stale chunks", async () => {
    let release!: () => void
    mocks.micStart.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        })
    )
    const h = setup()
    const { pending } = await h.start()
    await act(async () => {
      h.result.current.handlePayload({
        ...h.prepare(),
        event: "voice_readiness",
        ready: true
      })
      await Promise.resolve()
    })
    const oldChunk = mocks.chunk
    h.rerender({ sessionId: "new-session" })
    await act(async () => {
      release()
      await pending
      oldChunk?.(new ArrayBuffer(4))
    })
    expect(h.sent().filter((p) => p.type === "audio_chunk")).toEqual([])
    expect(h.result.current.state).toBe("idle")
    expect(mocks.micStop).toHaveBeenCalled()
  })

  it("the main voice toggle stops a thinking turn instead of starting another capture", async () => {
    const h = setup()
    const { pending } = await h.start()
    await h.ready(pending)
    act(() =>
      h.result.current.handlePayload({
        ...h.prepare(),
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "hello"
      })
    )
    expect(h.result.current.state).toBe("thinking")
    act(() => h.result.current.toggleListening())
    expect(h.result.current.state).toBe("idle")
    expect(h.sent().filter((p) => p.type === "voice_prepare")).toHaveLength(1)
    expect(h.sent().some((p) => p.type === "voice_stop")).toBe(true)
  })

  it("a matching session terminal notice cancels preparation without a turn identifier", async () => {
    const h = setup()
    const { pending } = await h.start()
    act(() =>
      h.result.current.handlePayload({
        event: "notice",
        reason_code: "SESSION_TERMINAL",
        session_id: "other"
      })
    )
    expect(h.result.current.isPreparing).toBe(true)
    act(() =>
      h.result.current.handlePayload({
        event: "notice",
        reason_code: "SESSION_TERMINAL",
        session_id: "session"
      })
    )
    await act(async () => {
      await pending
    })
    expect(h.result.current.isVoiceActive).toBe(false)
    expect(mocks.micStart).not.toHaveBeenCalled()
  })

  it("marks retired voice replies for exclusion from the shared transcript", async () => {
    const h = setup()
    const { pending } = await h.start()
    await h.ready(pending)
    const owner = h.prepare()
    act(() => h.result.current.resetTurn())
    expect(
      h.result.current.handlePayload({
        ...owner,
        event: "assistant_delta",
        text_delta: "late"
      })
    ).toBe(false)
    expect(
      h.result.current.handlePayload({
        session_id: "session",
        client_message_id: "text-request",
        event: "assistant_delta",
        text_delta: "text"
      })
    ).not.toBe(false)
  })

  it("waits for slow owned speech instead of auto-resuming after assistant text", async () => {
    vi.useFakeTimers()
    const h = setup()
    const { pending } = await h.start()
    await h.ready(pending)
    const owner = h.prepare()
    act(() =>
      h.result.current.handlePayload({
        ...owner,
        event: "notice",
        reason_code: "VOICE_TURN_COMMITTED",
        transcript: "hello"
      })
    )
    act(() =>
      h.result.current.handlePayload({
        ...owner,
        event: "assistant_delta",
        text_delta: "Hello back"
      })
    )
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })
    expect(h.sent().filter((p) => p.type === "voice_prepare")).toHaveLength(1)
    act(() => {
      h.result.current.handlePayload({
        ...owner,
        event: "tts_audio",
        chunk_index: 0,
        chunk_count: 2
      })
      h.result.current.handleBinaryPayload(new ArrayBuffer(4))
    })
    expect(mocks.audioAppend).toHaveBeenCalledOnce()
    expect(h.result.current.state).toBe("speaking")
  })

  it("returns to an explicit Start after a completed turn when auto-resume is off", async () => {
    const h = setup({ autoResume: false })
    const { pending } = await h.start()
    await h.ready(pending)
    act(() =>
      h.result.current.handlePayload({
        ...h.prepare(),
        event: "notice",
        reason_code: "TTS_UNAVAILABLE_TEXT_ONLY"
      })
    )
    expect(h.result.current.state).toBe("idle")
    expect(h.result.current.isVoiceActive).toBe(false)
  })

  it("accepts actual speech after a successful preparation recovers a degraded turn", async () => {
    const h = setup()
    const { pending } = await h.start()
    await h.ready(pending)
    const failedOwner = h.prepare()
    act(() => {
      h.result.current.handlePayload({
        ...failedOwner,
        event: "notice",
        reason_code: "VOICE_MANUAL_MODE_REQUIRED"
      })
      h.result.current.handlePayload({
        ...failedOwner,
        event: "notice",
        reason_code: "TTS_UNAVAILABLE_TEXT_ONLY"
      })
    })
    expect(h.result.current.textOnlyDueToTtsFailure).toBe(true)
    expect(h.result.current.manualModeRequired).toBe(true)
    expect(h.prepare().client_message_id).not.toBe(
      failedOwner.client_message_id
    )
    await h.ready(Promise.resolve())
    const recoveredOwner = h.prepare()
    act(() =>
      h.result.current.handlePayload({
        ...recoveredOwner,
        event: "assistant_delta",
        text_delta: "Recovered reply"
      })
    )
    act(() => {
      h.result.current.handlePayload({
        ...recoveredOwner,
        event: "tts_audio",
        chunk_index: 0,
        chunk_count: 2
      })
      h.result.current.handleBinaryPayload(new ArrayBuffer(4))
    })
    expect(mocks.audioAppend).toHaveBeenCalledOnce()
    expect(h.result.current.state).toBe("speaking")
    expect(h.result.current.textOnlyDueToTtsFailure).toBe(false)
    expect(h.result.current.manualModeRequired).toBe(false)
    expect(h.result.current.warning).toBeNull()
  })

  it.each(["preparing", "playing"])(
    "retires a server-cancelled %s owner without cancelling its successor",
    async (phase) => {
      const h = setup()
      const { pending } = await h.start()
      if (phase === "playing") {
        await h.ready(pending)
        act(() =>
          h.result.current.handlePayload({
            ...h.prepare(),
            event: "tts_audio",
            chunk_index: 0,
            chunk_count: 2
          })
        )
      }
      const owner = h.prepare()
      act(() =>
        h.result.current.handlePayload({
          ...owner,
          client_message_id: "other-turn",
          event: "notice",
          reason_code: "TURN_CANCELLED"
        })
      )
      expect(h.result.current.isVoiceActive).toBe(true)
      const sentBeforeCancellation = h.sent()
      const micStops = mocks.micStop.mock.calls.length
      const audioStops = mocks.audioStop.mock.calls.length
      act(() =>
        h.result.current.handlePayload({
          ...owner,
          event: "notice",
          reason_code: "TURN_CANCELLED"
        })
      )
      expect(h.result.current.isVoiceActive).toBe(false)
      await act(async () => {
        await pending
      })
      act(() => {
        h.result.current.handlePayload({
          ...owner,
          event: "voice_readiness",
          ready: true
        })
        h.result.current.handleBinaryPayload(new ArrayBuffer(4))
      })
      expect(h.sent()).toEqual(sentBeforeCancellation)
      expect(mocks.micStop.mock.calls.length).toBeGreaterThan(micStops)
      expect(mocks.audioStop.mock.calls.length).toBeGreaterThan(audioStops)
      expect(mocks.audioAppend).not.toHaveBeenCalled()
      expect(h.result.current.state).toBe("idle")
    }
  )

  it.each([true, false])(
    "stops on owned playback errors even when player.playing remains %s",
    async (playing) => {
      const h = setup()
      const { pending } = await h.start()
      await h.ready(pending)
      mocks.audioState.playing = true
      act(() => {
        h.result.current.handlePayload({
          ...h.prepare(),
          event: "tts_audio",
          chunk_index: 0,
          chunk_count: 1
        })
        h.result.current.handleBinaryPayload(new ArrayBuffer(4))
      })
      const prepareCount = h
        .sent()
        .filter((p) => p.type === "voice_prepare").length
      mocks.audioState.playing = playing
      mocks.audioState.error = "Audio playback blocked"
      h.rerender({ sessionId: "session" })
      expect(h.result.current.state).toBe("error")
      expect(h.result.current.warning).toMatch(/audio.*retry Start/i)
      expect(h.result.current.isVoiceActive).toBe(false)
      expect(h.sent().filter((p) => p.type === "voice_prepare")).toHaveLength(
        prepareCount
      )
      const retry = await h.start()
      await h.ready(retry.pending)
      expect(h.result.current.state).toBe("listening")
      expect(h.result.current.warning).toBeNull()
    }
  )

  it.each([
    ["USER_TURN_FAILED", false],
    ["CONVERSATION_UNAVAILABLE", false],
    ["TTS_SEND_FAILED", true]
  ] as const)(
    "retires owned %s failures and permits explicit retry",
    async (reasonCode, partialAudio) => {
      const h = setup()
      const { pending } = await h.start()
      await h.ready(pending)
      const owner = h.prepare()
      act(() => {
        h.result.current.handlePayload({
          ...owner,
          event: "notice",
          reason_code: "VOICE_TURN_COMMITTED"
        })
        if (partialAudio) {
          mocks.audioState.playing = true
          h.result.current.handlePayload({
            ...owner,
            event: "tts_audio",
            chunk_index: 0,
            chunk_count: 2
          })
          h.result.current.handleBinaryPayload(new ArrayBuffer(4))
        }
      })
      const micStops = mocks.micStop.mock.calls.length
      const audioStops = mocks.audioStop.mock.calls.length
      const sentBeforeFailure = h.sent()
      act(() =>
        h.result.current.handlePayload({
          ...owner,
          event: "notice",
          reason_code: reasonCode,
          message: "Check server settings and retry Start."
        })
      )
      expect(h.result.current.state).toBe("error")
      expect(h.result.current.warning).toBe(
        "Check server settings and retry Start."
      )
      expect(h.result.current.isVoiceActive).toBe(false)
      expect(mocks.micStop.mock.calls.length).toBeGreaterThan(micStops)
      expect(mocks.audioStop.mock.calls.length).toBeGreaterThan(audioStops)
      expect(h.sent()).toEqual(sentBeforeFailure)
      mocks.audioState.playing = false
      h.rerender({ sessionId: "session" })
      expect(h.sent()).toEqual(sentBeforeFailure)
      const retry = await h.start()
      await h.ready(retry.pending)
      expect(h.result.current.state).toBe("listening")
      expect(h.result.current.warning).toBeNull()
      const audioAppends = mocks.audioAppend.mock.calls.length
      act(() => {
        h.result.current.handlePayload({
          ...owner,
          event: "notice",
          reason_code: reasonCode,
          message: "Stale failure"
        })
        h.result.current.handlePayload({
          ...owner,
          event: "tts_audio",
          chunk_index: 1,
          chunk_count: 2
        })
        h.result.current.handleBinaryPayload(new ArrayBuffer(4))
      })
      expect(h.result.current.state).toBe("listening")
      expect(h.result.current.warning).toBeNull()
      expect(h.result.current.isVoiceActive).toBe(true)
      expect(mocks.audioAppend).toHaveBeenCalledTimes(audioAppends)
    }
  )

  it("ignores another session's wake notice while accepting this session without a turn ID", () => {
    const h = setup()
    act(() =>
      h.result.current.handlePayload({
        event: "notice",
        session_id: "other-session",
        reason_code: "WAKE_ACTIVATION_REJECTED",
        message: "Rejected for another session"
      })
    )
    expect(h.result.current.wakeWarning).toBeNull()
    act(() =>
      h.result.current.handlePayload({
        event: "notice",
        session_id: "session",
        reason_code: "WAKE_ACTIVATION_REJECTED",
        message: "Rejected for this session"
      })
    )
    expect(h.result.current.wakeWarning).toBe("Rejected for this session")
  })
})
