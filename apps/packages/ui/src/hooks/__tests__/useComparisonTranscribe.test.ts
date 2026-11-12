import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { transcribeAudio: vi.fn() }
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useComparisonTranscribe } from "@/hooks/useComparisonTranscribe"

const { resolveScopeMock } = vi.hoisted(() => ({ resolveScopeMock: vi.fn() }))
vi.mock("@/services/service-prompts", () => ({ resolveServicePromptScope: resolveScopeMock }))
const scopeOptions = { signal: expect.any(AbortSignal), requestScope: expect.objectContaining({userId:"unit-owner"}) }
const mockTranscribe = vi.mocked(tldwClient.transcribeAudio)

function makeBlob(content = "audio"): Blob {
  return new Blob([content], { type: "audio/wav" })
}

describe("useComparisonTranscribe", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resolveScopeMock.mockReset().mockResolvedValue({config:{serverUrl:"http://unit-server",authMode:"multi-user"},userId:"unit-owner"})
  })

  it("B05 invalid audio keeps the source row failed with actionable file recovery", async () => {
    mockTranscribe.mockRejectedValueOnce(Object.assign(new Error("Upload failed: 400"), {status:400,details:{detail:{status:"invalid_audio"}}}))
    const {result}=renderHook(()=>useComparisonTranscribe(()=>true))
    await act(async()=>{await result.current.transcribeAll(makeBlob("corrupt bytes"),["parakeet"],{})})
    expect(result.current.results[0]).toMatchObject({status:"error",text:"",errorCategory:"invalid_audio",error:"Audio file cannot be read",errorRecovery:"Choose a valid audio recording, then transcribe again."})
    expect(mockTranscribe).toHaveBeenCalledTimes(1)
  })

  it("B05 source handoff retains the uploaded filename in result provenance", async () => {
    mockTranscribe.mockResolvedValueOnce({text:"Original provider text"})
    const {result}=renderHook(()=>useComparisonTranscribe(()=>true))
    const file=new File(["unit audio"],"New Recording 6.m4a",{type:"audio/mp4"})
    await act(async()=>{await result.current.transcribeAll(file,["parakeet"],{})})
    expect(result.current.results[0].metadata?.audioSourceLabel).toBe(file.name)
    expect(result.current.results[0].text).toBe("Original provider text")
  })

  it("B05 no speech response has an empty transcript and zero words", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "[No speech detected]", duration: 3 })
    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["parakeet"], {})
    })
    expect(result.current.results[0]).toMatchObject({ status: "done", text: "", wordCount: 0 })
    expect(mockTranscribe).toHaveBeenCalledTimes(1)
  })

  it("B05 retry activity remains busy until a single Retry settles", async () => {
    mockTranscribe.mockRejectedValueOnce(new Error("unit provider failure"))
    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob()
    await act(async () => { await result.current.transcribeAll(blob, ["parakeet"], {}) })
    let finish!: (value: { text: string }) => void
    mockTranscribe.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    let retry!: Promise<void>
    await act(async () => { retry = result.current.retryModel(blob, "parakeet", {}) })
    const busyDuringRetry = result.current.isRunning
    await act(async () => { finish({ text: "Recovered transcript" }); await retry })
    expect(busyDuringRetry).toBe(true)
    expect(result.current.isRunning).toBe(false)
    expect(result.current.results[0].text).toBe("Recovered transcript")
  })

  it("B05 retry activity stays busy when another Retry is still pending", async () => {
    mockTranscribe.mockRejectedValueOnce(new Error("first unit failure"))
      .mockRejectedValueOnce(new Error("second unit failure"))
    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob()
    await act(async () => { await result.current.transcribeAll(blob, ["one", "two"], {}) })
    let finishOne!: (value: { text: string }) => void
    let finishTwo!: (value: { text: string }) => void
    mockTranscribe.mockImplementationOnce(() => new Promise(resolve => { finishOne = resolve }))
      .mockImplementationOnce(() => new Promise(resolve => { finishTwo = resolve }))
    let first!: Promise<void>, second!: Promise<void>
    await act(async () => {
      first = result.current.retryModel(blob, "one", {})
      second = result.current.retryModel(blob, "two", {})
    })
    const busy = [result.current.isRunning]
    await act(async () => { finishOne({ text: "First recovered" }); await first })
    busy.push(result.current.isRunning)
    await act(async () => { finishTwo({ text: "Second recovered" }); await second })
    busy.push(result.current.isRunning)
    expect(busy).toEqual([true, true, false])
  })

  it("returns empty initial state", () => {
    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    expect(result.current.results).toEqual([])
    expect(result.current.isRunning).toBe(false)
    expect(typeof result.current.transcribeAll).toBe("function")
    expect(typeof result.current.retryModel).toBe("function")
    expect(typeof result.current.clearResults).toBe("function")
  })

  it("transcribes multiple models in parallel", async () => {
    mockTranscribe
      .mockResolvedValueOnce({ text: "hello from whisper" })
      .mockResolvedValueOnce({ text: "hello from nemo" })

    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob()

    await act(async () => {
      await result.current.transcribeAll(blob, ["whisper-1", "nemo"], {})
    })

    expect(result.current.results).toHaveLength(2)
    expect(result.current.results[0].model).toBe("whisper-1")
    expect(result.current.results[0].text).toBe("hello from whisper")
    expect(result.current.results[0].status).toBe("done")
    expect(result.current.results[0].wordCount).toBe(3)
    expect(typeof result.current.results[0].latencyMs).toBe("number")

    expect(result.current.results[1].model).toBe("nemo")
    expect(result.current.results[1].text).toBe("hello from nemo")
    expect(result.current.results[1].status).toBe("done")
    expect(result.current.results[1].wordCount).toBe(3)

    expect(result.current.isRunning).toBe(false)

    expect(mockTranscribe).toHaveBeenCalledTimes(2)
    expect(mockTranscribe).toHaveBeenCalledWith(blob, { ...scopeOptions, model: "whisper-1" })
    expect(mockTranscribe).toHaveBeenCalledWith(blob, { ...scopeOptions, model: "nemo" })
  })

  it("passes sttOptions through to transcribeAudio", async () => {
    mockTranscribe.mockResolvedValue({ text: "ok" })

    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const opts = { language: "en", temperature: 0.2 }

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], opts)
    })

    expect(mockTranscribe).toHaveBeenCalledWith(expect.any(Blob), {
      ...scopeOptions,
      model: "m1",
      language: "en",
      temperature: 0.2
    })
  })

  it("stores request configuration and available response metadata", async () => {
    mockTranscribe.mockResolvedValueOnce({
      text: "hello world",
      language: "en",
      duration: 2.5,
      segments: [{ text: "hello" }, { text: "world" }],
      words: [{ word: "hello" }, { word: "world" }]
    })

    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob("audio-content")

    await act(async () => {
      await result.current.transcribeAll(blob, ["whisper-large"], {
        language: "en",
        task: "translate",
        response_format: "verbose_json",
        timestamp_granularities: ["word", "segment"],
        segment: true
      })
    })

    expect(result.current.results[0].config).toEqual({
      model: "whisper-large",
      language: "en",
      task: "translate",
      responseFormat: "verbose_json",
      timestampGranularities: ["word", "segment"],
      segmentationEnabled: true
    })
    expect(result.current.results[0].metadata).toMatchObject({
      audioSourceLabel: "Recorded audio",
      audioSizeBytes: blob.size,
      language: "en",
      durationSeconds: 2.5,
      segmentCount: 2,
      wordCount: 2
    })
    expect(result.current.results[0].metadata?.createdAt).toMatch(
      /^\d{4}-\d{2}-\d{2}T/
    )
    expect(typeof result.current.results[0].metadata?.clientLatencyMs).toBe(
      "number"
    )
  })

  it("isolates per-model errors", async () => {
    mockTranscribe
      .mockResolvedValueOnce({ text: "success" })
      .mockRejectedValueOnce(new Error("model not found: sk_secret_inline"))

    const { result } = renderHook(() => useComparisonTranscribe(() => true))

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["good", "bad"], {})
    })

    expect(result.current.results[0].status).toBe("done")
    expect(result.current.results[0].text).toBe("success")

    expect(result.current.results[1].status).toBe("error")
    expect(result.current.results[1].error).toBe("Model is not available")
    expect(result.current.results[1].errorCategory).toBe("missing_model")
    expect(result.current.results[1].errorRecovery).toContain("Audio Setup Guide")
    expect(result.current.results[1].errorDebugMessage).toContain("[redacted]")
    expect(result.current.results[1].errorDebugMessage).not.toContain("sk_secret_inline")
    expect(result.current.results[1].text).toBe("")

    expect(result.current.isRunning).toBe(false)
  })

  it("preserves a settings recovery link for credential failures", async () => {
    mockTranscribe.mockRejectedValueOnce(
      Object.assign(new Error("Request failed for sk_secret_inline"), {
        response: { status: 401 }
      })
    )

    const { result } = renderHook(() => useComparisonTranscribe(() => true))

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["whisper-1"], {})
    })

    expect(result.current.results[0].error).toBe("Credentials need attention")
    expect(result.current.results[0].errorSettingsHref).toBe("/settings/speech")
  })

  it("clears stale error recovery fields after a successful rerun", async () => {
    mockTranscribe
      .mockRejectedValueOnce(
        Object.assign(new Error("Request failed for sk_secret_inline"), {
          response: { status: 401 }
        })
      )
      .mockResolvedValueOnce({ text: "rerun succeeded" })

    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob()

    await act(async () => {
      await result.current.transcribeAll(blob, ["whisper-1"], {})
    })
    expect(result.current.results[0].errorSettingsHref).toBe("/settings/speech")

    await act(async () => {
      await result.current.retryModel(blob, result.current.results[0].id, {})
    })

    expect(result.current.results[0].status).toBe("done")
    expect(result.current.results[0].text).toBe("rerun succeeded")
    expect(result.current.results[0].error).toBeUndefined()
    expect(result.current.results[0].errorSettingsHref).toBeUndefined()
    expect(result.current.results[0].errorDebugMessage).toBeUndefined()
  })

  it("retries a single model with the original row configuration", async () => {
    mockTranscribe
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce({ text: "ok" })
      // For initial second model
      .mockResolvedValueOnce({ text: "other" })

    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob()

    // Initial run: first fails, second succeeds
    // Note: allSettled runs both concurrently, so mock order is call order
    mockTranscribe.mockReset()
    mockTranscribe
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce({ text: "other" })

    await act(async () => {
      await result.current.transcribeAll(blob, ["m1", "m2"], { language: "de" })
    })

    expect(result.current.results[0].status).toBe("error")
    expect(result.current.results[1].status).toBe("done")

    // Retry the failed model
    mockTranscribe.mockResolvedValueOnce({ text: "retry success" })

    await act(async () => {
      await result.current.retryModel(blob, "m1", { language: "en" })
    })

    expect(result.current.results[0].status).toBe("done")
    expect(result.current.results[0].text).toBe("retry success")
    expect(result.current.results[1].text).toBe("other") // unchanged

    expect(mockTranscribe).toHaveBeenLastCalledWith(blob, {
      ...scopeOptions,
      model: "m1",
      language: "de"
    })
  })

  it("retryModel is a no-op for unknown model", async () => {
    const { result } = renderHook(() => useComparisonTranscribe(() => true))

    await act(async () => {
      await result.current.retryModel(makeBlob(), "nonexistent", {})
    })

    expect(mockTranscribe).not.toHaveBeenCalled()
    expect(result.current.results).toEqual([])
  })

  it("duplicates rows and skips disabled rows during repeat runs", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "first result" })

    const { result } = renderHook(() => useComparisonTranscribe(() => true))
    const blob = makeBlob()

    await act(async () => {
      await result.current.transcribeAll(blob, ["m1"], { language: "de" })
    })

    const originalId = result.current.results[0].id

    act(() => {
      result.current.duplicateResult(originalId)
      result.current.setResultDisabled(originalId, true)
    })

    expect(result.current.results).toHaveLength(2)
    expect(result.current.results[0].disabled).toBe(true)
    expect(result.current.results[1].id).not.toBe(originalId)
    expect(result.current.results[1].model).toBe("m1")
    expect(result.current.results[1].requestOptions).toEqual({ language: "de" })

    mockTranscribe.mockReset()
    mockTranscribe.mockResolvedValueOnce({ text: "duplicate rerun" })

    await act(async () => {
      await result.current.transcribeAll(blob, ["m1"], { language: "en" })
    })

    expect(mockTranscribe).toHaveBeenCalledTimes(1)
    expect(mockTranscribe).toHaveBeenCalledWith(blob, {
      ...scopeOptions,
      model: "m1",
      language: "de"
    })
    expect(result.current.results[0].disabled).toBe(true)
    expect(result.current.results[0].text).toBe("first result")
    expect(result.current.results[1].text).toBe("duplicate rerun")
  })

  it("clears results", async () => {
    mockTranscribe.mockResolvedValue({ text: "hi" })
    const { result } = renderHook(() => useComparisonTranscribe(() => true))

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], {})
    })
    expect(result.current.results).toHaveLength(1)

    act(() => {
      result.current.clearResults()
    })

    expect(result.current.results).toEqual([])
    expect(result.current.isRunning).toBe(false)
  })

  describe("response text extraction", () => {
    it("handles plain string response", async () => {
      mockTranscribe.mockResolvedValueOnce("plain text")
      const { result } = renderHook(() => useComparisonTranscribe(() => true))

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("plain text")
    })

    it("handles {text} response", async () => {
      mockTranscribe.mockResolvedValueOnce({ text: "from text field" })
      const { result } = renderHook(() => useComparisonTranscribe(() => true))

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("from text field")
    })

    it("handles {transcript} response", async () => {
      mockTranscribe.mockResolvedValueOnce({ transcript: "from transcript" })
      const { result } = renderHook(() => useComparisonTranscribe(() => true))

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("from transcript")
    })

    it("handles {segments} response", async () => {
      mockTranscribe.mockResolvedValueOnce({
        segments: [{ text: "seg one" }, { text: "seg two" }]
      })
      const { result } = renderHook(() => useComparisonTranscribe(() => true))

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("seg one seg two")
    })

    it("falls back to empty string for unrecognized response", async () => {
      mockTranscribe.mockResolvedValueOnce({ foo: "bar" })
      const { result } = renderHook(() => useComparisonTranscribe(() => true))

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("")
    })
  })

  it("computes word count correctly", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "  one  two   three  " })
    const { result } = renderHook(() => useComparisonTranscribe(() => true))

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], {})
    })

    expect(result.current.results[0].wordCount).toBe(3)
  })

  it("word count is 0 for empty text", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "" })
    const { result } = renderHook(() => useComparisonTranscribe(() => true))

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], {})
    })

    expect(result.current.results[0].wordCount).toBe(0)
  })
  it("B05 clear aborts the upload and late work cannot retire a newer comparison", async () => {
    let oldResolve!: (value: {text: string}) => void;
    let newResolve!: (value: {text: string}) => void;
    mockTranscribe.mockImplementationOnce(() => new Promise(resolve => { oldResolve=resolve; }))
      .mockImplementationOnce(() => new Promise(resolve => { newResolve=resolve; }));
    const hook=renderHook(() => useComparisonTranscribe(() => true));
    let oldRun!: Promise<void>, newRun!: Promise<void>;
    await act(async () => { oldRun=hook.result.current.transcribeAll(makeBlob(), ["old"], {}); });
    const signal=mockTranscribe.mock.calls[0][1]?.signal;
    act(() => { hook.result.current.clearResults(); });
    expect(signal?.aborted).toBe(true);
    await act(async () => { newRun=hook.result.current.transcribeAll(makeBlob(), ["new"], {}); });
    await act(async () => { oldResolve({text:"Obsolete private transcript"}); await oldRun; });
    expect(hook.result.current.isRunning).toBe(true);
    expect(hook.result.current.results.map(r => r.model)).toEqual(["new"]);
    await act(async () => { newResolve({text:"Current transcript"}); await newRun; });
    expect(hook.result.current.results[0].text).toBe("Current transcript");
    hook.unmount();
  })

  it("B05 unmount aborts a pending transcription", async () => {
    let finish!: (value: {text: string}) => void;
    mockTranscribe.mockImplementationOnce(() => new Promise(resolve => { finish=resolve; }));
    const hook=renderHook(() => useComparisonTranscribe(() => true));
    let pending!: Promise<void>;
    await act(async () => { pending=hook.result.current.transcribeAll(makeBlob(), ["old"], {}); });
    const signal=mockTranscribe.mock.calls[0][1]?.signal;
    hook.unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { finish({text:"late"}); await pending; });
  })

  it("B05 lease stops the previous audio before React cleanup when scope returns a new account", async () => {
    let release!: (scope: unknown) => void;
    resolveScopeMock.mockImplementationOnce(() => new Promise(resolve => {release=resolve;}));
    let active=true;
    const hook=renderHook(() => useComparisonTranscribe(() => active));
    let pending!: Promise<void>;
    await act(async () => { pending=hook.result.current.transcribeAll(makeBlob("Alice private audio"), ["parakeet"], {}); });
    active=false; // Synchronous account invalidation, deliberately before any effect cleanup.
    await act(async () => { release({config:{serverUrl:"http://unit-server",authMode:"multi-user"},userId:"bob"}); await pending; });
    expect(mockTranscribe).not.toHaveBeenCalled();
    hook.unmount();
  })

})
