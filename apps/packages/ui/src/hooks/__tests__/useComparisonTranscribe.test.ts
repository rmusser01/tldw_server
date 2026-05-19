import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { transcribeAudio: vi.fn() }
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useComparisonTranscribe } from "@/hooks/useComparisonTranscribe"

const mockTranscribe = vi.mocked(tldwClient.transcribeAudio)

function makeBlob(content = "audio"): Blob {
  return new Blob([content], { type: "audio/wav" })
}

describe("useComparisonTranscribe", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("returns empty initial state", () => {
    const { result } = renderHook(() => useComparisonTranscribe())
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

    const { result } = renderHook(() => useComparisonTranscribe())
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
    expect(mockTranscribe).toHaveBeenCalledWith(blob, { model: "whisper-1" })
    expect(mockTranscribe).toHaveBeenCalledWith(blob, { model: "nemo" })
  })

  it("passes sttOptions through to transcribeAudio", async () => {
    mockTranscribe.mockResolvedValue({ text: "ok" })

    const { result } = renderHook(() => useComparisonTranscribe())
    const opts = { language: "en", temperature: 0.2 }

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], opts)
    })

    expect(mockTranscribe).toHaveBeenCalledWith(expect.any(Blob), {
      model: "m1",
      language: "en",
      temperature: 0.2
    })
  })

  it("isolates per-model errors", async () => {
    mockTranscribe
      .mockResolvedValueOnce({ text: "success" })
      .mockRejectedValueOnce(new Error("model not found: sk_secret_inline"))

    const { result } = renderHook(() => useComparisonTranscribe())

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

    const { result } = renderHook(() => useComparisonTranscribe())

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["whisper-1"], {})
    })

    expect(result.current.results[0].error).toBe("Credentials need attention")
    expect(result.current.results[0].errorSettingsHref).toBe("/settings/speech")
  })

  it("retries a single model", async () => {
    mockTranscribe
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce({ text: "ok" })
      // For initial second model
      .mockResolvedValueOnce({ text: "other" })

    const { result } = renderHook(() => useComparisonTranscribe())
    const blob = makeBlob()

    // Initial run: first fails, second succeeds
    // Note: allSettled runs both concurrently, so mock order is call order
    mockTranscribe.mockReset()
    mockTranscribe
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce({ text: "other" })

    await act(async () => {
      await result.current.transcribeAll(blob, ["m1", "m2"], {})
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
      model: "m1",
      language: "en"
    })
  })

  it("retryModel is a no-op for unknown model", async () => {
    const { result } = renderHook(() => useComparisonTranscribe())

    await act(async () => {
      await result.current.retryModel(makeBlob(), "nonexistent", {})
    })

    expect(mockTranscribe).not.toHaveBeenCalled()
    expect(result.current.results).toEqual([])
  })

  it("clears results", async () => {
    mockTranscribe.mockResolvedValue({ text: "hi" })
    const { result } = renderHook(() => useComparisonTranscribe())

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
      const { result } = renderHook(() => useComparisonTranscribe())

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("plain text")
    })

    it("handles {text} response", async () => {
      mockTranscribe.mockResolvedValueOnce({ text: "from text field" })
      const { result } = renderHook(() => useComparisonTranscribe())

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("from text field")
    })

    it("handles {transcript} response", async () => {
      mockTranscribe.mockResolvedValueOnce({ transcript: "from transcript" })
      const { result } = renderHook(() => useComparisonTranscribe())

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("from transcript")
    })

    it("handles {segments} response", async () => {
      mockTranscribe.mockResolvedValueOnce({
        segments: [{ text: "seg one" }, { text: "seg two" }]
      })
      const { result } = renderHook(() => useComparisonTranscribe())

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("seg one seg two")
    })

    it("falls back to empty string for unrecognized response", async () => {
      mockTranscribe.mockResolvedValueOnce({ foo: "bar" })
      const { result } = renderHook(() => useComparisonTranscribe())

      await act(async () => {
        await result.current.transcribeAll(makeBlob(), ["m1"], {})
      })

      expect(result.current.results[0].text).toBe("")
    })
  })

  it("computes word count correctly", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "  one  two   three  " })
    const { result } = renderHook(() => useComparisonTranscribe())

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], {})
    })

    expect(result.current.results[0].wordCount).toBe(3)
  })

  it("word count is 0 for empty text", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "" })
    const { result } = renderHook(() => useComparisonTranscribe())

    await act(async () => {
      await result.current.transcribeAll(makeBlob(), ["m1"], {})
    })

    expect(result.current.results[0].wordCount).toBe(0)
  })
})
