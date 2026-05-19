// @vitest-environment jsdom

import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useTranscriptionModelsCatalog } from "../useTranscriptionModelsCatalog"

const {
  getTranscriptionModelsMock,
  getTranscriptionModelHealthMock,
  tMock,
  unstableTranslationRef
} = vi.hoisted(() => ({
  getTranscriptionModelsMock: vi.fn(),
  getTranscriptionModelHealthMock: vi.fn(),
  tMock: vi.fn((_key: string, fallback?: string) => fallback || _key),
  unstableTranslationRef: { current: false }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: unstableTranslationRef.current
      ? vi.fn((_key: string, fallback: string) => fallback)
      : tMock
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getTranscriptionModels: getTranscriptionModelsMock,
    getTranscriptionModelHealth: getTranscriptionModelHealthMock
  }
}))

vi.mock("@/utils/request-timeout", () => ({
  isTimeoutLikeError: vi.fn((error: unknown) =>
    error instanceof Error && error.message.includes("timeout")
  )
}))

describe("useTranscriptionModelsCatalog", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    unstableTranslationRef.current = false
    getTranscriptionModelsMock.mockResolvedValue({
      categories: {
        "Whisper Models": [
          {
            value: "whisper-small",
            label: "Whisper Small",
            description: "Balanced speed/accuracy"
          }
        ],
        "Nemo Models": [
          {
            value: "nemo-parakeet-1.1b",
            label: "Nemo Parakeet 1.1B",
            description: "Standard model"
          }
        ]
      },
      all_models: ["whisper-small", "nemo-parakeet-1.1b"]
    })
    getTranscriptionModelHealthMock.mockResolvedValue({
      available: true,
      usable: true,
      on_demand: false,
      message: "Ready",
      provider: "whisper"
    })
  })

  it("retries model loading through the shared retry callback", async () => {
    getTranscriptionModelsMock
      .mockRejectedValueOnce(new Error("timeout while loading transcription models"))
      .mockResolvedValueOnce({ all_models: ["whisper-1", "parakeet-tdt"] })

    const { result } = renderHook(() => useTranscriptionModelsCatalog())

    await waitFor(() => {
      expect(result.current.serverModelsError).toBe(
        "Model list took longer than 10 seconds. Check server health and retry."
      )
    })
    expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(1)

    act(() => {
      result.current.retryServerModels()
    })

    await waitFor(() => {
      expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(2)
      expect(result.current.serverModels).toEqual(["parakeet-tdt", "whisper-1"])
      expect(result.current.serverModelsError).toBeNull()
    })
  })

  it("supports one automatic retry before surfacing the inline error", async () => {
    getTranscriptionModelsMock
      .mockRejectedValueOnce(new Error("timeout-1"))
      .mockRejectedValueOnce(new Error("timeout-2"))
      .mockResolvedValueOnce({ all_models: ["whisper-1", "parakeet-tdt"] })

    const { result } = renderHook(() =>
      useTranscriptionModelsCatalog({
        autoRetryOnFailureCount: 1
      })
    )

    await waitFor(() => {
      expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(2)
      expect(result.current.serverModelsError).toBe(
        "Model list took longer than 10 seconds. Check server health and retry."
      )
    })

    act(() => {
      result.current.retryServerModels()
    })

    await waitFor(() => {
      expect(getTranscriptionModelsMock).toHaveBeenCalledTimes(3)
      expect(result.current.serverModels).toEqual(["parakeet-tdt", "whisper-1"])
      expect(result.current.serverModelsError).toBeNull()
    })
  })

  it("sets an initial model when provided a setter callback", async () => {
    getTranscriptionModelsMock.mockResolvedValue({
      all_models: ["whisper-1", "parakeet-tdt"]
    })
    const setInitialModel = vi.fn()

    renderHook(() =>
      useTranscriptionModelsCatalog({
        activeModel: undefined,
        defaultModel: "parakeet-tdt",
        onInitialModel: setInitialModel
      })
    )

    await waitFor(() => {
      expect(setInitialModel).toHaveBeenCalledWith("parakeet-tdt")
    })
  })

  it("preserves serverModels while exposing modelOptions metadata", async () => {
    const { result } = renderHook(() =>
      useTranscriptionModelsCatalog({ defaultModel: "whisper-small" })
    )

    await waitFor(() => {
      expect(result.current.serverModelsLoading).toBe(false)
    })

    expect(result.current.serverModels).toEqual([
      "nemo-parakeet-1.1b",
      "whisper-small"
    ])
    expect(result.current.modelOptions).toEqual([
      expect.objectContaining({
        id: "nemo-parakeet-1.1b",
        label: "Nemo Parakeet 1.1B",
        availability: "unknown"
      }),
      expect.objectContaining({
        id: "whisper-small",
        label: "Whisper Small",
        availability: "ready",
        readinessMessage: "Ready"
      })
    ])
  })

  it("checks health only for the default or selected readiness model", async () => {
    const { result } = renderHook(() =>
      useTranscriptionModelsCatalog({ defaultModel: "whisper-small" })
    )

    await waitFor(() => {
      expect(result.current.serverModelsLoading).toBe(false)
    })

    expect(getTranscriptionModelHealthMock).toHaveBeenCalledTimes(1)
    expect(getTranscriptionModelHealthMock).toHaveBeenCalledWith("whisper-small")
  })

  it("does not loop when disabled and translation references are unstable", async () => {
    unstableTranslationRef.current = true
    let renderCount = 0

    const { result } = renderHook(() => {
      renderCount += 1
      return useTranscriptionModelsCatalog({ enabled: false })
    })

    await waitFor(() => {
      expect(result.current.serverModelsLoading).toBe(false)
    })

    expect(result.current.modelOptions).toEqual([])
    expect(getTranscriptionModelsMock).not.toHaveBeenCalled()
    expect(renderCount).toBeLessThan(5)
  })
})
