// @vitest-environment jsdom

import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useTranscriptionModelsCatalog } from "../useTranscriptionModelsCatalog"

const {
  getTranscriptionModelsMock,
  getTranscriptionModelHealthMock,
  getTranscriptionCapabilitiesMock,
  tMock,
  unstableTranslationRef
} = vi.hoisted(() => ({
  getTranscriptionModelsMock: vi.fn(),
  getTranscriptionModelHealthMock: vi.fn(),
  getTranscriptionCapabilitiesMock: vi.fn(),
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
    getTranscriptionModelHealth: getTranscriptionModelHealthMock,
    getTranscriptionCapabilities: getTranscriptionCapabilitiesMock
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
    getTranscriptionCapabilitiesMock.mockRejectedValue(new Error("not available"))
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

  it("falls back to selected-model health when capability summary is empty", async () => {
    getTranscriptionCapabilitiesMock.mockResolvedValue({ models: [] })

    const { result } = renderHook(() =>
      useTranscriptionModelsCatalog({ defaultModel: "whisper-small" })
    )

    await waitFor(() => {
      expect(result.current.serverModelsLoading).toBe(false)
    })

    expect(getTranscriptionCapabilitiesMock).toHaveBeenCalledTimes(1)
    expect(getTranscriptionModelHealthMock).toHaveBeenCalledTimes(1)
    expect(getTranscriptionModelHealthMock).toHaveBeenCalledWith("whisper-small")
    expect(result.current.modelOptions).toEqual([
      expect.objectContaining({
        id: "nemo-parakeet-1.1b",
        availability: "unknown"
      }),
      expect.objectContaining({
        id: "whisper-small",
        availability: "ready"
      })
    ])
  })

  it("uses capability summary as an enhancement when available", async () => {
    getTranscriptionCapabilitiesMock.mockResolvedValue({
      models: [
        {
          id: "whisper-small",
          label: "Whisper Small",
          description: "Balanced speed/accuracy",
          category: "Whisper Models",
          provider: "faster-whisper",
          availability: "ready",
          availability_source: "health",
          capabilities: {
            batch: "supported",
            streaming: "supported",
            diarization: "supported",
            timestamps: "supported",
            segments: "supported"
          },
          sources: {
            availability: "health",
            batch: "provider",
            streaming: "provider",
            diarization: "provider",
            timestamps: "response_schema",
            segments: "response_schema"
          },
          message: "Ready"
        },
        {
          id: "nemo-parakeet-1.1b",
          label: "Nemo Parakeet 1.1B",
          provider: "parakeet",
          availability: "on_demand",
          availability_source: "health",
          capabilities: {
            batch: "supported",
            streaming: "supported",
            diarization: "unsupported",
            timestamps: "supported",
            segments: "supported"
          },
          sources: {
            availability: "health",
            batch: "provider",
            streaming: "provider",
            diarization: "provider",
            timestamps: "response_schema",
            segments: "response_schema"
          },
          message: "Initializes on first use"
        }
      ]
    })

    const { result } = renderHook(() =>
      useTranscriptionModelsCatalog({ defaultModel: "whisper-small" })
    )

    await waitFor(() => {
      expect(result.current.serverModelsLoading).toBe(false)
    })

    expect(getTranscriptionCapabilitiesMock).toHaveBeenCalledTimes(1)
    expect(getTranscriptionModelHealthMock).not.toHaveBeenCalled()
    expect(result.current.modelOptions).toEqual([
      expect.objectContaining({
        id: "nemo-parakeet-1.1b",
        provider: "parakeet",
        availability: "on_demand",
        capabilities: expect.objectContaining({
          streaming: "supported",
          diarization: "unsupported"
        })
      }),
      expect.objectContaining({
        id: "whisper-small",
        provider: "faster-whisper",
        availability: "ready",
        capabilities: expect.objectContaining({
          timestamps: "supported"
        }),
        sources: expect.objectContaining({
          timestamps: "response_schema"
        })
      })
    ])
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
