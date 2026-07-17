import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useTtsProviderData } from "@/hooks/useTtsProviderData"
import { getModels, getVoices } from "@/services/elevenlabs"
import { fetchTldwTtsModels } from "@/services/tldw/audio-models"
import { fetchTtsProviders } from "@/services/tldw/audio-providers"
import { fetchTldwVoiceCatalog } from "@/services/tldw/audio-voices"

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasAudio: true },
    loading: false
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/elevenlabs", () => ({
  getVoices: vi.fn(),
  getModels: vi.fn()
}))

vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: vi.fn()
}))

vi.mock("@/services/tldw/audio-models", () => ({
  fetchTldwTtsModels: vi.fn()
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: vi.fn()
}))

const buildHarness = () => {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false }
    }
  })
  return {
    client,
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )
  }
}

describe("useTtsProviderData", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(fetchTtsProviders).mockResolvedValue(null)
    vi.mocked(fetchTldwTtsModels).mockResolvedValue([])
    vi.mocked(fetchTldwVoiceCatalog).mockResolvedValue([])
  })

  it("loads ElevenLabs metadata with a bounded timeout", async () => {
    vi.mocked(getVoices).mockResolvedValue([
      { voice_id: "voice-1", name: "Voice 1" }
    ])
    vi.mocked(getModels).mockResolvedValue([
      { model_id: "model-1", name: "Model 1" }
    ])

    const { wrapper } = buildHarness()
    const { result } = renderHook(
      () =>
        useTtsProviderData({
          provider: "elevenlabs",
          elevenLabsApiKey: "test-key"
        }),
      { wrapper }
    )

    await waitFor(() => {
      expect(result.current.elevenLabsLoading).toBe(false)
      expect(result.current.elevenLabsData?.voices).toHaveLength(1)
      expect(result.current.elevenLabsData?.models).toHaveLength(1)
    })

    expect(getVoices).toHaveBeenCalledWith("test-key", { timeoutMs: 10_000 })
    expect(getModels).toHaveBeenCalledWith("test-key", { timeoutMs: 10_000 })
    expect(result.current.elevenLabsError).toBeNull()
    expect(typeof result.current.refetchElevenLabs).toBe("function")
  })

  it("surfaces ElevenLabs metadata failures for retry UX", async () => {
    vi.mocked(getVoices).mockRejectedValue(new Error("Request timeout"))
    vi.mocked(getModels).mockResolvedValue([
      { model_id: "model-1", name: "Model 1" }
    ])

    const { wrapper } = buildHarness()
    const { result } = renderHook(
      () =>
        useTtsProviderData({
          provider: "elevenlabs",
          elevenLabsApiKey: "test-key"
        }),
      { wrapper }
    )

    await waitFor(() => {
      expect(result.current.elevenLabsLoading).toBe(false)
      expect(result.current.elevenLabsError).toBeTruthy()
    })

    expect(getVoices).toHaveBeenCalledWith("test-key", { timeoutMs: 10_000 })
    expect(getModels).toHaveBeenCalledWith("test-key", { timeoutMs: 10_000 })
    expect(result.current.elevenLabsData).toBeUndefined()
  })

  it("scopes model and voice queries by exact backend and model", async () => {
    const { client, wrapper } = buildHarness()

    const { result } = renderHook(
      () =>
        useTtsProviderData({
          provider: "tldw",
          backend: "gateway:Company",
          model: "Vendor/Exact-Case",
          inferredProviderKey: "openai"
        }),
      { wrapper }
    )

    await waitFor(() => {
      expect(fetchTldwTtsModels).toHaveBeenCalledWith("gateway:Company")
      expect(fetchTldwVoiceCatalog).toHaveBeenCalledWith("gateway:Company", {
        model: "Vendor/Exact-Case"
      })
      expect(result.current.tldwTtsModels).toEqual([])
      expect(result.current.tldwVoiceCatalog).toEqual([])
    })

    expect(
      client
        .getQueryCache()
        .findAll()
        .map((query) => query.queryKey)
    ).toEqual(
      expect.arrayContaining([
        ["tldw-tts-models", "gateway:Company"],
        [
          "tldw-voice-catalog",
          "gateway:Company",
          "Vendor/Exact-Case"
        ]
      ])
    )
  })

  it("ignores late model and voice results from a previous selection", async () => {
    let resolveOldModels!: (value: { id: string; label: string }[]) => void
    let resolveOldVoices!: (value: { id: string; name: string }[]) => void
    vi.mocked(fetchTldwTtsModels).mockImplementation((backend?: string) => {
      if (backend === "gateway:Old") {
        return new Promise((resolve) => {
          resolveOldModels = resolve
        })
      }
      return Promise.resolve([{ id: "New/Model", label: "New/Model" }])
    })
    vi.mocked(fetchTldwVoiceCatalog).mockImplementation((provider: string) => {
      if (provider === "gateway:Old") {
        return new Promise((resolve) => {
          resolveOldVoices = resolve
        })
      }
      return Promise.resolve([{ id: "NewVoice", name: "New Voice" }])
    })
    const { wrapper } = buildHarness()

    const { result, rerender } = renderHook(
      ({ backend, model }) =>
        useTtsProviderData({
          provider: "tldw",
          backend,
          model,
          inferredProviderKey: null
        }),
      {
        initialProps: { backend: "gateway:Old", model: "Old/Model" },
        wrapper
      }
    )

    rerender({ backend: "gateway:New", model: "New/Model" })

    await waitFor(() => {
      expect(result.current.tldwTtsModels?.[0]?.id).toBe("New/Model")
      expect(result.current.tldwVoiceCatalog?.[0]?.id).toBe("NewVoice")
    })

    resolveOldModels([{ id: "Old/Model", label: "Old/Model" }])
    resolveOldVoices([{ id: "OldVoice", name: "Old Voice" }])

    await waitFor(() => {
      expect(result.current.tldwTtsModels?.[0]?.id).toBe("New/Model")
      expect(result.current.tldwVoiceCatalog?.[0]?.id).toBe("NewVoice")
    })
  })
})
