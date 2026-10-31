import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useTldwAudioStatus } from "@/hooks/useTldwAudioStatus"

const state = vi.hoisted(() => ({
  capabilities: {
    hasAudio: true,
    hasStt: true,
    hasTts: true,
    hasVoiceChat: true,
    hasVoiceConversationTransport: true
  } as any,
  loading: false,
  apiSend: vi.fn(),
  fetchVoices: vi.fn(),
  fetchVoiceCatalog: vi.fn(),
  inferProviderFromModel: vi.fn(() => null)
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: state.capabilities,
    loading: state.loading
  })
}))

vi.mock("@/services/api-send", () => ({
  apiSend: (...args: unknown[]) =>
    (state.apiSend as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoices: (...args: unknown[]) =>
    (state.fetchVoices as (...args: unknown[]) => unknown)(...args),
  fetchTldwVoiceCatalog: (...args: unknown[]) =>
    (state.fetchVoiceCatalog as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: (...args: unknown[]) =>
    (state.inferProviderFromModel as (...args: unknown[]) => unknown)(...args)
}))

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useTldwAudioStatus", () => {
  beforeEach(() => {
    state.capabilities = {
      hasAudio: true,
      hasStt: true,
      hasTts: true,
      hasVoiceChat: true,
      hasVoiceConversationTransport: true
    }
    state.loading = false
    state.apiSend.mockReset()
    state.apiSend.mockResolvedValue({ ok: false, status: 404 })
    state.fetchVoices.mockReset()
    state.fetchVoiceCatalog.mockReset()
    state.inferProviderFromModel.mockReset()
    state.inferProviderFromModel.mockReturnValue(null)
    state.fetchVoices.mockResolvedValue([])
    state.fetchVoiceCatalog.mockResolvedValue([])
  })

  it("runs split STT/TTS probes when capabilities are available", async () => {
    state.apiSend.mockImplementation(async (payload: { path: string }) => {
      if (payload.path === "/api/v1/audio/transcriptions/health") {
        return { ok: true, status: 200, data: { available: true } }
      }
      if (payload.path === "/api/v1/audio/health") {
        return { ok: true, status: 200 }
      }
      return { ok: false, status: 404 }
    })

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthState).toBe("healthy")
      expect(result.current.ttsHealthState).toBe("healthy")
    })

    expect(result.current.hasStt).toBe(true)
    expect(result.current.hasTts).toBe(true)
    expect(result.current.hasVoiceChat).toBe(true)
    expect(result.current.hasVoiceConversationTransport).toBe(true)
    expect(result.current.hasAudio).toBe(true)
    expect(result.current.healthState).toBe("healthy")
    expect(state.apiSend).toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/audio/transcriptions/health" })
    )
    expect(state.apiSend).toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/audio/health" })
    )
  })

  it("does not probe audio health until the surface is explicitly enabled", async () => {
    const { result } = renderHook(() => useTldwAudioStatus({ enabled: false }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthLoading).toBe(false)
      expect(result.current.ttsHealthLoading).toBe(false)
    })

    expect(result.current.sttHealthState).toBe("unknown")
    expect(result.current.ttsHealthState).toBe("unknown")
    expect(state.apiSend).not.toHaveBeenCalled()
  })

  it("treats missing STT health endpoint as unknown instead of unhealthy", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: true,
      hasTts: false,
      hasVoiceChat: false
    }
    state.apiSend.mockResolvedValue({ ok: false, status: 404 })

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthLoading).toBe(false)
    })

    expect(result.current.sttHealthState).toBe("unknown")
    expect(result.current.ttsHealthState).toBe("unavailable")
  })

  it("marks STT as unhealthy when probe reports unavailable model", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: true,
      hasTts: false,
      hasVoiceChat: false
    }
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { available: false }
    })

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthLoading).toBe(false)
    })

    expect(result.current.sttHealthState).toBe("unhealthy")
  })

  it("treats on-demand Whisper models as healthy for first-use downloads", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: true,
      hasTts: false,
      hasVoiceChat: false
    }
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { available: false, provider: "whisper", on_demand: true }
    })

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthLoading).toBe(false)
    })

    expect(result.current.sttHealthState).toBe("healthy")
  })

  it("marks TTS unhealthy when the selected provider is not runtime-ready", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: false,
      hasTts: true,
      hasVoiceChat: false
    }
    state.inferProviderFromModel.mockReturnValue("qwen3_tts")
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        status: "healthy",
        providers: {
          details: {
            qwen3_tts: {
              status: "disabled",
              availability: "disabled"
            }
          }
        }
      }
    })

    const { result } = renderHook(
      () =>
        useTldwAudioStatus({
          ttsProvider: "browser",
          tldwTtsModel: "Qwen3-TTS-0.8B"
        } as any),
      {
        wrapper: buildWrapper()
      }
    )

    await waitFor(() => {
      expect(result.current.ttsHealthLoading).toBe(false)
    })

    expect(result.current.ttsHealthState).toBe("unhealthy")
  })

  it("does not reuse tldw provider health when voice chat is configured for openai", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: false,
      hasTts: true,
      hasVoiceChat: false
    }
    state.inferProviderFromModel.mockReturnValue("kokoro")
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        status: "healthy",
        providers: {
          details: {
            kokoro: {
              status: "disabled",
              availability: "disabled"
            },
            openai: {
              status: "enabled",
              availability: "enabled"
            }
          }
        }
      }
    })

    const { result } = renderHook(
      () =>
        useTldwAudioStatus({
          ttsProvider: "openai",
          tldwTtsModel: "kokoro"
        } as any),
      {
        wrapper: buildWrapper()
      }
    )

    await waitFor(() => {
      expect(result.current.ttsHealthLoading).toBe(false)
    })

    expect(result.current.ttsHealthState).toBe("healthy")
  })

  it("fails open when the selected provider is not described in the health payload", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: false,
      hasTts: true,
      hasVoiceChat: false
    }
    state.inferProviderFromModel.mockReturnValue("kokoro")
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        status: "healthy",
        providers: {
          details: {}
        }
      }
    })

    const { result } = renderHook(
      () =>
        useTldwAudioStatus({
          ttsProvider: "browser",
          tldwTtsModel: "kokoro"
        } as any),
      {
        wrapper: buildWrapper()
      }
    )

    await waitFor(() => {
      expect(result.current.ttsHealthLoading).toBe(false)
    })

    expect(result.current.ttsHealthState).toBe("healthy")
  })

  it("fail-opens STT health for non-whisper providers that report not-ready models", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: true,
      hasTts: false,
      hasVoiceChat: false
    }
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { available: false, provider: "parakeet" }
    })

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthLoading).toBe(false)
    })

    expect(result.current.sttHealthState).toBe("healthy")
  })

  it("derives split capability flags from legacy hasAudio-only responses", async () => {
    state.capabilities = {
      hasAudio: true
    }
    state.apiSend.mockImplementation(async (payload: { path: string }) => {
      if (payload.path === "/api/v1/audio/transcriptions/health") {
        return { ok: true, status: 200, data: { available: true } }
      }
      if (payload.path === "/api/v1/audio/health") {
        return { ok: true, status: 200 }
      }
      return { ok: false, status: 404 }
    })

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthState).toBe("healthy")
      expect(result.current.ttsHealthState).toBe("healthy")
    })

    expect(result.current.hasStt).toBe(true)
    expect(result.current.hasTts).toBe(true)
    expect(result.current.hasVoiceConversationTransport).toBe(false)
    expect(result.current.hasAudio).toBe(true)
  })

  it("keeps strict voice conversation transport disabled when only broad voice flags exist", async () => {
    state.capabilities = {
      hasAudio: true,
      hasVoiceChat: true
    }

    const { result } = renderHook(() => useTldwAudioStatus(), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.sttHealthLoading).toBe(false)
      expect(result.current.ttsHealthLoading).toBe(false)
    })

    expect(result.current.hasVoiceChat).toBe(true)
    expect(result.current.hasVoiceConversationTransport).toBe(false)
  })

  it("treats provider catalog voices as available before falling back to custom voices", async () => {
    state.fetchVoices.mockResolvedValue([])
    state.fetchVoiceCatalog.mockResolvedValue([
      { voice_id: "Bella", name: "Bella", provider: "kitten_tts" }
    ])
    state.inferProviderFromModel.mockReturnValue("kitten_tts")

    const { result } = renderHook(
      () =>
        useTldwAudioStatus({
          requireVoices: true,
          tldwTtsModel: "KittenML/kitten-tts-nano-0.8"
        } as any),
      {
        wrapper: buildWrapper()
      }
    )

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    expect(result.current.voicesAvailable).toBe(true)
    expect(state.fetchVoiceCatalog).toHaveBeenCalledWith("kitten_tts")
    expect(state.fetchVoices).not.toHaveBeenCalled()
  })

  it("does not request a voice catalog for a provider explicitly disabled by audio health", async () => {
    state.capabilities = {
      hasAudio: true,
      hasStt: false,
      hasTts: true,
      hasVoiceChat: false
    }
    state.loading = true
    state.inferProviderFromModel.mockReturnValue("kitten_tts")
    state.apiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        status: "unhealthy",
        providers: {
          details: {
            kitten_tts: {
              status: "not_initialized"
            }
          }
        },
        capabilities_envelope: [
          {
            provider: "kitten-tts",
            availability: "disabled"
          }
        ]
      }
    })

    const { result, rerender } = renderHook(
      () =>
        useTldwAudioStatus({
          requireVoices: true,
          tldwTtsModel: "KittenML/kitten-tts-nano-0.8"
        } as any),
      {
        wrapper: buildWrapper()
      }
    )

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(true)
    })
    expect(state.fetchVoiceCatalog).not.toHaveBeenCalled()

    state.loading = false
    rerender()

    await waitFor(() => {
      expect(result.current.ttsHealthLoading).toBe(false)
    })

    expect(state.fetchVoiceCatalog).not.toHaveBeenCalled()
    expect(state.fetchVoices).not.toHaveBeenCalled()
    expect(result.current.voicesAvailable).toBe(false)
  })

  it("does not mark TTS unavailable when catalog voices prove the selected provider is reachable", async () => {
    state.capabilities = {
      hasAudio: false,
      hasStt: false,
      hasTts: false,
      hasVoiceChat: false
    }
    state.fetchVoiceCatalog.mockResolvedValue([
      { voice_id: "Bella", name: "Bella", provider: "kitten_tts" }
    ])
    state.inferProviderFromModel.mockReturnValue("kitten_tts")

    const { result } = renderHook(
      () =>
        useTldwAudioStatus({
          requireVoices: true,
          tldwTtsModel: "KittenML/kitten-tts-nano-0.8"
        } as any),
      {
        wrapper: buildWrapper()
      }
    )

    await waitFor(() => {
      expect(result.current.voicesLoading).toBe(false)
    })

    expect(result.current.voicesAvailable).toBe(true)
    expect(result.current.ttsHealthState).toBe("unknown")
    expect(state.fetchVoiceCatalog).toHaveBeenCalledWith("kitten_tts")
    expect(state.apiSend).not.toHaveBeenCalled()
  })
})
