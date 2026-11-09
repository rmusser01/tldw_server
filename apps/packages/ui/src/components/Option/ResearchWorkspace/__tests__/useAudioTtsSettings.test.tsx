import { useState } from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { UseAudioTtsSettingsDeps } from "../StudioPane/hooks/useAudioTtsSettings"
import { useAudioTtsSettings } from "../StudioPane/hooks/useAudioTtsSettings"

const mocks = vi.hoisted(() => ({
  providers: vi.fn(),
  catalog: vi.fn(),
  custom: vi.fn(),
  accountChanged: null as null | (() => void)
}))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (listener: () => void) => {
    mocks.accountChanged = listener
    return () => {
      mocks.accountChanged = null
    }
  }
}))
vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: (...args: unknown[]) => mocks.catalog(...args)
}))
vi.mock("@/services/tldw/voice-cloning", () => ({
  listCustomVoices: () => mocks.custom()
}))
vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: () => mocks.providers()
}))
vi.mock("@/services/tldw/audio-models", () => ({
  fetchTldwTtsModels: async () => []
}))
vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: (model: string) =>
    model === "kitten_tts"
      ? "kitten_tts"
      : model === "kokoro"
        ? "kokoro"
        : "pockettts"
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))

const voice = (voice_id: string, provider = "pocket_tts") => ({
  voice_id,
  provider,
  name: voice_id
})
const deps = (
  patch: Partial<UseAudioTtsSettingsDeps["audioSettings"]> = {}
): UseAudioTtsSettingsDeps => ({
  audioSettings: {
    provider: "tldw",
    model: "pocket_tts",
    voice: "",
    speed: 1,
    format: "wav",
    ...patch
  },
  setAudioSettings: vi.fn(),
  messageApi: {} as UseAudioTtsSettingsDeps["messageApi"],
  t: ((key: string) => key) as UseAudioTtsSettingsDeps["t"]
})

describe("Studio owned custom TTS voices", () => {
  beforeEach(() => {
    mocks.catalog
      .mockReset()
      .mockResolvedValue([{ id: "built-in", name: "Built in" }])
    mocks.custom
      .mockReset()
      .mockResolvedValue([
        voice("own"),
        voice("foreign-provider", "kitten_tts")
      ])
    mocks.accountChanged = null
    mocks.providers.mockReset().mockResolvedValue({
      supports_explicit_backend: true,
      providers: { "gateway:Exact": { models: ["vendor/model"] } }
    })
  })

  it("offers the owned compatible reference with the custom selector while retaining built-ins", async () => {
    const { result } = renderHook(() => useAudioTtsSettings(deps()))
    await waitFor(() =>
      expect(result.current.getVoiceOptions()).toContainEqual({
        value: "custom:own",
        label: "Custom: own"
      })
    )
    expect(result.current.getVoiceOptions()).toContainEqual({
      value: "built-in",
      label: "Built in"
    })
    expect(
      result.current
        .getVoiceOptions()
        .some((row) => row.value.includes("foreign-provider"))
    ).toBe(false)
  })

  it("normalizes provider aliases and filters again when the model changes", async () => {
    const { result, rerender } = renderHook(
      ({ model }) => useAudioTtsSettings(deps({ model })),
      { initialProps: { model: "pocket_tts" } }
    )
    await waitFor(() =>
      expect(
        result.current
          .getVoiceOptions()
          .some((row) => row.value === "custom:own")
      ).toBe(true)
    )
    rerender({ model: "kitten_tts" })
    await waitFor(() =>
      expect(
        result.current
          .getVoiceOptions()
          .some((row) => row.value === "custom:foreign-provider")
      ).toBe(true)
    )
    expect(
      result.current.getVoiceOptions().some((row) => row.value === "custom:own")
    ).toBe(false)
  })

  it("keeps custom references out of an explicit gateway catalog", async () => {
    const { result } = renderHook(() =>
      useAudioTtsSettings(
        deps({ backend: "gateway:Exact", model: "vendor/model" })
      )
    )
    await waitFor(() =>
      expect(result.current.explicitBackendSupported).toBe(true)
    )
    expect(result.current.getVoiceOptions()).toEqual([
      { value: "built-in", label: "Built in" }
    ])
    expect(mocks.custom).not.toHaveBeenCalled()
  })

  it("retains custom voices when the built-in catalog is unavailable", async () => {
    mocks.catalog.mockRejectedValue(new Error("unavailable"))
    const { result } = renderHook(() => useAudioTtsSettings(deps()))
    await waitFor(() =>
      expect(result.current.getVoiceOptions()).toContainEqual({
        value: "custom:own",
        label: "Custom: own"
      })
    )
  })

  it("retires an old account response and refreshes the new account without exposing the old reference", async () => {
    let finish!: (value: unknown) => void
    mocks.custom
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve
          })
      )
      .mockResolvedValue([voice("new-owner")])
    const { result } = renderHook(() => useAudioTtsSettings(deps()))
    await waitFor(() => expect(mocks.custom).toHaveBeenCalledTimes(1))
    act(() => {
      mocks.accountChanged!()
    })
    await waitFor(() =>
      expect(result.current.getVoiceOptions()).toContainEqual({
        value: "custom:new-owner",
        label: "Custom: new-owner"
      })
    )
    await act(async () => {
      finish([voice("old-owner")])
    })
    expect(
      result.current
        .getVoiceOptions()
        .some((row) => row.value === "custom:old-owner")
    ).toBe(false)
  })

  it("ignores custom-voice failures and still supplies the built-in catalog", async () => {
    mocks.custom.mockRejectedValue(new Error("unavailable"))
    const { result } = renderHook(() => useAudioTtsSettings(deps()))
    await waitFor(() =>
      expect(result.current.getVoiceOptions()).toEqual([
        { value: "built-in", label: "Built in" }
      ])
    )
  })
  it("does not publish a held custom reference after switching to an explicit backend", async () => {
    let finish!: (value: unknown) => void
    mocks.custom.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const { result, rerender } = renderHook(
      ({ backend }) => useAudioTtsSettings(deps({ backend })),
      { initialProps: { backend: "" } }
    )
    await waitFor(() => expect(mocks.custom).toHaveBeenCalledTimes(1))
    rerender({ backend: "gateway:Exact" })
    await act(async () => {
      finish([voice("retired-reference")])
    })
    expect(
      result.current
        .getVoiceOptions()
        .some((row) => row.value.startsWith("custom:"))
    ).toBe(false)
  })

  it("does not let the retired account built-in response replace the refreshed catalog", async () => {
    let finish!: (value: unknown) => void
    mocks.catalog
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve
          })
      )
      .mockResolvedValue([{ id: "new-built-in", name: "New built in" }])
    const { result } = renderHook(() => useAudioTtsSettings(deps()))
    await waitFor(() => expect(mocks.catalog).toHaveBeenCalledTimes(1))
    act(() => {
      mocks.accountChanged!()
    })
    await waitFor(() =>
      expect(
        result.current
          .getVoiceOptions()
          .some((row) => row.value === "new-built-in")
      ).toBe(true)
    )
    await act(async () => {
      finish([{ id: "old-built-in", name: "Old built in" }])
    })
    expect(
      result.current
        .getVoiceOptions()
        .some((row) => row.value === "old-built-in")
    ).toBe(false)
  })
  it("preserves a saved custom choice while its owned catalog is still loading", async () => {
    let finish!: (value: unknown) => void
    mocks.custom.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const { result } = renderHook(() => {
      const [audioSettings, update] = useState(
        deps({ voice: "custom:own" }).audioSettings
      )
      return {
        audioSettings,
        hook: useAudioTtsSettings({
          ...deps(),
          audioSettings,
          setAudioSettings: (patch) =>
            update((current) => ({ ...current, ...patch }))
        })
      }
    })
    await waitFor(() => expect(mocks.custom).toHaveBeenCalledTimes(1))
    await act(async () => {})
    expect(result.current.audioSettings.voice).toBe("custom:own")
    await act(async () => {
      finish([voice("own")])
    })
    await waitFor(() =>
      expect(
        result.current.hook
          .getVoiceOptions()
          .some((row) => row.value === "custom:own")
      ).toBe(true)
    )
    expect(result.current.audioSettings.voice).toBe("custom:own")
  })
  it("keeps existing provider fallback choices alongside compatible uploaded voices", async () => {
    mocks.catalog.mockResolvedValue([])
    const { result } = renderHook(() =>
      useAudioTtsSettings(deps({ model: "kitten_tts" }))
    )
    await waitFor(() =>
      expect(result.current.getVoiceOptions()).toContainEqual({
        value: "custom:foreign-provider",
        label: "Custom: foreign-provider"
      })
    )
    expect(result.current.getVoiceOptions()).toContainEqual({
      value: "Bella",
      label: "Bella"
    })
  })
  it("preserves the saved explicit-backend voice until backend discovery settles", async () => {
    let finish!: (value: unknown) => void
    let finishCatalog!: (value: unknown) => void
    mocks.catalog
      .mockResolvedValueOnce([{ id: "old-inferred", name: "Old inferred" }])
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finishCatalog = resolve
          })
      )
    mocks.providers.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    const { result } = renderHook(() => {
      const [audioSettings, update] = useState(
        deps({
          backend: "gateway:Exact",
          model: "vendor/model",
          voice: "saved-gateway-voice"
        }).audioSettings
      )
      return {
        audioSettings,
        hook: useAudioTtsSettings({
          ...deps(),
          audioSettings,
          setAudioSettings: (patch) =>
            update((current) => ({ ...current, ...patch }))
        })
      }
    })
    await waitFor(() => expect(mocks.catalog).toHaveBeenCalledTimes(1))
    await act(async () => {})
    expect(result.current.audioSettings.voice).toBe("saved-gateway-voice")
    await act(async () => {
      finish({
        supports_explicit_backend: true,
        providers: { "gateway:Exact": { models: ["vendor/model"] } }
      })
    })
    await waitFor(() =>
      expect(result.current.hook.explicitBackendSupported).toBe(true)
    )
    await waitFor(() => expect(mocks.catalog).toHaveBeenCalledTimes(2))
    expect(result.current.audioSettings.voice).toBe("saved-gateway-voice")
    await act(async () => {
      finishCatalog([
        { id: "saved-gateway-voice", name: "Saved gateway voice" }
      ])
    })
    await waitFor(() =>
      expect(
        result.current.hook
          .getVoiceOptions()
          .some((row) => row.value === "saved-gateway-voice")
      ).toBe(true)
    )
    expect(result.current.audioSettings.voice).toBe("saved-gateway-voice")
    expect(mocks.custom).not.toHaveBeenCalled()
  })
  it("does not keep a Kokoro-only voice selected for Pocket when an owned reference is available", async () => {
    mocks.catalog.mockResolvedValue([])
    const { result } = renderHook(() => {
      const [audioSettings, update] = useState(
        deps({ voice: "af_heart" }).audioSettings
      )
      return {
        audioSettings,
        hook: useAudioTtsSettings({
          ...deps(),
          audioSettings,
          setAudioSettings: (patch) =>
            update((current) => ({ ...current, ...patch }))
        })
      }
    })
    await waitFor(() =>
      expect(result.current.audioSettings.voice).toBe("custom:own")
    )
    expect(
      result.current.hook
        .getVoiceOptions()
        .some((row) => row.value === "af_heart")
    ).toBe(false)
  })

  it("retains actual Kokoro fallback voices alongside its compatible custom references", async () => {
    mocks.catalog.mockResolvedValue([])
    mocks.custom.mockResolvedValue([voice("own-kokoro", "kokoro")])
    const { result } = renderHook(() =>
      useAudioTtsSettings(deps({ model: "kokoro" }))
    )
    await waitFor(() =>
      expect(result.current.getVoiceOptions()).toContainEqual({
        value: "custom:own-kokoro",
        label: "Custom: own-kokoro"
      })
    )
    expect(result.current.getVoiceOptions()).toContainEqual({
      value: "af_heart",
      label: "Heart (Female)"
    })
  })
})
