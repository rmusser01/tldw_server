import React, { StrictMode, useState } from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useWorkspaceStore } from "@/store/workspace"
import {
  fetchTldwVoiceCatalog,
  type TldwVoice
} from "@/services/tldw/audio-voices"
import { fetchTtsProviders } from "@/services/tldw/audio-providers"
import { fetchTldwTtsModels } from "@/services/tldw/audio-models"
import type { AudioGenerationSettings } from "@/types/workspace"
import {
  useAudioTtsSettings,
  type UseAudioTtsSettingsDeps
} from "../useAudioTtsSettings"

vi.mock("@/store/workspace", async () => {
  const { create } = await import("zustand")
  return {
    useWorkspaceStore: create(() => ({
      activeWorkspaceOrigin: { kind: "legacy-local" }
    }))
  }
})
vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: vi.fn()
}))
vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: vi.fn()
}))
vi.mock("@/services/tldw/audio-models", () => ({ fetchTldwTtsModels: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { synthesizeSpeechDetailed: vi.fn() }
}))

const ownedOrigin = {
  kind: "server-owned" as const,
  scope: {
    serverBase: "https://server.test",
    principalId: "owner-a",
    organizationId: null
  }
}

const initialSettings: AudioGenerationSettings = {
  provider: "tldw",
  model: "kitten_tts",
  voice: "server-only-voice",
  speed: 1.25,
  format: "flac",
  backend: "",
  allowFallback: false
}
const messageApi = {
  error: vi.fn()
} as unknown as UseAudioTtsSettingsDeps["messageApi"]
const t = ((_key: string, fallback: string) =>
  fallback) as UseAudioTtsSettingsDeps["t"]

function mountSettings(settings: AudioGenerationSettings, strict: boolean) {
  const writes = vi.fn()
  const hook = renderHook(
    () => {
      const [audioSettings, updateSettings] = useState(settings)
      const setAudioSettings = React.useCallback(
        (patch: Partial<AudioGenerationSettings>) => {
          writes(patch)
          updateSettings((current) => ({ ...current, ...patch }))
        },
        []
      )
      return {
        audioSettings,
        ...useAudioTtsSettings({
          audioSettings,
          setAudioSettings,
          messageApi,
          t
        })
      }
    },
    { wrapper: strict ? StrictMode : undefined }
  )
  return { ...hook, writes }
}

beforeEach(() => {
  vi.resetAllMocks()
  useWorkspaceStore.setState({ activeWorkspaceOrigin: ownedOrigin })
  vi.mocked(fetchTtsProviders).mockResolvedValue(null)
  vi.mocked(fetchTldwTtsModels).mockResolvedValue([])
  vi.mocked(fetchTldwVoiceCatalog).mockResolvedValue([
    { voice_id: "catalog-voice" }
  ])
})

afterEach(cleanup)

describe.each([false, true])(
  "owned audio automatic-write fence (StrictMode=%s)",
  (strict) => {
    it.each([
      ["available", "server-only-voice"],
      ["available", ""],
      ["empty", "server-only-voice"],
      ["empty", ""],
      ["unavailable", "server-only-voice"],
      ["unavailable", ""]
    ])(
      "preserves canonical settings with %s catalogs and voice %j",
      async (catalog, voice) => {
        if (catalog === "empty")
          vi.mocked(fetchTldwVoiceCatalog).mockResolvedValue([])
        if (catalog === "unavailable") {
          vi.mocked(fetchTldwVoiceCatalog).mockRejectedValue(
            new Error("offline")
          )
          vi.mocked(fetchTtsProviders).mockRejectedValue(new Error("offline"))
          vi.mocked(fetchTldwTtsModels).mockRejectedValue(new Error("offline"))
        }
        const settings = { ...initialSettings, voice }
        const { result, writes, unmount } = mountSettings(settings, strict)

        await waitFor(() => expect(result.current.loadingVoices).toBe(false))
        expect(result.current.getVoiceOptions()[0].value).toBe(
          catalog === "available" ? "catalog-voice" : "Bella"
        )
        expect(result.current.audioSettings).toEqual(settings)
        unmount()
        expect(writes).not.toHaveBeenCalled()
      }
    )

    it("preserves missing voice and an undiscovered model/backend", async () => {
      const settings = {
        provider: "tldw",
        model: "unlisted-model",
        backend: "unlisted-backend",
        speed: 0.75,
        format: "wav",
        allowFallback: false
      } as AudioGenerationSettings
      const { result, writes } = mountSettings(settings, strict)
      await act(async () => {})

      expect(result.current.audioSettings).toEqual(settings)
      expect(writes).not.toHaveBeenCalled()
    })

    it.each(["browser", "openai", "elevenlabs"] as const)(
      "preserves %s provider settings",
      async (provider) => {
        const settings = {
          ...initialSettings,
          provider,
          model: "unlisted-model",
          voice: ""
        }
        const { result, writes } = mountSettings(settings, strict)
        await act(async () => {})

        expect(result.current.audioSettings).toEqual(settings)
        expect(writes).not.toHaveBeenCalled()
      }
    )

    it.each(["legacy-local", "omitted"])(
      "retains normalization for %s origin",
      async (origin) => {
        useWorkspaceStore.setState({
          activeWorkspaceOrigin:
            origin === "omitted" ? undefined : { kind: "legacy-local" }
        })
        const { result, writes } = mountSettings(
          { ...initialSettings, voice: "" },
          strict
        )

        await waitFor(() =>
          expect(result.current.audioSettings.voice).toBe("catalog-voice")
        )
        expect(writes).toHaveBeenCalledWith({ voice: "catalog-voice" })
      }
    )

    it("retains legacy fallback normalization when the catalog is unavailable", async () => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "legacy-local" }
      })
      vi.mocked(fetchTldwVoiceCatalog).mockRejectedValue(new Error("offline"))
      const { result } = mountSettings(initialSettings, strict)

      await waitFor(() =>
        expect(result.current.audioSettings.voice).toBe("Bella")
      )
    })

    it("fences a pending catalog completion after switching from legacy to owned", async () => {
      let resolveCatalog!: (voices: TldwVoice[]) => void
      vi.mocked(fetchTldwVoiceCatalog).mockReturnValue(
        new Promise((resolve) => {
          resolveCatalog = resolve
        })
      )
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "legacy-local" }
      })
      const { result, writes } = mountSettings(initialSettings, strict)

      act(() =>
        useWorkspaceStore.setState({ activeWorkspaceOrigin: ownedOrigin })
      )
      await act(async () => resolveCatalog([{ voice_id: "late-voice" }]))

      expect(result.current.loadingVoices).toBe(false)
      expect(result.current.audioSettings).toEqual(initialSettings)
      expect(writes).not.toHaveBeenCalled()
    })

    it("resumes normalization when switching from owned to legacy", async () => {
      const { result, writes } = mountSettings(initialSettings, strict)
      await waitFor(() => expect(result.current.loadingVoices).toBe(false))
      expect(writes).not.toHaveBeenCalled()

      act(() =>
        useWorkspaceStore.setState({
          activeWorkspaceOrigin: { kind: "legacy-local" }
        })
      )

      await waitFor(() =>
        expect(result.current.audioSettings.voice).toBe("catalog-voice")
      )
    })

    it.each(["server-owned", "legacy-local"])(
      "preserves explicit backend selection in %s mode",
      async (kind) => {
        useWorkspaceStore.setState({
          activeWorkspaceOrigin:
            kind === "server-owned" ? ownedOrigin : { kind: "legacy-local" }
        })
        vi.mocked(fetchTtsProviders).mockResolvedValue({
          supports_explicit_backend: true,
          voices: {},
          providers: {
            "gateway:test": {
              display_name: "Test backend",
              default_model: "selected-model",
              model_capabilities: {
                "selected-model": { default_voice: "selected-voice" }
              }
            }
          }
        })
        const { result, writes } = mountSettings(
          { ...initialSettings, voice: "catalog-voice" },
          strict
        )
        await waitFor(() =>
          expect(result.current.explicitBackendSupported).toBe(true)
        )
        await waitFor(() => expect(result.current.loadingVoices).toBe(false))
        expect(writes).not.toHaveBeenCalled()

        act(() => result.current.handleBackendChange("gateway:test"))

        expect(writes).toHaveBeenCalledWith({
          backend: "gateway:test",
          allowFallback: false,
          model: "selected-model",
          voice: "selected-voice"
        })
        expect(result.current.audioSettings).toEqual({
          ...initialSettings,
          backend: "gateway:test",
          model: "selected-model",
          voice: kind === "server-owned" ? "selected-voice" : "catalog-voice"
        })
        await waitFor(() => expect(result.current.loadingVoices).toBe(false))
        if (kind === "server-owned") expect(writes).toHaveBeenCalledTimes(1)
      }
    )
  }
)
