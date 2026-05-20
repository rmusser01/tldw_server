// @vitest-environment jsdom

import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useAudioPresets } from "../useAudioPresets"

const {
  listAudioPresetsMock,
  createAudioPresetMock,
  updateAudioPresetMock,
  deleteAudioPresetMock,
  validateAudioPresetMock
} = vi.hoisted(() => ({
  listAudioPresetsMock: vi.fn(),
  createAudioPresetMock: vi.fn(),
  updateAudioPresetMock: vi.fn(),
  deleteAudioPresetMock: vi.fn(),
  validateAudioPresetMock: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listAudioPresets: listAudioPresetsMock,
    createAudioPreset: createAudioPresetMock,
    updateAudioPreset: updateAudioPresetMock,
    deleteAudioPreset: deleteAudioPresetMock,
    validateAudioPreset: validateAudioPresetMock
  }
}))

const preset = {
  id: "preset-1",
  owner_user_id: "1",
  kind: "tts" as const,
  name: "OpenAI Alloy",
  favorite: false,
  is_default: true,
  config: { provider: "openai", model: "tts-1", voice: "alloy" },
  capability_assumptions: {},
  created_at: "2026-05-19T00:00:00Z",
  updated_at: "2026-05-19T00:00:00Z"
}

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useAudioPresets", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    listAudioPresetsMock.mockResolvedValue({
      items: [preset],
      total: 1,
      limit: 100,
      offset: 0
    })
    createAudioPresetMock.mockResolvedValue(preset)
    updateAudioPresetMock.mockResolvedValue({ ...preset, name: "Updated" })
    deleteAudioPresetMock.mockResolvedValue(undefined)
    validateAudioPresetMock.mockResolvedValue({
      preset,
      valid: true,
      warnings: []
    })
  })

  it("loads presets by kind through the shared tldw client", async () => {
    const { result } = renderHook(() => useAudioPresets({ kind: "tts" }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(listAudioPresetsMock).toHaveBeenCalledWith({ kind: "tts" })
    expect(result.current.presets).toEqual([preset])
    expect(result.current.total).toBe(1)
  })

  it("invalidates preset lists after create, update, and delete mutations", async () => {
    const { result } = renderHook(() => useAudioPresets({ kind: "stt" }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    await act(async () => {
      await result.current.createPreset({
        kind: "stt",
        name: "Whisper English",
        config: { model: "whisper-small", language: "en" }
      })
    })
    await waitFor(() => {
      expect(listAudioPresetsMock).toHaveBeenCalledTimes(2)
    })

    await act(async () => {
      await result.current.updatePreset("preset-1", { name: "Updated" })
    })
    await waitFor(() => {
      expect(listAudioPresetsMock).toHaveBeenCalledTimes(3)
    })

    await act(async () => {
      await result.current.deletePreset("preset-1")
    })
    await waitFor(() => {
      expect(listAudioPresetsMock).toHaveBeenCalledTimes(4)
    })

    expect(createAudioPresetMock).toHaveBeenCalledWith({
      kind: "stt",
      name: "Whisper English",
      config: { model: "whisper-small", language: "en" }
    })
    expect(updateAudioPresetMock).toHaveBeenCalledWith("preset-1", {
      name: "Updated"
    })
    expect(deleteAudioPresetMock).toHaveBeenCalledWith("preset-1")
  })

  it("validates presets without invalidating the list", async () => {
    const { result } = renderHook(() => useAudioPresets({ kind: "tts" }), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    await act(async () => {
      const validation = await result.current.validatePreset("preset-1")
      expect(validation.valid).toBe(true)
      expect(validation.warnings).toEqual([])
    })

    expect(validateAudioPresetMock).toHaveBeenCalledWith("preset-1")
    expect(listAudioPresetsMock).toHaveBeenCalledTimes(1)
  })
})
