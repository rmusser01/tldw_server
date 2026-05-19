// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { AudioPreset } from "@/types/audio-presets"

const makePreset = (overrides: Partial<AudioPreset> = {}): AudioPreset => ({
  id: "preset-1",
  owner_user_id: "user-1",
  kind: "tts",
  name: "Warm narrator",
  favorite: false,
  is_default: false,
  config: { provider: "tldw", model: "kokoro", voice: "af_heart" },
  capability_assumptions: { provider: "tldw" },
  created_at: "2026-05-19T00:00:00Z",
  updated_at: "2026-05-19T00:00:00Z",
  ...overrides
})

const {
  hookState,
  createPresetMock,
  updatePresetMock,
  deletePresetMock,
  validatePresetMock,
  notificationMock
} = vi.hoisted(() => ({
  hookState: {
    current: {
      presets: [] as AudioPreset[],
      loading: false,
      createPreset: vi.fn(),
      updatePreset: vi.fn(),
      deletePreset: vi.fn(),
      validatePreset: vi.fn(),
      creating: false,
      updating: false,
      deleting: false,
      validating: false
    }
  },
  createPresetMock: vi.fn(),
  updatePresetMock: vi.fn(),
  deletePresetMock: vi.fn(),
  validatePresetMock: vi.fn(),
  notificationMock: {
    success: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    error: vi.fn()
  }
}))

vi.mock("@/hooks/useAudioPresets", () => ({
  useAudioPresets: () => hookState.current
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

import { AudioPresetControls } from "../AudioPresetControls"

describe("AudioPresetControls", () => {
  beforeEach(() => {
    createPresetMock.mockReset()
    updatePresetMock.mockReset()
    deletePresetMock.mockReset()
    validatePresetMock.mockReset()
    notificationMock.success.mockReset()
    notificationMock.warning.mockReset()
    notificationMock.info.mockReset()
    notificationMock.error.mockReset()

    const preset = makePreset()
    createPresetMock.mockResolvedValue(makePreset({ id: "preset-2", name: "Saved preset" }))
    updatePresetMock.mockImplementation(async (_id: string, patch: Partial<AudioPreset>) => ({
      ...preset,
      ...patch
    }))
    deletePresetMock.mockResolvedValue(undefined)
    validatePresetMock.mockResolvedValue({
      preset,
      valid: true,
      warnings: []
    })

    hookState.current = {
      presets: [],
      loading: false,
      createPreset: createPresetMock,
      updatePreset: updatePresetMock,
      deletePreset: deletePresetMock,
      validatePreset: validatePresetMock,
      creating: false,
      updating: false,
      deleting: false,
      validating: false
    }
  })

  it("saves the current settings with capability assumptions", async () => {
    render(
      <AudioPresetControls
        kind="tts"
        currentConfig={{ provider: "tldw", model: "kokoro" }}
        capabilityAssumptions={{ provider: "tldw", model: "ready" }}
        onApply={vi.fn()}
      />
    )

    fireEvent.change(screen.getByPlaceholderText("Preset name"), {
      target: { value: "Warm narrator" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save current settings" }))

    await waitFor(() => {
      expect(createPresetMock).toHaveBeenCalledWith({
        kind: "tts",
        name: "Warm narrator",
        config: { provider: "tldw", model: "kokoro" },
        capability_assumptions: { provider: "tldw", model: "ready" },
        is_default: true
      })
    })
    expect(notificationMock.success).toHaveBeenCalledWith({ message: "Preset saved" })
  })

  it("validates before applying the selected preset", async () => {
    const preset = makePreset()
    hookState.current.presets = [preset]
    validatePresetMock.mockResolvedValue({
      preset,
      valid: false,
      warnings: [{ code: "browser_revalidation", message: "Re-select browser voice." }]
    })
    const onApply = vi.fn()

    render(
      <AudioPresetControls
        kind="tts"
        currentConfig={{ provider: "browser" }}
        onApply={onApply}
      />
    )

    await screen.findByDisplayValue("Warm narrator")
    fireEvent.click(screen.getByRole("button", { name: "Apply preset" }))

    await waitFor(() => {
      expect(validatePresetMock).toHaveBeenCalledWith("preset-1")
      expect(onApply).toHaveBeenCalledWith(preset.config, preset)
    })
    expect(notificationMock.warning).toHaveBeenCalledWith({
      message: "Preset needs attention",
      description: "Re-select browser voice."
    })
  })

  it("renames, favorites, defaults, duplicates, and deletes the selected preset", async () => {
    const preset = makePreset()
    hookState.current.presets = [preset]

    render(
      <AudioPresetControls
        kind="tts"
        currentConfig={{ provider: "tldw" }}
        onApply={vi.fn()}
      />
    )

    await screen.findByDisplayValue("Warm narrator")
    fireEvent.change(screen.getByPlaceholderText("Preset name"), {
      target: { value: "Sharper narrator" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Rename selected preset" }))
    fireEvent.click(screen.getByRole("button", { name: "Favorite" }))
    fireEvent.click(screen.getByRole("button", { name: "Set as default" }))
    fireEvent.click(screen.getByRole("button", { name: "Duplicate selected preset" }))
    fireEvent.click(screen.getByRole("button", { name: "Delete selected preset" }))

    await waitFor(() => {
      expect(updatePresetMock).toHaveBeenCalledWith("preset-1", {
        name: "Sharper narrator"
      })
      expect(updatePresetMock).toHaveBeenCalledWith("preset-1", {
        favorite: true
      })
      expect(updatePresetMock).toHaveBeenCalledWith("preset-1", {
        is_default: true
      })
      expect(createPresetMock).toHaveBeenCalledWith({
        kind: "tts",
        name: "Warm narrator copy",
        description: preset.description,
        favorite: false,
        config: preset.config,
        capability_assumptions: preset.capability_assumptions
      })
      expect(deletePresetMock).toHaveBeenCalledWith("preset-1")
    })
  })
})
