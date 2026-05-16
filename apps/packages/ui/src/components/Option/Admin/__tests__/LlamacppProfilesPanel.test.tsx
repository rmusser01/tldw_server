import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { LlamacppProfilesPanel } from "../LlamacppProfilesPanel"
import type {
  LlamacppAsset,
  LlamacppProfile
} from "@/types/llamacpp-admin"

const ggufAsset: LlamacppAsset = {
  asset_id: "gguf:toy",
  kind: "gguf",
  identity_basis: "resolved_path",
  path: "/models/toy.gguf",
  resolved_path: "/models/toy.gguf",
  display_name: "Toy 7B",
  source: "models_dir",
  size_bytes: 4_200_000_000,
  modified_at: null,
  metadata: {},
  capabilities: ["unknown"],
  mmproj_asset_ids: ["mmproj:toy"],
  base_model_asset_ids: [],
  warnings: []
}

const mmprojAsset: LlamacppAsset = {
  ...ggufAsset,
  asset_id: "mmproj:toy",
  kind: "mmproj",
  path: "/models/mmproj-toy.gguf",
  resolved_path: "/models/mmproj-toy.gguf",
  display_name: "Toy projector",
  size_bytes: 50_000_000,
  capabilities: ["vision_projector"],
  mmproj_asset_ids: [],
  base_model_asset_ids: ["gguf:toy"]
}

const profiles: LlamacppProfile[] = [
  {
    profile_id: "default",
    name: "Default runtime",
    enabled: true,
    mode: "chat",
    model_id: "gguf:toy",
    model_path: "/models/toy.gguf",
    host: "127.0.0.1",
    port: 8181,
    port_policy: "explicit",
    server_args: {},
    autostart: false,
    restart_policy: {},
    provider_alias: null,
    tags: ["default"]
  },
  {
    profile_id: "analysis",
    name: "Analysis runtime",
    enabled: true,
    mode: "chat",
    model_id: "gguf:analysis",
    model_path: "/models/analysis.gguf",
    host: "127.0.0.1",
    port: 8182,
    port_policy: "explicit",
    server_args: { ctx_size: 8192 },
    autostart: true,
    restart_policy: {},
    provider_alias: "analysis-local",
    tags: ["analysis"]
  }
]

const renderPanel = (overrides = {}) => {
  const onCreate = vi.fn().mockResolvedValue(true)
  const onUpdate = vi.fn().mockResolvedValue(true)
  const onDelete = vi.fn().mockResolvedValue(true)

  render(
    <LlamacppProfilesPanel
      profiles={profiles}
      assets={{ assets: [ggufAsset, mmprojAsset], warnings: [], scan_limited: false }}
      loading={false}
      savingProfileId={null}
      error={null}
      onRefresh={vi.fn()}
      onCreate={onCreate}
      onUpdate={onUpdate}
      onDelete={onDelete}
      {...overrides}
    />
  )

  return { onCreate, onUpdate, onDelete }
}

describe("LlamacppProfilesPanel", () => {
  it("creates a saved launch profile from local assets", async () => {
    const { onCreate } = renderPanel()

    fireEvent.click(screen.getByRole("button", { name: "New profile" }))
    fireEvent.change(screen.getByLabelText("Profile name"), {
      target: { value: "Vision runtime" }
    })
    fireEvent.change(screen.getByLabelText("Profile port"), {
      target: { value: "8190" }
    })
    fireEvent.change(screen.getByLabelText("Profile tags"), {
      target: { value: "vision, local" }
    })
    fireEvent.change(screen.getByLabelText("Profile server args JSON"), {
      target: { value: '{ "ctx_size": 4096 }' }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    await waitFor(() => {
      expect(onCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Vision runtime",
          mode: "chat",
          model_id: "gguf:toy",
          host: "127.0.0.1",
          port: 8190,
          port_policy: "explicit",
          enabled: true,
          autostart: false,
          server_args: { ctx_size: 4096 },
          tags: ["vision", "local"]
        })
      )
    })
  })

  it("updates an existing profile without changing runtime state", async () => {
    const { onUpdate } = renderPanel()

    fireEvent.click(screen.getByRole("button", { name: "Edit Analysis runtime" }))
    fireEvent.change(screen.getByLabelText("Profile name"), {
      target: { value: "Analysis edited" }
    })
    fireEvent.change(screen.getByLabelText("Profile server args JSON"), {
      target: { value: '{ "ctx_size": 16384 }' }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    await waitFor(() => {
      expect(onUpdate).toHaveBeenCalledWith(
        "analysis",
        expect.objectContaining({
          name: "Analysis edited",
          server_args: { ctx_size: 16384 }
        })
      )
    })
  })

  it("duplicates an existing profile as a create request", async () => {
    const { onCreate, onUpdate } = renderPanel()

    fireEvent.click(screen.getByRole("button", { name: "Duplicate Default runtime" }))
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    await waitFor(() => {
      expect(onCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Default runtime copy",
          model_id: "gguf:toy",
          port: 8181,
          tags: ["default"]
        })
      )
    })
    expect(onUpdate).not.toHaveBeenCalled()
  })

  it("confirms before deleting a profile", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    const { onDelete } = renderPanel()

    fireEvent.click(screen.getByRole("button", { name: "Delete Analysis runtime" }))

    await waitFor(() => {
      expect(onDelete).toHaveBeenCalledWith("analysis")
    })
    expect(confirmSpy).toHaveBeenCalled()

    confirmSpy.mockRestore()
  })

  it("keeps the form open when server args JSON is invalid", async () => {
    const { onCreate } = renderPanel()

    fireEvent.click(screen.getByRole("button", { name: "New profile" }))
    fireEvent.change(screen.getByLabelText("Profile name"), {
      target: { value: "Broken args" }
    })
    fireEvent.change(screen.getByLabelText("Profile server args JSON"), {
      target: { value: "{" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    expect(await screen.findByText("Invalid server args JSON.")).toBeTruthy()
    expect(onCreate).not.toHaveBeenCalled()
  })

  it("blocks saving when no model asset or model path is available", async () => {
    const { onCreate } = renderPanel({
      assets: { assets: [], warnings: [], scan_limited: false }
    })

    fireEvent.click(screen.getByRole("button", { name: "New profile" }))
    fireEvent.change(screen.getByLabelText("Profile name"), {
      target: { value: "No model" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    expect(await screen.findByText("Model asset or model path is required.")).toBeTruthy()
    expect(onCreate).not.toHaveBeenCalled()
  })

  it("blocks conflicting mmproj asset and server args projector settings", async () => {
    const { onUpdate } = renderPanel({
      profiles: [
        {
          ...profiles[0],
          profile_id: "vision",
          name: "Vision runtime",
          mmproj_model_id: "mmproj:toy"
        }
      ]
    })

    fireEvent.click(screen.getByRole("button", { name: "Edit Vision runtime" }))
    fireEvent.change(screen.getByLabelText("Profile server args JSON"), {
      target: { value: '{ "mmproj": "/models/other-projector.gguf" }' }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    expect(
      await screen.findByText("mmproj asset conflicts with server args mmproj path.")
    ).toBeTruthy()
    expect(onUpdate).not.toHaveBeenCalled()
  })

  it("surfaces unserializable saved server args instead of silently replacing them", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const circularArgs: Record<string, unknown> = {}
    circularArgs.self = circularArgs
    const { onUpdate } = renderPanel({
      profiles: [
        {
          ...profiles[0],
          profile_id: "circular",
          name: "Circular args",
          server_args: circularArgs
        }
      ]
    })

    fireEvent.click(screen.getByRole("button", { name: "Edit Circular args" }))

    expect(
      await screen.findByText("Saved server args could not be displayed. Re-enter server args before saving.")
    ).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Save profile" }))

    expect(onUpdate).not.toHaveBeenCalled()
    expect(warnSpy).toHaveBeenCalledWith(
      "[LlamacppProfilesPanel] Failed to serialize saved server_args",
      expect.any(TypeError)
    )
    warnSpy.mockRestore()
  })
})
