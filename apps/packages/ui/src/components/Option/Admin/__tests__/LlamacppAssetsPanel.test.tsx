import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { LlamacppAssetsPanel } from "../LlamacppAssetsPanel"
import type { LlamacppAssetsResponse } from "@/types/llamacpp-admin"

const mockAssets: LlamacppAssetsResponse = {
  assets: [
    {
      asset_id: "gguf:base",
      kind: "gguf",
      identity_basis: "resolved_path",
      path: "/models/base-q4_k_m.gguf",
      resolved_path: "/models/base-q4_k_m.gguf",
      display_name: "base-q4_k_m",
      source: "models_dir",
      size_bytes: 4_000_000_000,
      modified_at: null,
      metadata: {
        quantization: "Q4_K_M",
        parameter_hint: "7B",
        context_hint: null,
        family_hint: "base"
      },
      capabilities: ["unknown"],
      mmproj_asset_ids: ["mmproj:vision"],
      base_model_asset_ids: [],
      warnings: ["Projector pairing is inferred."]
    },
    {
      asset_id: "mmproj:vision",
      kind: "mmproj",
      identity_basis: "resolved_path",
      path: "/models/mmproj-base.gguf",
      resolved_path: "/models/mmproj-base.gguf",
      display_name: "mmproj-base",
      source: "models_dir",
      size_bytes: 50_000_000,
      modified_at: null,
      metadata: {},
      capabilities: ["vision_projector"],
      mmproj_asset_ids: [],
      base_model_asset_ids: ["gguf:base"],
      warnings: []
    },
    {
      asset_id: "folder:external",
      kind: "folder",
      identity_basis: "resolved_path",
      path: "/external/models",
      resolved_path: "/external/models",
      display_name: "models",
      source: "imported_folder",
      size_bytes: null,
      modified_at: null,
      metadata: {},
      capabilities: ["asset_folder"],
      mmproj_asset_ids: [],
      base_model_asset_ids: [],
      warnings: ["Folder is outside the configured models directory but allowlisted."]
    }
  ],
  warnings: ["One imported folder could not be read."],
  scan_limited: false
}

describe("LlamacppAssetsPanel", () => {
  it("renders asset groups warnings and inferred projector candidates", () => {
    render(
      <LlamacppAssetsPanel
        assets={mockAssets}
        loading={false}
        registeringPath={false}
        importingFolder={false}
        error={null}
        onRegisterPath={vi.fn()}
        onImportFolder={vi.fn()}
        onReload={vi.fn()}
      />
    )

    expect(screen.getByText("GGUF models")).toBeTruthy()
    expect(screen.getByText("mmproj projectors")).toBeTruthy()
    expect(screen.getByText("Imported folders")).toBeTruthy()
    expect(screen.getByText("One imported folder could not be read.")).toBeTruthy()
    expect(screen.getByText("Projector pairing is inferred.")).toBeTruthy()
    expect(screen.getByText("Projector candidates: mmproj:vision")).toBeTruthy()
    expect(screen.getByText("Base model candidates: gguf:base")).toBeTruthy()
    expect(screen.getByLabelText("Register local asset path")).toBeTruthy()
    expect(screen.getByLabelText("Import local asset folder")).toBeTruthy()
  })

  it("submits register and import actions and clears successful inputs", async () => {
    const onRegisterPath = vi.fn().mockResolvedValue(true)
    const onImportFolder = vi.fn().mockResolvedValue(true)

    render(
      <LlamacppAssetsPanel
        assets={{ assets: [], warnings: [], scan_limited: false }}
        loading={false}
        registeringPath={false}
        importingFolder={false}
        error={null}
        onRegisterPath={onRegisterPath}
        onImportFolder={onImportFolder}
        onReload={vi.fn()}
      />
    )

    const assetInput = screen.getByLabelText("Register local asset path") as HTMLInputElement
    fireEvent.change(assetInput, { target: { value: "/external/model.gguf" } })
    fireEvent.click(screen.getByRole("button", { name: "Register asset" }))

    await waitFor(() => {
      expect(onRegisterPath).toHaveBeenCalledWith("/external/model.gguf")
    })
    expect(assetInput.value).toBe("")

    const folderInput = screen.getByLabelText("Import local asset folder") as HTMLInputElement
    fireEvent.change(folderInput, { target: { value: "/external/models" } })
    fireEvent.click(screen.getByRole("button", { name: "Import folder" }))

    await waitFor(() => {
      expect(onImportFolder).toHaveBeenCalledWith("/external/models")
    })
    expect(folderInput.value).toBe("")
  })

  it("keeps inputs available when actions report failure", async () => {
    const onRegisterPath = vi.fn().mockResolvedValue(false)
    const onImportFolder = vi.fn().mockResolvedValue(false)

    render(
      <LlamacppAssetsPanel
        assets={{ assets: [], warnings: [], scan_limited: false }}
        loading={false}
        registeringPath={false}
        importingFolder={false}
        error={null}
        onRegisterPath={onRegisterPath}
        onImportFolder={onImportFolder}
        onReload={vi.fn()}
      />
    )

    const assetInput = screen.getByLabelText("Register local asset path") as HTMLInputElement
    fireEvent.change(assetInput, { target: { value: "/external/model.gguf" } })
    fireEvent.click(screen.getByRole("button", { name: "Register asset" }))

    await waitFor(() => {
      expect(onRegisterPath).toHaveBeenCalledWith("/external/model.gguf")
    })
    expect(assetInput.value).toBe("/external/model.gguf")

    const folderInput = screen.getByLabelText("Import local asset folder") as HTMLInputElement
    fireEvent.change(folderInput, { target: { value: "/external/models" } })
    fireEvent.click(screen.getByRole("button", { name: "Import folder" }))

    await waitFor(() => {
      expect(onImportFolder).toHaveBeenCalledWith("/external/models")
    })
    expect(folderInput.value).toBe("/external/models")
  })
})
