import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { LlamacppAssetsPanel } from "../LlamacppAssetsPanel"
import type {
  LlamacppAcquisitionJobListResponse,
  LlamacppAssetImportPreviewResponse,
  LlamacppAssetDownloadRequest,
  LlamacppAssetsResponse
} from "@/types/llamacpp-admin"

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

const mockImportPreview: LlamacppAssetImportPreviewResponse = {
  folder: {
    asset_id: "folder:preview",
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
    warnings: []
  },
  assets: [mockAssets.assets[0]!, mockAssets.assets[1]!],
  asset_counts: {
    gguf: 1,
    mmproj: 1
  },
  warnings: ["Preview skipped unreadable sidecar file."],
  scan_limited: false,
  will_persist: false
}

const mockDownloads: LlamacppAcquisitionJobListResponse = {
  jobs: [
    {
      job_id: "42",
      status: "running",
      operation: "download",
      queue: "acquisition",
      source_label: "Toy model",
      destination_path: "/models/toy.gguf",
      asset_id: null,
      progress: {
        progress_percent: 25,
        progress_message: "downloading"
      },
      warnings: ["Checksum will be verified after download."],
      error_message: null
    }
  ]
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

  it("previews local folder imports before confirming persistence", async () => {
    const onPreviewImportFolder = vi.fn().mockResolvedValue(true)
    const onImportFolder = vi.fn().mockResolvedValue(true)

    render(
      <LlamacppAssetsPanel
        assets={{ assets: [], warnings: [], scan_limited: false }}
        loading={false}
        registeringPath={false}
        importingFolder={false}
        previewingFolder={false}
        importPreview={mockImportPreview}
        error={null}
        onRegisterPath={vi.fn()}
        onPreviewImportFolder={onPreviewImportFolder}
        onImportFolder={onImportFolder}
        onReload={vi.fn()}
      />
    )

    const folderInput = screen.getByLabelText("Import local asset folder") as HTMLInputElement
    fireEvent.change(folderInput, { target: { value: "/external/models" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview folder" }))

    await waitFor(() => {
      expect(onPreviewImportFolder).toHaveBeenCalledWith("/external/models")
    })

    expect(screen.getByText("Import preview")).toBeTruthy()
    expect(screen.getByText("GGUF: 1")).toBeTruthy()
    expect(screen.getByText("mmproj: 1")).toBeTruthy()
    expect(screen.getByText("Preview skipped unreadable sidecar file.")).toBeTruthy()
    expect(onImportFolder).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(onImportFolder).toHaveBeenCalledWith("/external/models")
    })
  })

  it("clears stale import previews when the folder path changes", async () => {
    const onClearImportPreview = vi.fn()
    const onPreviewImportFolder = vi.fn().mockResolvedValue(true)
    const onImportFolder = vi.fn().mockResolvedValue(true)

    render(
      <LlamacppAssetsPanel
        assets={{ assets: [], warnings: [], scan_limited: false }}
        loading={false}
        registeringPath={false}
        importingFolder={false}
        previewingFolder={false}
        importPreview={mockImportPreview}
        error={null}
        onRegisterPath={vi.fn()}
        onPreviewImportFolder={onPreviewImportFolder}
        onClearImportPreview={onClearImportPreview}
        onImportFolder={onImportFolder}
        onReload={vi.fn()}
      />
    )

    fireEvent.change(screen.getByLabelText("Import local asset folder"), {
      target: { value: "/external/other-models" }
    })

    expect(onClearImportPreview).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("Import preview")).toBeNull()
    expect(screen.queryByRole("button", { name: "Confirm import" })).toBeNull()
    expect(onImportFolder).not.toHaveBeenCalled()
  })

  it("renders duplicate warning strings without React key collisions", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)

    try {
      render(
        <LlamacppAssetsPanel
          assets={{
            ...mockAssets,
            warnings: ["Repeated warning.", "Repeated warning."],
            assets: [
              {
                ...mockAssets.assets[0]!,
                warnings: ["Repeated asset warning.", "Repeated asset warning."]
              }
            ]
          }}
          loading={false}
          registeringPath={false}
          importingFolder={false}
          previewingFolder={false}
          importPreview={{
            ...mockImportPreview,
            warnings: ["Repeated preview warning.", "Repeated preview warning."]
          }}
          downloads={{
            jobs: [
              {
                ...mockDownloads.jobs[0]!,
                warnings: ["Repeated job warning.", "Repeated job warning."]
              }
            ]
          }}
          error={null}
          onRegisterPath={vi.fn()}
          onPreviewImportFolder={vi.fn()}
          onImportFolder={vi.fn()}
          onStartDownload={vi.fn()}
          onReload={vi.fn()}
        />
      )

      const duplicateKeyWarnings = consoleError.mock.calls.filter((call) =>
        String(call[0]).includes("Encountered two children with the same key")
      )
      expect(duplicateKeyWarnings).toHaveLength(0)
    } finally {
      consoleError.mockRestore()
    }
  })

  it("queues downloads and renders cancellable acquisition status", async () => {
    const onStartDownload =
      vi.fn<(payload: LlamacppAssetDownloadRequest) => Promise<boolean>>()
        .mockResolvedValue(true)
    const onCancelDownload = vi.fn().mockResolvedValue(true)

    render(
      <LlamacppAssetsPanel
        assets={{ assets: [], warnings: [], scan_limited: false }}
        loading={false}
        registeringPath={false}
        importingFolder={false}
        error={null}
        downloads={mockDownloads}
        startingDownload={false}
        cancelingDownloadId={null}
        onRegisterPath={vi.fn()}
        onImportFolder={vi.fn()}
        onStartDownload={onStartDownload}
        onCancelDownload={onCancelDownload}
        onReload={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: "Queue download" })).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Download source URL"), {
      target: { value: "https://example.com/toy.gguf" }
    })
    fireEvent.change(screen.getByLabelText("Download destination directory"), {
      target: { value: "/models" }
    })
    fireEvent.change(screen.getByLabelText("Download filename"), {
      target: { value: "toy.gguf" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Queue download" }))

    await waitFor(() => {
      expect(onStartDownload).toHaveBeenCalledWith({
        url: "https://example.com/toy.gguf",
        destination_dir: "/models",
        filename: "toy.gguf"
      } satisfies Partial<LlamacppAssetDownloadRequest>)
    })

    expect(screen.getByText("Toy model")).toBeTruthy()
    expect(screen.getByText("running")).toBeTruthy()
    expect(screen.getByText("25%")).toBeTruthy()
    expect(screen.getByText("Checksum will be verified after download.")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Cancel download 42" }))

    await waitFor(() => {
      expect(onCancelDownload).toHaveBeenCalledWith("42")
    })
  })
})
