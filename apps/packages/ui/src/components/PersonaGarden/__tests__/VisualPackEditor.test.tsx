import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn(),
  translate: vi.fn(
    (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (...args: Parameters<typeof mocks.translate>) => mocks.translate(...args)
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) =>
      (mocks.fetchWithAuth as (...args: unknown[]) => unknown)(...args)
  }
}))

import { VisualPackEditor } from "../VisualPackEditor"

const okResponse = (payload: unknown) =>
  Promise.resolve({
    ok: true,
    json: async () => payload
  })

const okBinaryResponse = (payload: ArrayBuffer) =>
  Promise.resolve({
    ok: true,
    data: payload,
    json: async () => payload
  })

const parseJsonBody = (body: unknown): any => {
  if (typeof body === "string") return JSON.parse(body)
  return body
}

const baseManifest = {
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {
    idle: { animation_id: "idle" },
    listening: { animation_id: "listening" },
    thinking: { animation_id: "thinking" },
    speaking: { animation_id: "speaking" },
    error: { animation_id: "error" }
  },
  animations: {
    idle: {
      frames: [{ asset_id: "asset-a" }, { asset_id: "asset-b" }],
      frame_rate: 1,
      loop: true,
      alignment: { x: 0.5, y: 1 },
      preview_frame: 0
    },
    listening: { frames: [{ asset_id: "asset-a" }], frame_rate: 8, loop: true },
    thinking: { frames: [{ asset_id: "asset-a" }], frame_rate: 8, loop: true },
    speaking: { frames: [{ asset_id: "asset-a" }], frame_rate: 12, loop: true },
    error: { frames: [{ asset_id: "asset-b" }], frame_rate: 1, loop: false }
  },
  fallbacks: { tool_running: ["thinking", "idle"] },
  authored_triggers: []
}

const visualAssets = [
  {
    id: "asset-a",
    pack_id: "pack-1",
    persona_id: "persona-1",
    asset_role: "frame",
    url: "/asset-a.png",
    original_filename: "idle-a.png",
    mime_type: "image/png",
    byte_size: 100,
    checksum_sha256: "sha-a",
    width: 128,
    height: 128
  },
  {
    id: "asset-b",
    pack_id: "pack-1",
    persona_id: "persona-1",
    asset_role: "frame",
    url: "/asset-b.png",
    original_filename: "idle-b.png",
    mime_type: "image/png",
    byte_size: 120,
    checksum_sha256: "sha-b",
    width: 128,
    height: 128
  }
]

describe("VisualPackEditor", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    mocks.fetchWithAuth.mockReset()
    mocks.translate.mockReset()
    mocks.translate.mockImplementation(
      (
        key: string,
        defaultValueOrOptions?:
          | string
          | {
              defaultValue?: string
            }
      ) => {
        if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
        if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
        return key
      }
    )
  })

  it("localizes loading and refresh labels while candidates are loading", async () => {
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    let resolveCandidates: (value: unknown) => void = () => undefined

    mocks.translate.mockImplementation(
      (
        key: string,
        defaultValueOrOptions?:
          | string
          | {
              defaultValue?: string
            }
      ) => {
        if (key === "common:loading") return "Localized loading"
        if (key === "common:refresh") return "Localized refresh"
        if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
        if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
        return key
      }
    )
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return new Promise((resolve) => {
          resolveCandidates = resolve
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    expect(await screen.findByTestId("persona-visual-pack-status")).toHaveTextContent("draft")
    await waitFor(() =>
      expect(screen.getAllByRole("button", { name: "Localized loading" })).toHaveLength(1)
    )

    resolveCandidates({
      ok: true,
      json: async () => ({ candidates: [] })
    })

    await waitFor(() =>
      expect(screen.getAllByRole("button", { name: "Localized refresh" })).toHaveLength(2)
    )
  })

  it("loads pack list, creates a draft pack, and uploads the selected asset role", async () => {
    let packs: any[] = []
    const uploadedAssets: any[] = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse(packs)
      }
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "POST") {
        const body = parseJsonBody(init?.body)
        const pack = {
          id: "pack-created",
          persona_id: "persona-1",
          title: body.title,
          renderer_type: "sprite_frames",
          status: "draft",
          manifest: body.manifest,
          assets: [],
          version: 1
        }
        packs = [pack]
        return okResponse(pack)
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-created/assets" &&
        method === "POST"
      ) {
        const form = init?.body as FormData
        const file = form.get("file") as File
        const asset = {
          id: "asset-uploaded",
          pack_id: "pack-created",
          persona_id: "persona-1",
          asset_role: form.get("asset_role"),
          url: "/uploaded.png",
          original_filename: file.name,
          mime_type: file.type,
          byte_size: file.size,
          checksum_sha256: "sha-uploaded"
        }
        uploadedAssets.push(asset)
        packs = [{ ...packs[0], assets: [asset] }]
        return okResponse(asset)
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    const emptyState = await screen.findByTestId("persona-visual-pack-empty")
    expect(emptyState).toHaveTextContent(
      "Garden Helper's Persona Buddy does not have a visual pack yet."
    )
    expect(emptyState).toHaveTextContent("Create a draft visual pack first.")
    expect(emptyState).toHaveTextContent(
      "After a draft exists, upload frames, map states, import or export packs, queue generation, review candidates, and activate a valid pack."
    )
    expect(emptyState).not.toHaveTextContent("VN")
    expect(emptyState).not.toHaveTextContent("CYOA")
    fireEvent.change(screen.getByTestId("persona-visual-pack-title-input"), {
      target: { value: "First pack" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-create-pack"))

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "draft"
      )
    )
    expect(screen.getAllByText("First pack").length).toBeGreaterThan(0)

    fireEvent.change(screen.getByTestId("persona-visual-upload-role-select"), {
      target: { value: "sprite_sheet" }
    })
    const file = new File(["fake image"], "sheet.png", { type: "image/png" })
    fireEvent.change(screen.getByTestId("persona-visual-upload-input"), {
      target: { files: [file] }
    })
    fireEvent.click(screen.getByTestId("persona-visual-upload-button"))

    await waitFor(() => expect(uploadedAssets).toHaveLength(1))
    expect(uploadedAssets[0].asset_role).toBe("sprite_sheet")
    expect(await screen.findByText("sheet.png")).toBeInTheDocument()
  })

  it("edits state mappings, frame order, sprite-sheet regions, preview frame, and authored triggers", async () => {
    const savedManifests: any[] = []
    let pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/manifest" &&
        method === "PATCH"
      ) {
        const body = parseJsonBody(init?.body)
        savedManifests.push(body.manifest)
        pack = { ...pack, manifest: body.manifest, version: 4 }
        return okResponse(pack)
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "draft"
      )
    )
    fireEvent.change(screen.getByTestId("persona-visual-state-speaking-select"), {
      target: { value: "idle" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-animation-select"), {
      target: { value: "idle" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-frame-move-down-0"))
    const firstFrame = screen.getByTestId("persona-visual-frame-row-0")
    fireEvent.change(within(firstFrame).getByTestId("persona-visual-frame-region-x"), {
      target: { value: "8" }
    })
    fireEvent.change(within(firstFrame).getByTestId("persona-visual-frame-region-y"), {
      target: { value: "4" }
    })
    fireEvent.change(within(firstFrame).getByTestId("persona-visual-frame-region-width"), {
      target: { value: "64" }
    })
    fireEvent.change(within(firstFrame).getByTestId("persona-visual-frame-region-height"), {
      target: { value: "48" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-preview-frame-select"), {
      target: { value: "1" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-trigger-source-select"), {
      target: { value: "tool_category" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-trigger-match-input"), {
      target: { value: "notes" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-trigger-state-select"), {
      target: { value: "tool_running" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-trigger-duration-input"), {
      target: { value: "2500" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-trigger-priority-input"), {
      target: { value: "20" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-add-trigger"))
    fireEvent.click(screen.getByTestId("persona-visual-save-manifest"))

    await waitFor(() => expect(savedManifests).toHaveLength(1))
    const saved = savedManifests[0]
    expect(saved.states.speaking.animation_id).toBe("idle")
    expect(saved.animations.idle.frames.map((frame: any) => frame.asset_id)).toEqual([
      "asset-b",
      "asset-a"
    ])
    expect(saved.animations.idle.frames[0].region).toEqual({
      x: 8,
      y: 4,
      width: 64,
      height: 48
    })
    expect(saved.animations.idle.preview_frame).toBe(1)
    expect(saved.authored_triggers[0]).toMatchObject({
      source: "tool_category",
      match: "notes",
      state: "tool_running",
      duration_ms: 2500,
      priority: 20
    })
  })

  it("blocks activation when required states are missing, then saves, activates, and deactivates", async () => {
    const calls: string[] = []
    let pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Incomplete pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: {
        manifest_version: 1,
        renderer_type: "sprite_frames",
        states: { idle: { animation_id: "idle" } },
        animations: {
          idle: { frames: [{ asset_id: "asset-a" }], frame_rate: 1, loop: true }
        },
        fallbacks: {},
        authored_triggers: []
      },
      assets: [visualAssets[0]],
      version: 1
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/manifest" &&
        method === "PATCH"
      ) {
        const body = parseJsonBody(init?.body)
        pack = { ...pack, manifest: body.manifest, version: 2 }
        return okResponse(pack)
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/activate" &&
        method === "POST"
      ) {
        pack = { ...pack, status: "active" }
        return okResponse(pack)
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/deactivate" &&
        method === "POST"
      ) {
        pack = { ...pack, status: "archived" }
        return okResponse({ status: "deactivated", persona_id: "persona-1" })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "draft"
      )
    )
    expect(screen.getByTestId("persona-visual-validation-errors")).toHaveTextContent(
      "listening"
    )
    expect(screen.getByTestId("persona-visual-activate-button")).toBeDisabled()

    for (const state of ["listening", "thinking", "speaking", "error"]) {
      fireEvent.change(screen.getByTestId(`persona-visual-state-${state}-select`), {
        target: { value: "idle" }
      })
    }
    fireEvent.click(screen.getByTestId("persona-visual-save-manifest"))
    await waitFor(() =>
      expect(calls).toContain(
        "PATCH /api/v1/persona/profiles/persona-1/visual-packs/pack-1/manifest"
      )
    )
    expect(screen.queryByTestId("persona-visual-validation-errors")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("persona-visual-activate-button"))
    await waitFor(() =>
      expect(calls).toContain(
        "POST /api/v1/persona/profiles/persona-1/visual-packs/pack-1/activate"
      )
    )
    expect(await screen.findByTestId("persona-visual-pack-status")).toHaveTextContent("active")

    fireEvent.click(screen.getByTestId("persona-visual-deactivate-button"))
    await waitFor(() =>
      expect(calls).toContain(
        "POST /api/v1/persona/profiles/persona-1/visual-packs/deactivate"
      )
    )
  })

  it("enqueues generation jobs and accepts or rejects review candidates", async () => {
    const calls: string[] = []
    const candidate = {
      id: "candidate-1",
      pack_id: "pack-1",
      persona_id: "persona-1",
      job_id: "job-1",
      status: "review",
      proposed_manifest_patch: {
        states: { thinking: { animation_id: "generated-thinking" } }
      },
      generated_asset_ids: ["asset-a"],
      generated_assets: [visualAssets[0]],
      prompt: "make a thinking pose",
      failure_reason: null,
      created_at: "2026-05-09T00:00:00Z",
      last_modified: "2026-05-09T00:00:00Z",
      version: 1
    }
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [candidate] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-jobs" &&
        method === "POST"
      ) {
        return okResponse({ job_id: "job-created", status: "queued" })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/candidates/candidate-1/review" &&
        method === "POST"
      ) {
        return okResponse({
          ...candidate,
          status: parseJsonBody(init?.body).status
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    expect(await screen.findByText("make a thinking pose")).toBeInTheDocument()
    fireEvent.change(screen.getByTestId("persona-visual-generation-prompt-input"), {
      target: { value: "make a speaking pose" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-generation-target-state-select"), {
      target: { value: "speaking" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-generation-enqueue-button"))

    await waitFor(() =>
      expect(calls).toContain(
        "POST /api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-jobs"
      )
    )
    fireEvent.click(screen.getByTestId("persona-visual-candidate-accept-candidate-1"))
    await waitFor(() =>
      expect(calls).toContain(
        "POST /api/v1/persona/profiles/persona-1/visual-packs/pack-1/candidates/candidate-1/review"
      )
    )
    fireEvent.click(screen.getByTestId("persona-visual-candidate-reject-candidate-1"))
    await waitFor(() =>
      expect(
        calls.filter((call) =>
          call.includes("/visual-packs/pack-1/candidates/candidate-1/review")
        )
      ).toHaveLength(2)
    )
  })

  it("queues, polls, and downloads visual pack exports through the authenticated client", async () => {
    const calls: string[] = []
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:persona-visual-export")
    const revokeObjectUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined)
    const clickDownload = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined)

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any; responseType?: string }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/export" &&
        method === "POST"
      ) {
        return okResponse({
          job_id: "export-job-1",
          portability_job_id: "portability-1",
          operation: "export",
          persona_id: "persona-1",
          pack_id: "pack-1",
          status: "queued",
          stage: "queued",
          download_url: null
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/exports/export-job-1" &&
        method === "GET"
      ) {
        return okResponse({
          job_id: "export-job-1",
          portability_job_id: "portability-1",
          operation: "export",
          persona_id: "persona-1",
          pack_id: "pack-1",
          status: "completed",
          visual_status: "completed",
          stage: "completed",
          progress: { assets: 2 },
          warnings: [],
          archive_sha256: "sha-export",
          canonical_payload_fingerprint: "fingerprint-export",
          download_url: "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/exports/export-job-1/download"
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/exports/export-job-1/download" &&
        method === "GET"
      ) {
        expect(init?.responseType).toBe("arrayBuffer")
        return okBinaryResponse(Uint8Array.from([1, 2, 3, 4]).buffer)
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "draft"
      )
    )
    fireEvent.click(screen.getByTestId("persona-visual-export-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-export-status")).toHaveTextContent(
        "queued"
      )
    )

    fireEvent.click(screen.getByTestId("persona-visual-export-refresh-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-export-status")).toHaveTextContent(
        "completed"
      )
    )
    expect(screen.getByTestId("persona-visual-export-stage")).toHaveTextContent(
      "completed"
    )

    fireEvent.click(screen.getByTestId("persona-visual-export-download-button"))
    await waitFor(() => expect(createObjectUrl).toHaveBeenCalledTimes(1))
    expect(clickDownload).toHaveBeenCalledTimes(1)
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:persona-visual-export")
    expect(calls).toContain(
      "GET /api/v1/persona/profiles/persona-1/visual-packs/pack-1/exports/export-job-1/download"
    )
  })

  it("uploads import-preview archives and renders the review summary without mutating packs", async () => {
    const calls: string[] = []
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    const importedPack = {
      id: "pack-imported",
      persona_id: "persona-1",
      title: "Imported Visuals",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 1
    }
    let importCommitted = false

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse(importCommitted ? [pack, importedPack] : [pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/import-previews" &&
        method === "POST"
      ) {
        const form = init?.body as FormData
        expect((form.get("file") as File).name).toBe("portable.tldw-persona-vpack")
        return okResponse({
          preview_id: "preview-1",
          job_id: "preview-job-1",
          portability_job_id: "portability-preview-1",
          operation: "import_preview",
          target_persona_id: "persona-1",
          status: "queued",
          stage: "queued"
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-1" &&
        method === "GET"
      ) {
        return okResponse({
          preview_id: "preview-1",
          job_id: "preview-job-1",
          portability_job_id: "portability-preview-1",
          operation: "import_preview",
          target_persona_id: "persona-1",
          status: "completed",
          visual_status: "completed",
          stage: "completed",
          archive_sha256: "sha-preview",
          canonical_payload_fingerprint: "fingerprint-preview",
          schema_version: "persona_visual_pack.v1",
          bundle_summary: {
            pack_title: "Imported Visuals",
            asset_count: 2,
            assets_with_bytes: 2
          },
          validation_warnings: ["Unsigned archive"],
          conflicts: [{ type: "title", value: "Animated pack" }],
          proposed_plan: { target_mode: "create_new" },
          quota_estimate: { asset_bytes: 512 },
          required_choices: [],
          target_warnings: ["Review target persona before import"]
        })
      }
      if (
        path ===
          "/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-1/commit" &&
        method === "POST"
      ) {
        expect(parseJsonBody(init?.body)).toMatchObject({
          trust_mode: "untrusted_import",
          target_mode: "create_new"
        })
        return okResponse({
          job_id: "import-job-1",
          portability_job_id: "portability-import-1",
          operation: "import_commit",
          preview_id: "preview-1",
          target_persona_id: "persona-1",
          status: "queued",
          stage: "queued"
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/imports/import-job-1" &&
        method === "GET"
      ) {
        importCommitted = true
        return okResponse({
          job_id: "import-job-1",
          portability_job_id: "portability-import-1",
          operation: "import_commit",
          persona_id: "persona-1",
          pack_id: "pack-imported",
          status: "completed",
          visual_status: "completed",
          stage: "completed",
          progress: { asset_count: 2 },
          warnings: []
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "draft"
      )
    )
    const archive = new File(["portable archive"], "portable.tldw-persona-vpack", {
      type: "application/octet-stream"
    })
    fireEvent.change(screen.getByTestId("persona-visual-import-preview-input"), {
      target: { files: [archive] }
    })
    fireEvent.click(screen.getByTestId("persona-visual-import-preview-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-preview-status")).toHaveTextContent(
        "queued"
      )
    )

    fireEvent.click(screen.getByTestId("persona-visual-import-preview-refresh-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-preview-status")).toHaveTextContent(
        "completed"
      )
    )
    expect(screen.getByTestId("persona-visual-import-preview-summary")).toHaveTextContent(
      "Imported Visuals"
    )
    expect(screen.getByTestId("persona-visual-import-preview-summary")).toHaveTextContent(
      "2 assets"
    )
    expect(screen.getByTestId("persona-visual-import-preview-warnings")).toHaveTextContent(
      "Unsigned archive"
    )
    expect(screen.getByTestId("persona-visual-import-preview-conflicts")).toHaveTextContent(
      "Animated pack"
    )
    expect(screen.getByTestId("persona-visual-import-preview-plan")).toHaveTextContent(
      "create_new"
    )

    expect(screen.getByTestId("persona-visual-import-commit-button")).toBeEnabled()
    fireEvent.click(screen.getByTestId("persona-visual-import-commit-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
        "queued"
      )
    )
    expect(screen.getByTestId("persona-visual-import-commit-stage")).toHaveTextContent(
      "queued"
    )
    expect(screen.getByTestId("persona-visual-import-commit-job-id")).toHaveTextContent(
      "import-job-1"
    )
    expect(screen.getByTestId("persona-visual-import-commit-refresh-button")).toBeEnabled()

    fireEvent.click(screen.getByTestId("persona-visual-import-commit-refresh-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
        "completed"
      )
    )
    expect(screen.getByTestId("persona-visual-import-commit-refresh-button")).toBeDisabled()
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-select")).toHaveTextContent(
        "Imported Visuals"
      )
    )
    expect(
      calls.some((call) => call.includes("/activate") || call.includes("/manifest"))
    ).toBe(false)
  })

  it("allows failed import commit jobs to be retried", async () => {
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    let commitAttempts = 0

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([pack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/import-previews" &&
        method === "POST"
      ) {
        return okResponse({
          preview_id: "preview-1",
          job_id: "preview-job-1",
          portability_job_id: "portability-preview-1",
          operation: "import_preview",
          target_persona_id: "persona-1",
          status: "queued",
          stage: "queued"
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-1" &&
        method === "GET"
      ) {
        return okResponse({
          preview_id: "preview-1",
          job_id: "preview-job-1",
          portability_job_id: "portability-preview-1",
          operation: "import_preview",
          target_persona_id: "persona-1",
          status: "completed",
          visual_status: "completed",
          stage: "completed",
          bundle_summary: {
            pack_title: "Imported Visuals",
            asset_count: 2,
            assets_with_bytes: 2
          },
          validation_warnings: [],
          conflicts: [],
          proposed_plan: { target_mode: "create_new" },
          quota_estimate: {},
          required_choices: [],
          target_warnings: []
        })
      }
      if (
        path ===
          "/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-1/commit" &&
        method === "POST"
      ) {
        commitAttempts += 1
        expect(parseJsonBody(init?.body)).toMatchObject({
          trust_mode: "untrusted_import",
          target_mode: "create_new"
        })
        return okResponse({
          job_id: `import-job-${commitAttempts}`,
          portability_job_id: `portability-import-${commitAttempts}`,
          operation: "import_commit",
          preview_id: "preview-1",
          target_persona_id: "persona-1",
          status: "queued",
          stage: "queued"
        })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/imports/import-job-1" &&
        method === "GET"
      ) {
        return okResponse({
          job_id: "import-job-1",
          portability_job_id: "portability-import-1",
          operation: "import_commit",
          persona_id: "persona-1",
          pack_id: null,
          status: "failed",
          visual_status: "failed",
          stage: "failed",
          error_message: "Archive failed validation"
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "draft"
      )
    )
    const archive = new File(["portable archive"], "portable.tldw-persona-vpack", {
      type: "application/octet-stream"
    })
    fireEvent.change(screen.getByTestId("persona-visual-import-preview-input"), {
      target: { files: [archive] }
    })
    fireEvent.click(screen.getByTestId("persona-visual-import-preview-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-preview-status")).toHaveTextContent(
        "queued"
      )
    )

    fireEvent.click(screen.getByTestId("persona-visual-import-preview-refresh-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-preview-status")).toHaveTextContent(
        "completed"
      )
    )

    fireEvent.click(screen.getByTestId("persona-visual-import-commit-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
        "queued"
      )
    )
    fireEvent.click(screen.getByTestId("persona-visual-import-commit-refresh-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
        "failed"
      )
    )
    expect(screen.getByTestId("persona-visual-import-commit-refresh-button")).toBeDisabled()
    expect(screen.getByTestId("persona-visual-import-commit-button")).toBeEnabled()

    fireEvent.click(screen.getByTestId("persona-visual-import-commit-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-job-id")).toHaveTextContent(
        "import-job-2"
      )
    )
    expect(commitAttempts).toBe(2)
    expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
      "queued"
    )
  })
})
