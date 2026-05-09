import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
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
    mocks.fetchWithAuth.mockReset()
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

    expect(await screen.findByTestId("persona-visual-pack-empty")).toBeInTheDocument()
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
})
