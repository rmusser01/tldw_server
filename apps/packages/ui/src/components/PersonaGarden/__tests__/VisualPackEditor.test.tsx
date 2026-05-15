import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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

const deferredResponse = <T,>() => {
  let resolve: (value: T) => void = () => undefined
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve
  })
  return { promise, resolve }
}

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

const readyGenerationReadiness = {
  available: true,
  worker_enabled: true,
  queue: "generation",
  image_backend_available: true,
  default_backend: "openrouter",
  requested_backend: null,
  requested_backend_available: null,
  enabled_backends: ["openrouter"],
  reasons: []
}

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

  it("shows selected pack health diagnostics from the shared visual pack classifier", async () => {
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "active",
      manifest: structuredClone(baseManifest),
      assets: [visualAssets[0]],
      version: 3
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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

    const health = await screen.findByTestId("persona-visual-pack-health")
    await waitFor(() => {
      expect(health).toHaveTextContent("Visual asset is missing")
      expect(health).toHaveTextContent("asset-b")
    })
    expect(health).toHaveClass("border-danger/30")
    expect(health).toHaveClass("text-danger")
  })

  it("explains visual pack ownership and active pack semantics", async () => {
    const pack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "active",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
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

    const ownership = await screen.findByTestId("persona-visual-ownership-copy")
    expect(ownership).toHaveTextContent("Assets are user-owned")
    expect(ownership).toHaveTextContent("attached to Garden Helper by default")
    expect(ownership).toHaveTextContent("stored as manifests")
    expect(ownership).toHaveTextContent(
      "The active pack is the one Persona Buddy renders now"
    )
  })

  it("clarifies import export and generated candidate review semantics", async () => {
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

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
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

    const portability = await screen.findByTestId("persona-visual-portability-copy")
    expect(portability).toHaveTextContent(
      "Import preview validates a portable pack archive before it changes this persona"
    )
    expect(portability).toHaveTextContent(
      "Commit import creates a reviewed draft pack"
    )
    expect(portability).toHaveTextContent(
      "Export downloads a portable archive and does not publish to a shared library"
    )

    expect(screen.getByTestId("persona-visual-generation-review-copy")).toHaveTextContent(
      "Generated candidates stay in review until accepted"
    )
  })

  it("surfaces reusable visual-pack routes through existing editor controls", async () => {
    const clickFileInput = vi
      .spyOn(HTMLInputElement.prototype, "click")
      .mockImplementation(() => undefined)
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
    const libraryItem = {
      id: "library-1",
      user_id: "user-1",
      source_persona_id: "persona-1",
      source_pack_id: "pack-1",
      title: "Saved animated pack",
      notes: "Reusable idle and speaking poses.",
      tags: ["idle"],
      source_persona_name: "Source Persona",
      source_pack_title: "Animated pack",
      source_pack_version: 3,
      source_current_version: 3,
      source_available: true,
      source_changed: false,
      created_at: "2026-05-09T00:00:00Z",
      last_modified: "2026-05-09T00:00:00Z",
      version: 2
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        return okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Research Buddy" }
        ])
      }
      if (path === "/api/v1/persona/visual-library" && method === "GET") {
        return okResponse({ items: [libraryItem] })
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
        selectedPersonaName="Source Persona"
        isActive
      />
    )

    const reusePanel = await screen.findByTestId("persona-visual-reuse-panel")
    expect(reusePanel).toHaveTextContent("Reuse visual packs")
    expect(reusePanel).toHaveTextContent("user-owned")
    expect(reusePanel).not.toHaveTextContent(/marketplace/i)
    expect(reusePanel).not.toHaveTextContent(/\bVN\b/)
    expect(reusePanel).not.toHaveTextContent(/CYOA/i)

    fireEvent.click(within(reusePanel).getByRole("button", { name: /create draft/i }))
    expect(screen.getByTestId("persona-visual-pack-title-input")).toHaveFocus()

    fireEvent.click(
      within(reusePanel).getByRole("button", { name: /use personal library/i })
    )
    expect(screen.getByTestId("persona-visual-library-panel")).toHaveFocus()

    await waitFor(() =>
      expect(
        within(reusePanel).getByRole("button", { name: /duplicate to persona/i })
      ).toBeEnabled()
    )
    fireEvent.click(
      within(reusePanel).getByRole("button", { name: /duplicate to persona/i })
    )
    expect(screen.getByTestId("persona-visual-duplicate-target-select")).toHaveFocus()

    fireEvent.click(within(reusePanel).getByRole("button", { name: /import archive/i }))
    expect(clickFileInput).toHaveBeenCalled()
  })

  it("keeps duplicate reuse unavailable until a pack and another persona exist", async () => {
    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([])
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        return okResponse([{ id: "persona-1", name: "Solo Persona" }])
      }
      if (path === "/api/v1/persona/visual-library" && method === "GET") {
        return okResponse({ items: [] })
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
        selectedPersonaName="Solo Persona"
        isActive
      />
    )

    const reusePanel = await screen.findByTestId("persona-visual-reuse-panel")
    expect(reusePanel).toHaveTextContent("No saved visual packs yet")
    expect(reusePanel).toHaveTextContent("Save a pack here first")
    expect(reusePanel).not.toHaveTextContent("Use one to create")
    expect(reusePanel).toHaveTextContent("Select a pack before duplicating")
    expect(
      within(reusePanel).getByRole("button", { name: /duplicate to persona/i })
    ).toBeDisabled()
    expect(within(reusePanel).getByRole("button", { name: /import archive/i })).toBeDisabled()
    expect(
      within(reusePanel).getByRole("button", { name: /use personal library/i })
    ).toBeEnabled()
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
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
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

  it("disables visual generation when the Jobs worker is unavailable", async () => {
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

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse({
          ...readyGenerationReadiness,
          available: false,
          worker_enabled: false,
          reasons: ["jobs_worker_disabled"]
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

    const readiness = await screen.findByTestId("persona-visual-generation-readiness")
    expect(readiness).toHaveTextContent("Generation worker is not enabled.")
    expect(readiness).toHaveTextContent("Jobs queue: generation")

    fireEvent.change(screen.getByTestId("persona-visual-generation-prompt-input"), {
      target: { value: "make an idle pose" }
    })

    expect(screen.getByTestId("persona-visual-generation-enqueue-button")).toBeDisabled()
    expect(calls).not.toContain(
      "POST /api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-jobs"
    )
  })

  it("disables visual generation when no image provider is configured", async () => {
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

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse({
          ...readyGenerationReadiness,
          available: false,
          image_backend_available: false,
          default_backend: null,
          enabled_backends: [],
          reasons: ["image_backend_unavailable"]
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

    const readiness = await screen.findByTestId("persona-visual-generation-readiness")
    expect(readiness).toHaveTextContent("No image generation provider is configured.")
    expect(readiness).toHaveTextContent("Enable an image backend before queueing a Persona Buddy visual generation job.")

    fireEvent.change(screen.getByTestId("persona-visual-generation-prompt-input"), {
      target: { value: "make an idle pose" }
    })

    expect(screen.getByTestId("persona-visual-generation-enqueue-button")).toBeDisabled()
    expect(calls).not.toContain(
      "POST /api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-jobs"
    )
  })

  it("ignores stale generation readiness responses after switching packs", async () => {
    const delayedPackOneReadiness = deferredResponse<Awaited<ReturnType<typeof okResponse>>>()
    const packOne = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    const packTwo = {
      ...packOne,
      id: "pack-2",
      title: "Second pack",
      manifest: structuredClone(baseManifest)
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([packOne, packTwo])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-2/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return delayedPackOneReadiness.promise
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-2/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse({
          ...readyGenerationReadiness,
          available: false,
          image_backend_available: false,
          default_backend: null,
          enabled_backends: [],
          reasons: ["image_backend_unavailable"]
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
      expect(screen.getByTestId("persona-visual-pack-select")).toHaveValue("pack-1")
    )
    fireEvent.change(screen.getByTestId("persona-visual-pack-select"), {
      target: { value: "pack-2" }
    })

    const readiness = await screen.findByTestId("persona-visual-generation-readiness")
    expect(readiness).toHaveTextContent("No image generation provider is configured.")

    await act(async () => {
      delayedPackOneReadiness.resolve(await okResponse(readyGenerationReadiness))
      await delayedPackOneReadiness.promise
    })

    expect(screen.getByTestId("persona-visual-pack-select")).toHaveValue("pack-2")
    expect(screen.getByTestId("persona-visual-generation-readiness")).toHaveTextContent(
      "No image generation provider is configured."
    )
  })

  it("ignores stale duplicate target responses after switching personas", async () => {
    const delayedPersonaOneTargets = deferredResponse<Awaited<ReturnType<typeof okResponse>>>()
    const personaOnePack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Source pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    const personaTwoPack = {
      ...personaOnePack,
      id: "pack-2",
      persona_id: "persona-2",
      title: "Target pack",
      assets: visualAssets.map((asset) => ({
        ...asset,
        pack_id: "pack-2",
        persona_id: "persona-2"
      }))
    }
    let catalogCalls = 0
    const calls: string[] = []

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([personaOnePack])
      }
      if (path === "/api/v1/persona/profiles/persona-2/visual-packs" && method === "GET") {
        return okResponse([personaTwoPack])
      }
      if (path.endsWith("/generated-candidates") && method === "GET") {
        return okResponse({ candidates: [] })
      }
      if (path.endsWith("/generation-readiness") && method === "GET") {
        return okResponse(readyGenerationReadiness)
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        catalogCalls += 1
        if (catalogCalls === 1) return delayedPersonaOneTargets.promise
        return okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Target Persona" }
        ])
      }
      if (
        path === "/api/v1/persona/profiles/persona-2/visual-packs/pack-2/duplicate" &&
        method === "POST"
      ) {
        expect(parseJsonBody(init?.body)).toEqual({
          target_persona_id: "persona-1",
          title: "Copy of Target pack"
        })
        return okResponse({
          ...personaTwoPack,
          id: "pack-copy",
          persona_id: "persona-1",
          status: "draft",
          parent_pack_id: "pack-2"
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        error: `Unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    const { rerender } = render(
      <VisualPackEditor
        selectedPersonaId="persona-1"
        selectedPersonaName="Source Persona"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-select")).toHaveValue("pack-1")
    )
    await waitFor(() => expect(catalogCalls).toBe(1))

    rerender(
      <VisualPackEditor
        selectedPersonaId="persona-2"
        selectedPersonaName="Target Persona"
        isActive
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-select")).toHaveValue("pack-2")
    )
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-duplicate-target-select")).toHaveValue(
        "persona-1"
      )
    )

    await act(async () => {
      delayedPersonaOneTargets.resolve(
        await okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Target Persona" }
        ])
      )
      await delayedPersonaOneTargets.promise
    })

    expect(screen.getByTestId("persona-visual-duplicate-target-select")).toHaveValue(
      "persona-1"
    )
    expect(screen.getByTestId("persona-visual-duplicate-target-select")).not.toHaveTextContent(
      "Target Persona"
    )
    fireEvent.click(screen.getByTestId("persona-visual-duplicate-button"))
    await waitFor(() =>
      expect(calls).toContain(
        "POST /api/v1/persona/profiles/persona-2/visual-packs/pack-2/duplicate"
      )
    )
  })

  it("duplicates a visual pack to another persona as a draft", async () => {
    const sourcePack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "active",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    const duplicatedPack = {
      ...sourcePack,
      id: "pack-duplicate",
      persona_id: "persona-2",
      title: "Research Buddy copy",
      status: "draft",
      parent_pack_id: "pack-1",
      assets: [
        {
          ...visualAssets[0],
          id: "asset-copy",
          pack_id: "pack-duplicate",
          persona_id: "persona-2"
        }
      ]
    }
    const openTarget = vi.fn()

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([sourcePack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        return okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Research Buddy" }
        ])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/duplicate" &&
        method === "POST"
      ) {
        expect(parseJsonBody(init?.body)).toEqual({
          target_persona_id: "persona-2",
          title: "Research Buddy copy"
        })
        return okResponse(duplicatedPack)
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
        selectedPersonaName="Source Persona"
        isActive
        onOpenPersonaVisuals={openTarget}
      />
    )

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent(
        "active"
      )
    )
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-duplicate-target-select")).toHaveTextContent(
        "Research Buddy"
      )
    )
    expect(screen.getByTestId("persona-visual-duplicate-target-select")).not.toHaveTextContent(
      "Source Persona"
    )

    fireEvent.change(screen.getByTestId("persona-visual-duplicate-title-input"), {
      target: { value: "Research Buddy copy" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-duplicate-target-select"), {
      target: { value: "persona-2" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-duplicate-button"))

    await waitFor(() =>
      expect(screen.getByText(/Duplicated as a draft for Research Buddy/)).toBeInTheDocument()
    )
    fireEvent.click(screen.getByTestId("persona-visual-duplicate-open-target"))
    expect(openTarget).toHaveBeenCalledWith("persona-2")
  })

  it("saves the selected pack to the personal library and shows source status", async () => {
    const calls: string[] = []
    const sourcePack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "active",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    const changedLibraryItem = {
      id: "library-1",
      user_id: "user-1",
      source_persona_id: "persona-1",
      source_pack_id: "pack-1",
      title: "Saved animated pack",
      notes: "Reusable idle and speaking poses.",
      tags: ["idle", "speaking"],
      source_persona_name: "Source Persona",
      source_pack_title: "Animated pack",
      source_pack_version: 2,
      source_current_version: 3,
      source_available: true,
      source_changed: true,
      created_at: "2026-05-09T00:00:00Z",
      last_modified: "2026-05-09T00:00:00Z",
      version: 2
    }
    const savedLibraryItem = {
      ...changedLibraryItem,
      source_pack_version: 3,
      source_changed: false,
      version: 3
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([sourcePack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        return okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Research Buddy" }
        ])
      }
      if (path === "/api/v1/persona/visual-library" && method === "GET") {
        return okResponse({ items: [changedLibraryItem] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/library" &&
        method === "POST"
      ) {
        expect(parseJsonBody(init?.body)).toEqual({
          title: "Saved animated pack",
          notes: "Reusable idle and speaking poses.",
          tags: ["idle", "speaking"]
        })
        return okResponse(savedLibraryItem)
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
        selectedPersonaName="Source Persona"
        isActive
      />
    )

    const libraryPanel = await screen.findByTestId("persona-visual-library-panel")
    expect(libraryPanel).toHaveTextContent("Personal library")
    expect(within(libraryPanel).getByText("Saved animated pack")).toBeInTheDocument()
    expect(within(libraryPanel).getByText("source changed")).toBeInTheDocument()
    expect(within(libraryPanel).getByText("idle")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("persona-visual-library-save-button"))

    await waitFor(() =>
      expect(calls).toContain(
        "POST /api/v1/persona/profiles/persona-1/visual-packs/pack-1/library"
      )
    )
    expect(await screen.findByText("Saved to personal library.")).toBeInTheDocument()
    expect(screen.getByTestId("persona-visual-library-item-library-1")).toHaveTextContent(
      "Saved animated pack"
    )
  })

  it("shows personal library items and uses them when the selected persona has no packs", async () => {
    const calls: string[] = []
    const sourceLibraryItem = {
      id: "library-1",
      user_id: "user-1",
      source_persona_id: "persona-1",
      source_pack_id: "pack-1",
      title: "Reusable source pack",
      notes: "Useful for new personas.",
      tags: ["idle"],
      source_persona_name: "Source Persona",
      source_pack_title: "Animated pack",
      source_pack_version: 3,
      source_current_version: 3,
      source_available: true,
      source_changed: false,
      created_at: "2026-05-09T00:00:00Z",
      last_modified: "2026-05-09T00:00:00Z",
      version: 2
    }
    const duplicatedPack = {
      id: "pack-library-copy",
      persona_id: "persona-2",
      title: "Reusable source pack",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      parent_pack_id: "pack-1",
      version: 1
    }

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-2/visual-packs" && method === "GET") {
        return okResponse([])
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        return okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Target Persona" }
        ])
      }
      if (path === "/api/v1/persona/visual-library" && method === "GET") {
        return okResponse({ items: [sourceLibraryItem] })
      }
      if (path === "/api/v1/persona/visual-library/library-1/use" && method === "POST") {
        expect(parseJsonBody(init?.body)).toEqual({
          target_persona_id: "persona-2"
        })
        return okResponse(duplicatedPack)
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
        selectedPersonaId="persona-2"
        selectedPersonaName="Target Persona"
        isActive
      />
    )

    const libraryPanel = await screen.findByTestId("persona-visual-library-panel")
    expect(libraryPanel).toHaveTextContent("Personal library")
    expect(within(libraryPanel).getByText("Reusable source pack")).toBeInTheDocument()

    fireEvent.change(screen.getByTestId("persona-visual-library-target-library-1"), {
      target: { value: "persona-2" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-library-use-library-1"))

    await waitFor(() =>
      expect(calls).toContain("POST /api/v1/persona/visual-library/library-1/use")
    )
    expect(
      await screen.findByText(/Library item copied as a draft for Target Persona/)
    ).toBeInTheDocument()
    expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent("draft")
  })

  it("edits, removes, and uses personal library entries as draft target packs", async () => {
    const calls: string[] = []
    const sourcePack = {
      id: "pack-1",
      persona_id: "persona-1",
      title: "Animated pack",
      renderer_type: "sprite_frames",
      status: "active",
      manifest: structuredClone(baseManifest),
      assets: visualAssets,
      version: 3
    }
    let libraryItems: any[] = [
      {
        id: "library-1",
        user_id: "user-1",
        source_persona_id: "persona-1",
        source_pack_id: "pack-1",
        title: "Reusable source pack",
        notes: "Useful for research persona.",
        tags: ["idle"],
        source_persona_name: "Source Persona",
        source_pack_title: "Animated pack",
        source_pack_version: 3,
        source_current_version: 3,
        source_available: true,
        source_changed: false,
        created_at: "2026-05-09T00:00:00Z",
        last_modified: "2026-05-09T00:00:00Z",
        version: 2
      },
      {
        id: "library-stale",
        user_id: "user-1",
        source_persona_id: null,
        source_pack_id: null,
        title: "Missing source pack",
        notes: null,
        tags: [],
        source_persona_name: null,
        source_pack_title: null,
        source_pack_version: 1,
        source_current_version: null,
        source_available: false,
        source_changed: false,
        created_at: "2026-05-08T00:00:00Z",
        last_modified: "2026-05-08T00:00:00Z",
        version: 1
      }
    ]
    const duplicatedPack = {
      ...sourcePack,
      id: "pack-library-copy",
      persona_id: "persona-2",
      title: "Reusable source pack",
      status: "draft",
      parent_pack_id: "pack-1"
    }
    const openTarget = vi.fn()

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      calls.push(`${method} ${path}`)
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        return okResponse([sourcePack])
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
        method === "GET"
      ) {
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generation-readiness" &&
        method === "GET"
      ) {
        return okResponse(readyGenerationReadiness)
      }
      if (path === "/api/v1/persona/catalog" && method === "GET") {
        return okResponse([
          { id: "persona-1", name: "Source Persona" },
          { id: "persona-2", name: "Research Buddy" }
        ])
      }
      if (path === "/api/v1/persona/visual-library" && method === "GET") {
        return okResponse({ items: libraryItems })
      }
      if (path === "/api/v1/persona/visual-library/library-1" && method === "PATCH") {
        expect(parseJsonBody(init?.body)).toEqual({
          title: "Edited library title",
          notes: "Ready for reuse.",
          tags: ["idle", "formal"],
          expected_version: 2
        })
        libraryItems = libraryItems.map((item) =>
          item.id === "library-1"
            ? {
                ...item,
                title: "Edited library title",
                notes: "Ready for reuse.",
                tags: ["idle", "formal"],
                version: 3
              }
            : item
        )
        return okResponse(libraryItems[0])
      }
      if (path === "/api/v1/persona/visual-library/library-stale" && method === "DELETE") {
        libraryItems = libraryItems.filter((item) => item.id !== "library-stale")
        return okResponse({ status: "deleted", item_id: "library-stale" })
      }
      if (path === "/api/v1/persona/visual-library/library-1/use" && method === "POST") {
        expect(parseJsonBody(init?.body)).toEqual({
          target_persona_id: "persona-2"
        })
        return okResponse(duplicatedPack)
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
        selectedPersonaName="Source Persona"
        isActive
        onOpenPersonaVisuals={openTarget}
      />
    )

    const staleItem = await screen.findByTestId(
      "persona-visual-library-item-library-stale"
    )
    expect(staleItem).toHaveTextContent("Missing source pack")
    expect(staleItem).toHaveTextContent("unavailable")
    expect(
      screen.getByTestId("persona-visual-library-use-library-stale")
    ).toBeDisabled()

    fireEvent.click(screen.getByTestId("persona-visual-library-edit-library-1"))
    fireEvent.change(screen.getByTestId("persona-visual-library-edit-title-library-1"), {
      target: { value: "Edited library title" }
    })
    fireEvent.change(screen.getByTestId("persona-visual-library-edit-notes-library-1"), {
      target: { value: "Ready for reuse." }
    })
    fireEvent.change(screen.getByTestId("persona-visual-library-edit-tags-library-1"), {
      target: { value: "idle, formal" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-library-save-edit-library-1"))

    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-library-item-library-1")).toHaveTextContent(
        "Edited library title"
      )
    )

    fireEvent.click(screen.getByTestId("persona-visual-library-remove-library-stale"))
    await waitFor(() =>
      expect(calls).toContain("DELETE /api/v1/persona/visual-library/library-stale")
    )
    expect(screen.queryByText("Missing source pack")).not.toBeInTheDocument()

    fireEvent.change(screen.getByTestId("persona-visual-library-target-library-1"), {
      target: { value: "persona-2" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-library-use-library-1"))

    await waitFor(() =>
      expect(calls).toContain("POST /api/v1/persona/visual-library/library-1/use")
    )
    expect(
      await screen.findByText(/Library item copied as a draft for Research Buddy/)
    ).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("persona-visual-duplicate-open-target"))
    expect(openTarget).toHaveBeenCalledWith("persona-2")
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
    let importCommitStarts = 0

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
        expect(form.get("file")).toBeNull()
        expect((form.get("archive") as File).name).toBe("portable.tldw-persona-vpack")
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
        importCommitStarts += 1
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

    const commitButton = screen.getByTestId("persona-visual-import-commit-button")
    expect(commitButton).toBeEnabled()
    fireEvent.click(commitButton)
    fireEvent.click(commitButton)
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
        "queued"
      )
    )
    expect(importCommitStarts).toBe(1)
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
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-pack-select")).toHaveValue(
        "pack-imported"
      )
    )
    expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent("draft")
    expect(
      calls.some((call) => call.includes("/activate") || call.includes("/manifest"))
    ).toBe(false)
  })

  it("rejects unsupported visual import archive filenames before upload", async () => {
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

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string }) => {
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
    const archive = new File(["not a portable pack"], "visuals.zip", {
      type: "application/zip"
    })
    fireEvent.change(screen.getByTestId("persona-visual-import-preview-input"), {
      target: { files: [archive] }
    })

    expect(
      await screen.findByTestId("persona-visual-import-preview-file-error")
    ).toHaveTextContent(".tldw-persona-vpack")
    expect(screen.getByTestId("persona-visual-import-preview-button")).toBeDisabled()

    fireEvent.click(screen.getByTestId("persona-visual-import-preview-button"))

    expect(
      calls.some((call) => call.includes("/visual-packs/import-previews"))
    ).toBe(false)
  })

  it("shows import preview failure copy from the job response", async () => {
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
          status: "failed",
          visual_status: "failed",
          stage: "validate_archive",
          bundle_summary: {},
          validation_warnings: [],
          conflicts: [],
          proposed_plan: {},
          quota_estimate: {},
          required_choices: [],
          target_warnings: [],
          error_code: "invalid_archive",
          error_message: "The archive could not be opened as a Persona Visual pack."
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

    expect(
      await screen.findByTestId("persona-visual-import-preview-job-copy")
    ).toHaveTextContent("The archive could not be opened as a Persona Visual pack.")
  })

  it("surfaces blocked renderer import diagnostics and disables commit", async () => {
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
        return okResponse({ candidates: [] })
      }
      if (
        path === "/api/v1/persona/profiles/persona-1/visual-packs/import-previews" &&
        method === "POST"
      ) {
        return okResponse({
          preview_id: "preview-live2d",
          job_id: "preview-job-live2d",
          portability_job_id: "portability-preview-live2d",
          operation: "import_preview",
          target_persona_id: "persona-1",
          status: "queued",
          stage: "queued"
        })
      }
      if (
        path ===
          "/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-live2d" &&
        method === "GET"
      ) {
        return okResponse({
          preview_id: "preview-live2d",
          job_id: "preview-job-live2d",
          portability_job_id: "portability-preview-live2d",
          operation: "import_preview",
          target_persona_id: "persona-1",
          status: "completed",
          visual_status: "blocked",
          stage: "blocked",
          archive_sha256: "sha-live2d-preview",
          canonical_payload_fingerprint: "fingerprint-live2d-preview",
          schema_version: "persona_visual_pack.v2",
          bundle_summary: {
            pack_title: "Imported Live2D Visuals",
            asset_count: 2,
            assets_with_bytes: 2
          },
          validation_warnings: [],
          conflicts: [],
          proposed_plan: {
            target_mode: "create_new",
            commit_eligible: false,
            activation_eligible: false,
            commit_blockers: ["runtime_adapter_not_implemented"],
            renderer_import_preview: {
              status: "feature_gated",
              renderer_type: "live2d",
              manifest_version: 2,
              renderer_contract_version: 1,
              can_commit: false,
              activation_eligible: false,
              blockers: ["runtime_adapter_not_implemented"],
              warnings: ["requires_license_ack"],
              normalized_role_categories: {
                model: ["source-model"],
                texture: ["source-texture"]
              },
              setup_status: "feature_gated",
              setup_blockers: [],
              disabled_reason: "runtime_adapter_not_implemented"
            }
          },
          quota_estimate: { asset_bytes: 2048 },
          required_choices: [],
          target_warnings: []
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

    const diagnostics = screen.getByTestId(
      "persona-visual-import-renderer-diagnostics"
    )
    expect(diagnostics).toHaveTextContent("live2d")
    expect(diagnostics).toHaveTextContent("feature_gated")
    expect(diagnostics).toHaveTextContent("runtime_adapter_not_implemented")
    expect(diagnostics).toHaveTextContent("requires_license_ack")
    expect(diagnostics).toHaveTextContent("source-model")
    expect(diagnostics).toHaveTextContent("Activation unavailable")
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.rendererDiagnosticsTitle",
      { defaultValue: "Renderer diagnostics" }
    )
    expect(mocks.translate).toHaveBeenCalledWith("common:unknown", {
      defaultValue: "unknown"
    })
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.manifestVersion",
      { defaultValue: "Manifest v" }
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.contractVersion",
      { defaultValue: "Contract v" }
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.activationUnavailable",
      { defaultValue: "Activation unavailable" }
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.commitBlockers",
      { defaultValue: "Commit blockers" }
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.rendererWarnings",
      { defaultValue: "Warnings" }
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.assetRoles",
      { defaultValue: "Asset roles" }
    )
    expect(mocks.translate).toHaveBeenCalledWith(
      "sidepanel:personaGarden.visuals.importCommitBlocked",
      {
        defaultValue: "Commit unavailable until preview blockers are resolved"
      }
    )

    expect(screen.getByTestId("persona-visual-import-commit-button")).toBeDisabled()
    fireEvent.click(screen.getByTestId("persona-visual-import-commit-button"))
    expect(
      calls.some((call) => call.includes("/commit"))
    ).toBe(false)
  })

  it("sends explicit replace-draft choices for conflicted visual imports", async () => {
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
    let commitPayload: any = null

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
          archive_sha256: "sha-preview",
          canonical_payload_fingerprint: "fingerprint-preview",
          schema_version: "persona_visual_pack.v1",
          bundle_summary: {
            pack_title: "Imported Visuals",
            asset_count: 2,
            assets_with_bytes: 2
          },
          validation_warnings: [],
          conflicts: [
            {
              conflict_id: "target_pack_title_match:draft-pack-1",
              type: "target_pack_title_match",
              message: "Target persona already has a draft visual pack named Imported Visuals.",
              pack_id: "draft-pack-1",
              pack_title: "Imported Visuals",
              pack_status: "draft",
              allowed_choices: ["create_new", "replace_draft"]
            }
          ],
          proposed_plan: {
            target_mode: "create_new",
            target_modes: ["create_new", "replace_draft"],
            replaceable_pack_ids: ["draft-pack-1"]
          },
          quota_estimate: { asset_bytes: 512 },
          required_choices: [
            {
              choice_id: "import_target_mode",
              reason: "target_pack_conflicts",
              default_target_mode: "create_new",
              allowed_target_modes: ["create_new", "replace_draft"],
              replaceable_pack_ids: ["draft-pack-1"]
            }
          ],
          target_warnings: []
        })
      }
      if (
        path ===
          "/api/v1/persona/profiles/persona-1/visual-packs/import-previews/preview-1/commit" &&
        method === "POST"
      ) {
        commitPayload = parseJsonBody(init?.body)
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
      expect(screen.getByTestId("persona-visual-import-conflict-choice")).toBeInTheDocument()
    )
    expect(screen.getByTestId("persona-visual-import-commit-button")).toBeDisabled()

    fireEvent.change(screen.getByTestId("persona-visual-import-target-mode"), {
      target: { value: "replace_draft" }
    })
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-replace-pack")).toHaveValue(
        "draft-pack-1"
      )
    )
    fireEvent.change(screen.getByTestId("persona-visual-import-draft-title"), {
      target: { value: "Replacement Visuals" }
    })
    fireEvent.click(screen.getByTestId("persona-visual-import-commit-button"))

    await waitFor(() => expect(commitPayload).not.toBeNull())
    expect(commitPayload).toMatchObject({
      trust_mode: "untrusted_import",
      target_mode: "replace_draft",
      target_pack_id: "draft-pack-1",
      title: "Replacement Visuals"
    })
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

  it("does not show import completion success when pack refresh fails", async () => {
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
    let packListRequests = 0

    mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
      const method = init?.method || "GET"
      if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
        packListRequests += 1
        if (packListRequests > 1) {
          return Promise.resolve({
            ok: false,
            status: 503,
            error: "Pack refresh failed",
            json: async () => ({ detail: "Pack refresh failed" })
          })
        }
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

    fireEvent.click(screen.getByTestId("persona-visual-import-commit-button"))
    await waitFor(() =>
      expect(screen.getByTestId("persona-visual-import-commit-status")).toHaveTextContent(
        "queued"
      )
    )
    fireEvent.click(screen.getByTestId("persona-visual-import-commit-refresh-button"))
    await waitFor(() => expect(screen.getByText("Pack refresh failed")).toBeInTheDocument())
    expect(
      screen.queryByText(
        "Import commit completed. Review and activate the new draft when ready."
      )
    ).not.toBeInTheDocument()
  })
})
