import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import {
  commitAudiobookMigration,
  createAudioStudioExport,
  createAudioStudioGeneration,
  createAudioStudioProject,
  createAudioStudioRender,
  listAudioStudioProjects,
  listAudioStudioWorkflows,
  previewAudiobookMigration,
  updateAudioStudioProject,
  upsertAudioStudioClip,
  upsertAudioStudioSection,
  upsertAudioStudioTrack
} from "@/services/audio-studio"

describe("audio-studio service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue({})
  })

  it("lists workflows through the shared request client", async () => {
    await listAudioStudioWorkflows()

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/workflows",
      method: "GET"
    })
  })

  it("lists projects with workflow filters encoded in the query string", async () => {
    await listAudioStudioProjects({ workflow: "podcast", includeArchived: false })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/projects?workflow=podcast&include_archived=false",
      method: "GET"
    })
  })

  it("creates and updates projects without accepting client secrets", async () => {
    await createAudioStudioProject({
      title: "Morning Brief",
      workflow: "briefing",
      settings: { voice: "Ava" }
    })
    await updateAudioStudioProject("project 1", {
      title: "Daily Brief",
      base_revision_id: "rev-1"
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/audio-studio/projects",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        title: "Morning Brief",
        workflow: "briefing",
        settings: { voice: "Ava" }
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/audio-studio/projects/project%201",
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: {
        title: "Daily Brief",
        base_revision_id: "rev-1"
      }
    })
  })

  it("upserts project resources on the typed section, track, and clip endpoints", async () => {
    await upsertAudioStudioSection("p1", "s1", {
      title: "Intro",
      body_text: "Welcome",
      workflow: "narration",
      order: 0,
      base_revision_id: "rev-1"
    })
    await upsertAudioStudioTrack("p1", "speech", {
      name: "Speech",
      kind: "speech",
      order: 0,
      base_revision_id: "rev-2"
    })
    await upsertAudioStudioClip("p1", "clip/a", {
      track_id: "speech",
      section_id: "s1",
      start_ms: 0,
      duration_ms: 1000,
      base_revision_id: "rev-3"
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/audio-studio/projects/p1/sections/s1",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        title: "Intro",
        body_text: "Welcome",
        workflow: "narration",
        order: 0,
        base_revision_id: "rev-1"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/audio-studio/projects/p1/tracks/speech",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        name: "Speech",
        kind: "speech",
        order: 0,
        base_revision_id: "rev-2"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/audio-studio/projects/p1/clips/clip%2Fa",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        track_id: "speech",
        section_id: "s1",
        start_ms: 0,
        duration_ms: 1000,
        base_revision_id: "rev-3"
      }
    })
  })

  it("creates generation, render, export, and migration jobs", async () => {
    await createAudioStudioGeneration("p1", {
      workflow: "music",
      operation: "music",
      provider: "ace_step",
      idempotency_key: "generation-key-1234",
      inputs: { prompt: "warm intro" }
    })
    await createAudioStudioRender("p1", {
      idempotency_key: "render-key-1234",
      timeline_revision_id: "timeline-1"
    })
    await createAudioStudioExport("p1", {
      idempotency_key: "export-key-1234",
      format: "zip"
    })
    await previewAudiobookMigration({ legacy_project_ids: ["legacy-1"] })
    await commitAudiobookMigration({
      legacy_project_ids: ["legacy-1"],
      idempotency_key: "migration-key-1234"
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/audio-studio/projects/p1/generations",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        workflow: "music",
        operation: "music",
        provider: "ace_step",
        idempotency_key: "generation-key-1234",
        inputs: { prompt: "warm intro" }
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/audio-studio/projects/p1/renders",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        idempotency_key: "render-key-1234",
        timeline_revision_id: "timeline-1"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/audio-studio/projects/p1/exports",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        idempotency_key: "export-key-1234",
        format: "zip"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(4, {
      path: "/api/v1/audio-studio/migrations/audiobook/preview",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { legacy_project_ids: ["legacy-1"] }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(5, {
      path: "/api/v1/audio-studio/migrations/audiobook/commit",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        legacy_project_ids: ["legacy-1"],
        idempotency_key: "migration-key-1234"
      }
    })
  })
})
