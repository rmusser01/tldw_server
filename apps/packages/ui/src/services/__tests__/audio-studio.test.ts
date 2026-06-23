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
    mocks.bgRequest.mockResolvedValueOnce({
      workflows: [
        {
          id: "music",
          label: "Music",
          description: "Prompt-based music generation"
        }
      ]
    })

    const workflows = await listAudioStudioWorkflows()

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/workflows",
      method: "GET"
    })
    expect(workflows).toEqual([
      {
        id: "music",
        label: "Music",
        description: "Prompt-based music generation"
      }
    ])
  })

  it("lists projects with workflow filters encoded in the query string", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      projects: [
        {
          project_id: "pod-1",
          title: "Interview",
          workflow: "podcast",
          status: "draft"
        }
      ],
      limit: 50,
      offset: 0
    })

    const projects = await listAudioStudioProjects({
      workflow: "podcast",
      includeArchived: false
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/projects?workflow=podcast&include_archived=false",
      method: "GET"
    })
    expect(projects).toEqual([
      {
        project_id: "pod-1",
        title: "Interview",
        workflow: "podcast",
        status: "draft"
      }
    ])
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
    mocks.bgRequest
      .mockResolvedValueOnce({
        section_id: "s1",
        workflow: "narration",
        title: "Intro",
        body_text: "Welcome",
        order_index: 0,
        settings: {},
        current_revision_id: "rev-section"
      })
      .mockResolvedValueOnce({
        track_id: "speech",
        name: "Speech",
        kind: "speech",
        order_index: 0,
        muted: false,
        solo: false,
        volume: 0.9,
        settings: {},
        current_revision_id: "rev-track"
      })
      .mockResolvedValueOnce({
        clip_id: "clip/a",
        track_id: "speech",
        section_id: "s1",
        title: "Intro clip",
        clip_type: "speech",
        start_ms: 0,
        duration_ms: 1000,
        volume: 1,
        fade_in_ms: 10,
        fade_out_ms: 20,
        muted: false,
        settings: {},
        current_revision_id: "rev-clip"
      })

    const section = await upsertAudioStudioSection("p1", "s1", {
      title: "Intro",
      body_text: "Welcome",
      speaker_id: "speaker-1",
      order_index: 0,
      base_revision_id: "rev-1",
      metadata: { source: "draft" }
    })
    const track = await upsertAudioStudioTrack("p1", "speech", {
      name: "Speech",
      kind: "speech",
      order_index: 0,
      base_revision_id: "rev-2",
      volume: 0.9,
      metadata: { role: "main" }
    })
    const clip = await upsertAudioStudioClip("p1", "clip/a", {
      track_id: "speech",
      section_id: "s1",
      title: "Intro clip",
      clip_type: "speech",
      start_ms: 0,
      duration_ms: 1000,
      fade_in_ms: 10,
      fade_out_ms: 20,
      muted: false,
      base_revision_id: "rev-3",
      metadata: { take: 1 }
    })

    expect(section).toMatchObject({
      section_id: "s1",
      current_revision_id: "rev-section"
    })
    expect(track).toMatchObject({
      track_id: "speech",
      current_revision_id: "rev-track"
    })
    expect(clip).toMatchObject({
      clip_id: "clip/a",
      current_revision_id: "rev-clip"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/audio-studio/projects/p1/sections/s1",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        title: "Intro",
        body_text: "Welcome",
        speaker_id: "speaker-1",
        order_index: 0,
        base_revision_id: "rev-1",
        metadata: { source: "draft" }
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/audio-studio/projects/p1/tracks/speech",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        name: "Speech",
        kind: "speech",
        order_index: 0,
        base_revision_id: "rev-2",
        volume: 0.9,
        metadata: { role: "main" }
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/audio-studio/projects/p1/clips/clip%2Fa",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        track_id: "speech",
        section_id: "s1",
        title: "Intro clip",
        clip_type: "speech",
        start_ms: 0,
        duration_ms: 1000,
        fade_in_ms: 10,
        fade_out_ms: 20,
        muted: false,
        base_revision_id: "rev-3",
        metadata: { take: 1 }
      }
    })
  })

  it("creates generation, render, export, and migration jobs", async () => {
    await createAudioStudioGeneration("p1", {
      kind: "music",
      provider: "ace_step",
      idempotency_key: "generation-key-1234",
      target_resource_kind: "track",
      target_resource_id: "track-1",
      target_revision_id: "rev-4",
      options: { prompt: "warm intro" }
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
        kind: "music",
        provider: "ace_step",
        idempotency_key: "generation-key-1234",
        target_resource_kind: "track",
        target_resource_id: "track-1",
        target_revision_id: "rev-4",
        options: { prompt: "warm intro" }
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
