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
  fetchAudioStudioArtifactBlob,
  getAudioStudioArtifactMediaPath,
  listAudioStudioArtifacts,
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

  it("lists project artifacts through the encoded project endpoint", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      artifacts: [
        {
          artifact_id: "artifact-1",
          artifact_type: "speech",
          provider: "kokoro",
          mime_type: "audio/wav",
          size_bytes: 123,
          source_resource_kind: "section",
          source_resource_id: "section-1",
          source_revision_id: "revision-1",
          metadata: { take: 1 },
          created_at: "2026-01-01T00:00:00Z"
        }
      ],
      limit: 50,
      offset: 0
    })

    const artifacts = await listAudioStudioArtifacts("project 1")

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/projects/project%201/artifacts",
      method: "GET"
    })
    expect(artifacts).toEqual([
      {
        artifact_id: "artifact-1",
        artifact_type: "speech",
        provider: "kokoro",
        mime_type: "audio/wav",
        size_bytes: 123,
        source_resource_kind: "section",
        source_resource_id: "section-1",
        source_revision_id: "revision-1",
        metadata: { take: 1 },
        created_at: "2026-01-01T00:00:00Z"
      }
    ])
  })

  it("builds encoded artifact media paths", () => {
    expect(getAudioStudioArtifactMediaPath("p 1", "a/1")).toBe(
      "/api/v1/audio-studio/projects/p%201/artifacts/a%2F1/media"
    )
  })

  it("adds a download flag only when requested for artifact media paths", () => {
    expect(
      getAudioStudioArtifactMediaPath("p1", "a1", { download: true })
    ).toBe("/api/v1/audio-studio/projects/p1/artifacts/a1/media?download=true")
  })

  it("fetches artifact media as a blob using the response MIME type", async () => {
    const buffer = new Uint8Array([1, 2, 3]).buffer
    mocks.bgRequest.mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: buffer,
      headers: { "content-type": "audio/wav" }
    })

    const blob = await fetchAudioStudioArtifactBlob("p1", {
      artifact_id: "a1",
      mime_type: "audio/mpeg"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/projects/p1/artifacts/a1/media",
      method: "GET",
      responseType: "arrayBuffer",
      returnResponse: true
    })
    expect(blob).toBeInstanceOf(Blob)
    expect(blob.type).toBe("audio/wav")
    expect(blob.size).toBe(3)
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
    await previewAudiobookMigration({
      legacy_project_id: "legacy-1",
      project_payload: { legacy_project_id: "legacy-1" }
    })
    await commitAudiobookMigration({
      legacy_project_id: "legacy-1",
      project_payload: { legacy_project_id: "legacy-1" },
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
      body: {
        legacy_project_id: "legacy-1",
        project_payload: { legacy_project_id: "legacy-1" }
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(5, {
      path: "/api/v1/audio-studio/migrations/audiobook/commit",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        legacy_project_id: "legacy-1",
        project_payload: { legacy_project_id: "legacy-1" },
        idempotency_key: "migration-key-1234"
      }
    })
  })
})
