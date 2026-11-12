import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

const tldwClientMocks = vi.hoisted(() => ({
  getConfig: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => tldwClientMocks.getConfig(...args)
  }
}))

import {
  getAudioStudioVoices,
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
  mintAudioStudioArtifactMediaTicket,
  previewAudiobookMigration,
  updateAudioStudioProject,
  upsertAudioStudioClip,
  upsertAudioStudioSection,
  upsertAudioStudioTrack
} from "@/services/audio-studio"

describe("audio-studio service", () => {
  beforeEach(() => {
    vi.resetAllMocks()
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
      path: "/api/v1/audio-studio/projects/project%201/artifacts?limit=100&offset=0",
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

  it("mints playback media tickets and resolves ticket paths against the configured server", async () => {
    tldwClientMocks.getConfig.mockResolvedValueOnce({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.bgRequest.mockResolvedValueOnce({
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-playback",
      ticket_url: null,
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "playback",
      artifact_id: "artifact-1"
    })

    const ticket = await mintAudioStudioArtifactMediaTicket(
      "project 1",
      "artifact/1",
      "playback"
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/audio-studio/projects/project%201/artifacts/artifact%2F1/tickets",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { purpose: "playback" }
    })
    expect(ticket.ticket_url).toBe(
      "http://127.0.0.1:8000/api/v1/audio-studio/media-tickets/ticket-playback"
    )
    expect(ticket.ticket_path).toBe("/api/v1/audio-studio/media-tickets/ticket-playback")
  })

  it("prefers backend ticket_url when provided", async () => {
    tldwClientMocks.getConfig.mockResolvedValueOnce({
      serverUrl: "http://127.0.0.1:8000"
    })
    mocks.bgRequest.mockResolvedValueOnce({
      ticket_path: "/api/v1/audio-studio/media-tickets/ticket-download",
      ticket_url: "https://public.example.test/api/v1/audio-studio/media-tickets/ticket-download",
      expires_at: "2026-06-24T12:00:00Z",
      purpose: "download",
      artifact_id: "artifact-1"
    })

    const ticket = await mintAudioStudioArtifactMediaTicket(
      "project-1",
      "artifact-1",
      "download"
    )

    expect(ticket.ticket_url).toBe(
      "https://public.example.test/api/v1/audio-studio/media-tickets/ticket-download"
    )
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
      target_resource_kind: "render", target_resource_id: "mix-1", target_revision_id: "rev-4", render_type: "preview_mix"
    })
    await createAudioStudioExport("p1", {
      idempotency_key: "export-key-1234",
      target_resource_kind: "export", target_resource_id: "export-1", target_revision_id: "rev-4", export_type: "zip_package"
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
        target_resource_kind: "render", target_resource_id: "mix-1", target_revision_id: "rev-4", render_type: "preview_mix"
      }
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/audio-studio/projects/p1/exports",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        idempotency_key: "export-key-1234",
        target_resource_kind: "export", target_resource_id: "export-1", target_revision_id: "rev-4", export_type: "zip_package"
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
  it("B06 captured owner fences a deferred project response", async () => {
    let finish!: (value: unknown) => void
    mocks.bgRequest.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    let current = true
    const options = { assertCurrent: () => { if (!current) throw new Error("Audio Studio account changed") } }
    const pending = createAudioStudioProject({ title: "Private", workflow: "narration" }, options)
    current = false
    finish({ project_id: "private-project" })
    await expect(pending).rejects.toThrow("Audio Studio account changed")
  })

  it("B06 lists only compatible owned custom voices when the provider has no builtin catalog", async () => {
    mocks.bgRequest.mockImplementation(async (request) => {
      if (request.path === "/api/v1/audio/voices") return { voices: [
        { voice_id: "owned-pocket", name: "Reference", provider: "PocketTTS" },
        { voice_id: "other-provider", name: "Other", provider: "kokoro" }
      ] };
      throw Object.assign(new Error("No builtin catalog"), { status: 404 });
    });
    expect(await getAudioStudioVoices("pocket_tts", "pocket_tts")).toEqual([
      expect.objectContaining({ voice_id: "custom:owned-pocket", name: "Custom: Reference" })
    ]);
  });

  it("B06 fences retained custom voices when the account changes during builtin404", async () => {
    let rejectCatalog!: (error: unknown) => void;
    mocks.bgRequest.mockResolvedValueOnce({ voices: [{ voice_id: "owned-pocket", provider: "pocket_tts" }] });
    mocks.bgRequest.mockReturnValueOnce(new Promise((_, reject) => { rejectCatalog = reject; }));
    let current = true;
    const pending = getAudioStudioVoices("pocket_tts", "pocket_tts", {
      assertCurrent: () => { if (!current) throw new Error("Audio Studio account changed"); }
    });
    await vi.waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(2));
    current = false;
    rejectCatalog(Object.assign(new Error("No builtin catalog"), { status: 404 }));
    await expect(pending).rejects.toThrow("Audio Studio account changed");
  });

  it.each([401, 403, 500])("B06 preserves voice catalog failure %s instead of using a stale sample", async (status) => {
    mocks.bgRequest.mockResolvedValueOnce({ voices: [{ voice_id: "owned-pocket", name: "Reference", provider: "pocket_tts" }] });
    mocks.bgRequest.mockRejectedValueOnce(Object.assign(new Error("Catalog failed"), { status }));
    await expect(getAudioStudioVoices("pocket_tts", "pocket_tts")).rejects.toMatchObject({ status });
  });

  it("B06 retains earlier audio across artifact pages", async () => {
    const first = Array.from({length: 100}, (_, index) => ({ artifact_id: `a-${index}` }))
    mocks.bgRequest.mockResolvedValueOnce({ artifacts: first, limit: 100, offset: 0 }).mockResolvedValueOnce({ artifacts: [{ artifact_id: "earlier-success" }], limit: 100, offset: 100 })
    const artifacts = await listAudioStudioArtifacts("p")
    expect(artifacts).toHaveLength(101)
    expect(artifacts[100].artifact_id).toBe("earlier-success")
    expect(mocks.bgRequest.mock.calls.map(([request]) => request.path)).toEqual(["/api/v1/audio-studio/projects/p/artifacts?limit=100&offset=0", "/api/v1/audio-studio/projects/p/artifacts?limit=100&offset=100"])
  })

})
