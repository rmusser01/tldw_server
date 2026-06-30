import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

const createClient = () => {
  const client = new TldwApiClient()
  ;(client as any).ensureConfigForRequest = vi.fn(async () => ({ ok: true }))
  return client
}

describe("TldwApiClient ingestion sources contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("lists ingestion sources from array responses and normalizes ids/counts", async () => {
    mocks.bgRequest.mockResolvedValueOnce([
      {
        id: 7,
        user_id: 3,
        source_type: "archive_snapshot",
        sink_type: "notes",
        policy: "canonical",
        enabled: true,
        schedule_enabled: false,
        schedule_config: { interval_hours: 12 },
        config: { label: "Exported notes" },
        last_successful_snapshot_id: 9,
        last_sync_status: "completed",
        last_successful_sync_summary: {
          changed_count: 2,
          degraded_count: 1,
          conflict_count: 0
        }
      }
    ])

    const client = createClient()
    const result = await client.listIngestionSources()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/ingestion-sources",
        method: "GET"
      })
    )
    expect(result.total).toBe(1)
    expect(result.sources[0]?.id).toBe("7")
    expect(result.sources[0]?.last_successful_snapshot_id).toBe("9")
    expect(result.sources[0]?.last_successful_sync_summary?.degraded_count).toBe(1)
  })

  it("browses ingestion source directories with normalized path entries", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      roots: [{ name: "imports", path: "/srv/tldw/imports", is_root: true }],
      current_path: "/srv/tldw/imports",
      parent_path: null,
      entries: [
        { name: "notes", path: "/srv/tldw/imports/notes", is_root: false },
        { name: "media", path: "/srv/tldw/imports/media" }
      ],
      error: null
    })

    const client = createClient()
    const result = await client.browseIngestionSourceDirectories("/srv/tldw/imports")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/ingestion-sources/browse-directories?path=%2Fsrv%2Ftldw%2Fimports",
        method: "GET"
      })
    )
    expect(result.current_path).toBe("/srv/tldw/imports")
    expect(result.entries).toEqual([
      { name: "notes", path: "/srv/tldw/imports/notes", is_root: false },
      { name: "media", path: "/srv/tldw/imports/media", is_root: false }
    ])
  })

  it("fetches source detail and item lists with normalized ids", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      id: 7,
      user_id: 3,
      source_type: "local_directory",
      sink_type: "media",
      policy: "import_only",
      enabled: true,
      schedule_enabled: true,
      schedule_config: { interval_hours: 6 },
      config: { path: "/srv/tldw/imports" },
      active_job_id: 19,
      last_sync_status: "running",
      last_successful_sync_summary: {
        changed_count: 4,
        degraded_count: 0,
        conflict_count: 0
      }
    })
    mocks.bgRequest.mockResolvedValueOnce([
      {
        id: 41,
        source_id: 7,
        normalized_relative_path: "docs/foo.md",
        content_hash: "abc123",
        sync_status: "conflict_detached",
        binding: { note_id: 88 },
        present_in_source: true
      }
    ])

    const client = createClient()
    const source = await client.getIngestionSource("7")
    const items = await client.listIngestionSourceItems("7", {
      sync_status: "conflict_detached",
      present_in_source: true
    })

    expect(source.id).toBe("7")
    expect(source.active_job_id).toBe("19")
    expect(items.total).toBe(1)
    expect(items.items[0]?.id).toBe("41")
    expect(items.items[0]?.source_id).toBe("7")
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/ingestion-sources/7/items?sync_status=conflict_detached&present_in_source=true",
        method: "GET"
      })
    )
  })

  it("preserves git repository as a recognized source type", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      id: 8,
      user_id: 3,
      source_type: "git_repository",
      sink_type: "notes",
      policy: "import_only",
      enabled: true,
      schedule_enabled: false,
      schedule_config: {},
      config: { repo_url: "https://github.com/example/repo" }
    })

    const client = createClient()
    const source = await client.getIngestionSource("8")

    expect(source.source_type).toBe("git_repository")
  })

  it("creates and updates ingestion sources with guarded JSON requests", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      id: 7,
      user_id: 3,
      source_type: "local_directory",
      sink_type: "notes",
      policy: "canonical",
      enabled: true,
      schedule_enabled: false,
      schedule_config: {},
      config: { path: "/srv/tldw/notes" }
    })
    mocks.bgRequest.mockResolvedValueOnce({
      id: 7,
      user_id: 3,
      source_type: "local_directory",
      sink_type: "notes",
      policy: "import_only",
      enabled: false,
      schedule_enabled: false,
      schedule_config: {},
      config: { path: "/srv/tldw/notes" }
    })

    const client = createClient()
    const createPayload = {
      source_type: "local_directory" as const,
      sink_type: "notes" as const,
      policy: "canonical" as const,
      enabled: true,
      schedule_enabled: false,
      schedule: {},
      config: { path: "/srv/tldw/notes" }
    }
    const updatePayload = {
      enabled: false,
      policy: "import_only" as const
    }

    const created = await client.createIngestionSource(createPayload)
    const updated = await client.updateIngestionSource("7", updatePayload)

    expect(created.id).toBe("7")
    expect(updated.enabled).toBe(false)
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/ingestion-sources",
        method: "POST",
        body: createPayload
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/ingestion-sources/7",
        method: "PATCH",
        body: updatePayload
      })
    )
  })

  it("syncs sources, uploads archives, and reattaches detached items", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      status: "queued",
      source_id: 7,
      job_id: 12
    })
    mocks.bgUpload.mockResolvedValueOnce({
      status: "queued",
      source_id: 7,
      job_id: 13,
      snapshot_status: "staged"
    })
    mocks.bgRequest.mockResolvedValueOnce({
      id: 41,
      source_id: 7,
      normalized_relative_path: "docs/foo.md",
      content_hash: "abc123",
      sync_status: "sync_managed",
      binding: { note_id: 88 },
      present_in_source: true
    })

    const client = createClient()
    const archiveBuffer = new Uint8Array([1, 2, 3]).buffer
    const file = {
      name: "notes.tar.gz",
      type: "application/gzip",
      arrayBuffer: vi.fn(async () => archiveBuffer)
    } as unknown as File

    const syncResult = await client.syncIngestionSource("7")
    const uploadResult = await client.uploadIngestionSourceArchive("7", file)
    const reattachedItem = await client.reattachIngestionSourceItem("7", "41")

    expect(syncResult).toEqual({
      status: "queued",
      source_id: "7",
      job_id: "12",
      snapshot_status: null
    })
    expect(uploadResult).toEqual({
      status: "queued",
      source_id: "7",
      job_id: "13",
      snapshot_status: "staged"
    })
    expect(reattachedItem.id).toBe("41")
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/ingestion-sources/7/sync",
        method: "POST"
      })
    )
    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/ingestion-sources/7/archive",
        method: "POST",
        fileFieldName: "archive",
        file: expect.objectContaining({
          name: "notes.tar.gz",
          type: "application/gzip",
          data: archiveBuffer
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/ingestion-sources/7/items/41/reattach",
        method: "POST"
      })
    )
  })
})
