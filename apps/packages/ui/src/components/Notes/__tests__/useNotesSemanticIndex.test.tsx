// @vitest-environment jsdom
import { NotesSemanticClientError } from "@/services/note-semantic-index"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, renderHook } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useNotesSemanticIndex } from "../hooks/useNotesSemanticIndex"

const mocks = vi.hoisted(() => ({
  capabilities: vi.fn(),
  status: vi.fn(),
  run: vi.fn(),
  enable: vi.fn(),
  createRun: vi.fn(),
  cancel: vi.fn(),
  deleteIndex: vi.fn(),
  command: vi.fn(() => ({ idempotencyKey: "stable-command-key" }))
}))

vi.mock("@/services/note-semantic-index", () => ({
  NotesSemanticClientError: class NotesSemanticClientError extends Error {
    status: number
    code: string
    constructor(status: number, code: string) {
      super(code)
      this.status = status
      this.code = code
    }
  },
  getNotesSemanticCapabilities: mocks.capabilities,
  getNotesSemanticStatus: mocks.status,
  getNotesSemanticRun: mocks.run,
  enableNotesSemanticIndex: mocks.enable,
  createNotesSemanticRun: mocks.createRun,
  cancelNotesSemanticRun: mocks.cancel,
  deleteNotesSemanticIndex: mocks.deleteIndex,
  createNotesSemanticCommand: mocks.command
}))

const capability = (manageAuthorized = true) => ({
  active_note_count: 12,
  estimated_chunk_count: 36,
  estimated_run_count: 2,
  provider_label: "OpenAI",
  model: "text-embedding-3-small",
  endpoint_display: "https://api.openai.com",
  execution_boundary: "external",
  storage_boundary: "local",
  storage_label: "ChromaDB",
  outbound_data_categories: ["note_content_chunks", "note_title"],
  capability_revision: `sha256:${"a".repeat(64)}`,
  indexing_available: true,
  unavailable_reason: null,
  metric: "cosine",
  resolved_dimensions: 1536,
  dimension_probe_required: false,
  renewal_requires_delete: false,
  manage_authorized: manageAuthorized
})

const semanticRun = (status = "processing", revision = 4) => ({
  run_id: "run-a",
  mode: "rebuild",
  status,
  revision,
  indexed_notes: status === "completed" ? 12 : 4,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: status === "completed" ? 0 : 8,
  published_chunks: status === "completed" ? 36 : 12,
  cleanup_complete: status === "completed",
  error_code: null,
  link: "/api/v1/notes/graph/semantic-index/runs/run-a"
})

const semanticStatus = (
  state = "updating",
  activeRun: ReturnType<typeof semanticRun> | null = semanticRun(),
  overrides: Record<string, unknown> = {}
) => ({
  state,
  detail_reason: state === "ready" ? null : "building",
  desired_state: state === "off" ? "disabled" : "enabled",
  configuration_revision: 7,
  semantic_index_revision: 3,
  active_generation_id: state === "off" ? null : "generation-a",
  active_generation_usable: state !== "off",
  indexed_notes: 4,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: 8,
  published_chunks: 12,
  cleanup_pending: false,
  active_run: activeRun,
  ...overrides
})

const wrapper = (client: QueryClient) =>
  function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }

const flush = async () => {
  for (let step = 0; step < 4; step += 1) {
    await act(async () => {
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(1)
    })
  }
}

describe("useNotesSemanticIndex", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.resetAllMocks()
    mocks.command.mockReturnValue({ idempotencyKey: "stable-command-key" })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it("scopes capability and status state by authority and preserves last-good data offline", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status.mockResolvedValue(semanticStatus("ready", null))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: Infinity } }
    })
    const { result, rerender } = renderHook(
      ({ authorityScope, isOnline }) =>
        useNotesSemanticIndex({
          authorityScope,
          enabled: true,
          isOnline,
          datasetId: "dataset-a",
          pollIntervalMs: 20
        }),
      {
        initialProps: { authorityScope: "authority-a", isOnline: true },
        wrapper: wrapper(client)
      }
    )
    await flush()

    expect(result.current.capabilities?.active_note_count).toBe(12)
    expect(result.current.status?.state).toBe("ready")
    expect(
      client
        .getQueryCache()
        .findAll()
        .map((query) => query.queryKey)
    ).toEqual(
      expect.arrayContaining([
        ["notes-semantic-index", "authority-a", "dataset-a", "capabilities"],
        ["notes-semantic-index", "authority-a", "dataset-a", "status"]
      ])
    )

    rerender({ authorityScope: "authority-a", isOnline: false })
    await flush()
    expect(result.current.isOffline).toBe(true)
    expect(result.current.status?.state).toBe("ready")
    expect(mocks.capabilities).toHaveBeenCalledTimes(1)
    expect(mocks.status).toHaveBeenCalledTimes(1)

    rerender({ authorityScope: "authority-b", isOnline: false })
    expect(result.current.capabilities).toBeNull()
    expect(result.current.status).toBeNull()
  })

  it("polls only an active domain run and stops after terminal publication", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status.mockResolvedValue(semanticStatus())
    mocks.run
      .mockResolvedValueOnce(semanticRun("processing", 4))
      .mockResolvedValueOnce(semanticRun("completed", 5))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidate = vi.spyOn(client, "invalidateQueries")
    const { result } = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          pollIntervalMs: 20
        }),
      { wrapper: wrapper(client) }
    )

    await flush()
    expect(result.current.activeRun?.status).toBe("processing")
    expect(mocks.run).toHaveBeenCalledWith({
      datasetId: undefined,
      runId: "run-a"
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20)
    })
    await flush()
    expect(result.current.activeRun).toBeNull()
    expect(result.current.lastTerminalRun?.status).toBe("completed")
    const callsAtTerminal = mocks.run.mock.calls.length

    await act(async () => {
      await vi.advanceTimersByTimeAsync(100)
    })
    expect(mocks.run).toHaveBeenCalledTimes(callsAtTerminal)
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["notes-graph-workspace", "authority-a"]
    })
  })

  it("binds commands to current revisions and proactively rejects revoked management", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status.mockResolvedValue(semanticStatus("off", null))
    mocks.enable.mockResolvedValue({
      resource: semanticStatus("preparing"),
      run: semanticRun("queued")
    })
    mocks.run.mockResolvedValue(semanticRun("queued"))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      ({ manageAuthorized }) => {
        mocks.capabilities.mockResolvedValue(capability(manageAuthorized))
        return useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          datasetId: "dataset-a"
        })
      },
      {
        initialProps: { manageAuthorized: true },
        wrapper: wrapper(client)
      }
    )
    await flush()

    await act(async () => {
      await hook.result.current.enable()
    })
    expect(mocks.enable).toHaveBeenCalledWith({
      datasetId: "dataset-a",
      expectedRevision: 7,
      capabilityRevision: `sha256:${"a".repeat(64)}`,
      idempotencyKey: "stable-command-key"
    })

    act(() => {
      client.setQueryData(
        ["notes-semantic-index", "authority-a", "dataset-a", "capabilities"],
        capability(false)
      )
    })
    await flush()
    await act(async () => {
      await expect(hook.result.current.rebuild()).rejects.toMatchObject({
        status: 403,
        code: "notes_semantic_permission_denied"
      })
    })
    expect(mocks.createRun).not.toHaveBeenCalled()
  })

  it("refreshes status on run admission and projects the admitted run counts", async () => {
    const admitted = {
      ...semanticRun("queued", 7),
      indexed_notes: 2,
      pending_notes: 10
    }
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status
      .mockResolvedValueOnce(semanticStatus("ready", null))
      .mockResolvedValue(
        semanticStatus("updating", admitted, {
          indexed_notes: 2,
          pending_notes: 10
        })
      )
    mocks.createRun.mockResolvedValue(admitted)
    mocks.run.mockResolvedValue(admitted)
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          datasetId: "dataset-a",
          pollIntervalMs: 100_000
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await hook.result.current.rebuild()
    })
    await flush()

    expect(mocks.status).toHaveBeenCalledTimes(2)
    expect(hook.result.current.status?.state).toBe("updating")
    expect(hook.result.current.activeRun).toMatchObject({
      indexed_notes: 2,
      pending_notes: 10
    })
  })

  it("reconciles configuration conflicts before retrying with refreshed request variables", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status
      .mockResolvedValueOnce(semanticStatus("ready", null))
      .mockResolvedValue(
        semanticStatus("ready", null, { configuration_revision: 8 })
      )
    mocks.createRun
      .mockRejectedValueOnce(
        new NotesSemanticClientError(
          409,
          "notes_semantic_configuration_revision_conflict"
        )
      )
      .mockResolvedValueOnce(semanticRun("queued", 8))
    mocks.run.mockResolvedValue(semanticRun("queued", 8))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          datasetId: "dataset-a",
          pollIntervalMs: 100_000
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await expect(hook.result.current.rebuild()).rejects.toMatchObject({
        code: "notes_semantic_configuration_revision_conflict"
      })
    })
    await flush()
    await act(async () => {
      await hook.result.current.rebuild()
    })

    expect(mocks.createRun.mock.calls.map(([input]) => input)).toEqual([
      expect.objectContaining({ expectedRevision: 7 }),
      expect.objectContaining({ expectedRevision: 8 })
    ])
  })

  it("reconciles capability conflicts before renewed consent uses fresh revisions", async () => {
    const refreshedCapability = {
      ...capability(),
      capability_revision: `sha256:${"b".repeat(64)}`
    }
    mocks.capabilities
      .mockResolvedValueOnce(capability())
      .mockResolvedValue(refreshedCapability)
    mocks.status
      .mockResolvedValueOnce(
        semanticStatus("needs_attention", null, {
          detail_reason: "stale_configuration"
        })
      )
      .mockResolvedValue(
        semanticStatus("needs_attention", null, {
          detail_reason: "stale_configuration",
          configuration_revision: 8
        })
      )
    mocks.enable
      .mockRejectedValueOnce(
        new NotesSemanticClientError(
          409,
          "notes_semantic_capability_revision_conflict"
        )
      )
      .mockResolvedValueOnce({
        resource: semanticStatus("updating", semanticRun("queued", 9)),
        run: semanticRun("queued", 9)
      })
    mocks.run.mockResolvedValue(semanticRun("queued", 9))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          datasetId: "dataset-a",
          pollIntervalMs: 100_000
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await expect(hook.result.current.enable()).rejects.toMatchObject({
        code: "notes_semantic_capability_revision_conflict"
      })
    })
    await flush()
    await act(async () => {
      await hook.result.current.enable()
    })

    expect(mocks.enable.mock.calls.map(([input]) => input)).toEqual([
      expect.objectContaining({
        expectedRevision: 7,
        capabilityRevision: `sha256:${"a".repeat(64)}`
      }),
      expect.objectContaining({
        expectedRevision: 8,
        capabilityRevision: `sha256:${"b".repeat(64)}`
      })
    ])
  })

  it("reconciles run conflicts before cancellation retries with the refreshed run revision", async () => {
    const refreshedRun = semanticRun("processing", 5)
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status.mockResolvedValue(semanticStatus())
    mocks.run
      .mockResolvedValueOnce(semanticRun("processing", 4))
      .mockResolvedValue(refreshedRun)
    mocks.cancel
      .mockRejectedValueOnce(
        new NotesSemanticClientError(
          409,
          "notes_semantic_run_revision_conflict"
        )
      )
      .mockResolvedValueOnce({
        resource: semanticStatus("updating", refreshedRun),
        run: refreshedRun
      })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          pollIntervalMs: 100_000
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await expect(hook.result.current.cancel()).rejects.toMatchObject({
        code: "notes_semantic_run_revision_conflict"
      })
    })
    await flush()
    await act(async () => {
      await hook.result.current.cancel()
    })

    expect(
      mocks.cancel.mock.calls.map(([input]) => input.expectedRevision)
    ).toEqual([4, 5])
  })

  it("reconciles writer conflicts before retrying with refreshed configuration", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status
      .mockResolvedValueOnce(semanticStatus("ready", null))
      .mockResolvedValue(
        semanticStatus("ready", null, { configuration_revision: 8 })
      )
    mocks.createRun
      .mockRejectedValueOnce(
        new NotesSemanticClientError(409, "notes_semantic_writer_conflict")
      )
      .mockResolvedValueOnce(semanticRun("queued", 8))
    mocks.run.mockResolvedValue(semanticRun("queued", 8))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          datasetId: "dataset-a",
          pollIntervalMs: 100_000
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    await act(async () => {
      await expect(hook.result.current.rebuild()).rejects.toMatchObject({
        code: "notes_semantic_writer_conflict"
      })
    })
    await flush()
    await act(async () => {
      await hook.result.current.rebuild()
    })

    expect(
      mocks.createRun.mock.calls.map(([input]) => input.expectedRevision)
    ).toEqual([7, 8])
  })

  it("reconciles a run missing on first read without projecting terminal history", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status.mockResolvedValue(semanticStatus())
    mocks.run.mockRejectedValue(
      new NotesSemanticClientError(404, "notes_semantic_run_not_found")
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          pollIntervalMs: 20
        }),
      { wrapper: wrapper(client) }
    )
    await flush()

    expect(hook.result.current.activeRun).toBeNull()
    expect(hook.result.current.lastTerminalRun).toBeNull()
    expect(mocks.status.mock.calls.length).toBeGreaterThan(1)
  })

  it("reconciles a later run 404 without retaining the previously polled run", async () => {
    mocks.capabilities.mockResolvedValue(capability())
    mocks.status.mockResolvedValue(semanticStatus())
    mocks.run
      .mockResolvedValueOnce(semanticRun("processing", 4))
      .mockRejectedValueOnce(
        new NotesSemanticClientError(404, "notes_semantic_run_not_found")
      )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const hook = renderHook(
      () =>
        useNotesSemanticIndex({
          authorityScope: "authority-a",
          enabled: true,
          isOnline: true,
          pollIntervalMs: 20
        }),
      { wrapper: wrapper(client) }
    )
    await flush()
    expect(hook.result.current.activeRun?.status).toBe("processing")

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20)
    })
    await flush()

    expect(hook.result.current.activeRun).toBeNull()
    expect(hook.result.current.lastTerminalRun).toBeNull()
    expect(mocks.status.mock.calls.length).toBeGreaterThan(1)
  })
})
