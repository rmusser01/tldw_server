import {
  NotesSemanticClientError,
  cancelNotesSemanticRun,
  createNotesSemanticCommand,
  createNotesSemanticRun,
  deleteNotesSemanticIndex,
  enableNotesSemanticIndex,
  getNotesSemanticCapabilities,
  getNotesSemanticRun,
  getNotesSemanticStatus
} from "@/services/note-semantic-index"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({ bgRequest: vi.fn() }))

vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))

const run = (status = "queued") => ({
  run_id: "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
  mode: "rebuild",
  status,
  revision: 9,
  indexed_notes: 1,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: 3,
  published_chunks: 2,
  cleanup_complete: false,
  error_code: null,
  link: "/api/v1/notes/graph/semantic-index/runs/6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
})

const status = (state = "ready") => ({
  state,
  detail_reason: null,
  desired_state: state === "off" ? "disabled" : "enabled",
  configuration_revision: 9,
  semantic_index_revision: 2,
  active_generation_id: state === "off" ? null : "generation-a",
  indexed_notes: 4,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: 0,
  published_chunks: 8,
  cleanup_pending: false,
  active_run: null
})

const capabilities = () => ({
  active_note_count: 4,
  estimated_chunk_count: 8,
  estimated_run_count: 1,
  provider_label: "OpenAI",
  model: "text-embedding-3-small",
  execution_boundary: "external",
  storage_boundary: "local",
  storage_label: "ChromaDB",
  outbound_data_categories: ["note_content_chunks", "note_title"],
  capability_revision: `sha256:${"a".repeat(64)}`,
  indexing_available: true,
  unavailable_reason: null,
  metric: "cosine",
  resolved_dimensions: 1536,
  manage_authorized: true
})

describe("Notes semantic index client", () => {
  beforeEach(() => vi.resetAllMocks())

  it("uses only the seven nested Notes semantic routes with dataset authority", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce(capabilities())
      .mockResolvedValueOnce(status())
      .mockResolvedValueOnce(run())
      .mockResolvedValueOnce({ resource: status("preparing"), run: run() })
      .mockResolvedValueOnce({
        resource: status("off"),
        run: run("processing")
      })
      .mockResolvedValueOnce(run("processing"))
      .mockResolvedValueOnce({
        resource: status("updating"),
        run: run("processing")
      })

    await getNotesSemanticCapabilities({ datasetId: "dataset / alpha" })
    await getNotesSemanticStatus({ datasetId: "dataset / alpha" })
    await createNotesSemanticRun({
      datasetId: "dataset / alpha",
      mode: "rebuild",
      expectedRevision: 9,
      idempotencyKey: "rebuild-key"
    })
    await enableNotesSemanticIndex({
      datasetId: "dataset / alpha",
      expectedRevision: 0,
      capabilityRevision: `sha256:${"a".repeat(64)}`,
      idempotencyKey: "enable-key"
    })
    await deleteNotesSemanticIndex({
      datasetId: "dataset / alpha",
      expectedRevision: 9,
      idempotencyKey: "delete-key"
    })
    await getNotesSemanticRun({
      datasetId: "dataset / alpha",
      runId: "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    })
    await cancelNotesSemanticRun({
      datasetId: "dataset / alpha",
      runId: "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
      expectedRevision: 9,
      idempotencyKey: "cancel-key"
    })

    const requests = mocks.bgRequest.mock.calls.map(([request]) => request)
    expect(requests.map((request) => request.path)).toEqual([
      "/api/v1/notes/graph/semantic-index/capabilities?dataset_id=dataset+%2F+alpha",
      "/api/v1/notes/graph/semantic-index?dataset_id=dataset+%2F+alpha",
      "/api/v1/notes/graph/semantic-index/runs?dataset_id=dataset+%2F+alpha",
      "/api/v1/notes/graph/semantic-index?dataset_id=dataset+%2F+alpha",
      "/api/v1/notes/graph/semantic-index?dataset_id=dataset+%2F+alpha",
      "/api/v1/notes/graph/semantic-index/runs/6ec1dfbe-f86f-4d2b-93af-f88f64cd9701?dataset_id=dataset+%2F+alpha",
      "/api/v1/notes/graph/semantic-index/runs/6ec1dfbe-f86f-4d2b-93af-f88f64cd9701/cancel?dataset_id=dataset+%2F+alpha"
    ])
    expect(
      requests.every((request) => !String(request.path).includes("/jobs"))
    ).toBe(true)
  })

  it("sends revision bodies and one idempotency header per command", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({ resource: status("preparing"), run: run() })
      .mockResolvedValueOnce(run())
      .mockResolvedValueOnce({
        resource: status("off"),
        run: run("processing")
      })

    await enableNotesSemanticIndex({
      expectedRevision: 0,
      capabilityRevision: `sha256:${"b".repeat(64)}`,
      idempotencyKey: "enable-key"
    })
    await createNotesSemanticRun({
      mode: "retry_failed",
      expectedRevision: 3,
      idempotencyKey: "retry-key"
    })
    await cancelNotesSemanticRun({
      runId: "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
      expectedRevision: 11,
      idempotencyKey: "cancel-key"
    })

    expect(mocks.bgRequest.mock.calls.map(([request]) => request.body)).toEqual(
      [
        {
          expected_revision: 0,
          capability_revision: `sha256:${"b".repeat(64)}`
        },
        { mode: "retry_failed", expected_revision: 3 },
        { expected_revision: 11 }
      ]
    )
    expect(
      mocks.bgRequest.mock.calls.map(
        ([request]) => request.headers["Idempotency-Key"]
      )
    ).toEqual(["enable-key", "retry-key", "cancel-key"])
  })

  it("creates a stable command key without exposing configuration overrides", () => {
    const command = createNotesSemanticCommand()
    expect(command.idempotencyKey).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
    )
    expect(command).toEqual({ idempotencyKey: command.idempotencyKey })
  })

  it("maps foreign runs, permission denial, and stale revisions to stable errors", async () => {
    mocks.bgRequest
      .mockRejectedValueOnce({
        status: 404,
        details: {
          detail: {
            error_code: "notes_semantic_run_not_found",
            message: "secret raw response"
          }
        }
      })
      .mockRejectedValueOnce({ status: 403, details: { detail: "forbidden" } })
      .mockRejectedValueOnce({
        status: 409,
        details: {
          detail: {
            error_code: "notes_semantic_configuration_revision_conflict"
          }
        }
      })

    await expect(
      getNotesSemanticRun({
        runId: "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
      })
    ).rejects.toMatchObject({
      status: 404,
      code: "notes_semantic_run_not_found"
    })
    await expect(
      deleteNotesSemanticIndex({
        expectedRevision: 9,
        idempotencyKey: "delete-key"
      })
    ).rejects.toMatchObject({
      status: 403,
      code: "notes_semantic_permission_denied"
    })
    await expect(
      createNotesSemanticRun({
        mode: "rebuild",
        expectedRevision: 9,
        idempotencyKey: "rebuild-key"
      })
    ).rejects.toMatchObject({
      status: 409,
      code: "notes_semantic_configuration_revision_conflict"
    })
  })

  it("fails closed on malformed capability, status, and run responses", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({ ...capabilities(), manage_authorized: "yes" })
      .mockResolvedValueOnce({ ...status(), configuration_revision: -1 })
      .mockResolvedValueOnce({ ...run(), status: "invented" })

    await expect(getNotesSemanticCapabilities({})).rejects.toBeInstanceOf(
      NotesSemanticClientError
    )
    await expect(getNotesSemanticStatus({})).rejects.toMatchObject({
      code: "notes_semantic_invalid_response"
    })
    await expect(
      getNotesSemanticRun({
        runId: "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
      })
    ).rejects.toMatchObject({ code: "notes_semantic_invalid_response" })
  })

  it.each([
    [
      "unknown outbound data",
      { outbound_data_categories: ["note_title", "raw_note_content"] }
    ],
    ["incomplete outbound data", { outbound_data_categories: ["note_title"] }],
    ["unavailable storage", { storage_boundary: "unavailable" }],
    [
      "an unavailable reason",
      { unavailable_reason: "notes_semantic_provider_unavailable" }
    ],
    ["unresolved dimensions", { resolved_dimensions: null }]
  ])(
    "rejects an available capability disclosure with %s",
    async (_label, override) => {
      mocks.bgRequest.mockResolvedValueOnce({
        ...capabilities(),
        ...override
      })

      await expect(getNotesSemanticCapabilities({})).rejects.toMatchObject({
        code: "notes_semantic_invalid_response"
      })
    }
  )

  it("accepts only the complete allowlisted outbound category set", async () => {
    mocks.bgRequest.mockResolvedValueOnce(capabilities())

    const disclosure = await getNotesSemanticCapabilities({})

    expect(disclosure.outbound_data_categories).toEqual([
      "note_content_chunks",
      "note_title"
    ])
  })
})
