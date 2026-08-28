import {
  type CancelNotesGraphSuggestionRunInput,
  type CreateNotesGraphSuggestionCommandInput,
  type DecideNotesGraphSuggestionInput,
  type GetNotesGraphSuggestionCapabilitiesInput,
  type GetNotesGraphSuggestionRunInput,
  type ListNotesGraphSuggestionRunsInput,
  type ListNotesGraphSuggestionsInput,
  NotesGraphSuggestionClientError,
  type ResetNotesGraphSuggestionRejectionsInput,
  acceptNotesGraphSuggestion,
  cancelNotesGraphSuggestionRun,
  createNotesGraphSuggestionCommand,
  createNotesGraphSuggestionRun,
  fetchNotesGraph,
  getNotesGraphSuggestionCapabilities,
  getNotesGraphSuggestionRun,
  isNotesGraphCapabilitiesChangedError,
  listNotesGraphSuggestionRuns,
  listNotesGraphSuggestions,
  rejectNotesGraphSuggestion,
  resetNotesGraphSuggestionRejections
} from "@/services/note-graph-suggestions"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

const fingerprint = (value: string) => `sha256:${value.repeat(64).slice(0, 64)}`

const capabilityPayload = (revision = fingerprint("a")) => ({
  provider: "configured provider",
  model: "model/one",
  endpoint_origin_revision: fingerprint("b"),
  data_boundary: "remote",
  disclosure_external: true,
  outbound_data_categories: ["selected_note_excerpt"],
  generation_available: true,
  unavailable_reason: null,
  limits: {
    max_candidates: 30,
    max_relationships: 5,
    max_tags: 5,
    max_new_tags: 2,
    max_tag_catalog: 100,
    max_estimated_input_tokens: 24000,
    max_output_tokens: 2000,
    provider_timeout_seconds: 120,
    response_candidates: 1
  },
  allowed_actions: ["generate", "accept", "reject"],
  revision
})

const runPayload = (id = "run/one", state = "queued") => ({
  id,
  provider: "configured provider",
  model: "model/one",
  state,
  revision: 3,
  created_at: "2026-08-27T12:00:00Z",
  started_at: null,
  completed_at: null,
  suggestion_count: 0,
  related_note_count: 0,
  tag_count: 0,
  invalid_item_count: 0,
  cancellation_available: true,
  error_code: null,
  guidance_key: null
})

const suggestionPayload = (id: string) => ({
  id,
  run_id: "run/one",
  kind: "related_note",
  state: "pending",
  revision: 2,
  source_note_id: "note/source",
  source_fingerprint: fingerprint("c"),
  target_note_id: "note/target",
  target_fingerprint: fingerprint("d"),
  normalized_tag: null,
  display_tag: null,
  existing_tag: false,
  match_strength: "strong",
  rationale: "R".repeat(500),
  evidence: Array.from({ length: 8 }, (_, index) => ({
    side: index % 2 ? "target" : "source",
    note_id: index % 2 ? "note/target" : "note/source",
    field: "content",
    start_offset: index,
    end_offset: index + 1,
    text: "E".repeat(700)
  })),
  updated_at: "2026-08-27T12:00:00Z"
})

describe("Notes graph suggestion client", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses every nested route with encoded path and query values", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { etag: `"${fingerprint("a")}"` },
        data: capabilityPayload()
      })
      .mockResolvedValueOnce({
        items: [runPayload()],
        next_cursor: "run cursor"
      })
      .mockResolvedValueOnce(runPayload())
      .mockResolvedValueOnce({
        items: [suggestionPayload("suggestion/one")],
        next_cursor: "suggestion cursor",
        current_source_fingerprint: fingerprint("c"),
        rejection_set_revision: 4,
        rejection_count: 1
      })
      .mockResolvedValue({
        resource_id: "resource",
        state: "completed",
        revision: 4
      })

    await getNotesGraphSuggestionCapabilities({
      noteId: "note / one",
      datasetId: "dataset / alpha",
      provider: "configured provider",
      model: "model/one"
    })
    await listNotesGraphSuggestionRuns({
      noteId: "note / one",
      datasetId: "dataset / alpha",
      states: ["queued", "running"],
      limit: 200,
      cursor: "run cursor"
    })
    await getNotesGraphSuggestionRun({
      noteId: "note / one",
      runId: "run / one",
      datasetId: "dataset / alpha"
    })
    await listNotesGraphSuggestions({
      noteId: "note / one",
      datasetId: "dataset / alpha",
      states: ["pending", "accepting"],
      limit: 200,
      cursor: "suggestion cursor"
    })
    await cancelNotesGraphSuggestionRun({
      noteId: "note / one",
      runId: "run / one",
      datasetId: "dataset / alpha",
      expectedRevision: 3,
      idempotencyKey: "cancel-key"
    })
    await resetNotesGraphSuggestionRejections({
      noteId: "note / one",
      datasetId: "dataset / alpha",
      expectedRejectionRevision: 4,
      sourceFingerprint: fingerprint("c"),
      idempotencyKey: "reset-key"
    })
    await acceptNotesGraphSuggestion({
      noteId: "note / one",
      suggestionId: "suggestion / one",
      datasetId: "dataset / alpha",
      expectedRevision: 2,
      expectedSourceFingerprint: fingerprint("c"),
      expectedTargetFingerprint: fingerprint("d"),
      idempotencyKey: "accept-key"
    })
    await rejectNotesGraphSuggestion({
      noteId: "note / one",
      suggestionId: "suggestion / one",
      datasetId: "dataset / alpha",
      expectedRevision: 2,
      expectedSourceFingerprint: fingerprint("c"),
      expectedTargetFingerprint: fingerprint("d"),
      idempotencyKey: "reject-key"
    })

    expect(mocks.bgRequest.mock.calls.map(([request]) => request.path)).toEqual(
      [
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/capabilities?provider=configured+provider&model=model%2Fone&dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/runs?state=queued%2Crunning&limit=100&cursor=run+cursor&dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/runs/run%20%2F%20one?dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions?state=pending%2Caccepting&limit=100&cursor=suggestion+cursor&dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/runs/run%20%2F%20one/cancel?dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/rejections/reset?dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/suggestion%20%2F%20one/accept?dataset_id=dataset+%2F+alpha",
        "/api/v1/notes/note%20%2F%20one/graph/suggestions/suggestion%20%2F%20one/reject?dataset_id=dataset+%2F+alpha"
      ]
    )
  })

  it("extracts a strict ETag and retains one idempotency key across a 412 retry", async () => {
    const firstRevision = fingerprint("a")
    const secondRevision = fingerprint("e")
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { ETag: `"${firstRevision}"` },
        data: capabilityPayload(firstRevision)
      })
      .mockRejectedValueOnce({
        status: 412,
        details: {
          detail: {
            error_code: "notes_graph_capabilities_changed",
            message: "raw provider details must not escape"
          }
        }
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { etag: `"${secondRevision}"` },
        data: capabilityPayload(secondRevision)
      })
      .mockResolvedValueOnce(runPayload())

    const capability = await getNotesGraphSuggestionCapabilities({
      noteId: "note-1"
    })
    const command = createNotesGraphSuggestionCommand({ noteId: "note-1" })
    const result = await createNotesGraphSuggestionRun(command, capability)

    expect(result.id).toBe("run/one")
    const postCalls = mocks.bgRequest.mock.calls.filter(
      ([request]) => request.method === "POST"
    )
    expect(postCalls).toHaveLength(2)
    expect(postCalls[0][0].path).toBe(
      "/api/v1/notes/note-1/graph/suggestions/runs"
    )
    expect(postCalls[0][0].headers).toEqual({
      "Content-Type": "application/json",
      "Idempotency-Key": command.idempotencyKey,
      "If-Match": `"${firstRevision}"`
    })
    expect(postCalls[1][0].headers).toEqual({
      "Content-Type": "application/json",
      "Idempotency-Key": command.idempotencyKey,
      "If-Match": `"${secondRevision}"`
    })
    expect(postCalls[0][0].body).toEqual({
      provider: "configured provider",
      model: "model/one"
    })
    expect(postCalls[1][0].body).toEqual(postCalls[0][0].body)
  })

  it("re-resolves the configured default after a capability 412", async () => {
    const firstRevision = fingerprint("a")
    const secondRevision = fingerprint("e")
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { etag: `"${firstRevision}"` },
        data: capabilityPayload(firstRevision)
      })
      .mockRejectedValueOnce({
        status: 412,
        details: {
          detail: { error_code: "notes_graph_capabilities_changed" }
        }
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { etag: `"${secondRevision}"` },
        data: {
          ...capabilityPayload(secondRevision),
          provider: "new configured provider",
          model: "model/two"
        }
      })
      .mockResolvedValueOnce(runPayload("run/new-default", "queued"))

    const capability = await getNotesGraphSuggestionCapabilities({
      noteId: "note-1"
    })
    const command = createNotesGraphSuggestionCommand({ noteId: "note-1" })
    await createNotesGraphSuggestionRun(command, capability)

    expect(mocks.bgRequest.mock.calls[2][0].path).toBe(
      "/api/v1/notes/note-1/graph/suggestions/capabilities"
    )
    expect(mocks.bgRequest.mock.calls[3][0].body).toEqual({
      provider: "new configured provider",
      model: "model/two"
    })
    expect(mocks.bgRequest.mock.calls[3][0].headers["Idempotency-Key"]).toBe(
      command.idempotencyKey
    )
  })

  it.each([
    ["missing", undefined],
    ["unquoted", fingerprint("a")],
    ["weak", `W/"${fingerprint("a")}"`],
    ["mismatched", `"${fingerprint("b")}"`]
  ])("rejects a %s capability ETag", async (_label, etag) => {
    mocks.bgRequest.mockResolvedValueOnce({
      ok: true,
      status: 200,
      headers: etag ? { etag } : {},
      data: capabilityPayload(fingerprint("a"))
    })

    await expect(
      getNotesGraphSuggestionCapabilities({ noteId: "note-1" })
    ).rejects.toMatchObject({
      code: "notes_graph_invalid_response"
    })
  })

  it("bounds graph and suggestion normalization to authoritative response limits", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        nodes: Array.from({ length: 12 }, (_, index) => ({
          id: `note:${index}`,
          type: "note",
          label: `Note ${index}`
        })),
        edges: Array.from({ length: 12 }, (_, index) => ({
          id: `edge:${index}`,
          source: "note:0",
          target: `note:${index + 1}`,
          type: "manual",
          directed: false
        })),
        truncated: true,
        truncated_by: ["nodes"],
        has_more: true,
        cursor: "next graph cursor",
        limits: { max_nodes: 3, max_edges: 4, max_degree: 2 },
        radius_cap_applied: false,
        active_note_count: 12,
        all_notes_note_cap: 3,
        all_notes_eligible: false
      })
      .mockResolvedValueOnce({
        items: Array.from({ length: 140 }, (_, index) =>
          suggestionPayload(`suggestion-${index}`)
        ),
        next_cursor: "next suggestion cursor",
        current_source_fingerprint: fingerprint("c"),
        rejection_set_revision: 4,
        rejection_count: 1
      })

    const graph = await fetchNotesGraph({
      centerNoteId: "note/source",
      radius: 1,
      maxNodes: 9999,
      maxEdges: 99999,
      edgeTypes: ["manual", "tag_membership"]
    })
    const suggestions = await listNotesGraphSuggestions({
      noteId: "note/source",
      limit: 999
    })

    expect(graph.nodes).toHaveLength(3)
    expect(graph.edges).toHaveLength(4)
    expect(graph.limits).toEqual({ max_nodes: 3, max_edges: 4, max_degree: 2 })
    expect(suggestions.items).toHaveLength(100)
    expect(suggestions.items[0].rationale).toHaveLength(240)
    expect(suggestions.items[0].evidence).toHaveLength(4)
    expect(suggestions.items[0].evidence[0].text).toHaveLength(480)
    expect(String(mocks.bgRequest.mock.calls[0][0].path)).toContain(
      "max_nodes=2000"
    )
    expect(String(mocks.bgRequest.mock.calls[0][0].path)).toContain(
      "max_edges=8000"
    )
  })

  it("maps raw failures to the stable sanitized error contract", async () => {
    mocks.bgRequest.mockRejectedValueOnce({
      status: 503,
      message: "provider https://secret.example failed with API key sk-secret",
      details: {
        detail: {
          error_code: "unrecognized_provider_failure",
          message: "SQL and provider response"
        },
        endpoint_url: "https://secret.example",
        credentials: "sk-secret"
      }
    })

    const error = await listNotesGraphSuggestionRuns({
      noteId: "note-1"
    }).catch((value) => value)

    expect(error).toBeInstanceOf(NotesGraphSuggestionClientError)
    expect(error).toMatchObject({
      status: 503,
      code: "notes_graph_suggestions_unavailable",
      message: "Notes graph suggestions are temporarily unavailable."
    })
    expect(JSON.stringify(error)).not.toContain("secret")
    expect(isNotesGraphCapabilitiesChangedError(error)).toBe(false)
  })

  it("preserves allowlisted status codes and drops unsafe capability and run text", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { etag: `"${fingerprint("a")}"` },
        data: {
          ...capabilityPayload(),
          generation_available: false,
          unavailable_reason:
            "provider https://secret.example failed with API key sk-secret"
        }
      })
      .mockResolvedValueOnce({
        items: [
          {
            ...runPayload(),
            error_code: "job-42 provider response: sk-secret",
            guidance_key: "open https://secret.example"
          }
        ],
        next_cursor: null
      })
      .mockRejectedValueOnce({
        status: 503,
        details: {
          detail: {
            error_code: "notes_graph_provider_not_configured",
            message: "provider configuration contained sk-secret"
          }
        }
      })

    const capabilities = await getNotesGraphSuggestionCapabilities({
      noteId: "note-1"
    })
    const runs = await listNotesGraphSuggestionRuns({ noteId: "note-1" })
    const error = await listNotesGraphSuggestionRuns({
      noteId: "note-1"
    }).catch((value) => value)

    expect(capabilities.unavailable_reason).toBeNull()
    expect(runs.items[0]).toMatchObject({
      error_code: null,
      guidance_key: null
    })
    expect(error).toMatchObject({
      code: "notes_graph_provider_not_configured",
      message: "The selected provider is not configured."
    })
    expect(JSON.stringify({ capabilities, runs, error })).not.toContain(
      "secret"
    )
  })

  it("recognizes only the stable 412 capability revision error", () => {
    expect(
      isNotesGraphCapabilitiesChangedError(
        new NotesGraphSuggestionClientError(
          412,
          "notes_graph_capabilities_changed",
          "Suggestion capabilities changed; refresh and retry."
        )
      )
    ).toBe(true)
    expect(
      isNotesGraphCapabilitiesChangedError(
        new NotesGraphSuggestionClientError(
          412,
          "notes_graph_suggestion_conflict",
          "The suggestion changed; refresh and retry."
        )
      )
    ).toBe(false)
  })

  it("keeps public run commands free of endpoint and credential fields", () => {
    type ForbiddenField =
      | "endpoint"
      | "endpointUrl"
      | "baseUrl"
      | "apiKey"
      | "credential"
      | "credentials"
      | "candidateIds"
    type IsClosed<T> =
      Extract<keyof T, ForbiddenField> extends never ? true : false
    const closedRequestTypes: [
      IsClosed<CreateNotesGraphSuggestionCommandInput>,
      IsClosed<GetNotesGraphSuggestionCapabilitiesInput>,
      IsClosed<ListNotesGraphSuggestionRunsInput>,
      IsClosed<GetNotesGraphSuggestionRunInput>,
      IsClosed<CancelNotesGraphSuggestionRunInput>,
      IsClosed<ListNotesGraphSuggestionsInput>,
      IsClosed<DecideNotesGraphSuggestionInput>,
      IsClosed<ResetNotesGraphSuggestionRejectionsInput>
    ] = [true, true, true, true, true, true, true, true]

    expect(closedRequestTypes.every(Boolean)).toBe(true)
    const command = createNotesGraphSuggestionCommand({
      noteId: "note-1",
      provider: "configured provider",
      model: "model/one"
    })
    expect(command).toMatchObject({
      noteId: "note-1",
      provider: "configured provider",
      model: "model/one",
      idempotencyKey: expect.any(String)
    })
    expect(command.idempotencyKey).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    )
  })
})
