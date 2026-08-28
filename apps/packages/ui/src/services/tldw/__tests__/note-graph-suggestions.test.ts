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
  outbound_data_categories: ["selected_note_excerpts"],
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
  allowed_actions: [
    "generate",
    "cancel",
    "accept",
    "reject",
    "reset_rejections"
  ],
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
  rationale: "Related concept",
  evidence: Array.from({ length: 6 }, (_, index) => ({
    side: index % 2 ? "target" : "source",
    note_id: index % 2 ? "note/target" : "note/source",
    field: "content",
    start_offset: index,
    end_offset: index + 1,
    text: `Evidence ${index}`
  })),
  updated_at: "2026-08-27T12:00:00Z"
})

describe("Notes graph suggestion client", () => {
  beforeEach(() => {
    vi.resetAllMocks()
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
        revision: 4,
        cleared_count: null
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
      limit: 100,
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
      limit: 100,
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

  it("does not start service-owned 412 recovery after retry authority is revoked", async () => {
    const revision = fingerprint("a")
    mocks.bgRequest
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        headers: { etag: `"${revision}"` },
        data: capabilityPayload(revision)
      })
      .mockRejectedValueOnce({
        status: 412,
        details: {
          detail: { error_code: "notes_graph_capabilities_changed" }
        }
      })

    const capability = await getNotesGraphSuggestionCapabilities({
      noteId: "note-1"
    })
    const command = createNotesGraphSuggestionCommand({ noteId: "note-1" })

    await expect(
      createNotesGraphSuggestionRun(command, capability, {
        canRetry: () => false
      })
    ).rejects.toMatchObject({
      status: 412,
      code: "notes_graph_capabilities_changed"
    })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    expect(
      mocks.bgRequest.mock.calls.filter(
        ([request]) => request.method === "POST"
      )
    ).toHaveLength(1)
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

  it("preserves valid Task 8 response fields without rewriting them", async () => {
    const longLabel = "Long graph label ".repeat(400)
    const longId = `note:${"x".repeat(5000)}`
    const rationale = "😀".repeat(240)
    const evidenceText = "😀".repeat(480)
    mocks.bgRequest
      .mockResolvedValueOnce({
        nodes: [
          {
            id: longId,
            type: "note",
            label: longLabel,
            created_at: null,
            deleted: false,
            degree: 9001,
            tag_count: 3001,
            primary_source_id: null
          }
        ],
        edges: [
          {
            id: "edge:0",
            source: longId,
            target: longId,
            type: "manual",
            directed: false,
            weight: 1,
            label: longLabel
          }
        ],
        truncated: false,
        truncated_by: [longLabel],
        has_more: false,
        cursor: null,
        limits: { max_nodes: 5000, max_edges: 9000, max_degree: 5000 },
        radius_cap_applied: false,
        active_note_count: 12,
        all_notes_note_cap: 5000,
        all_notes_eligible: true
      })
      .mockResolvedValueOnce({
        items: Array.from({ length: 100 }, (_, index) => ({
          ...suggestionPayload(`suggestion-${index}`),
          rationale,
          evidence: Array.from({ length: 6 }, (_, evidenceIndex) => ({
            side: evidenceIndex % 2 ? "target" : "source",
            note_id: evidenceIndex % 2 ? "note/target" : "note/source",
            field: "content",
            start_offset: evidenceIndex,
            end_offset: evidenceIndex + 1,
            text: evidenceText
          }))
        })),
        next_cursor: "next suggestion cursor",
        current_source_fingerprint: fingerprint("c"),
        rejection_set_revision: 4,
        rejection_count: 1
      })
      .mockResolvedValueOnce({
        items: [
          {
            ...runPayload("run-large", "succeeded"),
            provider: longLabel,
            model: longLabel,
            suggestion_count: 101,
            related_note_count: 101,
            tag_count: 101,
            invalid_item_count: 2001
          }
        ],
        next_cursor: null
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
      limit: 100
    })
    const runs = await listNotesGraphSuggestionRuns({ noteId: "note/source" })

    expect(graph.nodes[0].label).toBe(longLabel)
    expect(graph.nodes[0].id).toBe(longId)
    expect(graph.nodes[0]).toMatchObject({ degree: 9001, tag_count: 3001 })
    expect(graph.edges[0]).toMatchObject({ weight: 1, label: longLabel })
    expect(graph.truncated_by).toEqual([longLabel])
    expect(graph.limits).toEqual({
      max_nodes: 5000,
      max_edges: 9000,
      max_degree: 5000
    })
    expect(suggestions.items).toHaveLength(100)
    expect(suggestions.items[0].rationale).toBe(rationale)
    expect(suggestions.items[0].evidence).toHaveLength(6)
    expect(suggestions.items[0].evidence[0].text).toBe(evidenceText)
    expect(runs.items[0]).toMatchObject({
      provider: longLabel,
      model: longLabel,
      suggestion_count: 101,
      related_note_count: 101,
      tag_count: 101,
      invalid_item_count: 2001
    })
    expect(String(mocks.bgRequest.mock.calls[0][0].path)).toContain(
      "max_nodes=9999"
    )
    expect(String(mocks.bgRequest.mock.calls[0][0].path)).toContain(
      "max_edges=99999"
    )
  })

  it.each([
    [
      "overlong rationale",
      () => ({
        items: [
          {
            ...suggestionPayload("suggestion-rationale"),
            rationale: "😀".repeat(241)
          }
        ],
        next_cursor: null,
        current_source_fingerprint: fingerprint("c"),
        rejection_set_revision: 0,
        rejection_count: 0
      }),
      () => listNotesGraphSuggestions({ noteId: "note-1" })
    ],
    [
      "overlong evidence text",
      () => ({
        items: [
          {
            ...suggestionPayload("suggestion-evidence-text"),
            evidence: [
              {
                ...suggestionPayload("unused").evidence[0],
                text: "😀".repeat(481)
              }
            ]
          }
        ],
        next_cursor: null,
        current_source_fingerprint: fingerprint("c"),
        rejection_set_revision: 0,
        rejection_count: 0
      }),
      () => listNotesGraphSuggestions({ noteId: "note-1" })
    ],
    [
      "too many evidence excerpts",
      () => ({
        items: [
          {
            ...suggestionPayload("suggestion-evidence-count"),
            evidence: Array.from({ length: 7 }, (_, index) => ({
              ...suggestionPayload("unused").evidence[0],
              start_offset: index,
              end_offset: index + 1
            }))
          }
        ],
        next_cursor: null,
        current_source_fingerprint: fingerprint("c"),
        rejection_set_revision: 0,
        rejection_count: 0
      }),
      () => listNotesGraphSuggestions({ noteId: "note-1" })
    ],
    [
      "nodes beyond reported graph limit",
      () => ({
        nodes: [
          {
            id: "note:1",
            type: "note",
            label: "One",
            created_at: null,
            deleted: false,
            degree: 0,
            tag_count: 0,
            primary_source_id: null
          },
          {
            id: "note:2",
            type: "note",
            label: "Two",
            created_at: null,
            deleted: false,
            degree: 0,
            tag_count: 0,
            primary_source_id: null
          }
        ],
        edges: [],
        truncated: true,
        truncated_by: ["max_nodes"],
        has_more: true,
        cursor: "next",
        limits: { max_nodes: 1, max_edges: 2, max_degree: 40 },
        radius_cap_applied: false,
        active_note_count: 2,
        all_notes_note_cap: 100,
        all_notes_eligible: true
      }),
      () => fetchNotesGraph({ centerNoteId: "note-1" })
    ],
    [
      "edges beyond reported graph limit",
      () => ({
        nodes: [
          {
            id: "note:1",
            type: "note",
            label: "One",
            created_at: null,
            deleted: false,
            degree: 2,
            tag_count: 0,
            primary_source_id: null
          }
        ],
        edges: [
          {
            id: "edge:1",
            source: "note:1",
            target: "note:1",
            type: "manual",
            directed: false,
            weight: 1,
            label: null
          },
          {
            id: "edge:2",
            source: "note:1",
            target: "note:1",
            type: "manual",
            directed: false,
            weight: 1,
            label: null
          }
        ],
        truncated: true,
        truncated_by: ["max_edges"],
        has_more: true,
        cursor: "next",
        limits: { max_nodes: 2, max_edges: 1, max_degree: 40 },
        radius_cap_applied: false,
        active_note_count: 1,
        all_notes_note_cap: 100,
        all_notes_eligible: true
      }),
      () => fetchNotesGraph({ centerNoteId: "note-1" })
    ]
  ])(
    "rejects a %s response instead of rewriting it",
    async (_label, response, call) => {
      mocks.bgRequest.mockResolvedValueOnce(response())

      await expect(call()).rejects.toMatchObject({
        status: 502,
        code: "notes_graph_invalid_response",
        message: "The Notes graph server returned an invalid response."
      })
    }
  )

  it("rejects a graph edge whose serialized Task 8 weight is omitted", async () => {
    mocks.bgRequest.mockResolvedValueOnce({
      nodes: [
        {
          id: "note:1",
          type: "note",
          label: "One",
          created_at: null,
          deleted: false,
          degree: 1,
          tag_count: 0,
          primary_source_id: null
        }
      ],
      edges: [
        {
          id: "edge:1",
          source: "note:1",
          target: "note:1",
          type: "manual",
          directed: false,
          label: null
        }
      ],
      truncated: false,
      truncated_by: [],
      has_more: false,
      cursor: null,
      limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
      radius_cap_applied: false,
      active_note_count: 1,
      all_notes_note_cap: 100,
      all_notes_eligible: true
    })

    await expect(
      fetchNotesGraph({ centerNoteId: "note:1" })
    ).rejects.toMatchObject({ code: "notes_graph_invalid_response" })
  })

  it("fails closed on malformed limits, counts, revisions, actions, categories, states, and mutation envelopes", async () => {
    const invalidResponses: Array<{
      response: unknown
      call: () => Promise<unknown>
    }> = [
      {
        response: {
          nodes: [],
          edges: [],
          truncated: false,
          truncated_by: [],
          has_more: false,
          cursor: null,
          limits: { max_nodes: "120", max_edges: 480, max_degree: 40 },
          radius_cap_applied: false,
          active_note_count: 1,
          all_notes_note_cap: 100,
          all_notes_eligible: true
        },
        call: () => fetchNotesGraph({ centerNoteId: "note-1" })
      },
      {
        response: {
          ok: true,
          status: 200,
          headers: { etag: `"${fingerprint("a")}"` },
          data: {
            ...capabilityPayload(),
            allowed_actions: ["generate", "exfiltrate"]
          }
        },
        call: () => getNotesGraphSuggestionCapabilities({ noteId: "note-1" })
      },
      {
        response: {
          ok: true,
          status: 200,
          headers: { etag: `"${fingerprint("a")}"` },
          data: {
            ...capabilityPayload(),
            outbound_data_categories: [
              "selected_note_excerpts",
              "raw_credentials"
            ]
          }
        },
        call: () => getNotesGraphSuggestionCapabilities({ noteId: "note-1" })
      },
      {
        response: {
          ok: true,
          status: 200,
          headers: { etag: `"${fingerprint("a")}"` },
          data: {
            ...capabilityPayload(),
            endpoint_url: "https://secret.example"
          }
        },
        call: () => getNotesGraphSuggestionCapabilities({ noteId: "note-1" })
      },
      {
        response: {
          items: [{ ...runPayload(), state: "job-internal" }],
          next_cursor: null
        },
        call: () => listNotesGraphSuggestionRuns({ noteId: "note-1" })
      },
      {
        response: {
          items: [{ ...runPayload(), suggestion_count: "0" }],
          next_cursor: null
        },
        call: () => listNotesGraphSuggestionRuns({ noteId: "note-1" })
      },
      {
        response: {
          items: [{ ...suggestionPayload("suggestion-1"), state: "hidden" }],
          next_cursor: null,
          current_source_fingerprint: fingerprint("c"),
          rejection_set_revision: 0,
          rejection_count: 0
        },
        call: () => listNotesGraphSuggestions({ noteId: "note-1" })
      },
      {
        response: {
          resource_id: "run-1",
          state: "job-internal",
          revision: "4",
          cleared_count: null
        },
        call: () =>
          cancelNotesGraphSuggestionRun({
            noteId: "note-1",
            runId: "run-1",
            expectedRevision: 3,
            idempotencyKey: "cancel-key"
          })
      }
    ]

    for (const invalid of invalidResponses) {
      mocks.bgRequest.mockResolvedValueOnce(invalid.response)
      await expect(invalid.call()).rejects.toMatchObject({
        code: "notes_graph_invalid_response",
        message: "The Notes graph server returned an invalid response."
      })
    }
  })

  it("rejects oversized graph cursors and malformed revision guards before transport", async () => {
    const invalidCalls = [
      () =>
        fetchNotesGraph({ centerNoteId: "note-1", cursor: "x".repeat(4097) }),
      () => listNotesGraphSuggestionRuns({ noteId: "note-1", limit: 101 }),
      () => listNotesGraphSuggestions({ noteId: "note-1", limit: 101 }),
      () =>
        cancelNotesGraphSuggestionRun({
          noteId: "note-1",
          runId: "run-1",
          expectedRevision: 0,
          idempotencyKey: "cancel-key"
        }),
      () =>
        acceptNotesGraphSuggestion({
          noteId: "note-1",
          suggestionId: "suggestion-1",
          expectedRevision: "2" as unknown as number,
          expectedSourceFingerprint: fingerprint("c"),
          expectedTargetFingerprint: fingerprint("d"),
          idempotencyKey: "accept-key"
        }),
      () =>
        resetNotesGraphSuggestionRejections({
          noteId: "note-1",
          expectedRejectionRevision: Number.NaN,
          sourceFingerprint: fingerprint("c"),
          idempotencyKey: "reset-key"
        })
    ]

    for (const call of invalidCalls) {
      await expect(call()).rejects.toMatchObject({
        status: 422,
        code: "notes_graph_invalid_request"
      })
    }
    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("preserves the complete Task 8 persisted public run error allowlist", async () => {
    const persistedTask8RunErrorCodes = [
      "notes_graph_admission_failed",
      "notes_graph_capabilities_changed_before_queue",
      "notes_graph_job_missing",
      "notes_graph_capabilities_changed_before_provider",
      "notes_graph_fingerprint_stale",
      "notes_graph_fts_not_ready",
      "notes_graph_provider_retry_policy_unsupported",
      "notes_graph_provider_unavailable",
      "notes_graph_source_too_large",
      "notes_graph_suggestion_no_valid_items",
      "notes_graph_suggestion_suppression_limit",
      "notes_graph_publication_state_missing",
      "notes_graph_publication_receipt_mismatch",
      "notes_graph_publication_receipt_missing",
      "notes_graph_source_changed",
      "notes_graph_target_changed"
    ]
    mocks.bgRequest.mockResolvedValueOnce({
      items: persistedTask8RunErrorCodes.map((errorCode, index) => ({
        ...runPayload(`run-${index}`, "failed"),
        error_code: errorCode,
        guidance_key: errorCode.endsWith("_changed") ? null : "retry_generation"
      })),
      next_cursor: null
    })

    const page = await listNotesGraphSuggestionRuns({ noteId: "note-1" })

    expect(page.items.map((item) => item.error_code)).toEqual(
      persistedTask8RunErrorCodes
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

  it("rejects unknown closed response codes and sanitizes transport errors", async () => {
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

    const capabilityError = await getNotesGraphSuggestionCapabilities({
      noteId: "note-1"
    }).catch((value) => value)
    const runError = await listNotesGraphSuggestionRuns({
      noteId: "note-1"
    }).catch((value) => value)
    const error = await listNotesGraphSuggestionRuns({
      noteId: "note-1"
    }).catch((value) => value)

    expect(capabilityError).toMatchObject({
      status: 502,
      code: "notes_graph_invalid_response",
      message: "The Notes graph server returned an invalid response."
    })
    expect(runError).toMatchObject({
      status: 502,
      code: "notes_graph_invalid_response",
      message: "The Notes graph server returned an invalid response."
    })
    expect(error).toMatchObject({
      code: "notes_graph_provider_not_configured",
      message: "The selected provider is not configured."
    })
    expect(JSON.stringify({ capabilityError, runError, error })).not.toContain(
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
