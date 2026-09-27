import { beforeEach, describe, expect, it, vi } from "vitest"
import { getStructuredApiErrorDetail, TldwApiError } from "../../api-error"
import {
  isSharedWorkspacePostCommitResponseError,
  SharedWorkspacePostCommitResponseError,
  sharedWorkspacesApi
} from "../shared-workspaces"

const fetchWithTldwAuth = vi.hoisted(() => vi.fn())
const getTldwServerURL = vi.hoisted(() =>
  vi.fn().mockResolvedValue("https://tldw.example")
)

vi.mock("@/services/tldw/auth-fetch", () => ({ fetchWithTldwAuth }))
vi.mock("@/services/tldw-server", () => ({ getTldwServerURL }))

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  })

const bootstrapPayload = (generationDefault: unknown = {
  provider: "openai",
  model: "gpt-5-mini",
  ready: true,
  reason_code: null
}) => ({
  schema_version: 1,
  generated_at: "2026-08-22T00:00:00Z",
  share: {
    share_id: 42,
    access_level: "view_chat",
    allow_clone: false,
    owner_display_name: "Owner",
    shared_at: "2026-08-21T00:00:00Z"
  },
  workspace: {
    workspace_id: "workspace-1",
    name: "Research",
    description: "Shared research"
  },
  allowed_actions: {
    inspect_sources: { allowed: true, reason_code: null },
    ask_grounded_questions: { allowed: true, reason_code: null },
    add_sources: { allowed: false, reason_code: "recipient_read_only" },
    edit_workspace: { allowed: false, reason_code: "recipient_read_only" },
    clone_workspace: { allowed: false, reason_code: "clone_not_allowed" }
  },
  generation_default: generationDefault,
  source_summary: { total: 0, queryable: 0, processing: 0, failed: 0 },
  sources: {
    items: [],
    pagination: { offset: 0, limit: 50, total: 0, has_more: false }
  },
  conversation: { conversation_id: null, messages: [], next_before: null },
  partial_errors: []
})

describe("sharedWorkspacesApi", () => {
  beforeEach(() => {
    fetchWithTldwAuth.mockReset()
    getTldwServerURL.mockClear()
  })

  it("uses authenticated fetch and propagates the bootstrap abort signal", async () => {
    const signal = new AbortController().signal
    fetchWithTldwAuth.mockResolvedValue(jsonResponse(bootstrapPayload()))

    await sharedWorkspacesApi.bootstrap(42, signal)

    expect(fetchWithTldwAuth).toHaveBeenCalledWith(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/workspace",
      { signal }
    )
  })

  it("uses only canonical source-list parameters and omits empty values", async () => {
    const signal = new AbortController().signal
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse({
        items: [],
        pagination: { offset: 5, limit: 25, total: 0, has_more: false },
        summary: { total: 0, queryable: 0, processing: 0, failed: 0 },
        partial_errors: []
      })
    )

    await sharedWorkspacesApi.listSources(
      42,
      { offset: 5, limit: 25, q: "  ", state: "queryable" },
      signal
    )

    expect(fetchWithTldwAuth).toHaveBeenCalledWith(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/sources?offset=5&limit=25&state=queryable",
      { signal }
    )
  })

  it("encodes non-empty source search values with URLSearchParams", async () => {
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse({
        items: [],
        pagination: { offset: 0, limit: 50, total: 0, has_more: false },
        summary: { total: 0, queryable: 0, processing: 0, failed: 0 },
        partial_errors: []
      })
    )

    await sharedWorkspacesApi.listSources(42, {
      offset: 0,
      limit: 50,
      q: "alpha & beta"
    })

    expect(fetchWithTldwAuth.mock.calls[0][0]).toBe(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/sources?offset=0&limit=50&q=alpha+%26+beta"
    )
  })

  it("URL-encodes source IDs and propagates preview chunk selection", async () => {
    const signal = new AbortController().signal
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse({
        source_id: "folder/a b",
        title: "Source",
        source_type: "document",
        origin_url: null,
        origin_host: null,
        state: "queryable",
        reason_code: null,
        content_available: true,
        preview_mode: "content_excerpt",
        unavailable_reason: null,
        text_preview: "Preview",
        text_total_chars: 7,
        text_truncated: false,
        snippets: [],
        generated_at: "2026-08-22T00:00:00Z"
      })
    )

    await sharedWorkspacesApi.previewSource(42, "folder/a b", 3, signal)

    expect(fetchWithTldwAuth).toHaveBeenCalledWith(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/sources/folder%2Fa%20b/preview?chunk_index=3",
      { signal }
    )
  })

  it("uses the canonical bounded history cursor without local workspace paths", async () => {
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse({ conversation_id: null, messages: [], next_before: null })
    )

    await sharedWorkspacesApi.listMessages(42, "cursor +/=")

    const [url] = fetchWithTldwAuth.mock.calls[0]
    expect(url).toBe(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/chat/messages?before=cursor+%2B%2F%3D"
    )
    expect(url).not.toContain("/workspaces/")
  })

  it("posts the exact chat payload once to the canonical recipient route", async () => {
    const signal = new AbortController().signal
    const request = {
      request_id: "00000000-0000-4000-8000-000000000042",
      query: "What changed?",
      source_scope: { mode: "include" as const, source_ids: ["source-1"] },
      provider: "openai",
      model: "gpt-5-mini"
    }
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse({
        schema_version: 1,
        request_id: request.request_id,
        conversation_id: "conversation-1",
        turn: {
          user_message: {
            message_id: "user-1",
            role: "user",
            content: request.query,
            created_at: "2026-08-22T00:00:01Z"
          },
          assistant_message: {
            message_id: "assistant-1",
            role: "assistant",
            content: "Answer",
            created_at: "2026-08-22T00:00:02Z"
          }
        },
        citations: [
          {
            citation_id: "citation-1",
            source_id: "source-1",
            source_title: "Source",
            locator: { chunk: 1, start_char: null, end_char: null },
            quote: "Evidence",
            score: 0.9
          }
        ],
        generation: { provider: "openai", model: "gpt-5-mini" },
        source_scope: { mode: "include", effective_source_count: 1 },
        replay: { replayed: false }
      })
    )

    await sharedWorkspacesApi.ask(42, request, signal)

    expect(fetchWithTldwAuth).toHaveBeenCalledWith(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/chat",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal
      }
    )
    expect(Object.keys(sharedWorkspacesApi).sort()).toEqual([
      "ask",
      "bootstrap",
      "listMessages",
      "listSources",
      "previewSource"
    ])
  })

  it("fails closed when a ready generation default is blank or over backend bounds", async () => {
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse(
        bootstrapPayload({
          provider: "p".repeat(129),
          model: " ",
          ready: true,
          reason_code: null
        })
      )
    )

    const result = await sharedWorkspacesApi.bootstrap(42)

    expect(result.generation_default).toEqual({
      provider: null,
      model: null,
      ready: false,
      reason_code: "generation_default_unavailable"
    })
  })

  it("preserves an allowed server chat action when generation is unavailable", async () => {
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse(
        bootstrapPayload({
          provider: null,
          model: null,
          ready: false,
          reason_code: "no_provider_configured"
        })
      )
    )

    const result = await sharedWorkspacesApi.bootstrap(42)

    expect(result.generation_default).toEqual({
      provider: null,
      model: null,
      ready: false,
      reason_code: "no_provider_configured"
    })
    expect(result.allowed_actions.ask_grounded_questions).toEqual({
      allowed: true,
      reason_code: null
    })
  })

  it("rejects malformed typed envelopes instead of asserting their shape", async () => {
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse({ items: [], pagination: null, summary: {}, partial_errors: [] })
    )

    const error = await sharedWorkspacesApi
      .listSources(42, { offset: 0, limit: 50 })
      .catch((cause: unknown) => cause)

    expect(error).toMatchObject({
      status: 502,
      detail: { code: "shared_workspace_unavailable", retryable: true }
    })
    expect(isSharedWorkspacePostCommitResponseError(error)).toBe(false)
  })

  it.each([
    [
      "truncated JSON",
      () =>
        new Response('{"schema_version":1', {
          status: 200,
          headers: { "Content-Type": "application/json" }
        })
    ],
    ["a strictly invalid chat envelope", () => jsonResponse({ schema_version: 1 })]
  ])("marks ask-only %s as post-commit ambiguous", async (_case, response) => {
    fetchWithTldwAuth.mockResolvedValue(response())

    const error = await sharedWorkspacesApi
      .ask(42, {
        request_id: "00000000-0000-4000-8000-000000000042",
        query: "What changed?",
        source_scope: { mode: "all", source_ids: [] },
        provider: "openai",
        model: "gpt-5-mini"
      })
      .catch((cause: unknown) => cause)

    expect(error).toBeInstanceOf(SharedWorkspacePostCommitResponseError)
    expect(isSharedWorkspacePostCommitResponseError(error)).toBe(true)
    expect(error).toMatchObject({
      status: 502,
      detail: {
        code: "shared_chat_response_unconfirmed",
        retryable: true,
        recovery_action: "retry"
      }
    })
  })

  it("does not mark a typed non-2xx ask error as post-commit ambiguous", async () => {
    fetchWithTldwAuth.mockResolvedValue(
      jsonResponse(
        {
          detail: {
            code: "generation_failed",
            message: "Generation failed.",
            retryable: true
          }
        },
        503
      )
    )

    const error = await sharedWorkspacesApi
      .ask(42, {
        request_id: "00000000-0000-4000-8000-000000000042",
        query: "What changed?",
        source_scope: { mode: "all", source_ids: [] },
        provider: "openai",
        model: "gpt-5-mini"
      })
      .catch((cause: unknown) => cause)

    expect(error).toBeInstanceOf(TldwApiError)
    expect(isSharedWorkspacePostCommitResponseError(error)).toBe(false)
    expect(error).toMatchObject({
      status: 503,
      detail: { code: "generation_failed" }
    })
  })

  it("normalizes shared error fields without changing unrelated detail semantics", () => {
    const error = new TldwApiError("limited", 429, {
      code: "shared_chat_rate_limited",
      message: "Wait",
      retryable: true,
      recovery_action: "retry",
      retry_after_ms: -4,
      ignored: "not part of the typed detail"
    })

    expect(getStructuredApiErrorDetail(error)).toEqual({
      code: "shared_chat_rate_limited",
      message: "Wait",
      retryable: true,
      recovery_action: "retry",
      retry_after_ms: undefined,
      category: undefined,
      frontend_state: undefined,
      ignored: "not part of the typed detail"
    })
  })

  it("preserves context-budget codes and does not mark a neutral 404 retryable", async () => {
    fetchWithTldwAuth
      .mockResolvedValueOnce(
        jsonResponse(
          {
            detail: {
              code: "shared_chat_context_too_large",
              message: "Too large",
              retryable: false
            }
          },
          422
        )
      )
      .mockResolvedValueOnce(
        jsonResponse(
          {
            detail: {
              code: "shared_workspace_not_found",
              message: "Shared workspace not found.",
              retryable: false
            }
          },
          404
        )
      )

    await expect(
      sharedWorkspacesApi.ask(42, {
        request_id: "00000000-0000-4000-8000-000000000042",
        query: "large",
        source_scope: { mode: "all", source_ids: [] },
        provider: null,
        model: null
      })
    ).rejects.toMatchObject({
      status: 422,
      detail: { code: "shared_chat_context_too_large", retryable: false }
    })

    await expect(sharedWorkspacesApi.bootstrap(42)).rejects.toMatchObject({
      status: 404,
      detail: { code: "shared_workspace_not_found", retryable: false }
    })
    expect(fetchWithTldwAuth).toHaveBeenCalledTimes(2)
  })
})
