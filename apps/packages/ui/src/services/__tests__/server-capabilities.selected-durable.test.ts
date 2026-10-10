import { beforeEach, describe, expect, it, vi } from "vitest"
import * as capabilities from "../tldw/server-capabilities"
const calls = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => calls.request(...args) }))
const bounds = { sources: 20, excerpt_scalars: 1000, excerpt_utf8_bytes: 4000, aggregate_excerpt_scalars: 16000,
  name_utf8_bytes: 1000, metadata_text_utf8_bytes: 1000, chunk_id_utf8_bytes: 512, compact_label_utf8_bytes: 128,
  media_id_utf8_bytes: 512, paired_character_ranges: true, chunk_index_within_total: true,
  url_utf8_bytes: 2048, result_canonical_utf8_bytes: 65536, distinct_excerpts: true }
const marker = { version: 1, history: "h1_single_input_v1", result: "rag_source_v1", request_digest: "history_context_wire_v1",
  recovery_read: "protected_live_v1", inference_guarantee: "multiple_results_possible" }
const recoveryMarker = { version: 1, projection: "protected_live_v1" }
type Schema = {
  [key: string]: unknown
  properties?: Record<string, Schema>
  required?: string[]
  additionalProperties?: boolean
  items?: Schema
  maximum?: number
  maxLength?: number
  maxItems?: number
}
const ref = (name: string) => ({ $ref: `#/components/schemas/${name}` })
const str = { type: "string" }
const literal = (value: string | number) => ({ type: typeof value === "number" ? "integer" : "string", const: value })
const integer = { type: "integer", minimum: 0, maximum: Number.MAX_SAFE_INTEGER }
const hex = { type: "string", pattern: "^[0-9a-f]{64}$", minLength: 64, maxLength: 64 }
const strict = (properties: Record<string, Schema>, required = Object.keys(properties)) => ({ type: "object", additionalProperties: false, properties, required })
const reference = { version: literal(1), owner_key: str, conversation_id: str, input_message_id: str,
  input_message_revision: str, selection_digest: str }
const nullable = (schema: Schema) => ({ anyOf: [schema, { type: "null" }] })
const spec = () => {
  const schemas: Record<string, Schema> = {
    Revision: strict({ id: str, revision: str }),
    Fences: strict({ conversation: str, history: str, settings: str }),
    Parent: strict({ kind: literal("parent_graph_v1") }),
    Legacy: strict({ kind: literal("legacy_linear_v1"), projection_id: str }),
    Empty: strict({ kind: literal("empty") }),
    Before: strict({ kind: literal("before_message"), message_id: str }),
    After: strict({ kind: literal("after_message"), message_id: str }),
    Selection: strict({ version: literal(1), owner_key: str, conversation_id: str,
      interpretation: { oneOf: [ref("Parent"), ref("Legacy")] }, cursor: { oneOf: [ref("Empty"), ref("Before"), ref("After")] },
      purpose: { type: "string", enum: ["send", "fork"] }, messages: { type: "array", items: ref("Revision") },
      selection_revision: { type: "integer", minimum: 0 }, fences: ref("Fences"), storage_context_digest: str,
      request_context_digest: str, selection_digest: str }),
    Reference: strict(reference),
    Admission: strict({ ...reference, messages: { type: "array", items: ref("Revision") },
      originating_selection_revision: { type: "integer", minimum: 0 } }),
    Selected: strict({ version: literal(1), kind: literal("selection"), selection: ref("Selection") }),
    Admitted: strict({ version: literal(1), kind: literal("admission"), admission: ref("Reference"), request_context_digest: hex }),
    Lines: strict({ from: integer, to: integer }), Loc: strict({ lines: ref("Lines") }),
    Metadata: strict({ source: str, title: str, chunk_id: str, retrieval_strategy: str, source_type: str,
      selection_reason: str, score: { anyOf: [{ type: "integer" }, { type: "number" }] }, page: integer, loc: ref("Loc"),
      media_id: str, author: str, chunk_index: integer, total_chunks: { ...integer, minimum: 1 },
      start_char: integer, end_char: integer, chunk_start: integer, chunk_end: integer }, []),
    Source: strict({ name: str, type: str, mode: literal("rag"), url: str, pageContent: { ...str, maxLength: 1000 }, metadata: ref("Metadata") }),
    Payload: { ...strict({ version: literal(1), sources: { type: "array", maxItems: 20, items: ref("Source") } }), "x-tldw-source-bounds": bounds },
    Result: strict({ version: literal(1), sources: { type: "array", maxItems: 20, items: ref("Source") },
      result_message_id: { type: "string", format: "uuid" }, result_message_revision: literal("1"),
      admission: ref("Reference"), request_context_digest: hex }),
    Workspace: strict({ scope_type: literal("workspace"), workspace_id: { ...str, minLength: 1 } }),
    Global: strict({ scope_type: literal("global"), workspace_id: { type: "null" } }),
    InputVerified: strict({ version: literal(1), status: literal("input_verified"),
      scope: { oneOf: [ref("Workspace"), ref("Global")] }, admission: ref("Admission") }),
    ResultVerified: strict({ version: literal(1), status: literal("result_verified"),
      scope: { oneOf: [ref("Workspace"), ref("Global")] }, result: ref("Result") }),
    Unverified: strict({ version: literal(1), status: literal("unverified"),
      code: { type: "string", enum: ["no_protected_binding", "live_state_mismatch", "unsupported_projection"] } }),
    Turn: strict({ user_message_id: { type: "string", format: "uuid" }, history_v1: nullable({ oneOf: [ref("Selected"), ref("Admitted")] }),
      result_v1: nullable(ref("Payload")) }, ["user_message_id"]),
    Request: { type: "object", properties: { tldw_turn: nullable(ref("Turn")) } },
    Message: { type: "object", properties: { tldw_history_recovery_v1: nullable({ oneOf: [ref("InputVerified"), ref("ResultVerified"), ref("Unverified")] }) } },
    Pagination: { type: "object", properties: { mode: { ...literal("offset"), default: "offset" },
      limit: { type: "integer", minimum: 1 }, offset: { type: "integer", minimum: 0 },
      total: nullable({ type: "integer", minimum: 0 }), has_more: { type: "boolean" },
      next_offset: nullable({ type: "integer", minimum: 0 }) }, required: ["limit", "offset", "has_more"] },
    MessageList: { type: "object", properties: { messages: { type: "array", items: ref("Message") },
      total: { type: "integer" }, limit: { type: "integer" }, offset: { type: "integer" }, pagination: ref("Pagination"),
      has_more: nullable({ type: "boolean" }), next_offset: nullable({ type: "integer", minimum: 0 }) },
      required: ["messages", "total", "limit", "offset", "pagination"] }
  }
  const query = (name: string, schema: Schema) => ({ name, in: "query", schema })
  const params = [query("scope_type", { type: "string", enum: ["global", "workspace"] }), query("workspace_id", nullable(str)),
    query("include_history_recovery_v1", { type: "boolean", default: false })]
  return { components: { schemas }, paths: {
    "/api/v1/chat/completions": { post: { "x-tldw-selected-durable-turn": marker,
      requestBody: { content: { "application/json": { schema: ref("Request") } } } } },
    "/api/v1/messages/{message_id}": { get: { "x-tldw-history-recovery-read": recoveryMarker, parameters: params,
      responses: { "200": { content: { "application/json": { schema: ref("Message") } } } } } },
    "/api/v1/chats/{chat_id}/messages": { get: { "x-tldw-history-recovery-read": recoveryMarker,
      parameters: [...params, query("limit", { type: "integer", default: 50, minimum: 1, maximum: 200 }),
        query("offset", { type: "integer", default: 0, minimum: 0 }),
        ...["format_for_completions", "include_character_context", "include_deleted", "render_placeholders"].map(name =>
          query(name, { type: "boolean", default: name === "render_placeholders" }))],
      responses: { "200": { content: { "application/json": { schema: ref("MessageList") as Schema } } } } } }
  } }
}
const scope = { config: { serverUrl: "https://pinned.example", authMode: "single-user" as const, apiKey: "private" }, userId: 7 }
beforeEach(() => {
  calls.request.mockReset()
  expect(capabilities.getSelectedDurableTurnSupport, "strict pinned capability feature is missing").toBeTypeOf("function")
})
describe("selected durable capability contract", () => {
  it("accepts only the coherent reachable strict contract and pins the request", async () => {
    calls.request.mockResolvedValue(spec())
    const controller = new AbortController()
    expect(await capabilities.getSelectedDurableTurnSupport(scope, controller.signal)).toBe(true)
    expect(calls.request).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ path: "/openapi.json", method: "GET",
      abortSignal: controller.signal, headers: { "X-TLDW-Expected-User-ID": "7" }, servicePromptConfig: expect.objectContaining({ expectedUserId: 7 }) }))
  })
  it("accepts the actual native nullable scope query representation on both protected reads", async () => {
    const value = spec()
    for (const path of ["/api/v1/messages/{message_id}", "/api/v1/chats/{chat_id}/messages"] as const) {
      value.paths[path].get.parameters.find(parameter => parameter.name === "scope_type").schema =
        nullable({ type: "string", enum: ["global", "workspace"] })
    }
    calls.request.mockResolvedValue(value)
    expect(await capabilities.getSelectedDurableTurnSupport(scope)).toBe(true)
  })
  it.each([
    { anyOf: [{ type: "string", enum: ["global", "workspace", "other"] }, { type: "null" }] },
    { anyOf: [{ type: "string", enum: ["global", "workspace"] }, { type: "integer" }, { type: "null" }] }
  ])("rejects widened nullable scope query branches %#", async schema => {
    const value = spec()
    value.paths["/api/v1/messages/{message_id}"].get.parameters.find(parameter => parameter.name === "scope_type").schema = schema
    calls.request.mockResolvedValue(value)
    expect(await capabilities.getSelectedDurableTurnSupport(scope)).toBe(false)
  })
  it.each([
    ["missing response", (value: ReturnType<typeof spec>) => { delete (value.paths["/api/v1/chats/{chat_id}/messages"].get as Schema).responses }],
    ["string response", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.responses["200"].content["application/json"].schema = str }],
    ["loose response union", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.responses["200"].content["application/json"].schema = { anyOf: [ref("MessageList"), str] } }],
    ["unreachable wrapper", (value: ReturnType<typeof spec>) => { delete value.components.schemas.MessageList }],
    ["wrong messages shape", (value: ReturnType<typeof spec>) => { value.components.schemas.MessageList.properties.messages = str }],
    ["non-recovery message items", (value: ReturnType<typeof spec>) => { value.components.schemas.MessageList.properties.messages.items = { type: "object", properties: {} } }],
    ["missing response pagination", (value: ReturnType<typeof spec>) => { delete value.components.schemas.MessageList.properties.pagination }],
    ["optional messages", (value: ReturnType<typeof spec>) => { value.components.schemas.MessageList.required = ["total", "limit", "offset", "pagination"] }],
    ["wrong response offset", (value: ReturnType<typeof spec>) => { value.components.schemas.MessageList.properties.offset = str }],
    ["wrong pagination mode", (value: ReturnType<typeof spec>) => { value.components.schemas.Pagination.properties.mode = literal("cursor") }],
    ["missing limit", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters = value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.filter(parameter => parameter.name !== "limit") }],
    ["widened limit", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.find(parameter => parameter.name === "limit").schema = { type: "integer", default: 50, minimum: -1, maximum: 1000000000 } }],
    ["missing limit maximum", (value: ReturnType<typeof spec>) => { delete value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.find(parameter => parameter.name === "limit").schema.maximum }],
    ["wrong limit default", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.find(parameter => parameter.name === "limit").schema.default = 200 }],
    ["missing offset", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters = value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.filter(parameter => parameter.name !== "offset") }],
    ["negative offset", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.find(parameter => parameter.name === "offset").schema.minimum = -1 }],
    ["wrong offset type", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.find(parameter => parameter.name === "offset").schema.type = "number" }],
    ["wrong exact-read flag type", (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.find(parameter => parameter.name === "render_placeholders").schema.type = "string" }]
  ] as const)("rejects incoherent standard list evidence: %s", async (_name, mutate) => {
    const value = spec()
    mutate(value)
    calls.request.mockResolvedValue(value)
    expect(await capabilities.getSelectedDurableTurnSupport(scope)).toBe(false)
  })
  it.each([
    (value: ReturnType<typeof spec>) => { delete (value.paths["/api/v1/chat/completions"].post as Schema)["x-tldw-selected-durable-turn"] },
    (value: ReturnType<typeof spec>) => { (value.paths["/api/v1/chat/completions"].post as Schema)["x-tldw-selected-durable-turn"] = { ...marker, version: 2 } },
    (value: ReturnType<typeof spec>) => { (value.paths["/api/v1/chat/completions"].post as Schema)["x-tldw-selected-durable-turn"] = { ...marker, extra: true } },
    (value: ReturnType<typeof spec>) => { delete (value.paths["/api/v1/messages/{message_id}"].get as Schema)["x-tldw-history-recovery-read"] },
    (value: ReturnType<typeof spec>) => { delete (value.paths["/api/v1/chats/{chat_id}/messages"].get as Schema)["x-tldw-history-recovery-read"] },
    (value: ReturnType<typeof spec>) => { value.paths["/api/v1/messages/{message_id}"].get.parameters = [] },
    (value: ReturnType<typeof spec>) => { value.paths["/api/v1/chats/{chat_id}/messages"].get.parameters.pop() },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Request.properties = {} },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Message.properties = {} },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Turn.additionalProperties = true },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Source.additionalProperties = true },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Source.required.pop() },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Metadata.properties.api_key = str },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Metadata.properties.source = nullable(str) },
    (value: ReturnType<typeof spec>) => { delete value.components.schemas.Metadata.properties.media_id },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Metadata.properties.author = nullable(str) },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Metadata.properties.start_char = str },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Metadata.properties.total_chunks.minimum = 0 },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Payload["x-tldw-source-bounds"] = { ...bounds, paired_character_ranges: false } },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Lines.properties.from.maximum = Number.MAX_SAFE_INTEGER + 1 },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Payload.properties.sources = {} },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Payload.properties.sources.maxItems = 21 },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Source.properties.pageContent.maxLength = 1001 },
    (value: ReturnType<typeof spec>) => { delete value.components.schemas.Payload["x-tldw-source-bounds"] },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Payload["x-tldw-source-bounds"] = { ...bounds, distinct_excerpts: false } },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Result.properties.admission = {} },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Result.properties.result_message_revision = str },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Result.properties.request_context_digest = str },
    (value: ReturnType<typeof spec>) => { value.components.schemas.InputVerified.additionalProperties = true },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Global.required = ["scope_type"] },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Admission.properties.messages.items = {} },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Selection.properties.interpretation = { oneOf: [ref("Parent"), ref("Parent")] } },
    (value: ReturnType<typeof spec>) => { value.components.schemas.Selection.properties.cursor = { oneOf: [ref("Empty"), ref("Empty"), ref("Empty")] } },
    (value: ReturnType<typeof spec>) => { value.components.schemas.InputVerified.properties.scope = { oneOf: [ref("Workspace"), ref("Workspace")] } }
  ])("rejects missing, widened or unreachable contract evidence %#", async mutate => {
    const value = structuredClone(spec())
    mutate(value)
    calls.request.mockResolvedValue(value)
    expect(await capabilities.getSelectedDurableTurnSupport(scope)).toBe(false)
  })
  it("does not reuse cached authority between serving nodes", async () => {
    calls.request.mockResolvedValueOnce(spec()).mockResolvedValueOnce({ paths: {} })
    expect(await capabilities.getSelectedDurableTurnSupport(scope)).toBe(true)
    expect(await capabilities.getSelectedDurableTurnSupport(scope)).toBe(false)
    expect(calls.request).toHaveBeenCalledTimes(2)
  })
  it("fails closed on discovery failure without fallback", async () => {
    calls.request.mockRejectedValue(new Error("offline"))
    await expect(capabilities.getSelectedDurableTurnSupport(scope)).rejects.toThrow("offline")
  })
  it("checks cancellation before and after pinned discovery", async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(capabilities.getSelectedDurableTurnSupport(scope, controller.signal)).rejects.toThrow()
    expect(calls.request).not.toHaveBeenCalled()
    const later = new AbortController()
    calls.request.mockImplementation(async () => { later.abort(); return spec() })
    await expect(capabilities.getSelectedDurableTurnSupport(scope, later.signal)).rejects.toThrow()
  })
})
