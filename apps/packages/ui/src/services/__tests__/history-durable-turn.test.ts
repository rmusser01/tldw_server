import { beforeEach, describe, expect, it, vi } from "vitest"
import { captureHistorySnapshot } from "../chat-history-selection"
import { historyDigest } from "@/db/dexie/history-selection"
import type { HistorySelectionCaptureV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import * as service from "../history-durable-turn"

const api = vi.hoisted(() => ({ capture: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { captureHistorySelection: api.capture }
}))
const uuid = "12345678-1234-4321-8123-123456789abc"
const digest = "a".repeat(64)
const source = (pageContent = "Evidence") => ({
  name: "Report", type: "pdf", mode: "rag" as const,
  url: "MEDIA:Exact/%2f?q=Two+Words#Page", pageContent, metadata: {}
})
const result = (sources = [source()]) => ({ version: 1, sources })
const body = () => ({
  stream: true, model: "actual-model", api_provider: "actual-provider",
  save_to_db: true, conversation_id: "chat",
  messages: [{ role: "system", content: "Frozen evidence" }, { role: "user", content: "Original input" }],
  temperature: null,
  tldw_turn: { user_message_id: uuid, result_v1: result([]) }
})
const owner = () => ({
  kind: "native" as const, owner_key: "native-key", conversation_id: "chat",
  request_scope: { config: { serverUrl: "https://server.test", authMode: "multi-user" as const }, userId: "alice" },
  validate_lease: () => true
})
const view: HistoryViewSelectionV1 = {
  owner_key: "native-key", conversation_id: "chat", view_session_id: "view",
  cursor: { kind: "empty" }, interpretation: { kind: "parent_graph_v1" }, selection_revision: 1
}
const capture = (): HistorySelectionCaptureV1 => ({
  status: "captured", snapshot: {
    version: 1, owner_key: "native-key", conversation_id: "chat",
    fences: { conversation: "1", history: "1", settings: "1" }, nodes: [],
    source_digest: "source", interpretation_status: { kind: "parent_graph_v1" }, storage_context_digest: "storage"
  }, rows: [], selected_content: [], view, purpose: "send", storage_context_digest: "storage"
})
const admission = () => ({
  version: 1 as const, owner_key: "native-key", conversation_id: "chat",
  input_message_id: uuid, input_message_revision: "1", selection_digest: digest
})
const retryOptions = () => ({
  owner: owner(), mode: "plain" as const, validate_lease: () => true,
  history: { kind: "admission" as const, admission: admission() }
})
beforeEach(() => {
  api.capture.mockResolvedValue(capture())
})

describe("bounded source result", () => {
  it("preserves exact inert URLs, evidence and source order in detached frozen JSON", () => {
    const input = result([source("Second"), source("First")])
    const parsed = service.parseHistoryDurableResult(input, "rag")
    input.sources[0].pageContent = "changed"
    expect(parsed.sources.map(s => s.pageContent)).toEqual(["Second", "First"])
    expect(parsed.sources[0].url).toBe("MEDIA:Exact/%2f?q=Two+Words#Page")
    expect(Object.isFrozen(parsed.sources[0].metadata)).toBe(true)
  })
  it("requires explicit empty sources for plain sends and evidence for RAG", () => {
    expect(service.parseHistoryDurableResult(result([]), "plain")).toEqual({ version: 1, sources: [] })
    for (const invalid of [undefined, { version: 1 }, result()]) {
      expect(() => service.parseHistoryDurableResult(invalid, "plain")).toThrow()
    }
    expect(() => service.parseHistoryDurableResult(result([]), "rag")).toThrow()
  })
  it.each(["name", "type", "mode", "url", "pageContent", "metadata"])("requires source key %s", key => {
    const invalid: Record<string, unknown> = { ...source() }
    delete invalid[key]
    expect(() => service.parseHistoryDurableResult({ version: 1, sources: [invalid] }, "rag")).toThrow()
  })
  it.each([
    { ...source(), headers: { Authorization: "secret" } },
    { ...source(), metadata: { api_key: "secret" } },
    { ...source(), metadata: { page: null } },
    { ...source(), metadata: { page: true } },
    { ...source(), metadata: { page: Number.MAX_SAFE_INTEGER + 1 } },
    { ...source(), metadata: { score: Infinity } },
    { ...source(), metadata: { score: false } },
    { ...source(), metadata: { loc: { lines: { from: 2, to: 1 } } } },
    { ...source(), metadata: { loc: { lines: { from: 0, to: 1, column: 3 } } } },
    { ...source(), metadata: { title: " " } },
    { ...source(), pageContent: "\ud800" },
    { ...source(), url: "\udfff" },
    { ...source(), name: "" },
    { ...source(), type: " " },
    { ...source(), mode: "chat" }
  ])("rejects malformed, credential-bearing or unsupported wire fields %#", invalid => {
    expect(() => service.parseHistoryDurableResult({ version: 1, sources: [invalid] }, "rag")).toThrow()
  })
  it("accepts safe locator endpoints and finite scores outside the unit interval", () => {
    expect(service.parseHistoryDurableResult({ version: 1, sources: [{ ...source(), metadata: {
      score: -12.5, page: Number.MAX_SAFE_INTEGER, loc: { lines: { from: 0, to: Number.MAX_SAFE_INTEGER } }
    } }] }, "rag").sources[0].metadata.score).toBe(-12.5)
  })
  it("enforces scalar, UTF8, source-count and aggregate limits without truncating", () => {
    expect(service.parseHistoryDurableResult(result([source("\u{1f600}".repeat(1000))]), "rag").sources[0].pageContent.length).toBe(2000)
    expect(service.parseHistoryDurableResult(result(Array.from({ length: 20 }, (_, i) => source(String(i)))), "rag").sources).toHaveLength(20)
    for (const input of [
      result([source("x".repeat(1001))]), result([source("\u{1f600}".repeat(1001))]),
      result([{ ...source(), name: "\u00e9".repeat(501) }]),
      result([{ ...source(), type: "x".repeat(129) }]),
      result([{ ...source(), url: "x".repeat(2049) }]),
      result(Array.from({ length: 21 }, (_, i) => source(String(i)))),
      result(Array.from({ length: 17 }, (_, i) => source(String(i).padEnd(1000, "x"))))
    ]) expect(() => service.parseHistoryDurableResult(input, "rag")).toThrow()
  })
  it("rejects escaped JSON budget overflow and duplicate excerpts with different locators", () => {
    expect(() => service.parseHistoryDurableResult(result(Array.from({ length: 16 }, (_, i) => source(String(i) + "\u0001".repeat(999)))), "rag")).toThrow()
    expect(() => service.parseHistoryDurableResult(result([source(), { ...source(), url: "different" }]), "rag")).toThrow()
  })
  it("accepts exactly 16000 scalars and exactly 65536 canonical UTF8 bytes", () => {
    const sources = Array.from({ length: 16 }, (_, i) => ({ ...source(String(i).padEnd(1000, "x")),
      name: "n".repeat(900), metadata: { source: "s".repeat(1000), title: "t".repeat(1000), selection_reason: "" }
    }))
    const input = { version: 1, sources }
    for (const item of sources) item.metadata.selection_reason = "r"
    const remaining = 65536 - new TextEncoder().encode(JSON.stringify(input)).length
    sources[0].metadata.selection_reason += "r".repeat(remaining)
    expect(service.parseHistoryDurableResult(input, "rag").sources).toHaveLength(16)
    sources[0].metadata.selection_reason += "r"
    expect(() => service.parseHistoryDurableResult(input, "rag")).toThrow()
  })
  it.each([
    ["source", 1000], ["title", 1000], ["selection_reason", 1000], ["chunk_id", 512],
    ["retrieval_strategy", 128], ["source_type", 128]
  ])("enforces UTF8 metadata budget for %s", (key, limit) => {
    const input = { version: 1, sources: [{ ...source(), metadata: { [key]: "\u00e9".repeat(Number(limit) / 2) } }] }
    expect(service.parseHistoryDurableResult(input, "rag").sources).toHaveLength(1)
    input.sources[0].metadata[key] += "x"
    expect(() => service.parseHistoryDurableResult(input, "rag")).toThrow()
  })
  it("projects aliases in MessageSource precedence without trimming", () => {
    const projected = service.projectHistoryDurableSources([{ ...source(),
      score: NaN, relevance: -2, chunk_id: " top chunk ", chunkId: "other",
      strategy: " first strategy ", retrieval_strategy: "second", source_type: "top type",
      rationale: " first reason ", reason: "second",
      metadata: { source: "Attribution", title: "Title", score: 0.1, chunk_id: "meta",
        retrieval_strategy: "meta", source_type: "meta", selection_reason: "meta", page: 0 }
    }])
    expect(projected[0].metadata).toEqual({ source: "Attribution", title: "Title", score: -2,
      chunk_id: " top chunk ", retrieval_strategy: " first strategy ", source_type: "top type",
      selection_reason: " first reason ", page: 0 })
  })
  it("projects fallback score and metadata aliases", () => {
    expect(service.projectHistoryDurableSources([{ ...source(), type: "pdf", metadata: {
      rerank_score: 85, bm25_norm: 0.5, chunkId: "chunk", reranking_strategy: "rerank",
      why_selected: "reason"
    } }])[0].metadata).toEqual({ score: 85, chunk_id: "chunk", retrieval_strategy: "rerank",
      source_type: "pdf", selection_reason: "reason" })
  })
  it.each(["chunk_id", "chunkId", "retrieval_strategy", "search_mode", "selection_reason", "why_selected"])(
    "rejects invalid raw %s rather than silently losing required display data", key => {
      expect(() => service.projectHistoryDurableSources([{ ...source(), metadata: { [key]: 0 } }])).toThrow()
    }
  )
  it("preserves zero score and exact empty URL while allowing redundant raw metadata", () => {
    expect(service.projectHistoryDurableSources([{ ...source(), url: "", score: 0,
      metadata: { score: 1, url: "", media_type: "pdf" }
    }])[0]).toEqual({ ...source(), url: "", metadata: { score: 0, source_type: "pdf" } })
    expect(() => service.projectHistoryDurableSources([{ ...source(), metadata: { url: "lost locator" } }])).toThrow()
  })
  it.each([{ media_id: 4 }, { page_number: 4 }, { line_range: [1, 3] }, { asset_url: "signed" }, { options: {} }, { headers: {} }])(
    "rejects raw unrepresentable required locators or execution data %#", metadata => {
      expect(() => service.projectHistoryDurableSources([{ ...source(), metadata }])).toThrow()
    }
  )
  it("adapts only allowlisted display metadata, without creating a server receipt", () => {
    const value = { version: 1, request_context_digest: digest, sources: [source()] }
    expect(service.historyResultV1ToMessageSources(value)).toEqual([source()])
    expect(() => service.historyResultV1ToMessageSources({ ...value, admission: admission() })).toThrow()
    expect(() => service.historyResultV1ToMessageSources({ ...value, request_context_digest: "bad" })).toThrow()
  })
})

describe("final body projection", () => {
  it("retains ECMAScript numeric-key, exponent, negative-zero and astral-string digest parity", () => {
    expect(service.historyDurableRequestDigest({ stream: true, model: "model", api_provider: "chosen",
      messages: [{ role: "user", content: "\u{1f600}" }], numeric: { "10": 1e21, "2": 1e-7, a: -0 },
      temperature: 0.125, explicit: null, omitted: undefined,
      tldw_turn: { user_message_id: uuid, result_v1: result([]), history_v1: { excluded: true } }
    })).toBe("22c91ead66e42ab9f80a6855b9e2525d48db9c76015942084b9b8b02ba73cf0d")
  })
  it("excludes only nested history, preserving every other body field and explicit null", () => {
    const input = body()
    expect(service.historyDurableRequestDigest({ ...input, tldw_turn: { ...input.tldw_turn, history_v1: { arbitrary: "excluded" } } })).toBe(historyDigest(input))
    for (const changed of [{ ...input, stream: false }, { ...input, model: "different" },
      { ...input, api_provider: "different" }, { ...input, temperature: 1 },
      { ...input, extra_body: { history_v1: "not excluded" } },
      { ...input, tldw_turn: { ...input.tldw_turn, user_message_id: "22345678-1234-4321-8123-123456789abc" } }
    ]) expect(service.historyDurableRequestDigest(changed)).not.toBe(service.historyDurableRequestDigest(input))
    expect(service.historyDurableRequestDigest({ ...input, absent: undefined })).toBe(service.historyDurableRequestDigest(input))
    const withoutNull = { ...input } as Record<string, unknown>
    delete withoutNull.temperature
    expect(service.historyDurableRequestDigest(withoutNull)).not.toBe(service.historyDurableRequestDigest(input))
    expect(service.historyDurableRequestDigest({ ...input, tldw_turn: { ...input.tldw_turn,
      result_v1: result([source()])
    } })).not.toBe(service.historyDurableRequestDigest(input))
  })
  it("finalizes the existing H1 selection around a detached frozen actual body", async () => {
    const native = owner()
    const captured = await captureHistorySnapshot(native, view, "send")
    if (captured.status !== "captured") throw Error(captured.status)
    const input = body()
    const prepared = service.prepareHistoryDurableTurn(input, {
      owner: native, mode: "plain", validate_lease: () => true,
      history: { kind: "selection", capture: captured, current_view: view }
    })
    input.messages[0].content = "changed"
    expect(prepared.body.messages[0].content).toBe("Frozen evidence")
    expect(Object.isFrozen(prepared.body.tldw_turn.history_v1)).toBe(true)
    expect(prepared.body.tldw_turn.history_v1).toMatchObject({ kind: "selection", selection: {
      purpose: "send", request_context_digest: prepared.request_context_digest
    } })
    expect(prepared.request_context_digest).toBe(service.historyDurableRequestDigest(prepared.body))
  })
  it("keeps the original admission digest while attaching a new retry body digest", () => {
    const prepared = service.prepareHistoryDurableTurn(body(), retryOptions())
    expect(prepared.body.tldw_turn.history_v1).toEqual({ version: 1, kind: "admission",
      admission: admission(), request_context_digest: prepared.request_context_digest })
  })
  it.each([
    { stream: undefined }, { stream: "true" }, { model: undefined }, { api_provider: null },
    { save_to_db: false }, { conversation_id: "other" }, { tldw_continuation: {} },
    { tldw_history_selection_v1: {} }, { tools: [] }, { functions: [] },
    { metadata: { tldw_retry_failed_turn: true } }, { extra_body: { stream: false } },
    { messages: [{ role: "assistant", content: "prefix" }, { role: "user", content: "new" }] },
    { messages: [{ role: "user", content: [{ type: "image_url", image_url: "asset" }] }] },
    { tldw_turn: { user_message_id: "not-uuid", result_v1: result([]) } },
    { tldw_turn: { user_message_id: uuid } }
  ])("rejects invalid or conflicting protocol controls %#", changed => {
    expect(() => service.prepareHistoryDurableTurn({ ...body(), ...changed }, retryOptions())).toThrow()
  })
  it.each([
    { input_message_id: "22345678-1234-4321-8123-123456789abc" },
    { owner_key: "other" }, { conversation_id: "other" }, { selection_digest: "A".repeat(64) },
    { headers: { authorization: "secret" } }
  ])("rejects invalid admission bindings %#", changed => {
    const options = retryOptions()
    expect(() => service.prepareHistoryDurableTurn(body(), { ...options,
      history: { kind: "admission", admission: { ...admission(), ...changed } }
    })).toThrow()
  })
  it("rejects expired request and owner leases", () => {
    const options = retryOptions()
    expect(() => service.prepareHistoryDurableTurn(body(), { ...options, validate_lease: () => false })).toThrow()
    expect(() => service.prepareHistoryDurableTurn(body(), { ...options, owner: { ...options.owner, validate_lease: () => false } })).toThrow()
  })
  it("rejects fork selections and changed views through the existing H1 finalizer", async () => {
    const native = owner()
    const captured = await captureHistorySnapshot(native, view, "send")
    if (captured.status !== "captured") throw Error(captured.status)
    const options = { owner: native, mode: "plain" as const, validate_lease: () => true,
      history: { kind: "selection" as const, capture: captured, current_view: view }
    }
    expect(() => service.prepareHistoryDurableTurn(body(), { ...options, history: { ...options.history,
      capture: { ...captured, purpose: "fork" }
    } })).toThrow()
    expect(() => service.prepareHistoryDurableTurn(body(), { ...options, history: { ...options.history,
      current_view: { ...view, selection_revision: 2 }
    } })).toThrow("stale_selection")
  })
})
