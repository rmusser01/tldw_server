import { describe, expect, it, vi } from "vitest"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, resolve } from "node:path"
// A pure source import must not initialize either DB authority or native transport.
vi.mock("@/services/chat-history-selection", () => { throw new Error("native service dependency") })
vi.mock("@/db/dexie/history-selection", () => { throw new Error("DB authority dependency") })
import { historyResultV1ToMessageSources, parseHistoryDurableResult, projectHistoryDurableSources } from "../history-durable-sources"

// Recorded document metadata from source23-native-url-generation.json, not native UAT.
const metadata = { title: "downloaded_323911489848964201", media_type: "document", url: "https://example.com/",
  created_at: "2026-09-30T23:08:27.793Z", transcription_model: "Imported", last_modified: "2026-09-30T23:35:33.096Z",
  source: "media_db", media_id: "2", chunk_index: 1, total_chunks: 3, retrieval_mode: "late_chunk",
  start_char: 46, end_char: 202, chunk_type: "text", paragraph_kind: "paragraph", ancestry_titles: [],
  highlighted: "**This** **domain**", match_count: 29, snippets: ["...rvice, avoid relying on it for testing and monitoring purposes."],
  chunk_id: "late_chunk:2:1" }
const excerpt = "This domain is for use in documentation examples without needing permission. This is not a service, avoid relying on it for testing and monitoring purposes."
const source = { name: metadata.title, type: metadata.media_type, mode: "rag", url: metadata.url, pageContent: excerpt, metadata }
describe("pure durable sources and observed retrieval metadata", () => {
  it("normalizes known serialized numeric IDs and omits bookkeeping outside the durable wire", () => {
    const raw = { ...source, metadata: { ...metadata, media_id: 7, chunk_id: 17,
      note_id: "note-7", record_id: 19, start: 0, end: 35 } }
    const projected = projectHistoryDurableSources([raw])[0]
    expect(projected).toEqual({ ...projectHistoryDurableSources([source])[0],
      metadata: { ...projectHistoryDurableSources([source])[0].metadata, media_id: "7", chunk_id: "17" } })
    expect(raw.metadata.media_id).toBe(7)
    expect(raw.metadata.chunk_id).toBe(17)
    expect(parseHistoryDurableResult({ version: 1, sources: [projected] }, "rag").sources[0]).toEqual(projected)
  })
  it("retains the selected semantic type when a distinct media type is serialized", () => {
    const projected = projectHistoryDurableSources([{ ...source, type: "note",
      metadata: { ...metadata, type: "note", media_type: "text" } }])[0]
    expect(projected.type).toBe("note")
    expect(projected.metadata.source_type).toBe("note")
    expect(projected.pageContent).toBe(excerpt)
    expect(projected.metadata).not.toHaveProperty("media_type")
    expect(projected.metadata).not.toHaveProperty("note_id")
  })
  it.each([0, Number.MAX_SAFE_INTEGER])("preserves the complete safe numeric identifier boundary %s", id => {
    const projected = projectHistoryDurableSources([{ ...source,
      metadata: { ...metadata, media_id: id, chunk_id: id, note_id: id, record_id: id } }])[0]
    expect(projected.metadata.media_id).toBe(String(id))
    expect(projected.metadata.chunk_id).toBe(String(id))
  })
  it.each([
    { note_id: null }, { note_id: " " }, { note_id: "x".repeat(513) },
    { record_id: true }, { record_id: -1 }, { record_id: 1.5 },
    { media_id: 1.5 }, { media_id: -1 }, { media_id: Number.MAX_SAFE_INTEGER + 1 },
    { chunk_id: 1.5 }, { chunk_id: Number.NaN }, { chunk_id: false },
    { start: -1 }, { end: "35" }, { start: 2, end: 1 },
    { media_type: " " }, { media_type: "x".repeat(129) }
  ])("rejects malformed recognized serializer bookkeeping %#", addition => {
    expect(() => projectHistoryDurableSources([{ ...source, metadata: { ...metadata, ...addition } }])).toThrow()
  })
  it("keeps numeric identifiers invalid on the already-projected durable wire", () => {
    expect(() => parseHistoryDurableResult({ version: 1, sources: [{ ...source,
      metadata: { media_id: 7 } }] }, "rag")).toThrow()
  })
  it("projects latest-dev retrieval bookkeeping without changing source evidence or locators", () => {
    const observed = { ...metadata, source_id: "2", evidence_origin: "local_library",
      section_path: "Lumen Project Field Memo", ancestry_titles: ["Lumen Project Field Memo"] }
    expect(projectHistoryDurableSources([{ ...source, metadata: observed }])).toEqual(
      projectHistoryDurableSources([source])
    )
  })
  it.each([
    { source_id: null }, { source_id: 2 }, { source_id: "x".repeat(513) },
    { evidence_origin: null }, { evidence_origin: 1 }, { evidence_origin: "x".repeat(129) },
    { section_path: null }, { section_path: "x".repeat(1001) },
    { ancestry_titles: null }, { ancestry_titles: [1] },
    { ancestry_titles: ["x".repeat(1001)] }, { ancestry_titles: Array(21).fill("Title") },
  ])("rejects malformed latest-dev retrieval bookkeeping %#", addition => {
    expect(() => projectHistoryDurableSources([{ ...source, metadata: { ...metadata, ...addition } }])).toThrow()
  })
  it("preserves the winning source_type through raw projection, strict validation and display restoration", () => {
    const raw = { name: "PDF evidence", type: "pdf", source_type: " vector-evidence ", mode: "rag", url: "urn:exact:evidence",
      pageContent: "Exact excerpt", metadata: { source: "Attribution", title: "Title", chunk_id: "chunk:1", page: 2 } }
    const wire = projectHistoryDurableSources([raw])
    const validated = parseHistoryDurableResult({ version: 1, sources: wire }, "rag")
    expect(validated.sources[0]).toEqual({ name: raw.name, type: "pdf", mode: "rag", url: raw.url,
      pageContent: raw.pageContent, metadata: { ...raw.metadata, source_type: " vector-evidence " } })
    expect(validated.sources[0]).not.toHaveProperty("source_type")
    const displayed = historyResultV1ToMessageSources({ ...validated, request_context_digest: "a".repeat(64) })
    // MessageSource's winning alias is top-level source_type before the media type.
    expect(displayed[0].source_type || displayed[0].type || displayed[0].metadata.source_type).toBe(" vector-evidence ")
    expect(displayed[0].type).toBe("pdf")
    expect(projectHistoryDurableSources(displayed)).toEqual(wire)
    expect(Object.isFrozen(displayed[0])).toBe(true)
    expect(Object.isFrozen(displayed[0].metadata)).toBe(true)
  })
  it("does not treat display-only aliases as eligible strict wire fields", () => {
    const raw = { name: "PDF evidence", type: "pdf", source_type: "vector", mode: "rag", url: "",
      pageContent: "Exact excerpt", metadata: {} }
    const result = { version: 1, sources: projectHistoryDurableSources([raw]) }
    const displayed = historyResultV1ToMessageSources({ ...result, request_context_digest: "a".repeat(64) })
    expect(() => parseHistoryDurableResult({ version: 1, sources: displayed }, "rag")).toThrow()
    expect(parseHistoryDurableResult(result, "rag")).toEqual(result)
  })
  it("returns detached JSON with absent optional members omitted", () => {
    const parsed = parseHistoryDurableResult({ version: 1, sources: [{ ...source, metadata: { source: undefined } }] }, "rag")
    expect(Object.keys(parsed.sources[0].metadata)).toEqual([])
  })
  it("preserves every approved locator in the complete recorded native document route", () => {
    const documents = JSON.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "fixtures/history-durable-source23.documents.json"), "utf8")) as
      Array<{ content: string; metadata: Record<string, unknown> }>
    const raw = documents.map(document => ({ name: String(document.metadata.title || document.metadata.source || "untitled"),
      type: String(document.metadata.type || document.metadata.media_type || "unknown"), mode: "rag", url: String(document.metadata.url || ""),
      pageContent: document.content, metadata: document.metadata }))
    expect(raw).toHaveLength(4)
    expect(raw.every(entry => entry.metadata.media_id === "2" && typeof entry.metadata.chunk_id === "string")).toBe(true)
    const projected = projectHistoryDurableSources(raw)
    expect(projected.map(entry => entry.pageContent)).toEqual(documents.map(entry => entry.content))
    for (const [index, entry] of projected.entries()) {
      for (const key of ["media_id", "author", "chunk_index", "total_chunks", "start_char", "end_char", "chunk_start", "chunk_end"]) {
        expect(entry.metadata[key as keyof typeof entry.metadata]).toEqual(documents[index].metadata[key])
      }
    }
    const restored = historyResultV1ToMessageSources({ version: 1, sources: projected, request_context_digest: "a".repeat(64) })
    expect(restored.map(entry => entry.metadata)).toEqual(projected.map(entry => entry.metadata))
  })
  it("preserves the recorded media ID and exact character/chunk locators", () => {
    const projected = projectHistoryDurableSources([source])[0]
    expect(projected.metadata).toMatchObject({ media_id: "2", start_char: 46, end_char: 202, chunk_index: 1, total_chunks: 3 })
  })
  it("omits only observed retrieval bookkeeping when no unsupported locator/attribution is present", () => {
    const { media_id: _mediaId, start_char: _start, end_char: _end, chunk_index: _index, total_chunks: _total, ...displayMetadata } = metadata
    const projected = projectHistoryDurableSources([{ ...source, metadata: displayMetadata },
      { ...source, pageContent: "Second exact excerpt", metadata: { ...displayMetadata, chunk_id: "late_chunk:2:2" } }])
    expect(projected.map(entry => entry.pageContent)).toEqual([excerpt, "Second exact excerpt"])
    expect(projected[0]).toEqual({ ...source, metadata: { title: metadata.title, source: "media_db", chunk_id: "late_chunk:2:1", source_type: "document" } })
  })
  it.each(["section_title", "asset_token", "api_key", "headers", "execution", "unknown_locator"])("keeps unknown %s as a rejection gate", key => {
    expect(() => projectHistoryDurableSources([{ ...source, metadata: { source: "media_db", chunk_id: "late_chunk:2:1", [key]: "required" } }])).toThrow()
  })
  it.each([
    { media_id: " " }, { media_id: "x".repeat(513) }, { media_id: null },
    { author: "x".repeat(1001) }, { author: null },
    { chunk_index: true }, { chunk_index: -1 }, { chunk_index: "0" }, { total_chunks: 0 },
    { chunk_index: 3, total_chunks: 3 }, { start_char: 1 }, { end_char: 1 },
    { start_char: 2, end_char: 1 }, { chunk_start: 1 }, { chunk_end: 1 },
    { chunk_start: 2, chunk_end: 1 }, { chunk_start: 0, chunk_end: Number.MAX_SAFE_INTEGER + 1 }
  ])("rejects malformed locator metadata %# without coercion or dropping fields", metadata => {
    expect(() => projectHistoryDurableSources([{ ...source, metadata }])).toThrow()
    expect(() => parseHistoryDurableResult({ version: 1, sources: [{ ...source, metadata }] }, "rag")).toThrow()
  })
  it("preserves author attribution and both complete character ranges exactly", () => {
    const metadata = { media_id: "1", author: " Mira Chen ", chunk_index: 0,
      start_char: 0, end_char: 10, chunk_start: 0, chunk_end: 10 }
    expect(projectHistoryDurableSources([{ ...source, metadata }])[0].metadata).toMatchObject(metadata)
  })
})
