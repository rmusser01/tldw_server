import { beforeEach, describe, expect, it, vi } from "vitest"
import { HumanMessage, SystemMessage } from "@/types/messages"
import { canonicalHistoryJson } from "@/db/dexie/history-selection"
import { prepareHistoryContext } from "@/services/chat-history-selection"
import { historyDurableRequestDigest } from "@/services/history-durable-turn"
import { selectionDigest } from "@/utils/history-selection"
import type { HistorySelectionV1 } from "@/types/history-selection"
import { consumeStreamingChunk, extractStreamingChunkError } from "@/utils/streaming-chunks"

const calls = vi.hoisted(() => ({ stream: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => undefined, getConfig: async () => null,
  streamChatCompletion: (...args: unknown[]) => calls.stream(...args)
} }))
vi.mock("@/services/tldw", async () => {
  const { TldwChatService } = await import("@/services/tldw/TldwChat")
  return { tldwChat: new TldwChatService() }
})
import { ChatTldw } from "../ChatTldw"

const inputId = "12345678-1234-4321-8123-123456789abc"
const resultId = "22345678-1234-4321-8123-123456789abc"
const source = { name: "Evidence", type: "pdf", mode: "rag" as const, url: "media:exact",
  pageContent: "Excerpt", metadata: {} }
const model = (extra = {}) => new ChatTldw({ model: "tldw:actual-model", apiProvider: "actual-provider",
  saveToDb: true, conversationId: "chat", tldwTurn: { user_message_id: inputId },
  originalUserMessage: "Original", ...extra })
const prepare = (chat: ChatTldw, sources = [source]) => {
  expect(chat.prepareSelectedDurableRequest, "selected durable prepare feature is missing").toBeTypeOf("function")
  const bare = chat.prepareSelectedDurableRequest([new SystemMessage("Instructions"), new HumanMessage("Evidence")], sources)
  const selection: HistorySelectionV1 = { version: 1, owner_key: "native-key", conversation_id: "chat",
    interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "empty" }, selection_revision: 1,
    purpose: "send", messages: [], fences: { conversation: "1", history: "1", settings: "1" },
    storage_context_digest: "storage", request_context_digest: historyDurableRequestDigest(bare), selection_digest: "" }
  const finalized = { ...selection, selection_digest: selectionDigest(selection) }
  return prepareHistoryContext({ ...bare, tldw_turn: { ...bare.tldw_turn,
    history_v1: { version: 1, kind: "selection", selection: finalized }
  } }, () => true).payload as ReturnType<ChatTldw["prepareSelectedDurableRequest"]>
}
const admission = (request: ReturnType<typeof prepare>) => ({
  version: 1, owner_key: "native-key", conversation_id: "chat", input_message_id: inputId,
  input_message_revision: "1", selection_digest: request.tldw_turn.history_v1!.kind === "selection"
    ? request.tldw_turn.history_v1!.selection.selection_digest : request.tldw_turn.history_v1!.admission.selection_digest,
  messages: [], originating_selection_revision: 1
})
const frame = (request: ReturnType<typeof prepare>) => ({ tldw_conversation_id: "chat",
  tldw_user_message_id: inputId, tldw_history_admission_v1: admission(request) })
const resultFrame = (request: ReturnType<typeof prepare>) => ({ ...frame(request), tldw_message_id: resultId,
  tldw_history_result_v1: { version: 1, result_message_id: resultId, result_message_revision: "1",
    admission: Object.fromEntries(Object.entries(admission(request)).filter(([key]) => !["messages", "originating_selection_revision"].includes(key))),
    request_context_digest: historyDurableRequestDigest(request), sources: request.tldw_turn.result_v1!.sources }
})
beforeEach(() => calls.stream.mockReset())
describe("selected durable model transport", () => {
  it("builds an exact bare durable body with explicit provider/model/stream/save and original text", () => {
    const chat = model({ originalUserMessage: "  Original\n" })
    expect(chat.prepareSelectedDurableRequest, "selected durable prepare feature is missing").toBeTypeOf("function")
    const request = chat.prepareSelectedDurableRequest([new HumanMessage("Grounded evidence")], [source])
    expect(request).toMatchObject({ api_provider: "actual-provider", model: "actual-model", stream: true,
      save_to_db: true, conversation_id: "chat", tldw_turn: { user_message_id: inputId,
        result_v1: { version: 1, sources: [source] } }, messages: [
        { role: "system", content: "Grounded evidence" }, { role: "user", content: "  Original\n" }
      ] })
    expect(Object.keys(request.tldw_turn!)).toEqual(["result_v1", "user_message_id"])
    expect(Object.isFrozen(request)).toBe(true)
  })
  it("observes admission-only and result-only frames without treating provider IDs as persistence", async () => {
    const chat = model()
    const request = prepare(chat)
    calls.stream.mockImplementation(async function* () {
      yield { id: "provider-id", choices: [{ delta: { content: "Answer" } }] }
      yield frame(request)
      yield resultFrame(request)
    })
    const chunks = []
    for await (const chunk of await chat.stream([], { preparedRequest: request })) chunks.push(chunk)
    expect(chunks[0]).toBe("Answer")
    expect(chunks.some(chunk => typeof chunk === "object" && chunk?.tldw_history_result_v1)).toBe(true)
    expect(chat.historyAdmission).toEqual(admission(request))
    expect(chat.historyResult).toEqual(resultFrame(request).tldw_history_result_v1)
    expect(chat.serverMessageId).toBe(resultId)
    expect(chat.serverUserMessageId).toBe(inputId)
    expect(chat.serverMessagesAlreadyPersisted).toBe(true)
    expect(calls.stream.mock.calls[0][0]).toBe(request)
  })
  it("makes an admission visible when a receipt-only stream ends without a result", async () => {
    const chat = model()
    const request = prepare(chat)
    calls.stream.mockImplementation(async function* () { yield frame(request) })
    const stream = await chat.stream([], { preparedRequest: request })
    const first = await stream.next()
    expect(first.value).toMatchObject({ tldw_history_admission_v1: admission(request) })
    expect(chat.historyResult).toBeUndefined()
    expect(chat.serverMessagesAlreadyPersisted).toBe(false)
    await stream.return()
  })
  it("yields a receipt-only admission before waiting for the next inference frame", async () => {
    const chat = model()
    const request = prepare(chat)
    let released = false
    calls.stream.mockImplementation(async function* () {
      yield frame(request)
      released = true
      yield { choices: [{ delta: { content: "Answer" } }] }
    })
    const stream = await chat.stream([], { preparedRequest: request })
    expect((await stream.next()).value).toMatchObject({ tldw_history_admission_v1: admission(request) })
    expect(released).toBe(false)
    await stream.return()
  })
  it("preserves a verified settlement error for existing pipeline recovery without completing or resending", async () => {
    const chat = model()
    const request = prepare(chat)
    const error = { code: "selected_durable_result_unverified", type: "history_result_error",
      message: "Selected durable result could not be verified." }
    calls.stream.mockImplementation(async function* () {
      yield frame(request)
      yield { choices: [{ delta: { content: "Partial answer" } }] }
      yield { ...frame(request), error }
    })
    const chunks = []
    for await (const chunk of await chat.stream([], { preparedRequest: request })) chunks.push(chunk)
    const failed = chunks.find(chunk => extractStreamingChunkError(chunk))
    expect(failed).toMatchObject({ error })
    let cause: unknown
    try {
      consumeStreamingChunk({ fullText: "Partial answer", contentToSave: "Partial answer", apiReasoning: false }, failed)
    } catch (caught) { cause = caught }
    expect(cause).toMatchObject({ code: error.code, message: error.message })
    expect(chat.historyAdmission).toEqual(admission(request))
    expect(chat.historyResult).toBeUndefined()
    expect(chat.serverMessagesAlreadyPersisted).toBe(false)
    expect(calls.stream).toHaveBeenCalledOnce()
  })
  it("validates receipt identity before forwarding a settlement error", async () => {
    const chat = model()
    const request = prepare(chat)
    calls.stream.mockImplementation(async function* () {
      yield { ...frame(request), tldw_conversation_id: "other", error: {
        code: "selected_durable_result_unverified", message: "Selected durable result could not be verified." } }
    })
    const chunks = []
    await expect(async () => {
      for await (const chunk of await chat.stream([], { preparedRequest: request })) chunks.push(chunk)
    }).rejects.toThrow()
    expect(chunks).toEqual([])
    expect(chat.historyAdmission).toBeUndefined()
  })
  it("forwards workspace scope through the real chat service without adding scope to the body", async () => {
    const scope = { type: "workspace" as const, workspaceId: "space" }
    const chat = model({ scope })
    const request = prepare(chat)
    calls.stream.mockImplementation(async function* () { yield resultFrame(request) })
    for await (const _chunk of await chat.stream([], { preparedRequest: request })) { /* drain */ }
    expect(calls.stream.mock.calls[0][1].scope).toBe(scope)
    expect(request).not.toHaveProperty("scope_type")
    expect(request).not.toHaveProperty("workspace_id")
  })
  it.each([
    (request: ReturnType<typeof prepare>) => ({ ...frame(request), tldw_conversation_id: "other" }),
    (request: ReturnType<typeof prepare>) => ({ ...frame(request), tldw_user_message_id: resultId }),
    (request: ReturnType<typeof prepare>) => ({ ...frame(request), tldw_history_admission_v1: { ...admission(request), messages: [{ id: "invented", revision: "1" }] } }),
    (request: ReturnType<typeof prepare>) => ({ ...frame(request), tldw_history_admission_v1: { ...admission(request), originating_selection_revision: 2 } }),
    (request: ReturnType<typeof prepare>) => ({ ...resultFrame(request), tldw_message_id: "provider-id" }),
    (request: ReturnType<typeof prepare>) => ({ ...resultFrame(request), tldw_history_result_v1: { ...resultFrame(request).tldw_history_result_v1, sources: [] } }),
    (request: ReturnType<typeof prepare>) => ({ ...resultFrame(request), tldw_history_result_v1: { ...resultFrame(request).tldw_history_result_v1, request_context_digest: "a".repeat(64) } })
  ])("throws on a wrong receipt rather than accepting legacy identity %#", async invalid => {
    const chat = model()
    const request = prepare(chat)
    calls.stream.mockImplementation(async function* () { yield invalid(request) })
    const consume = async () => { for await (const _chunk of await chat.stream([], { preparedRequest: request })) { /* drain */ } }
    await expect(consume()).rejects.toThrow()
    expect(chat.historyResult).toBeUndefined()
    expect(chat.serverMessagesAlreadyPersisted).toBe(false)
  })
  it("rejects a conflicting observed result identity instead of replacing it", async () => {
    const chat = model()
    const request = prepare(chat)
    calls.stream.mockImplementation(async function* () {
      yield resultFrame(request)
      yield { ...resultFrame(request), tldw_message_id: "32345678-1234-4321-8123-123456789abc",
        tldw_history_result_v1: { ...resultFrame(request).tldw_history_result_v1, result_message_id: "32345678-1234-4321-8123-123456789abc" } }
    })
    await expect(async () => { for await (const _chunk of await chat.stream([], { preparedRequest: request })) { /* drain */ } }).rejects.toThrow()
    expect(chat.historyResult?.result_message_id).toBe(resultId)
  })
  it("matches a Retry admission reference without replacing its originating selection digest", async () => {
    const chat = model()
    const initial = prepare(chat)
    const reference = resultFrame(initial).tldw_history_result_v1.admission
    const bare = chat.prepareSelectedDurableRequest([new HumanMessage("Original")], [source])
    const retry = prepareHistoryContext({ ...bare, tldw_turn: { ...bare.tldw_turn,
      history_v1: { version: 1, kind: "admission", admission: reference, request_context_digest: historyDurableRequestDigest(bare) }
    } }, () => true).payload as ReturnType<typeof prepare>
    calls.stream.mockImplementation(async function* () { yield resultFrame(retry) })
    for await (const _chunk of await chat.stream([], { preparedRequest: retry })) { /* drain */ }
    expect(chat.historyResult?.admission.selection_digest).toBe(reference.selection_digest)
    expect(chat.historyResult?.request_context_digest).toBe(historyDurableRequestDigest(retry))
  })
  it("rejects changed prepared inference values before transport", async () => {
    const chat = model()
    const request = prepare(chat)
    await expect(chat.stream([], { preparedRequest: { ...request, stream: false } })).rejects.toThrow()
    expect(calls.stream).not.toHaveBeenCalled()
    expect(canonicalHistoryJson(request)).toContain('"stream":true')
  })
  it("does not downgrade an explicit null nested envelope to the legacy receipt path", async () => {
    const chat = model()
    const request = prepare(chat)
    await expect(chat.stream([], { preparedRequest: { ...request, tldw_turn: { ...request.tldw_turn, history_v1: null } } })).rejects.toThrow()
    expect(calls.stream).not.toHaveBeenCalled()
  })
})
