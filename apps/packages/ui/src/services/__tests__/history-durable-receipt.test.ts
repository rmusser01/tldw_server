import { beforeEach, describe, expect, it } from "vitest"
import * as wire from "../history-durable-turn"

const inputId = "12345678-1234-4321-8123-123456789abc"
const resultId = "22345678-1234-4321-8123-123456789abc"
const owner = { owner_key: "native-key", conversation_id: "chat" }
const admission = { version: 1 as const, ...owner, input_message_id: inputId,
  input_message_revision: "1", selection_digest: "a".repeat(64) }
const requestDigest = "b".repeat(64)
const sources = [{ name: "Evidence", type: "pdf", mode: "rag" as const, url: "media:exact",
  pageContent: "Excerpt", metadata: { page: 1 } }]
const receipt = () => ({ version: 1, result_message_id: resultId, result_message_revision: "1",
  admission, request_context_digest: requestDigest, sources: structuredClone(sources) })
const parse = (value: unknown) => wire.validateHistoryDurableResultReceipt(owner, admission, requestDigest, sources, value)
beforeEach(() => {
  expect(wire.validateHistoryDurableResultReceipt, "strict receipt parser feature is missing").toBeTypeOf("function")
})
describe("strict selected durable result receipt", () => {
  it("returns a detached frozen observation without changing source URLs", () => {
    const value = receipt()
    const parsed = parse(value)
    value.sources[0].pageContent = "changed"
    expect(parsed.sources[0].pageContent).toBe("Excerpt")
    expect(parsed.sources[0].url).toBe("media:exact")
    expect(Object.isFrozen(parsed.admission)).toBe(true)
  })
  it.each([
    { version: 2 }, { version: true }, { result_message_id: "provider-id" },
    { result_message_id: inputId }, { result_message_revision: "2" },
    { request_context_digest: "c".repeat(64) }, { request_context_digest: "B".repeat(64) },
    { admission: { ...admission, input_message_id: resultId } },
    { admission: { ...admission, input_message_revision: "2" } },
    { admission: { ...admission, selection_digest: "c".repeat(64) } },
    { admission: { ...admission, conversation_id: "other" } },
    { admission: { ...admission, owner_key: "other" } },
    { admission: { ...admission, headers: {} } }, { headers: {} },
    { sources: [] }, { sources: [{ ...sources[0], pageContent: "different" }] },
    { sources: [{ ...sources[0], metadata: { api_key: "secret" } }] }
  ])("rejects mismatched, forged or unsupported receipt fields %#", changed => {
    expect(() => parse({ ...receipt(), ...changed })).toThrow()
  })
  it("rejects mismatched owner binding even when the receipt matches the supplied reference", () => {
    expect(() => wire.validateHistoryDurableResultReceipt({ ...owner, owner_key: "other" }, admission,
      requestDigest, sources, receipt())).toThrow()
  })
  it("accepts mandatory empty sources for a plain result", () => {
    expect(wire.validateHistoryDurableResultReceipt(owner, admission, requestDigest, [],
      { ...receipt(), sources: [] }).sources).toEqual([])
  })
})
