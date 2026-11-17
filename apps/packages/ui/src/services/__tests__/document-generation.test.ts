import type { QuickIngestOperation } from "../tldw/quick-ingest-authority"
import type { ServicePromptRequestScope } from "../tldw/domains/service-prompts"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  createDocumentClient,
  parseDocumentDraft,
  parseGeneratedDocument
} from "../document-generation"
const request = vi.hoisted(() => vi.fn())
vi.mock("../background-proxy", () => ({ bgRequest: request }))
const draft = {
  title: "Rowan",
  pages: [{ heading: "Telescope", text: "Selene" }]
}
const reply = (value: unknown, finish = "stop") => ({
  choices: [
    { finish_reason: finish, message: { content: JSON.stringify(value) } }
  ]
})
beforeEach(() => request.mockReset())
describe("generated document trust boundary", () => {
  it("preserves exact editable plain text", () =>
    expect(parseGeneratedDocument(reply(draft))).toEqual(draft))
  it.each([
    reply({ ...draft, id: 7 }),
    reply(draft, "length"),
    reply({ ...draft, pages: [] }),
    reply({ ...draft, pages: [{ heading: "Bad", text: "a\u0000b" }] }),
    reply({ ...draft, pages: [{ heading: "Bad", text: "a".repeat(6001) }] }),
    reply({ ...draft, title: "\ud800" }),
    { choices: [{ message: { content: "{}" }, finish_reason: "tool_calls" }] }
  ])("refuses malformed, clipped or foreign-identity output", (value) =>
    expect(() => parseGeneratedDocument(value)).toThrow(
      "invalid document draft"
    )
  )
  it("validates edited text by the same contract", () =>
    expect(() => parseDocumentDraft({ ...draft, title: " " })).toThrow())
})
describe("owned document requests", () => {
  const scope = {
    config: { serverUrl: "http://owner.invalid", authMode: "multi-user" },
    userId: "401"
  } satisfies ServicePromptRequestScope
  const owner = () =>
    ({
      authorityKey: "owner",
      authorityRevision: 1,
      requestScope: scope,
      signal: new AbortController().signal,
      assertCurrent: vi.fn(),
      isCurrent: () => true
    }) satisfies QuickIngestOperation
  it("uses the captured account for generation and bounded PDF requests", async () => {
    request
      .mockResolvedValueOnce(reply(draft))
      .mockResolvedValueOnce(
        new TextEncoder().encode("%PDF-complete\n%%EOF").buffer
      )
    const operation = owner(),
      controller = new AbortController()
    const client = createDocumentClient(operation)
    expect(
      await client.generate(
        { instructions: "Rowan", model: "local", provider: "llamacpp" },
        controller.signal
      )
    ).toEqual(draft)
    expect((await client.render(draft, controller.signal)).type).toBe(
      "application/pdf"
    )
    expect(
      request.mock.calls.map(([value]) => [
        value.path,
        value.method,
        value.servicePromptConfig,
        value.abortSignal.aborted
      ])
    ).toEqual([
      [
        "/api/v1/chat/completions",
        "POST",
        { ...scope.config, expectedUserId: "401" },
        false
      ],
      [
        "/api/v1/media/documents/render",
        "POST",
        { ...scope.config, expectedUserId: "401" },
        false
      ]
    ])
  })
  it("abandons an old owner before sending", async () => {
    const operation = owner()
    operation.assertCurrent.mockImplementation(() => {
      throw new Error("retired")
    })
    await expect(
      createDocumentClient(operation).exportPdf(1, new AbortController().signal)
    ).rejects.toThrow("retired")
    expect(request).not.toHaveBeenCalled()
  })
  it("does not publish a late result after Stop", async () => {
    const controller = new AbortController()
    request.mockImplementation(async () => {
      controller.abort()
      return reply(draft)
    })
    await expect(
      createDocumentClient(owner()).generate(
        { instructions: "Rowan", model: "local", provider: "llamacpp" },
        controller.signal
      )
    ).rejects.toThrow()
  })
  it("refuses partial PDF bytes without a complete trailer", async () => {
    request.mockResolvedValue(
      new TextEncoder().encode("%PDF-incomplete").buffer
    )
    await expect(
      createDocumentClient(owner()).render(draft, new AbortController().signal)
    ).rejects.toThrow("complete PDF")
  })
})
