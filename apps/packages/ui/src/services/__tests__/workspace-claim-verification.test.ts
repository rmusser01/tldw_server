import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ServicePromptSnapshot } from "../service-prompts"
import { claimTextHash } from "@/utils/claim-verification"
import { verifyWorkspaceAnswer } from "../workspace-claim-verification"
const mocks = vi.hoisted(() => ({ sources: vi.fn(), request: vi.fn() }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getWorkspaceSources: mocks.sources } }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.request }))
const text = "The source proposes a transfer [1]."
const receipt = () => ({ version: 1, text_sha256: claimTextHash(text), verdict: "needs_revision", provider: "openai", model: "test",
  coverage: { total: 1, evaluated: 1, supported: 0, uncovered: 0, complete: true }, findings: [{ id: "C1", text, start: 0, end: text.length, evidence_ids: ["1"], status: "unsupported", reason: "The cited source describes another agency." }], evidence: { "1": { text_sha256: "a".repeat(64) } }, limits: {}, context: {} })
const input = () => ({ workspaceId: "owned", text, provider: "openai", model: "test", sources: [{ pageContent: "Exact source text", metadata: { media_id: 7, citation_index: 1, start_char: 10, end_char: 27 } }], signal: new AbortController().signal,
  scope: { requestScope: { userId: "1", config: { serverUrl: "https://owned.test", authMode: "multi-user" } }, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal } as unknown as ServicePromptSnapshot })
beforeEach(() => { vi.clearAllMocks(); mocks.sources.mockResolvedValue([{ id: "source-1", workspace_id: "owned", media_id: 7 }]); mocks.request.mockResolvedValue(receipt()) })
describe("final workspace answer citation check", () => {
  it("sends exact final text and enrolled source locators with captured ownership", async () => {
    const args = input()
    expect(await verifyWorkspaceAnswer(args)).toEqual(receipt())
    expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/research-workspace/citations/verify", method: "POST",
      servicePromptConfig: expect.objectContaining({ serverUrl: "https://owned.test", expectedUserId: "1" }),
      body: { workspace_id: "owned", text, api_provider: "openai", model: "test", evidence: [{ id: "1", source_id: "source-1", quote: "Exact source text", start_char: 10, end_char: 27 }] } }))
  })
  it("never uses a source from another workspace or invented locators", async () => {
    mocks.sources.mockResolvedValue([{ id: "foreign", workspace_id: "other", media_id: 7 }])
    await verifyWorkspaceAnswer(input())
    expect(mocks.request.mock.calls[0][0].body.evidence).toEqual([])
  })
  it.each(["provider", "malformed", "stale", "identity"]) ("retains an explicit unchecked report after %s failure", async failure => {
    if (failure === "provider") mocks.request.mockRejectedValue(new Error("private-provider-message"))
    else mocks.request.mockResolvedValue(failure === "stale" ? { ...receipt(), text_sha256: "f".repeat(64) } : failure === "identity" ? { ...receipt(), model: "different-model" } : {})
    const report = await verifyWorkspaceAnswer(input())
    expect(report.verdict).toBe("unverified")
    expect(report.text_sha256).toBe(claimTextHash(text))
    expect(report.coverage.complete).toBe(false)
    expect(JSON.stringify(report)).not.toContain("private-provider-message")
  })
  it("does not dispatch after the captured account changes during source lookup", async () => {
    const owner = new AbortController(), args = input()
    args.scope.scopeInvalidatedSignal = owner.signal
    mocks.sources.mockImplementation(async () => { owner.abort(); return [] })
    await expect(verifyWorkspaceAnswer(args)).rejects.toThrow()
    expect(mocks.request).not.toHaveBeenCalled()
  })
})
