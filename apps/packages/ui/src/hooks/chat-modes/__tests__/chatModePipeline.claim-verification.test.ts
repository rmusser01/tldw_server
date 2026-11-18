import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Message } from "@/store/option"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
const mocks = vi.hoisted(() => ({ model: vi.fn(), verify: vi.fn(), save: vi.fn(async () => "history"), error: vi.fn(async () => "history") }))
vi.mock("@/models", () => ({ pageAssistModel: mocks.model }))
vi.mock("@/services/workspace-claim-verification", () => ({ verifyWorkspaceAnswer: mocks.verify, uncheckedAnswerVerification: (text: string) => ({ verdict: "unverified", text }) }))
vi.mock("@/db/dexie/helpers", () => ({ generateID: vi.fn(() => crypto.randomUUID()) }))
vi.mock("@/db/dexie/nickname", () => ({ getModelNicknameByID: vi.fn(async () => null) }))
vi.mock("@/utils/mcp-disclosure", () => ({ applyMcpModuleDisclosureFromToolCalls: vi.fn() }))
vi.mock("@/store/option", () => ({ useStoreMessageOption: { getState: () => ({ setHistory: vi.fn() }) } }))
import { runChatPipeline, type ChatModeParamsBase, type ChatModeDefinition } from "../chatModePipeline"
const sources = [{ pageContent: "Exact evidence.", metadata: { media_id: 7, title: "Source PDF", media_type: "pdf", citation_index: 1, page_number: 5 } }]
const mode: ChatModeDefinition<ChatModeParamsBase> = { id: "rag", buildUserMessage: context => ({ id: context.resolvedUserMessageId, isBot: false, name: "You", message: context.message, sources: [] }),
  buildAssistantMessage: context => ({ id: context.resolvedAssistantMessageId, isBot: true, name: "Assistant", message: "", sources: [] }),
  preparePrompt: async () => ({ chatHistory: [], humanMessage: { role: "user", content: "Question" }, sources }) }
const report = { verdict: "needs_revision", findings: [{ reason: "Wrong attribution", evidence_ids: ["1"] }] }
const params = () => ({ selectedModel: "openai/test", useOCR: false, saveToDb: false, claimVerificationWorkspaceId: "workspace-1",
  setMessages: vi.fn<ChatModeParamsBase["setMessages"]>(), saveMessageOnSuccess: mocks.save, saveMessageOnError: mocks.error, setHistory: vi.fn(),
  setIsProcessing: vi.fn(), setStreaming: vi.fn(), setAbortController: vi.fn(), historyId: "history", setHistoryId: vi.fn() })
beforeEach(() => { vi.clearAllMocks(); mocks.verify.mockResolvedValue(report); mocks.model.mockResolvedValue({ model: "test", apiProvider: "openai", saveToDb: false,
  stream: async function* () { yield "Final "; yield "answer [1]." } }) })
describe("final QA claim verification", () => {
  it("checks the complete generated answer and actual sources before canonical persistence", async () => {
    await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, params())
    expect(mocks.verify).toHaveBeenCalledWith(expect.objectContaining({ workspaceId: "workspace-1", text: "Final answer [1].", sources, provider: "openai", model: "test" }))
    expect(mocks.save).toHaveBeenCalledWith(expect.objectContaining({ fullText: "Final answer [1].", assistantMetadataExtra: expect.objectContaining({ claim_verification: report, workspace_rag_sources_v1: expect.any(Object) }) }))
    expect(mocks.verify.mock.invocationCallOrder[0]).toBeLessThan(mocks.save.mock.invocationCallOrder[0])
  })
  it("publishes pending state while retaining answer text, then saves findings", async () => {
    let resolve!: (value: unknown) => void
    mocks.verify.mockReturnValue(new Promise(done => { resolve = done }))
    const options = params()
    const rows: Message[] = []
    options.setMessages.mockImplementation(next => { const value = typeof next === "function" ? next(rows) : next; rows.splice(0, rows.length, ...value) })
    const running = runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, options)
    await vi.waitFor(() => expect(mocks.verify).toHaveBeenCalledOnce())
    expect(rows.find(row => row.isBot)).toMatchObject({ message: "Final answer [1].", metadataExtra: { claim_verification_pending: true } })
    expect(mocks.save).not.toHaveBeenCalled()
    resolve(report); await running
    expect(rows.find(row => row.isBot).metadataExtra.claim_verification_pending).toBeUndefined()
  })
  it("leaves non-workspace generation unchanged", async () => {
    await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, { ...params(), claimVerificationWorkspaceId: undefined })
    expect(mocks.verify).not.toHaveBeenCalled()
  })
  it("does not save verification to another account when ownership changes during checking", async () => {
    const owner = new AbortController()
    const options = { ...params(), servicePromptSnapshot: { scopeSignal: new AbortController().signal, scopeInvalidatedSignal: owner.signal } as unknown as ServicePromptSnapshot }
    mocks.verify.mockImplementation(async () => { owner.abort(); return report })
    await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, options)
    expect(mocks.verify).toHaveBeenCalledOnce()
    expect(mocks.save).not.toHaveBeenCalled()
    expect(mocks.error).not.toHaveBeenCalled()
  })
  it("saves a finished answer with an explicit unverified report when checks are unavailable", async () => {
    mocks.verify.mockResolvedValue({ verdict: "unverified", coverage: { complete: false } })
    await runChatPipeline(mode, "Question", "", false, [], [], new AbortController().signal, params())
    expect(mocks.save).toHaveBeenCalledWith(expect.objectContaining({ fullText: "Final answer [1].",
      assistantMetadataExtra: expect.objectContaining({ claim_verification: { verdict: "unverified", coverage: { complete: false } } }) }))
  })
})
