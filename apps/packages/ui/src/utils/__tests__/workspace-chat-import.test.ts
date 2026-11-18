import { describe, expect, it } from "vitest"
import { remapImportedWorkspaceChat, buildWorkspaceChatImport } from "../workspace-chat-import"
import { portableFixture } from "./fixtures/portable-workspace"
import type { WorkspaceBundleChatSession } from "@/store/workspace-bundle"
import { claimTextHash } from "../claim-verification"

const session = (): WorkspaceBundleChatSession => ({ historyId: "old-browser", serverChatId: "old-server", history: [
  { role: "user", content: "Question" }, { role: "assistant", content: "Answer" }
], messages: [
  { id: "local-user", serverMessageId: "remote-user", isBot: false, name: "You", message: "Question", createdAt: 1000, sources: [], parentMessageId: null },
  { id: "local-answer", serverMessageId: "remote-answer", serverMessageVersion: 9, isBot: true, name: "AI", message: "Answer", createdAt: 2000, sources: [], parentMessageId: "remote-user" }
] })

describe("portable chat graph", () => {
  it("retains exact verification findings and marks a restored check as historical", async () => {
    const original = session()
    const report = { version: 1, text_sha256: claimTextHash("Answer"), verdict: "needs_revision", provider: "openai", model: "test",
      coverage: { total: 1, evaluated: 1, supported: 0, uncovered: 0, complete: true }, findings: [{ id: "C1", text: "Answer", start: 0, end: 6,
        evidence_ids: ["1"], status: "unsupported", reason: "The source describes another agency." }], evidence: { "1": { text_sha256: "a".repeat(64) } }, limits: {}, context: {} }
    original.messages[1].metadataExtra = { claim_verification: report }
    const before = JSON.stringify(original)
    let next = 0
    const restored = remapImportedWorkspaceChat(original, () => `new-${++next}`)
    expect(restored.messages[1].metadataExtra).toEqual({ claim_verification: report, claim_verification_imported: true })
    const bundle = portableFixture()
    bundle.workspace.chatSession = restored
    const imported = await buildWorkspaceChatImport(bundle, "destination")
    expect(imported!.messages[1].metadata_extra).toMatchObject({ claim_verification: report, claim_verification_imported: true })
    expect(JSON.stringify(original)).toBe(before)
  })

  it("restores ID-less legacy rows without inventing parent links or timestamps", async () => {
    const original = session()
    for (const row of original.messages) {
      delete row.id
      delete row.serverMessageId
      delete row.serverMessageVersion
      delete row.parentMessageId
      delete row.createdAt
    }
    let sequence = 0
    const result = remapImportedWorkspaceChat(original, () => `new-${++sequence}`)
    expect(result.messages).toEqual([
      { id: "new-1", parentMessageId: null, isBot: false, name: "You", message: "Question", sources: [] },
      { id: "new-2", parentMessageId: null, isBot: true, name: "AI", message: "Answer", sources: [] },
    ])
    expect(result).toMatchObject({ historyId: null, serverChatId: null, history: original.history })
    const bundle = portableFixture()
    bundle.workspace.chatSession = result
    await expect(buildWorkspaceChatImport(bundle, "destination")).rejects.toThrow("original message timestamps")
  })

  it("resolves an explicit server parent alias when the legacy row has no local ID", () => {
    const original = session()
    delete original.messages[0].id
    let sequence = 0
    const result = remapImportedWorkspaceChat(original, () => `new-${++sequence}`)
    expect(result.messages.map(row => [row.id, row.parentMessageId])).toEqual([["new-1", null], ["new-2", "new-1"]])
    expect(result.messages.map(row => [row.message, row.createdAt])).toEqual([["Question", 1000], ["Answer", 2000]])
    expect(result.messages.every(row => !row.serverMessageId && !row.serverMessageVersion)).toBe(true)
  })

  it("preserves branching graph edges across reordered rows and mixed source aliases", () => {
    for (let size = 2; size <= 40; size++) {
      const original = session()
      original.history = []
      const rows = Array.from({ length: size }, (_, index) => ({
        id: `local-${index}`, serverMessageId: `server-${index}`, isBot: index % 2 === 1,
        name: index % 2 ? "AI" : "You", message: `Message ${index}`, createdAt: index, sources: [],
        parentMessageId: index ? `${index % 2 ? "local" : "server"}-${Math.floor((index - 1) / 2)}` : null,
      }))
      original.messages = [...rows.slice(1), rows[0]]
      let sequence = 0
      const result = remapImportedWorkspaceChat(original, () => `new-${++sequence}`)
      const byText = new Map(result.messages.map(row => [row.message, row]))
      for (let index = 1; index < size; index++)
        expect(byText.get(`Message ${index}`)!.parentMessageId).toBe(byText.get(`Message ${Math.floor((index - 1) / 2)}`)!.id)
      expect(result.messages.map(row => [row.message, row.createdAt])).toEqual(original.messages.map(row => [row.message, row.createdAt]))
    }
  })
  it.each(["remote-user", "local-user"])("resolves the %s parent alias before clearing source authority", parent => {
    const original = session()
    original.messages[1].parentMessageId = parent
    let sequence = 0
    const result = remapImportedWorkspaceChat(original, () => `new-${++sequence}`)
    expect(result.messages.map(row => [row.id, row.parentMessageId])).toEqual([["new-1", null], ["new-2", "new-1"]])
    expect(result.messages.map(row => [row.message, row.createdAt])).toEqual([["Question", 1000], ["Answer", 2000]])
    expect(result.messages.every(row => !row.serverMessageId && !row.serverMessageVersion)).toBe(true)
    expect(result).toMatchObject({ historyId: null, serverChatId: null, history: original.history })
  })

  it.each(["dangling", "ambiguous", "cycle", "duplicate", "hidden-history"])("refuses %s instead of changing the conversation", kind => {
    const original = session()
    if (kind === "dangling") original.messages[1].parentMessageId = "unknown"
    if (kind === "ambiguous") original.messages[1].id = "remote-user"
    if (kind === "cycle") original.messages[0].parentMessageId = "local-answer"
    if (kind === "duplicate") original.messages.push({ ...original.messages[1] })
    if (kind === "hidden-history") original.history.push({ role: "system", content: "Private instructions" })
    expect(() => remapImportedWorkspaceChat(original)).toThrow(/chat|message|parent|history/i)
  })

  it("builds a retry-stable scoped import with original timestamps and citation provenance", async () => {
    const bundle = portableFixture()
    bundle.workspace.chatSession = session()
    let sequence = 0
    bundle.workspace.chatSession = remapImportedWorkspaceChat(bundle.workspace.chatSession, () => `new-${++sequence}`)
    const first = await buildWorkspaceChatImport(bundle, "destination")
    const second = await buildWorkspaceChatImport(bundle, "destination")
    expect(first).toEqual(second)
    expect(first).toMatchObject({ scope_type: "workspace", workspace_id: "destination", created_at: "1970-01-01T00:00:01.000Z", messages: [
      { id: "new-1", role: "user", timestamp: "1970-01-01T00:00:01.000Z", parent_message_id: null },
      { id: "new-2", role: "assistant", timestamp: "1970-01-01T00:00:02.000Z", parent_message_id: "new-1" }
    ] })
    expect(JSON.stringify(first)).not.toMatch(/old-server|remote-user/)
  })
  it("preserves recorded generation facts without inferring a model from its display label", async () => {
    const bundle = portableFixture()
    bundle.workspace.chatSession!.messages[0].modelId = "deepseek-flash"
    bundle.workspace.chatSession!.messages[0].metadataExtra = { provider: "deepseek", generation_status: "complete", usage: { completion_tokens: 7 } }
    const request = await buildWorkspaceChatImport(bundle, "destination")
    expect(request!.messages[0].metadata).toEqual({ model_id: "deepseek-flash", provider: "deepseek", generation_status: "complete", usage: { completion_tokens: 7 } })
    delete bundle.workspace.chatSession!.messages[0].modelId
    delete bundle.workspace.chatSession!.messages[0].metadataExtra
    bundle.workspace.chatSession!.messages[0].modelName = "Assistant"
    expect((await buildWorkspaceChatImport(bundle, "destination"))!.messages[0].metadata).toBeUndefined()
  })
})
