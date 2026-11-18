import { beforeEach, describe, expect, it, vi } from "vitest"
import { exportWorkspaceResearchPack, restoreWorkspaceResearchPack, restoreLegacyWorkspaceBundle } from "../workspace-research-pack"
import { portableFixture } from "@/utils/__tests__/fixtures/portable-workspace"
import type { ServicePromptSnapshot } from "../service-prompts"
import type { CompleteResearchPack } from "@/utils/workspace-research-pack"
import { isServicePromptRequestPath } from "../tldw/service-prompt-scope-error"
import { servicePromptAuthorityKey } from "../tldw/domains/service-prompts"

const mocks = vi.hoisted(() => ({ request: vi.fn(), chatImport: vi.fn(), chatMessages: vi.fn(), upload: vi.fn(), export: vi.fn(), download: vi.fn(), build: vi.fn(), persist: vi.fn(), current: true }))
vi.mock("@/services/background-proxy", () => {
  const guard = (request: { path: string; method?: string; servicePromptConfig?: unknown }) => {
    if (request.servicePromptConfig && !isServicePromptRequestPath(request.path, request.method))
      throw new Error("A Service Prompt config can only be used with Service Prompt requests.")
  }
  return {
    bgRequest: (request: Parameters<typeof guard>[0]) => { guard(request); return request.path === "/api/v1/chats/import" ? mocks.chatImport(request) : mocks.request(request) },
    bgUpload: (request: Parameters<typeof guard>[0]) => { guard(request); return mocks.upload(request) }
  }
})
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { exportChatbook: mocks.export, downloadChatbookExport: mocks.download, listChatMessages: mocks.chatMessages } }))
vi.mock("@/utils/workspace-research-pack", async original => ({ ...(await original<typeof import("@/utils/workspace-research-pack")>()), buildCompleteResearchPack: mocks.build }))
vi.mock("@/services/workspace-artifact-persistence", () => ({ rememberCanonicalArtifactWorkspace: vi.fn(), persistWorkspaceArtifact: mocks.persist }))
const controller = new AbortController()
const scope = { scopeKey: "owner", requestScope: { config: { serverUrl: "https://destination.test", authMode: "single-user", apiKey: "never-export-this-secret" }, userId: "owner" }, scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, release: vi.fn(), definitions: {}, capability: "unchecked" } as ServicePromptSnapshot
const options = () => ({ scope, isCurrent: () => mocks.current, onProgress: vi.fn(), pollIntervalMs: 0 })
const pack = (): CompleteResearchPack => ({ bundle: portableFixture(), notes: [portableFixture().workspace.snapshot.currentNote], chatbook: new Uint8Array([1, 2]), fingerprint: "a".repeat(64), manifest: { format: "tldw.complete-research-pack", schemaVersion: 1, sourceWorkspaceId: "old", mediaIds: ["1"], noteIds: ["note-old"], files: [] } })

describe("scoped complete research pack orchestration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
    mocks.current = true
    mocks.export.mockResolvedValue({ success: true, job_id: "export-1" })
    mocks.download.mockResolvedValue({ blob: new Blob(["chatbook"]), filename: "archive.zip" })
    mocks.build.mockResolvedValue(new Blob(["complete pack"]))
    mocks.upload.mockResolvedValue({ success: true, job_id: "import-1" })
    mocks.persist.mockImplementation(async artifact => ({ ...artifact, workspaceArtifactId: artifact.id, workspaceArtifactVersion: 1 }))
    mocks.chatImport.mockImplementation(async request => ({ id: request.body.id, scope_type: "workspace", workspace_id: request.body.workspace_id }))
    mocks.chatMessages.mockImplementation(async () => structuredClone(mocks.chatImport.mock.calls.at(-1)![0].body.messages).map(row => ({ ...row,
      created_at: row.timestamp, version: 1 })))
  })

  it("paginates associated Notes and uses explicit Media/Note allowlists with original files", async () => {
    mocks.request.mockImplementation(async request => {
      if (request.path.includes("/notes/search/")) {
        const offset = new URL(request.path, "https://local").searchParams.get("offset")
        return offset === "0" ? { items: [pack().notes[0]], pagination: { has_more: true, next_offset: 1 } } : { items: [{ ...pack().notes[0], id: "second-note" }], pagination: { has_more: false } }
      }
      if (request.path.includes("/notes/note-old")) return pack().notes[0]
      if (request.path.includes("/export/jobs/")) return { status: "completed" }
      throw new Error(`Unexpected ${request.path}`)
    })
    await exportWorkspaceResearchPack(portableFixture(), "workspace-old", options())
    expect(mocks.export).toHaveBeenCalledWith(expect.objectContaining({ content_selections: { media: ["1"], note: ["note-old", "second-note"] }, include_media: true, media_quality: "original", include_generated_content: false }), expect.objectContaining({ requestScope: scope.requestScope }))
    expect(mocks.build.mock.calls[0][0].notes).toHaveLength(2)
    expect(JSON.stringify(mocks.build.mock.calls[0][0])).not.toContain("never-export-this-secret")
    expect(mocks.request.mock.calls.every(([request]) => request.servicePromptConfig?.serverUrl === scope.requestScope.config.serverUrl && request.servicePromptConfig?.expectedUserId === "owner")).toBe(true)
  })

  it.each([
    "Project 2025 — PDF acceptance",
    "研究 📚",
    "Policy / Justice: <review> & questions",
    "A".repeat(300)
  ])("exports a validator-safe backup while preserving the workspace title %s", async title => {
    const bundle = portableFixture()
    bundle.workspace.name = bundle.workspace.snapshot.workspaceName = title
    const completed = new Blob(["complete pack"])
    mocks.build.mockResolvedValueOnce(completed)
    mocks.export.mockImplementationOnce(async ({ name }) => {
      // ChatbookValidator accepts this punctuation subset and at most 255 characters.
      if (!name.trim() || name.length > 255 || !/^[\w\s\-.,!?'"()[\]{}@]+$/.test(name))
        throw new Error("Name contains invalid characters or is too long")
      return { success: true, job_id: "export-1" }
    })
    mocks.request.mockImplementation(async request => {
      if (request.path.includes("/notes/search/")) return { items: [], pagination: { has_more: false } }
      if (request.path.includes("/notes/note-old")) return pack().notes[0]
      if (request.path.includes("/export/jobs/")) return { status: "completed" }
      throw new Error(`Unexpected ${request.path}`)
    })

    await expect(exportWorkspaceResearchPack(bundle, "workspace-old", options())).resolves.toBe(completed)
    expect(mocks.build.mock.calls[0][0].bundle.workspace).toMatchObject({ name: title, snapshot: { workspaceName: title } })
  })

  it("refuses stale scope before exporting or restoring anything", async () => {
    mocks.current = false
    await expect(exportWorkspaceResearchPack(portableFixture(), "old", options())).rejects.toThrow(/changed|current/i)
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/changed|current/i)
    expect(mocks.upload).not.toHaveBeenCalled()
    expect(mocks.export).not.toHaveBeenCalled()
  })

  it("requires complete verified mappings before creating a destination workspace", async () => {
    mocks.request.mockResolvedValue({ status: "completed", id_maps: { media: {}, notes: { "note-old": "note-new" } } })
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/mapping/i)
    expect(mocks.request.mock.calls.some(([request]) => request.method === "PUT" && request.path.includes("/workspaces/"))).toBe(false)
  })

  it("uploads one archive under the Chatbooks singular file field", async () => {
    mocks.request.mockResolvedValue({ status: "failed" })
    const operation = options()
    await expect(restoreWorkspaceResearchPack(pack(), operation)).rejects.toThrow(/Partial/)
    expect(mocks.upload).toHaveBeenCalledWith(expect.objectContaining({ fileFieldName: "file", file: expect.objectContaining({ data: expect.any(ArrayBuffer) }) }))
    expect(operation.onProgress).toHaveBeenCalledWith("Uploading original sources and saved Notes…")
  })

  it("rejects a compressed Chatbook above the backend100MiB quota before saving a submission receipt", async () => {
    const archive = pack()
    archive.chatbook = new Uint8Array(100 * 1024 * 1024 + 1)
    await expect(restoreWorkspaceResearchPack(archive, options())).rejects.toThrow(/100 MiB/)
    expect(mocks.upload).not.toHaveBeenCalled()
    expect(sessionStorage.length).toBe(0)
  })

  it("reconnects an interrupted import receipt instead of uploading the same archive twice", async () => {
    mocks.request.mockRejectedValueOnce(new Error("Network unavailable"))
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/import|Network/i)
    mocks.request.mockResolvedValue({ status: "failed", id_maps: { media: { "1": 42 }, notes: {} }, failed_items: 1 })
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/partial|failed/i)
    expect(mocks.upload).toHaveBeenCalledOnce()
    expect(sessionStorage.length).toBe(1)
    expect(sessionStorage.getItem(sessionStorage.key(0)!)).not.toContain("never-export-this-secret")
  })

  it.each(["ready", "missing", "different identity", "different context", "deleted", "archived", "workspace changed", "account changed"])("checks a ready receipt read-only: %s", async outcome => {
    const archive = pack()
    const original = JSON.stringify(archive)
    const identity = { id: "restored-workspace", name: "Original import name", tag: "workspace:restored-workspace" }
    const key = `tldw:research-pack-import:v1:${servicePromptAuthorityKey(scope.requestScope)}:${archive.fingerprint}`
    const receipt = JSON.stringify({ version: 1, fingerprint: archive.fingerprint, identity, phase: "ready", jobId: "import-1", maps: { media: { "1": 42 }, notes: { "note-old": "note-new" } } })
    sessionStorage.setItem(key, receipt)
    const invalidation = new AbortController()
    const operation = { ...options(), scope: { ...scope, scopeInvalidatedSignal: invalidation.signal } }
    mocks.request.mockImplementation(async () => {
      if (outcome === "missing") throw new Error("Workspace not found")
      if (outcome === "workspace changed") mocks.current = false
      if (outcome === "account changed") invalidation.abort()
      return { workspace_id: outcome === "different context" ? "other-workspace" : identity.id, workspace: { id: outcome === "different identity" ? "other-workspace" : identity.id, name: "Renamed restored workspace", deleted: outcome === "deleted", archived: outcome === "archived" } }
    })

    if (outcome === "ready") {
      await expect(restoreWorkspaceResearchPack(archive, operation)).resolves.toEqual({ alreadyRestored: true, identity: { ...identity, name: "Renamed restored workspace" } })
    } else {
      await expect(restoreWorkspaceResearchPack(archive, operation)).rejects.toThrow(/not found|identity|unavailable|changed/i)
    }
    expect(mocks.request).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ path: "/api/v1/workspaces/restored-workspace/context", method: "GET", servicePromptConfig: expect.objectContaining({ expectedUserId: "owner" }) }))
    expect(mocks.upload).not.toHaveBeenCalled()
    expect(mocks.persist).not.toHaveBeenCalled()
    expect(mocks.chatImport).not.toHaveBeenCalled()
    expect(sessionStorage.getItem(key)).toBe(receipt)
    expect(JSON.stringify(archive)).toBe(original)
  })

  it("does not write after the account changes while an import status request is pending", async () => {
    mocks.request.mockImplementation(async () => { mocks.current = false; return { status: "completed", id_maps: { media: { "1": 42 }, notes: { "note-old": "note-new" } } } })
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/changed|current/i)
    expect(mocks.request).toHaveBeenCalledOnce()
    expect(mocks.persist).not.toHaveBeenCalled()
  })

  it.each(["failed", "cancelled"])("does not treat a retained %s receipt with complete maps as success on reopen", async status => {
    mocks.request.mockResolvedValue({ status, id_maps: { media: { "1": 42 }, notes: { "note-old": "note-new" } } })
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/Partial/)
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/Partial/)
    expect(mocks.request.mock.calls.every(([request]) => request.method === "GET" && request.path.includes("/import/jobs/"))).toBe(true)
    expect(mocks.upload).toHaveBeenCalledOnce()
  })
  it("enrolls restored editable artifacts and mapped Notes before returning the destination", async () => {
    mocks.request.mockImplementation(async request => {
      if (request.path.includes("/import/jobs/")) return { status: "completed", id_maps: { media: { "1": 42 }, notes: { "note-old": "note-new" } } }
      if (request.path.includes("/notes/")) return request.method === "GET" ? { id: "note-new", title: "Note", content: "Reviewed text", version: 1, keywords: [] } : { id: "note-new", version: 2, ...request.body }
      if (request.path.endsWith("/sources")) return request.method === "GET" ? [] : request.body
      return { id: request.path.split("/").pop() }
    })
    const restored = await restoreWorkspaceResearchPack(pack(), options())
    if (!("bundle" in restored)) throw new Error("A first restore must return its verified bundle")
    expect(restored.bundle.workspace.snapshot.sources[0].mediaId).toBe(42)
    expect(restored.bundle.workspace.snapshot.currentNote).toMatchObject({ id: "note-new", version: 2, isDirty: true })
    expect(restored.bundle.workspace.snapshot.generatedArtifacts[0].workspaceArtifactVersion).toBe(1)
    expect(mocks.persist).toHaveBeenCalledOnce()
    expect(mocks.request.mock.calls.find(([r]) => r.method === "PUT" && r.path.includes("/notes/"))?.[0].headers["expected-version"]).toBe("1")
    const imported = mocks.chatImport.mock.calls[0]?.[0]
    expect(imported?.body).toMatchObject({ scope_type: "workspace", workspace_id: restored.identity.id, messages: [
      { role: "assistant", content: "Answer [1]", timestamp: "1970-01-01T00:00:00.001Z", metadata_extra: { workspace_rag_sources_v1: { sources: [{ mediaId: 42 }] } } }
    ] })
    expect(restored.bundle.workspace.chatSession?.serverChatId).toBe(imported.body.id)
    expect(restored.bundle.workspace.chatSession?.messages[0].serverMessageId).toBe(imported.body.messages[0].id)
  })

  it("enrolls a legacy snapshot as new editable artifacts without invoking Chatbooks or reconnecting its old Note", async () => {
    mocks.request.mockImplementation(async request => request.path.endsWith("/sources") ? (request.method === "GET" ? [] : request.body) : { id: request.path.split("/").pop() })
    const restored = await restoreLegacyWorkspaceBundle(portableFixture(), options())
    expect(restored.bundle.workspace.snapshot.generatedArtifacts[0]).toMatchObject({ workspaceArtifactVersion: 1, content: "Unchanged draft" })
    expect(restored.bundle.workspace.snapshot.currentNote.id).toBeUndefined()
    expect(restored.bundle.workspace.chatSession).toMatchObject({ historyId: null, serverChatId: mocks.chatImport.mock.calls[0]?.[0].body.id })
    expect(mocks.upload).not.toHaveBeenCalled()
    expect(mocks.request.mock.calls.some(([r]) => r.path.includes("/notes/"))).toBe(false)
  })

  it.each(["wrong-workspace", "changed-message", "changed-citation", "account-change"])("does not adopt an imported conversation after %s", async kind => {
    mocks.request.mockImplementation(async request => request.path.endsWith("/sources") ? (request.method === "GET" ? [] : request.body) : { id: request.path.split("/").pop() })
    if (kind === "wrong-workspace") mocks.chatImport.mockImplementation(async request => ({ id: request.body.id, scope_type: "workspace", workspace_id: "foreign" }))
    else mocks.chatMessages.mockImplementation(async () => {
      const messages = structuredClone(mocks.chatImport.mock.calls.at(-1)![0].body.messages).map(row => ({ ...row, created_at: row.timestamp, version: 1 }))
      if (kind === "changed-message") messages[0].content = "Someone edited this"
      if (kind === "changed-citation") messages[0].metadata_extra.workspace_rag_sources_v1.sources[0].mediaId = 999
      if (kind === "account-change") mocks.current = false
      return messages
    })
    await expect(restoreLegacyWorkspaceBundle(portableFixture(), options())).rejects.toThrow(/changed|match|canonical|conversation/i)
  })

  it("replays the same atomic conversation after a lost import response without changing message identities", async () => {
    mocks.request.mockImplementation(async request => request.path.endsWith("/sources") ? (request.method === "GET" ? [] : request.body) : { id: request.path.split("/").pop() })
    mocks.chatImport.mockRejectedValueOnce(new Error("response lost"))
    await expect(restoreLegacyWorkspaceBundle(portableFixture(), options())).rejects.toThrow(/lost/)
    const restored = await restoreLegacyWorkspaceBundle(portableFixture(), options())
    expect(mocks.chatImport.mock.calls[0][0].body).toEqual(mocks.chatImport.mock.calls[1][0].body)
    expect(restored.bundle.workspace.chatSession?.serverChatId).toBe(mocks.chatImport.mock.calls[1][0].body.id)
  })

  it.each(["valid", "wrong-derived-role", "unknown-metadata"])("checks the actual API readback through the real normalizer: %s", async kind => {
    const { chatRagMethods } = await import("../tldw/domains/chat-rag")
    mocks.chatMessages.mockImplementation((id, params, requestOptions) => chatRagMethods.listChatMessages.call(
      { getChatMessagesCacheKey: (chatId: string, query: string) => chatId + query } as never,
      id, params, requestOptions,
    ))
    mocks.request.mockImplementation(async request => {
      if (request.path.includes("/chats/") && request.path.includes("/messages")) {
        const rows = mocks.chatImport.mock.calls.at(-1)![0].body.messages
        return { messages: rows.map(row => ({ id: row.id, sender: row.role, content: row.content,
          timestamp: row.timestamp.replace("Z", "000Z"), parent_message_id: row.parent_message_id,
          images: row.images, version: 1, metadata_extra: { sender_role: kind === "wrong-derived-role" ? "user" : row.role,
            ...row.metadata, ...row.metadata_extra, ...(kind === "unknown-metadata" ? { unrecognized: "retain strict validation" } : {}) },
        })), total: rows.length }
      }
      return request.path.endsWith("/sources") ? (request.method === "GET" ? [] : request.body) : { id: request.path.split("/").pop() }
    })
    const original = portableFixture()
    original.workspace.chatSession!.messages[0].metadataExtra = { model_id: "deepseek-flash", provider: "deepseek" }
    const restored = restoreLegacyWorkspaceBundle(original, options())
    if (kind !== "valid") await expect(restored).rejects.toThrow(/canonical|match/i)
    else {
      const result = await restored
      expect(result.bundle.workspace.chatSession?.messages[0]).toMatchObject({ role: "assistant",
        metadataExtra: { sender_role: "assistant", model_id: "deepseek-flash", provider: "deepseek", workspace_rag_sources_v1: { sources: [{ mediaId: 1 }] } } })
    }
  })

  it("does not rewrite an acknowledged Note after later artifact enrollment fails", async () => {
    let saved: Record<string, unknown> = { ...pack().notes[0], id: "note-new", version: 1, keywords: [], knowledge_provenance_version: 1 }
    mocks.request.mockImplementation(async request => {
      if (request.path.includes("/import/jobs/")) return { status: "completed", id_maps: { media: { "1": 42 }, notes: { "note-old": "note-new" } } }
      if (request.path.includes("/notes/")) {
        if (request.method === "PUT") saved = { ...request.body, id: "note-new", version: 2, knowledge_provenance_version: 2, knowledge_provenance_state: "active" }
        return saved
      }
      if (request.path.endsWith("/sources")) return request.method === "GET" ? [] : request.body
      return { id: request.path.split("/").pop() }
    })
    mocks.persist.mockRejectedValueOnce(new Error("Artifact save interrupted"))
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/interrupted/)
    await restoreWorkspaceResearchPack(pack(), options())
    expect(mocks.request.mock.calls.filter(([r]) => r.method === "PUT" && r.path.includes("/notes/"))).toHaveLength(1)
    expect(mocks.upload).toHaveBeenCalledOnce()
  })

  it("preserves a competing destination Note edit when enrollment is resumed", async () => {
    let saved: Record<string, unknown> = { ...pack().notes[0], id: "note-new", version: 1, keywords: [], knowledge_provenance_version: 1 }
    mocks.request.mockImplementation(async request => {
      if (request.path.includes("/import/jobs/")) return { status: "completed", id_maps: { media: { "1": 42 }, notes: { "note-old": "note-new" } } }
      if (request.path.includes("/notes/")) {
        if (request.method === "PUT") saved = { ...request.body, id: "note-new", version: 2, knowledge_provenance_version: 2, knowledge_provenance_state: "active" }
        return saved
      }
      if (request.path.endsWith("/sources")) return request.method === "GET" ? [] : request.body
      return { id: request.path.split("/").pop() }
    })
    mocks.persist.mockRejectedValueOnce(new Error("Artifact save interrupted"))
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/interrupted/)
    saved = { ...saved, content: "A later human edit", version: 3 }
    await expect(restoreWorkspaceResearchPack(pack(), options())).rejects.toThrow(/changed|edited|conflict/i)
    expect(mocks.request.mock.calls.filter(([r]) => r.method === "PUT" && r.path.includes("/notes/"))).toHaveLength(1)
  })

})
