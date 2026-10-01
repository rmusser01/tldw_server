import { beforeEach, describe, expect, it, vi } from "vitest"
import { readFileSync } from "node:fs"
import { URL } from "node:url"
import { TextDecoder } from "node:util"
import vm from "node:vm"
import ts from "typescript"
import {
  buildResearchWorkspaceMigrationPlan,
  runResearchWorkspaceMigration,
  type ResearchWorkspaceMigrationRunInput
} from "../workspace-migration"

const workspaceId = "native-workspace"
const snapshotKey = `tldw-workspace:workspace:${workspaceId}:snapshot`
const legacySnapshot = {
  workspaceId,
  sources: [{ id: "source-1", mediaId: 17, title: "Lumen" }],
  selectedSourceIds: ["source-1"],
  currentNote: { title: "Draft", content: "Keep this draft", isDirty: true }
}
const serverWorkspace = {
  scopeKey: "verified-owner",
  metadata: { id: workspaceId },
  sourceSignature: "old-baseline",
  selectedSourceSignature: "",
  notes: [{ id: 42, workspace_id: workspaceId, content: "Native note" }]
}
const indexPayload = (snapshot: Record<string, unknown> = legacySnapshot) => ({
  schema: "workspace_split_v1",
  splitVersion: 1,
  version: 1,
  state: {
    workspaceId,
    workspaceIds: [workspaceId],
    workspaceSnapshots: { [workspaceId]: snapshot },
    workspaceChatSessions: {},
    savedWorkspaces: [{ id: workspaceId }],
    archivedWorkspaces: []
  }
})

const migrationInput = (values: Record<string, string>) => {
  for (const [key, value] of Object.entries(values)) localStorage.setItem(key, value)
  const receipt = { id: "migration", status: "finalized", client_delete_eligible: true }
  return {
    targetWorkspaceId: workspaceId,
    targetWorkspaceName: "Native Research",
    discoveredLocalStorageKeys: Object.keys(values),
    readLocalStorageValue: vi.fn(async (key: string) => localStorage.getItem(key)),
    readLocalStorageValueSync: vi.fn((key: string) => localStorage.getItem(key)),
    compareAndDeleteLocalStorageValue: vi.fn((key: string, expected: string) => {
      if (localStorage.getItem(key) !== expected) return false
      localStorage.removeItem(key)
      return true
    }),
    subscribeToWorkspaceChanges: (_listener: () => void) => () => {},
    writeLocalStorageValue: vi.fn((key: string, value: string) => localStorage.setItem(key, value)),
    api: {
      createWorkspaceMigration: vi.fn(async () => receipt),
      putWorkspaceMigrationChunk: vi.fn(async () => ({})),
      finalizeWorkspaceMigration: vi.fn(async () => receipt),
      getWorkspaceMigration: vi.fn(async () => receipt),
      ackWorkspaceMigrationClientDelete: vi.fn(async () => ({}))
    }
  }
}

const readSource = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8")
const compile = (source: string) => ts.transpileModule(source.replaceAll("import.meta", "({env:{}})"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS }
}).outputText
const variableDeclarations = (source: string, keep: (name: string) => boolean) => {
  const ast = ts.createSourceFile("source.ts", source, ts.ScriptTarget.Latest, true)
  return ast.statements.filter(ts.isVariableStatement).flatMap((statement) =>
    statement.declarationList.declarations.filter((declaration) => keep(declaration.name.getText(ast)))
      .map((declaration) => `let ${declaration.getText(ast)};`)).join("\n")
}

// Execute the real migration callbacks and consumer, without React,
// the store's initialization side effects, IndexedDB, or network adapters.
const actualMigrationCallbacks = (storage: Pick<Storage, "getItem" | "setItem" | "removeItem">) => {
  const ast = ts.createSourceFile("ResearchWorkspace.tsx", readSource("../../components/Option/ResearchWorkspace/index.tsx"), ts.ScriptTarget.Latest, true)
  const callbacks = new Map<string, string>()
  const visit = (node: ts.Node) => {
    if (ts.isPropertyAssignment(node)) callbacks.set(node.name.getText(ast), node.initializer.getText(ast))
    ts.forEachChild(node, visit)
  }
  visit(ast)
  const context = vm.createContext({ window: { localStorage: storage }, useWorkspaceStore: {
    getState: () => ({ workspaceId, serverWorkspace: null }),
    subscribe: () => () => {}
  } })
  const names = ["getCurrentWorkspace", "subscribeToWorkspaceChanges", "readLocalStorageValue"] as const
  const actual = Object.fromEntries(names.map((name) => {
    if (!callbacks.has(name)) throw new Error(`Missing production callback: ${name}`)
    return [name, vm.runInContext(compile(`(${callbacks.get(name)})`), context)]
  })) as Pick<ResearchWorkspaceMigrationRunInput, typeof names[number]>
  return actual
}

const actualWorkspaceConsumer = (storage: Pick<Storage, "getItem" | "setItem" | "removeItem">) => {
  const shared = variableDeclarations(readSource("../../types/workspace.ts"), (name) => name.startsWith("DEFAULT_")) + "\n" +
    variableDeclarations(readSource("../../types/workspace-assistant-defaults.ts"), (name) => name === "normalizeWorkspaceAssistantDefaults") + "\n" +
    variableDeclarations(readSource("../workspace-chat-session-key.ts"), () => true)
  const context = vm.createContext({ console, TextEncoder, TextDecoder, structuredClone, localStorage: storage,
    window: { localStorage: storage, dispatchEvent: () => true }, process: { env: { NODE_ENV: "production" } },
    WORKSPACE_STORAGE_KEY: "tldw-workspace", isWorkspaceBroadcastSyncEnabled: () => false,
    buildResearchWorkspaceMigrationTombstoneKey: (id: string) => `tldw:research-workspace:migration:tombstone:${encodeURIComponent(id)}`
  })
  vm.runInContext(compile(shared + "\n" + variableDeclarations(readSource("../workspace.ts"), (name) => name !== "useWorkspaceStore")), context)
  const adapter = { isAvailable: () => false }
  return {
    read: () => vm.runInContext("rebuildWorkspaceEnvelopeFromStorage", context)("tldw-workspace", adapter, adapter) as string | null,
    persist: (value: string) => vm.runInContext("writeSplitWorkspacePersistence", context)("tldw-workspace", value, adapter) as Promise<boolean>
  }
}

describe("Research Workspace migration eligibility", () => {
  beforeEach(() => localStorage.clear())

  it("skips an edited canonical target before planning or reading storage", async () => {
    const input = migrationInput({ "tldw-workspace": JSON.stringify(indexPayload()) })
    const result = await runResearchWorkspaceMigration({ ...input, serverWorkspace })

    expect(result.status).toBe("not_needed")
    expect(input.readLocalStorageValue).not.toHaveBeenCalled()
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
  })

  it("does not plan canonical persisted snapshots without runtime provenance", async () => {
    const input = migrationInput({
      "tldw-workspace": JSON.stringify(indexPayload({ ...legacySnapshot, serverWorkspace })),
      [snapshotKey]: JSON.stringify({ ...legacySnapshot, serverWorkspace })
    })
    const plan = await buildResearchWorkspaceMigrationPlan(input)
    const result = await runResearchWorkspaceMigration(input)

    expect(plan.chunks).toEqual([])
    expect(result.status).toBe("not_needed")
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
    expect(localStorage.getItem(snapshotKey)).toBeTruthy()
  })

  it("retains mismatched canonical provenance before planning", async () => {
    const input = migrationInput({ "tldw-workspace": JSON.stringify(indexPayload()) })
    const result = await runResearchWorkspaceMigration({
      ...input,
      serverWorkspace: { ...serverWorkspace, metadata: { id: "another-workspace" } }
    })

    expect(result.status).toBe("blocked")
    expect(input.readLocalStorageValue).not.toHaveBeenCalled()
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("does not overwrite an existing tombstone for a canonical target", async () => {
    const tombstoneKey = `tldw:research-workspace:migration:tombstone:${workspaceId}`
    const existingTombstone = JSON.stringify({ migrationId: "previous-migration" })
    localStorage.setItem(tombstoneKey, existingTombstone)
    const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })

    await runResearchWorkspaceMigration({ ...input, serverWorkspace })

    expect(localStorage.getItem(tombstoneKey)).toBe(existingTombstone)
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("stops before API creation when canonical provenance arrives during planning", async () => {
    const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })
    let currentProvenance: unknown = null
    input.readLocalStorageValue.mockImplementationOnce(async (key) => {
      currentProvenance = serverWorkspace
      return localStorage.getItem(key)
    })

    const result = await runResearchWorkspaceMigration({
      ...input,
      getCurrentWorkspace: () => ({ workspaceId, serverWorkspace: currentProvenance })
    })

    expect(result.status).toBe("blocked")
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it.each(["canonical", "different-active-target"])(
    "retains payloads and tombstones when %s activation occurs during receipt handling",
    async (activation) => {
      const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })
      let currentWorkspace = { workspaceId, serverWorkspace: null as unknown }
      input.api.getWorkspaceMigration.mockImplementationOnce(async () => {
        currentWorkspace = activation === "canonical"
          ? { workspaceId, serverWorkspace }
          : { workspaceId: "other", serverWorkspace: null }
        return { id: "migration", status: "finalized", client_delete_eligible: true }
      })

      const result = await runResearchWorkspaceMigration({
        ...input,
        getCurrentWorkspace: () => currentWorkspace
      })

      expect(result.status).toBe("blocked")
      expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
      expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
      expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
      expect(localStorage.getItem(snapshotKey)).toBe(JSON.stringify(legacySnapshot))
    }
  )

  it.each([
    ["different active identity", { state: { workspaceId: "another-workspace" } }],
    ["conflicting active identity aliases", { state: { workspaceId, activeWorkspaceId: "other" } }],
    ["invalid persisted state", { workspaceId, state: [] }],
    ["mixed snapshot identities", { state: { ...indexPayload().state,
      workspaceSnapshots: { [workspaceId]: legacySnapshot, other: { workspaceId: "other", serverWorkspace } } } }],
    ["mismatched snapshot identity", indexPayload({ ...legacySnapshot, workspaceId: "other" })],
    ["mixed index identities", { state: { ...indexPayload().state, workspaceIds: [workspaceId, "other"] } }],
    ["mixed saved identities", { state: { ...indexPayload().state, savedWorkspaces: [{ id: "other" }] } }],
    ["mixed chat identities", { state: { ...indexPayload().state, workspaceChatSessions: { other: { messages: [] } } } }],
    ["invalid provenance", indexPayload({ ...legacySnapshot, serverWorkspace: { metadata: { id: workspaceId } } })],
    ["missing active identity", { state: { sources: [] } }]
  ])("retains %s without bulk deletion", async (_name, payload) => {
    const raw = JSON.stringify(payload)
    const input = migrationInput({ "tldw-workspace": raw })
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(localStorage.getItem("tldw-workspace")).toBe(raw)
  })

  it.each([
    [snapshotKey, JSON.stringify({ ...legacySnapshot, workspaceId: "other" })],
    ["tldw-workspace:workspace:other:snapshot", JSON.stringify(legacySnapshot)],
    ["tldw-workspace", "not-json"]
  ])("retains mismatched or unreadable split/root payload %s", async (key, raw) => {
    const input = migrationInput({ [key]: raw })
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
    expect(localStorage.getItem(key)).toBe(raw)
  })

  it("retains all covered keys when their bytes change before cleanup", async () => {
    const input = migrationInput({
      "tldw-workspace": JSON.stringify(indexPayload()),
      [snapshotKey]: JSON.stringify(legacySnapshot)
    })
    input.api.getWorkspaceMigration.mockImplementationOnce(async () => {
      localStorage.setItem(snapshotKey, JSON.stringify({ ...legacySnapshot, currentNote: { content: "New edit" } }))
      return { id: "migration", status: "finalized", client_delete_eligible: true }
    })
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(localStorage.getItem(snapshotKey)).toContain("New edit")
  })

  it.each(["workspace-chat-sessions", "workspace-artifact-payloads"])(
    "retains shared IndexedDB %s before planning or whole-store deletion",
    async (storeName) => {
      const input = migrationInput({ "tldw-workspace": JSON.stringify(indexPayload()) })
      const records = [workspaceId, "canonical-elsewhere"].map((id) =>
        storeName === "workspace-chat-sessions"
          ? {
              key: `workspace:${id}:chat`, workspaceId: id, updatedAt: 1,
              session: { historyId: null, serverChatId: null, messages: [{ role: "user", content: "Keep" }] }
            }
          : {
              key: `workspace:${id}:artifact:output`, workspaceId: id, updatedAt: 1,
              artifactId: "output", payload: { content: "Keep" }
            }
      )
      const readIndexedDbStorePayload = vi.fn(async () => records)
      const deleteIndexedDbStorePayload = vi.fn(async () => {})
      const indexedInput = {
        ...input,
        discoveredIndexedDbStores: [{ databaseName: "tldw-workspace-storage", storeName }],
        readIndexedDbStorePayload,
        deleteIndexedDbStorePayload
      }

      const plan = await buildResearchWorkspaceMigrationPlan(indexedInput)
      const result = await runResearchWorkspaceMigration(indexedInput)

      expect(result.status).toBe("blocked")
      expect(plan.chunks).toEqual([])
      expect(readIndexedDbStorePayload).not.toHaveBeenCalled()
      expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
      expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
      expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
      expect(deleteIndexedDbStorePayload).not.toHaveBeenCalled()
    }
  )

  it("records an eligible legacy receipt but retains exact writable bytes without automatic cleanup", async () => {
    const values = {
      "tldw-workspace": JSON.stringify(indexPayload()),
      [snapshotKey]: JSON.stringify(legacySnapshot)
    }
    const input = migrationInput(values)
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(result.serverMigration?.client_delete_eligible).toBe(true)
    expect(result.message).toMatch(/automatic.*cleanup.*disabled/i)
    expect(result.deletedSurfaceIds).toEqual([])
    expect(input.api.putWorkspaceMigrationChunk).toHaveBeenCalledTimes(2)
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(input.compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(Object.fromEntries(Object.keys(values).map((key) => [key, localStorage.getItem(key)]))).toEqual(values)
    expect(localStorage.length).toBe(2)
    // The chunk API receives metadata, not a durable import of source content.
    expect(JSON.stringify(input.api.putWorkspaceMigrationChunk.mock.calls)).not.toContain("Keep this draft")
  })

  it("retains a Page-shaped edit queued while the metadata receipt is awaited", async () => {
    const input = migrationInput({
      "tldw-workspace": JSON.stringify(indexPayload()),
      [snapshotKey]: JSON.stringify(legacySnapshot)
    })
    const edited = JSON.stringify({ ...legacySnapshot, currentNote: { content: "After hash check" } })
    input.api.getWorkspaceMigration.mockImplementation(async () => {
      queueMicrotask(() => localStorage.setItem(snapshotKey, edited))
      return { id: "migration", status: "finalized", client_delete_eligible: true }
    })

    const result = await runResearchWorkspaceMigration(input)

    expect(localStorage.getItem(snapshotKey)).toBe(edited)
    expect(result.status).toBe("blocked")
    expect(result.deletedSurfaceIds).toEqual([])
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
    expect(localStorage.getItem(`tldw:research-workspace:migration:tombstone:${workspaceId}`)).toBeNull()
  })

  it.each(["canonical", "target", "target-ABA", "canonical-ABA"])(
    "revokes metadata work on %s transition without removing any bytes",
    async (transition) => {
      const input = migrationInput({
        "tldw-workspace": JSON.stringify(indexPayload()),
        [snapshotKey]: JSON.stringify(legacySnapshot)
      })
      let current = { workspaceId, serverWorkspace: null as unknown }
      const listeners = new Set<() => void>()
      const edited = JSON.stringify({ ...legacySnapshot, currentNote: { content: "Canonical draft" }, serverWorkspace })
      input.api.putWorkspaceMigrationChunk.mockImplementationOnce(async () => {
        queueMicrotask(() => {
          current = transition.startsWith("canonical")
            ? { workspaceId, serverWorkspace }
            : { workspaceId: "other", serverWorkspace: null }
          listeners.forEach((listener) => listener())
          // A returned target must not revive an attempt revoked during the await.
          if (transition.endsWith("ABA")) {
            current = { workspaceId, serverWorkspace: null }
            listeners.forEach((listener) => listener())
          } else localStorage.setItem(snapshotKey, edited)
        })
        return {}
      })

      const result = await runResearchWorkspaceMigration({
        ...input,
        getCurrentWorkspace: () => current,
        subscribeToWorkspaceChanges: (listener: () => void) => {
          listeners.add(listener)
          return () => listeners.delete(listener)
        }
      })

      expect(localStorage.getItem(snapshotKey)).toBe(transition.endsWith("ABA") ? JSON.stringify(legacySnapshot) : edited)
      expect(result.status).toBe("blocked")
      expect(result.deletedSurfaceIds).toEqual([])
      expect(input.api.putWorkspaceMigrationChunk).toHaveBeenCalledOnce()
      expect(input.api.finalizeWorkspaceMigration).not.toHaveBeenCalled()
      expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
      expect(localStorage.getItem(`tldw:research-workspace:migration:tombstone:${workspaceId}`)).toBeNull()
      expect(listeners.size).toBe(0)
    }
  )

  it.each(["workspace_chat_session_v1", "workspace_artifact_payload_v1"])(
    "rejects actual %s payloads arriving after empty offload discovery",
    async (offloadType) => {
      const chatKey = `tldw-workspace:workspace:${workspaceId}:chat`
      const input = migrationInput({ [chatKey]: JSON.stringify({ messages: [] }) })
      const pointer = JSON.stringify({ offloadType, key: `workspace:${workspaceId}:chat`, historyId: null, serverChatId: null, updatedAt: 1 })
      // Discovery has completed with no pointer; the helper awaits the actual read.
      queueMicrotask(() => localStorage.setItem(chatKey, pointer))
      input.readLocalStorageValue.mockImplementation(async (key) => {
        await Promise.resolve()
        return localStorage.getItem(key)
      })

      const result = await runResearchWorkspaceMigration({ ...input, discoveredIndexedDbStores: [] })

      expect(localStorage.getItem(chatKey)).toBe(pointer)
      expect(result.status).toBe("blocked")
      expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
      expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
    }
  )

  it("rejects an embedded artifact pointer even when the caller omitted store discovery", async () => {
    const payload = JSON.stringify(indexPayload({
      ...legacySnapshot,
      generatedArtifacts: [{ id: "artifact", payload: { offloadType: "workspace_artifact_payload_v1", key: "workspace:other:artifact:artifact", updatedAt: 1 } }]
    }))
    const input = migrationInput({ "tldw-workspace": payload })

    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(localStorage.getItem("tldw-workspace")).toBe(payload)
    expect(input.api.createWorkspaceMigration).not.toHaveBeenCalled()
  })

  it("retains existing preflight metadata without cleanup or acknowledgement", async () => {
    const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })
    const preflightKey = `tldw:research-workspace:migration:tombstone-preflight:${workspaceId}`
    localStorage.setItem(preflightKey, "new attempt marker")

    const result = await runResearchWorkspaceMigration(input)

    expect(localStorage.getItem(preflightKey)).toBe("new attempt marker")
    expect(result.status).toBe("blocked")
    expect(result.deletedSurfaceIds).toEqual([])
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains historical preflight metadata across a canonical transition during receipt handling", async () => {
    const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })
    let current = { workspaceId, serverWorkspace: null as unknown }
    const preflightKey = `tldw:research-workspace:migration:tombstone-preflight:${workspaceId}`
    localStorage.setItem(preflightKey, "historical preflight")
    input.api.getWorkspaceMigration.mockImplementation(async () => {
      queueMicrotask(() => {
        current = { workspaceId, serverWorkspace }
      })
      return { id: "migration", status: "finalized", client_delete_eligible: true }
    })

    const result = await runResearchWorkspaceMigration({ ...input, getCurrentWorkspace: () => current })

    expect(localStorage.getItem(preflightKey)).toBe("historical preflight")
    expect(result.status).toBe("blocked")
    expect(result.deletedSurfaceIds).toEqual([])
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains all bytes and reports no removals when a later metadata chunk fails", async () => {
    const input = migrationInput({ "tldw-workspace": JSON.stringify(indexPayload()), [snapshotKey]: JSON.stringify(legacySnapshot) })
    input.api.putWorkspaceMigrationChunk.mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error("metadata chunk unavailable"))

    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("failed")
    expect(result.deletedSurfaceIds).toEqual([])
    expect(localStorage.getItem("tldw-workspace")).toBe(JSON.stringify(indexPayload()))
    expect(localStorage.getItem(snapshotKey)).toBe(JSON.stringify(legacySnapshot))
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it.each(["atomic-remover", "transition-subscription"])("retains writable copies without the former cleanup dependency %s", async (missing) => {
    const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })
    const result = await runResearchWorkspaceMigration({
      ...input,
      getCurrentWorkspace: () => ({ workspaceId, serverWorkspace: null }),
      ...(missing === "atomic-remover"
        ? { compareAndDeleteLocalStorageValue: undefined }
        : { subscribeToWorkspaceChanges: undefined })
    })

    expect(result.status).toBe("blocked")
    expect(localStorage.getItem(snapshotKey)).toBe(JSON.stringify(legacySnapshot))
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains writable copies without synchronous cleanup reads", async () => {
    const input = migrationInput({ [snapshotKey]: JSON.stringify(legacySnapshot) })
    const result = await runResearchWorkspaceMigration({ ...input, readLocalStorageValueSync: undefined })

    expect(result.status).toBe("blocked")
    expect(localStorage.getItem(snapshotKey)).toBe(JSON.stringify(legacySnapshot))
    expect(input.writeLocalStorageValue).not.toHaveBeenCalled()
    expect(input.api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  // The approved policy removes delete/marker timing boundaries entirely.
  // Keep late-writer and consumer coverage at the remaining awaited receipt boundary.
  it.each(["stable-inline-control", "chunk-writer", "receipt-writer", "immediate-consumer", "historical-marker-consumer", "replacement-marker"])(
    "retains %s content through actual read-only callbacks and the workspace consumer",
    async (mode) => {
      const originalRoot = JSON.stringify(indexPayload())
      const freshRoot = JSON.stringify(indexPayload({ ...legacySnapshot, currentNote: { title: "Draft", content: "Fresh repopulated draft", isDirty: true } }))
      const freshSnapshot = JSON.stringify(JSON.parse(freshRoot).state.workspaceSnapshots[workspaceId])
      const tombstoneKey = `tldw:research-workspace:migration:tombstone:${workspaceId}`
      const historicalMarker = JSON.stringify({ migrationId: "old-attempt", legacyWorkspaceId: workspaceId, contentRetained: false })
      const replacementMarker = JSON.stringify({ migrationId: "other-attempt", legacyWorkspaceId: workspaceId, contentRetained: false })
      const values = new Map([["tldw-workspace", originalRoot], [snapshotKey, JSON.stringify(legacySnapshot)]])
      if (mode === "historical-marker-consumer") values.set(tombstoneKey, historicalMarker)
      const repopulate = () => {
        values.set("tldw-workspace", freshRoot)
        values.set(snapshotKey, freshSnapshot)
        if (mode === "replacement-marker") values.set(tombstoneKey, replacementMarker)
      }
      const storage = {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
        removeItem: (key: string) => { values.delete(key) }
      }
      const receipt = { id: "receipt", status: "finalized", client_delete_eligible: true }
      const ack = vi.fn(async () => ({}))
      const consumer = actualWorkspaceConsumer(storage)
      const api = {
        createWorkspaceMigration: async () => receipt,
        putWorkspaceMigrationChunk: async () => {
          if (mode === "chunk-writer") queueMicrotask(repopulate)
          return {}
        },
        finalizeWorkspaceMigration: async () => receipt,
        getWorkspaceMigration: async () => {
          if (mode !== "stable-inline-control" && mode !== "chunk-writer") repopulate()
          // Exercise the real consumer before the helper resumes from its await.
          if (mode.includes("consumer")) {
            const duringReceipt = await consumer.read()
            expect(duringReceipt).toContain("Fresh repopulated draft")
            await consumer.persist(freshRoot)
          }
          return receipt
        },
        ackWorkspaceMigrationClientDelete: ack
      }
      const result = await runResearchWorkspaceMigration({
        targetWorkspaceId: workspaceId, targetWorkspaceName: "Legacy",
        discoveredLocalStorageKeys: ["tldw-workspace", snapshotKey],
        ...actualMigrationCallbacks(storage),
        api
      })
      const hydrated = await consumer.read()
      if (mode === "stable-inline-control") {
        expect(hydrated).toContain("Keep this draft")
        expect(values.get("tldw-workspace")).toBe(originalRoot)
        expect(values.get(snapshotKey)).toBe(JSON.stringify(legacySnapshot))
      } else {
        expect(hydrated).toContain("Fresh repopulated draft")
        expect(values.get("tldw-workspace")).toContain("Fresh repopulated draft")
        expect(values.get(snapshotKey)).toContain("Fresh repopulated draft")
      }
      expect(result.status).toBe("blocked")
      expect(result.deletedSurfaceIds).toEqual([])
      expect(ack).not.toHaveBeenCalled()
      expect(values.get(tombstoneKey) ?? null).toBe(mode === "replacement-marker" ? replacementMarker : mode === "historical-marker-consumer" ? historicalMarker : null)
    }
  )
})
