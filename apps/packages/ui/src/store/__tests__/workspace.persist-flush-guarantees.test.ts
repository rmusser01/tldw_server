// WP-16 Stage 2 review round 1 — flush guarantees for the coalescing writer.
//
// IMPORTANT 2: a lifecycle flush (pagehide/beforeunload/tab hidden) must be
// DURABLE when IndexedDB offload is enabled. The snapshot/chat/index writes
// used to sit behind `await offloadWorkspaceSnapshotArtifacts(...)` (and the
// chat-record put), so a page death before those puts resolved lost BOTH the
// snapshot keys and the split index. The lifecycle drain now skips every
// IndexedDB await: oversized artifacts stay INLINE for that pass and every
// localStorage write lands synchronously before the flush returns. The next
// coalesced pass re-offloads and pointer-izes again (self-heal).
//
// IMPORTANT 4: reads through the storage adapter observe AT-REST storage
// (rehydration/reload semantics — storage wins over a buffered envelope that
// reflects older in-memory state), so getItem deliberately does NOT flush.
// The old comment justified this with the false premise that store creation
// was the only reader — `utils/use-research-workspace-prefill.ts` also reads
// WORKSPACE_STORAGE_KEY, to verify a just-run import persisted. That reader
// now drains the coalescer first via `flushWorkspacePersistence()`. These
// tests pin both sides: the flushed prefill read sees the buffered value;
// the unflushed read keeps serving out-of-band storage.
//
// jsdom note: Storage.prototype spies only (instance spies are broken under
// jsdom's Storage proxy). The fake adapter gates its puts on a manually
// released deferred so a hung IndexedDB write is observable rather than
// fatal.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import {
  DEFAULT_AUDIO_SETTINGS,
  DEFAULT_WORKSPACE_NOTE,
  type GeneratedArtifact
} from "@/types/workspace"
import {
  createWorkspaceStorage,
  flushWorkspacePersistence,
  workspacePersistWriterInternals
} from "../workspace"

const STORAGE_KEY = WORKSPACE_STORAGE_KEY
const FIXTURE_DATE = new Date("2026-10-06T00:00:00.000Z")
const ARTIFACT_BYTES = 13 * 1024 // above WORKSPACE_ARTIFACT_OFFLOAD_MIN_BYTES

const snapshotKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`

const artifactContent = (seed: string): string =>
  `lifecycle ${seed}: ${"x".repeat(ARTIFACT_BYTES - `lifecycle ${seed}: `.length)}`

const makeArtifact = (id: string, seed: string): GeneratedArtifact => ({
  id,
  type: "summary",
  title: `Artifact ${id}`,
  status: "completed",
  content: artifactContent(seed),
  createdAt: FIXTURE_DATE
})

const makeSnapshot = (
  workspaceId: string,
  notes: string,
  artifacts: GeneratedArtifact[]
) => ({
  workspaceId,
  workspaceName: "Flush Guarantees Workspace",
  workspaceTag: "workspace:flush-guarantees",
  studyMaterialsPolicy: null,
  assistantDefaults: null,
  workspaceCreatedAt: FIXTURE_DATE,
  workspaceChatReferenceId: workspaceId,
  sources: [],
  selectedSourceIds: [],
  sourceFolders: [],
  sourceFolderMemberships: [],
  selectedSourceFolderIds: [],
  activeFolderId: null,
  generatedArtifacts: artifacts,
  notes,
  currentNote: { ...DEFAULT_WORKSPACE_NOTE },
  workspaceBanner: { title: "", subtitle: "", image: null },
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  audioSettings: { ...DEFAULT_AUDIO_SETTINGS }
})

const buildEnvelope = (
  workspaceId: string,
  notes: string,
  artifacts: GeneratedArtifact[]
): string =>
  JSON.stringify({
    state: {
      workspaceId,
      savedWorkspaces: [
        {
          id: workspaceId,
          name: "Flush Guarantees Workspace",
          tag: "workspace:flush-guarantees",
          collectionId: null,
          createdAt: FIXTURE_DATE,
          lastAccessedAt: FIXTURE_DATE,
          sourceCount: 0
        }
      ],
      archivedWorkspaces: [],
      workspaceCollections: [],
      workspaceSnapshots: {
        [workspaceId]: makeSnapshot(workspaceId, notes, artifacts)
      },
      workspaceChatSessions: {}
    },
    version: 1
  })

type GatedAdapterState = {
  putCalls: Array<{ key: string; artifactId: string }>
  releasePuts: () => void
}

const createGatedAdapter = (): {
  adapter: Parameters<typeof createWorkspaceStorage>[0]["indexedDbAdapter"]
  state: GatedAdapterState
} => {
  const state: GatedAdapterState = {
    putCalls: [],
    releasePuts: () => {}
  }
  // The gate starts CLOSED: any put issued while it is closed never settles,
  // standing in for an IndexedDB write the dying page will not wait for.
  let releaseGate: () => void = () => {}
  const gate = new Promise<void>((resolve) => {
    releaseGate = resolve
  })
  state.releasePuts = () => releaseGate()
  return {
    state,
    adapter: {
      isAvailable: () => true,
      putChatRecord: async () => true,
      getChatRecord: async () => null,
      deleteChatRecord: async () => true,
      putArtifactPayloadRecord: async (record) => {
        state.putCalls.push({ key: record.key, artifactId: record.artifactId })
        await gate
        return true
      },
      getArtifactPayloadRecord: async () => null,
      deleteArtifactPayloadRecord: async () => true
    }
  }
}

const settlePersistWrites = async (): Promise<void> => {
  workspacePersistWriterInternals.flush()
  await Promise.resolve()
  for (let tick = 0; tick < 8; tick += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

describe("workspace persist lifecycle flush durability", () => {
  beforeEach(() => {
    workspacePersistWriterInternals.cancelPendingWrite()
    localStorage.clear()
  })

  afterEach(async () => {
    workspacePersistWriterInternals.cancelPendingWrite()
    await settlePersistWrites()
    workspacePersistWriterInternals.cancelPendingWrite()
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it("pagehide lands snapshot and index synchronously without waiting for IndexedDB offload", async () => {
    const workspaceId = "lifecycle-ws"
    const { adapter, state } = createGatedAdapter()
    const storage = createWorkspaceStorage({ indexedDbAdapter: adapter })
    const artifacts = [makeArtifact("art-1", "one")]

    const write = storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v1", artifacts)
    )
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)

    window.dispatchEvent(new Event("pagehide"))

    // NO awaits: the snapshot key and the split index must already be on
    // disk, with the artifact INLINE (offload skipped for the lifecycle pass)
    // and zero IndexedDB puts issued — the gated puts can never block it.
    const stored = JSON.parse(
      localStorage.getItem(snapshotKey(workspaceId)) || "{}"
    )
    expect(stored.notes).toBe("notes v1")
    expect(stored.generatedArtifacts[0]?.content).toBe(artifacts[0].content)
    expect(
      stored.generatedArtifacts[0]?.__tldwArtifactPayloadRef
    ).toBeUndefined()
    expect(
      JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}").schema
    ).toBe("workspace_split_v1")
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}").state
      ?.workspaceSnapshots?.[workspaceId]?.notes).toBe("notes v1")
    expect(state.putCalls).toHaveLength(0)
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)

    await write

    // Self-heal: the next COALESCED pass re-offloads and pointer-izes again.
    state.releasePuts()
    await storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v2", artifacts)
    )
    await settlePersistWrites()

    const healed = JSON.parse(
      localStorage.getItem(snapshotKey(workspaceId)) || "{}"
    )
    expect(healed.notes).toBe("notes v2")
    expect(healed.generatedArtifacts[0]?.__tldwArtifactPayloadRef?.key).toBe(
      state.putCalls[0]?.key
    )
    expect(healed.generatedArtifacts[0]?.content).toBeUndefined()
    expect(state.putCalls).toHaveLength(1)
  })

  it("removeItem awaits an in-flight drain instead of racing it (no resurrection of cleared storage)", async () => {
    const workspaceId = "remove-race-ws"
    const { adapter, state } = createGatedAdapter()
    const storage = createWorkspaceStorage({ indexedDbAdapter: adapter })
    const artifacts = [makeArtifact("art-1", "one")]

    // Deliver the drain (microtask) and let it suspend inside the gated
    // IndexedDB put: the write is IN FLIGHT — claimed off the pending map,
    // blocked mid-perform — exactly the state a user's clearStorage() hits
    // when it fires while an offload is settling.
    const write = storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v1", artifacts)
    )
    for (let hop = 0; hop < 3; hop += 1) {
      await Promise.resolve()
    }
    expect(state.putCalls).toHaveLength(1)
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)

    // User clears storage while the drain is suspended on IndexedDB.
    const removal = storage.removeItem(STORAGE_KEY)

    // The suspension settles: the drain resumes and finishes its write pass
    // BEFORE the removal proceeds (removeItem must await the full drain, not
    // race it fire-and-forget).
    state.releasePuts()
    await removal

    // Storage stays empty across a subsequent read/drain cycle — the resumed
    // drain must not have resurrected the cleared keys.
    await settlePersistWrites()
    expect(localStorage.getItem(snapshotKey(workspaceId))).toBeNull()
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull()
    const readBack = await Promise.resolve(storage.getItem(STORAGE_KEY))
    expect(readBack).toBeNull()
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)

    await Promise.allSettled([write])
  })

  it("serves a buffered write to a prefill-style read that flushes first, while plain getItem stays storage-wins", async () => {
    const workspaceId = "read-flush-ws"
    const storage = createWorkspaceStorage()

    // Buffer a write through the adapter (the import flow's persist writes
    // go exactly this way) — it has NOT landed in storage yet.
    const write = storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes fresh", [])
    )
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)

    // Prefill parity: flushWorkspacePersistence() drains the buffered write
    // (and awaits an in-flight chain), so the subsequent read sees the NEW
    // value instead of stale storage.
    await flushWorkspacePersistence()
    const value = await Promise.resolve(storage.getItem(STORAGE_KEY))
    expect(value).toBeTruthy()
    const parsed = JSON.parse(value as string)
    expect(parsed?.state?.workspaceSnapshots?.[workspaceId]?.notes).toBe(
      "notes fresh"
    )
    await write

    // Reload parity: a SECOND buffered write must NOT shadow storage that a
    // reloader seeded out-of-band — plain getItem keeps serving at-rest
    // storage (this is the rehydrate/reload pattern; storage wins over the
    // older buffered envelope).
    const secondWrite = storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "buffered v2", [])
    )
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)
    localStorage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "seeded by reloader", [])
    )
    const reloaded = await Promise.resolve(storage.getItem(STORAGE_KEY))
    const reloadedParsed = JSON.parse(reloaded as string)
    expect(
      reloadedParsed?.state?.workspaceSnapshots?.[workspaceId]?.notes
    ).toBe("seeded by reloader")

    // Drop the buffered envelope without letting it clobber the seed.
    workspacePersistWriterInternals.cancelPendingWrite()
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)
    await Promise.allSettled([secondWrite])
  })
})
