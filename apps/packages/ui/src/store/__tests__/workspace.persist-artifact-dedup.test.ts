// WP-16 Stage 2 sub-step 3 — artifact offload dedup + parallelization.
//
// `offloadWorkspaceSnapshotArtifacts` used to await one IndexedDB
// `putArtifactPayloadRecord` per oversized artifact SEQUENTIALLY, and re-put
// every artifact on every dirty-workspace pass (the active workspace is dirty
// on every keystroke). These tests pin the deduped contract:
//
//   - eligible artifacts are offloaded with PARALLEL puts,
//   - an artifact whose (id + payload content hash) already has an offload
//     record is NOT re-put — the snapshot is still pointer-ized,
//   - only artifacts whose content changed are re-put, exactly once each,
//     in parallel with the other changed artifacts.
//
// The fake adapter defers each put by a macrotask so concurrency is
// observable. jsdom note: Storage.prototype spies only.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import {
  DEFAULT_AUDIO_SETTINGS,
  DEFAULT_WORKSPACE_NOTE,
  type GeneratedArtifact
} from "@/types/workspace"
import {
  createWorkspaceStorage,
  workspacePersistWriterInternals
} from "../workspace"

const STORAGE_KEY = WORKSPACE_STORAGE_KEY
const workspaceIdFor = (variant: string) => `artifact-dedup-ws-${variant}`
const FIXTURE_DATE = new Date("2026-10-06T00:00:00.000Z")
const ARTIFACT_BYTES = 13 * 1024 // above WORKSPACE_ARTIFACT_OFFLOAD_MIN_BYTES

const snapshotKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`

const artifactContent = (seed: string, salt = ""): string => {
  const prefix = `dedup ${seed}${salt}: `
  return `${prefix}${"x".repeat(ARTIFACT_BYTES - prefix.length)}`
}

const makeArtifact = (id: string, seed: string, salt = ""): GeneratedArtifact => ({
  id,
  type: "summary",
  title: `Artifact ${id}`,
  status: "completed",
  content: artifactContent(seed, salt),
  createdAt: FIXTURE_DATE
})

const makeSnapshot = (
  workspaceId: string,
  notes: string,
  artifacts: GeneratedArtifact[]
) => ({
  workspaceId,
  workspaceName: "Artifact Dedup Workspace",
  workspaceTag: "workspace:artifact-dedup",
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
          name: "Artifact Dedup Workspace",
          tag: "workspace:artifact-dedup",
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

type FakeAdapterState = {
  putCalls: Array<{ key: string; artifactId: string }>
  activePuts: number
  maxActivePuts: number
}

const createFakeAdapter = (): {
  adapter: Parameters<typeof createWorkspaceStorage>[0]["indexedDbAdapter"]
  state: FakeAdapterState
} => {
  const state: FakeAdapterState = {
    putCalls: [],
    activePuts: 0,
    maxActivePuts: 0
  }
  return {
    state,
    adapter: {
      isAvailable: () => true,
      putChatRecord: async () => true,
      getChatRecord: async () => null,
      deleteChatRecord: async () => true,
      putArtifactPayloadRecord: async (record) => {
        state.activePuts += 1
        state.maxActivePuts = Math.max(state.maxActivePuts, state.activePuts)
        state.putCalls.push({ key: record.key, artifactId: record.artifactId })
        await new Promise((resolve) => setTimeout(resolve, 0))
        state.activePuts -= 1
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

describe("workspace persist artifact offload dedup", () => {
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

  it("offloads eligible artifacts with parallel puts", async () => {
    const workspaceId = workspaceIdFor("parallel")
    const { adapter, state } = createFakeAdapter()
    const storage = createWorkspaceStorage({ indexedDbAdapter: adapter })

    const artifacts = [
      makeArtifact("art-1", "one"),
      makeArtifact("art-2", "two"),
      makeArtifact("art-3", "three")
    ]
    await storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v1", artifacts)
    )
    await settlePersistWrites()

    expect(state.putCalls).toHaveLength(3)
    expect(state.maxActivePuts).toBe(3)

    const stored = JSON.parse(
      localStorage.getItem(snapshotKey(workspaceId)) || "{}"
    )
    for (const artifact of artifacts) {
      const storedArtifact = stored.generatedArtifacts.find(
        (candidate: { id: string }) => candidate.id === artifact.id
      )
      expect(storedArtifact?.__tldwArtifactPayloadRef?.key).toBeTruthy()
      expect(storedArtifact?.content).toBeUndefined()
    }
  })

  it("does not re-put artifacts whose content hash is unchanged", async () => {
    const workspaceId = workspaceIdFor("dedup")
    const { adapter, state } = createFakeAdapter()
    const storage = createWorkspaceStorage({ indexedDbAdapter: adapter })

    const artifacts = [
      makeArtifact("art-1", "one"),
      makeArtifact("art-2", "two")
    ]
    await storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v1", artifacts)
    )
    await settlePersistWrites()
    expect(state.putCalls).toHaveLength(2)
    state.putCalls.length = 0

    // Same artifacts, changed workspace content (notes): the workspace is
    // dirty, but the artifact payloads are byte-identical.
    await storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v2", artifacts)
    )
    await settlePersistWrites()

    expect(state.putCalls).toHaveLength(0)

    // The snapshot is still pointer-ized.
    const stored = JSON.parse(
      localStorage.getItem(snapshotKey(workspaceId)) || "{}"
    )
    expect(stored.notes).toBe("notes v2")
    expect(stored.generatedArtifacts).toHaveLength(2)
    for (const artifact of stored.generatedArtifacts) {
      expect(artifact.__tldwArtifactPayloadRef?.key).toBeTruthy()
      expect(artifact.content).toBeUndefined()
    }
  })

  it("re-puts only the changed artifact, exactly once, in parallel with other changed artifacts", async () => {
    const workspaceId = workspaceIdFor("changed")
    const { adapter, state } = createFakeAdapter()
    const storage = createWorkspaceStorage({ indexedDbAdapter: adapter })

    const artifacts = [
      makeArtifact("art-1", "one"),
      makeArtifact("art-2", "two"),
      makeArtifact("art-3", "three")
    ]
    await storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v1", artifacts)
    )
    await settlePersistWrites()
    state.putCalls.length = 0
    state.maxActivePuts = 0

    // Change two of three artifacts; the third keeps its content.
    const changed = [
      makeArtifact("art-1", "one", " REVISED"),
      makeArtifact("art-2", "two", " REVISED"),
      artifacts[2]
    ]
    await storage.setItem(
      STORAGE_KEY,
      buildEnvelope(workspaceId, "notes v2", changed)
    )
    await settlePersistWrites()

    expect(state.putCalls).toHaveLength(2)
    expect(new Set(state.putCalls.map((call) => call.artifactId))).toEqual(
      new Set(["art-1", "art-2"])
    )
    expect(state.maxActivePuts).toBe(2)
  })
})
