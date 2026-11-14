// WP-16 Stage 2 sub-step 1 — coalescing persist writer for the workspace store.
//
// A trivial `set()` used to run the ENTIRE split-key persist chain
// (partialize -> stringify -> parse -> per-workspace stringify + compare)
// synchronously on the setState path, once per mutation. These tests pin the
// coalescing-writer contract (pattern reused from
// `entries/background-session-store.ts` `createSerializedSessionStateWriter`):
//
//   - rapid mutations coalesce into ONE persist pass carrying the latest value,
//   - no storage write happens synchronously on the set() path (deferred off
//     the keystroke's critical path),
//   - a pending write is flushed SYNCHRONOUSLY on pagehide/beforeunload so a
//     page close cannot lose the buffered envelope (jsdom has no
//     requestIdleCallback, so the microtask delivery path is what runs here,
//     with the trailing timer as the worst-case bound),
//   - a write for a DIFFERENT storage key arriving inside the coalescing
//     window never displaces the buffered workspace envelope (the Research
//     onboarding sidecar shares this adapter),
//   - the trailing timer lands a buffered write within
//     WORKSPACE_PERSIST_TRAILING_FLUSH_MS even when idle delivery never fires,
//   - overlapping writes can never land out of order (an older envelope may
//     never overwrite a newer one).
//
// jsdom note: Storage.prototype spies only (instance spies become named
// storage entries under jsdom's WebIDL proxy Storage).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import {
  DEFAULT_AUDIO_SETTINGS,
  DEFAULT_WORKSPACE_NOTE,
  type SavedWorkspace
} from "@/types/workspace"
import {
  createWorkspaceStorage,
  useWorkspaceStore,
  WORKSPACE_PERSIST_TRAILING_FLUSH_MS,
  workspacePersistWriterInternals
} from "../workspace"

const STORAGE_KEY = WORKSPACE_STORAGE_KEY
const WORKSPACE_ID = "coalesce-ws-01"
const snapshotKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`

const FIXTURE_DATE = new Date("2026-10-06T00:00:00.000Z")

// Mirrors the bench's drain wait: enough macrotask ticks for the deferred
// write chain (and its microtask-scheduled drain) to land.
const settlePersistWrites = async (): Promise<void> => {
  await Promise.resolve()
  for (let tick = 0; tick < 5; tick += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

const savedWorkspaceEntry = (): SavedWorkspace => ({
  id: WORKSPACE_ID,
  name: "Coalescing Workspace",
  tag: "workspace:coalescing-workspace",
  collectionId: null,
  createdAt: FIXTURE_DATE,
  lastAccessedAt: FIXTURE_DATE,
  sourceCount: 0
})

const seedStore = (notes: string): void => {
  useWorkspaceStore.setState({
    workspaceId: WORKSPACE_ID,
    workspaceName: "Coalescing Workspace",
    workspaceTag: "workspace:coalescing-workspace",
    workspaceCreatedAt: FIXTURE_DATE,
    workspaceChatReferenceId: WORKSPACE_ID,
    studyMaterialsPolicy: null,
    assistantDefaults: null,
    sources: [],
    selectedSourceIds: [],
    sourceFolders: [],
    sourceFolderMemberships: [],
    selectedSourceFolderIds: [],
    activeFolderId: null,
    generatedArtifacts: [],
    notes,
    currentNote: { ...DEFAULT_WORKSPACE_NOTE },
    workspaceBanner: { title: "", subtitle: "", image: null },
    leftPaneCollapsed: false,
    rightPaneCollapsed: false,
    audioSettings: { ...DEFAULT_AUDIO_SETTINGS },
    savedWorkspaces: [savedWorkspaceEntry()],
    archivedWorkspaces: [],
    workspaceCollections: [],
    workspaceSnapshots: {},
    workspaceChatSessions: {}
  })
}

const readStoredSnapshotNotes = (): string => {
  const raw = localStorage.getItem(snapshotKey(WORKSPACE_ID))
  const parsed = raw ? JSON.parse(raw) : {}
  return typeof parsed.notes === "string" ? parsed.notes : "<missing>"
}

const buildEnvelope = (notes: string): string =>
  JSON.stringify({
    state: {
      workspaceId: WORKSPACE_ID,
      savedWorkspaces: [savedWorkspaceEntry()],
      archivedWorkspaces: [],
      workspaceCollections: [],
      workspaceSnapshots: {
        [WORKSPACE_ID]: {
          workspaceId: WORKSPACE_ID,
          workspaceName: "Coalescing Workspace",
          workspaceTag: "workspace:coalescing-workspace",
          workspaceCreatedAt: FIXTURE_DATE.toISOString(),
          workspaceChatReferenceId: WORKSPACE_ID,
          sources: [],
          selectedSourceIds: [],
          generatedArtifacts: [],
          notes,
          currentNote: { ...DEFAULT_WORKSPACE_NOTE },
          leftPaneCollapsed: false,
          rightPaneCollapsed: false,
          audioSettings: { provider: "openai", model: "x", voice: "alloy", speed: 1, format: "mp3" }
        }
      },
      workspaceChatSessions: {}
    },
    version: 1
  })

describe("workspace persist coalescing writer", () => {
  beforeEach(() => {
    workspacePersistWriterInternals.cancelPendingWrite()
    localStorage.clear()
    seedStore("seeded")
  })

  afterEach(async () => {
    workspacePersistWriterInternals.cancelPendingWrite()
    await settlePersistWrites()
    workspacePersistWriterInternals.cancelPendingWrite()
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it("collapses 10 rapid mutations into one persist pass with the latest value", async () => {
    await settlePersistWrites()
    expect(readStoredSnapshotNotes()).toBe("seeded")

    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")
    for (let index = 0; index < 10; index += 1) {
      useWorkspaceStore.setState({ notes: `burst ${index}` })
    }

    // The write is deferred off the setState path: nothing landed yet.
    expect(readStoredSnapshotNotes()).toBe("seeded")

    await settlePersistWrites()

    const snapshotWrites = setItemSpy.mock.calls.filter(
      ([name]) => name === snapshotKey(WORKSPACE_ID)
    )
    const indexWrites = setItemSpy.mock.calls.filter(
      ([name]) => name === STORAGE_KEY
    )
    expect(snapshotWrites).toHaveLength(1)
    expect(indexWrites).toHaveLength(1)
    expect(readStoredSnapshotNotes()).toBe("burst 9")
  })

  it("flushes a pending write synchronously on pagehide", async () => {
    await settlePersistWrites()
    useWorkspaceStore.setState({ notes: "before pagehide" })
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)

    window.dispatchEvent(new Event("pagehide"))

    // No awaits: the flush must land synchronously (unload cannot wait for a
    // drain scheduled on a later tick).
    expect(readStoredSnapshotNotes()).toBe("before pagehide")
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)
    expect(
      JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}").state
        ?.workspaceSnapshots?.[WORKSPACE_ID]?.notes
    ).toBe("before pagehide")
  })

  it("flushes a pending write synchronously on beforeunload", async () => {
    await settlePersistWrites()
    useWorkspaceStore.setState({ notes: "before beforeunload" })
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)

    window.dispatchEvent(new Event("beforeunload"))

    expect(readStoredSnapshotNotes()).toBe("before beforeunload")
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)
  })

  it("overlapping writes coalesce to a single in-order pass (latest wins)", async () => {
    const storage = createWorkspaceStorage()

    await storage.setItem(STORAGE_KEY, buildEnvelope("first"))
    await settlePersistWrites()

    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")
    // Two overlapping, un-awaited writes: the older must never overwrite the
    // newer, and both must be covered by a single pass.
    const firstWrite = storage.setItem(STORAGE_KEY, buildEnvelope("older"))
    const secondWrite = storage.setItem(STORAGE_KEY, buildEnvelope("newer"))
    await Promise.all([firstWrite, secondWrite])
    await settlePersistWrites()

    expect(readStoredSnapshotNotes()).toBe("newer")
    const snapshotWrites = setItemSpy.mock.calls.filter(
      ([name]) => name === snapshotKey(WORKSPACE_ID)
    )
    expect(snapshotWrites).toHaveLength(1)
  })

  it("a different-key write in the window does not displace the pending workspace write", async () => {
    const storage = createWorkspaceStorage()
    await settlePersistWrites()
    expect(readStoredSnapshotNotes()).toBe("seeded")

    // Exact cross-key sequence from review CRITICAL 1: a workspace mutation is
    // buffered, then a DIFFERENT key sharing the module-level writer (the
    // Research onboarding sidecar does exactly this via createWorkspaceStorage)
    // is written inside the coalescing window.
    const workspaceWrite = storage.setItem(
      STORAGE_KEY,
      buildEnvelope("cross-key v1")
    )
    const sidecarWrite = storage.setItem(
      "tldw:research-workspace:onboarding-dismissed:v1",
      "1"
    )
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)

    // Lifecycle flush while both jobs are still buffered: the buffered
    // workspace envelope must NOT have been displaced — it lands
    // synchronously, inside the pagehide handler, with the split index.
    window.dispatchEvent(new Event("pagehide"))

    expect(readStoredSnapshotNotes()).toBe("cross-key v1")
    expect(
      JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}").state
        ?.workspaceSnapshots?.[WORKSPACE_ID]?.notes
    ).toBe("cross-key v1")

    // The sidecar job drains right behind it (one microtask later — jobs
    // perform sequentially) and must land too.
    await Promise.allSettled([workspaceWrite, sidecarWrite])
    expect(
      localStorage.getItem("tldw:research-workspace:onboarding-dismissed:v1")
    ).toBe("1")
    expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)
  })

  it("lands a buffered write by the 300ms trailing timer when idle delivery never fires", async () => {
    vi.useFakeTimers()
    // Simulate a permanently-busy main thread: requestIdleCallback is
    // available but never fires, so only the trailing timer can deliver.
    vi.stubGlobal(
      "requestIdleCallback",
      (callback: () => void): number => {
        void callback
        return 0
      }
    )
    try {
      useWorkspaceStore.setState({ notes: "trailing edge" })
      expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(true)
      expect(readStoredSnapshotNotes()).toBe("seeded")

      // One tick before the trailing edge: still buffered.
      await vi.advanceTimersByTimeAsync(
        WORKSPACE_PERSIST_TRAILING_FLUSH_MS - 1
      )
      expect(readStoredSnapshotNotes()).toBe("seeded")

      // At exactly the trailing deadline the write lands (≤300ms + ε).
      await vi.advanceTimersByTimeAsync(1)
      expect(readStoredSnapshotNotes()).toBe("trailing edge")
      expect(workspacePersistWriterInternals.hasPendingWrite()).toBe(false)
    } finally {
      // Unstub BEFORE re-enabling real timers: useRealTimers then restores the
      // true originals, otherwise the sinon-faked requestIdleCallback that
      // stubGlobal captured could leak past this test.
      vi.unstubAllGlobals()
      vi.useRealTimers()
      workspacePersistWriterInternals.cancelPendingWrite()
    }
  })
})
