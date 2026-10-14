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
//     requestIdleCallback, so the microtask/flush fallback is what runs here),
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
    const buildEnvelope = (notes: string) =>
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
})
