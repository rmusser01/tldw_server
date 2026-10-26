// WP-16 Stage 2 sub-step 2 — per-workspace revision diff for the persist chain.
//
// `writeSplitWorkspacePersistence` used to re-serialize EVERY workspace
// snapshot on every persist pass and compare the full string against
// localStorage (after reading every workspace snapshot key back out of
// storage). These tests pin the diffed contract:
//
//   - a mutation re-serializes only the changed workspace: unchanged
//     workspaces' snapshot keys are neither READ from nor WRITTEN to
//     localStorage during the pass,
//   - the active workspace is the single re-serialization candidate (its
//     snapshot is rebuilt from top-level state on every pass),
//   - a persisted snapshot with no known revision (legacy storage, or the
//     first write after load) is treated as DIRTY: it is serialized once,
//     rewritten carrying a revision, and only then tracked,
//   - the hydration read path accepts BOTH legacy (no-revision) and
//     revision-carrying snapshots, and strips the internal revision fields
//     so they never leak into store state.
//
// jsdom note: Storage.prototype spies only (instance spies are broken under
// jsdom's Storage proxy).
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
const WORKSPACE_IDS = ["diff-ws-a", "diff-ws-b", "diff-ws-c"] as const
const ACTIVE_WORKSPACE_ID = WORKSPACE_IDS[0]
const FIXTURE_DATE = new Date("2026-10-06T00:00:00.000Z")

const snapshotKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`
const chatKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:chat`

const makeSnapshot = (workspaceId: string, notes: string) => ({
  workspaceId,
  workspaceName: `Diff Workspace ${workspaceId}`,
  workspaceTag: `workspace:${workspaceId}`,
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
  generatedArtifacts: [],
  notes,
  currentNote: { ...DEFAULT_WORKSPACE_NOTE },
  workspaceBanner: { title: "", subtitle: "", image: null },
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  audioSettings: { ...DEFAULT_AUDIO_SETTINGS }
})

const savedWorkspaceEntries = (): SavedWorkspace[] =>
  WORKSPACE_IDS.map((workspaceId) => ({
    id: workspaceId,
    name: `Diff Workspace ${workspaceId}`,
    tag: `workspace:${workspaceId}`,
    collectionId: null,
    createdAt: FIXTURE_DATE,
    lastAccessedAt: FIXTURE_DATE,
    sourceCount: 0
  }))

const seedStore = (): void => {
  const active = makeSnapshot(ACTIVE_WORKSPACE_ID, "A notes")
  useWorkspaceStore.setState({
    workspaceId: active.workspaceId,
    workspaceName: active.workspaceName,
    workspaceTag: active.workspaceTag,
    workspaceCreatedAt: active.workspaceCreatedAt,
    workspaceChatReferenceId: active.workspaceChatReferenceId,
    studyMaterialsPolicy: null,
    assistantDefaults: null,
    sources: [],
    selectedSourceIds: [],
    sourceFolders: [],
    sourceFolderMemberships: [],
    selectedSourceFolderIds: [],
    activeFolderId: null,
    generatedArtifacts: [],
    notes: active.notes,
    currentNote: { ...DEFAULT_WORKSPACE_NOTE },
    workspaceBanner: { title: "", subtitle: "", image: null },
    leftPaneCollapsed: false,
    rightPaneCollapsed: false,
    audioSettings: { ...DEFAULT_AUDIO_SETTINGS },
    savedWorkspaces: savedWorkspaceEntries(),
    archivedWorkspaces: [],
    workspaceCollections: [],
    workspaceSnapshots: {
      [WORKSPACE_IDS[1]]: makeSnapshot(WORKSPACE_IDS[1], "B notes"),
      [WORKSPACE_IDS[2]]: makeSnapshot(WORKSPACE_IDS[2], "C notes")
    },
    workspaceChatSessions: {}
  })
}

const settlePersistWrites = async (): Promise<void> => {
  workspacePersistWriterInternals.flush()
  await Promise.resolve()
  for (let tick = 0; tick < 5; tick += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

describe("workspace persist per-workspace diff", () => {
  beforeEach(() => {
    workspacePersistWriterInternals.cancelPendingWrite()
    localStorage.clear()
    seedStore()
  })

  afterEach(async () => {
    workspacePersistWriterInternals.cancelPendingWrite()
    await settlePersistWrites()
    workspacePersistWriterInternals.cancelPendingWrite()
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it("mutating one workspace neither reads nor writes unchanged workspaces' keys", async () => {
    await settlePersistWrites()
    for (const workspaceId of WORKSPACE_IDS) {
      expect(localStorage.getItem(snapshotKey(workspaceId))).toBeTruthy()
    }

    const getItemSpy = vi.spyOn(Storage.prototype, "getItem")
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")

    useWorkspaceStore.setState({ notes: "A notes changed" })
    await settlePersistWrites()

    const unchangedIds = [WORKSPACE_IDS[1], WORKSPACE_IDS[2]]
    for (const workspaceId of unchangedIds) {
      expect(
        getItemSpy.mock.calls.filter(([name]) => name === snapshotKey(workspaceId))
      ).toHaveLength(0)
      expect(
        setItemSpy.mock.calls.filter(([name]) => name === snapshotKey(workspaceId))
      ).toHaveLength(0)
      expect(
        getItemSpy.mock.calls.filter(([name]) => name === chatKey(workspaceId))
      ).toHaveLength(0)
    }

    // The changed workspace snapshot and the split index are still written.
    expect(
      setItemSpy.mock.calls.filter(
        ([name]) => name === snapshotKey(ACTIVE_WORKSPACE_ID)
      )
    ).toHaveLength(1)
    expect(
      setItemSpy.mock.calls.filter(([name]) => name === STORAGE_KEY)
    ).toHaveLength(1)
    expect(
      JSON.parse(
        localStorage.getItem(snapshotKey(ACTIVE_WORKSPACE_ID)) || "{}"
      ).notes
    ).toBe("A notes changed")
  })

  it("re-serializes exactly one workspace for a single-workspace mutation", async () => {
    await settlePersistWrites()

    const stringifySpy = vi.spyOn(JSON, "stringify")
    useWorkspaceStore.setState({ notes: "A notes changed again" })
    await settlePersistWrites()

    // Identify the per-workspace snapshot serialization calls by their
    // argument (the envelope, the dev-diagnostics state probe, and the index
    // carry other shapes — the diagnostics probe stringifies the whole
    // partialized state, which also has a `workspaceId` field).
    const isSnapshotShape = (
      candidate: unknown
    ): candidate is { workspaceId: string } =>
      Boolean(candidate) &&
      typeof candidate === "object" &&
      "workspaceId" in (candidate as Record<string, unknown>) &&
      "generatedArtifacts" in (candidate as Record<string, unknown>)
    const serializeCallsFor = (workspaceId: string): number =>
      stringifySpy.mock.calls.filter(
        ([candidate]) =>
          isSnapshotShape(candidate) && candidate.workspaceId === workspaceId
      ).length

    expect(serializeCallsFor(ACTIVE_WORKSPACE_ID)).toBe(1)
    expect(serializeCallsFor(WORKSPACE_IDS[1])).toBe(0)
    expect(serializeCallsFor(WORKSPACE_IDS[2])).toBe(0)
  })

  it("treats a legacy no-revision snapshot as dirty: serializes once, rewrites with a revision, then tracks it", async () => {
    const legacyWorkspaceId = "legacy-diff-ws"
    const legacySnapshot = makeSnapshot(legacyWorkspaceId, "Legacy notes")

    // Seed legacy-shaped storage directly: split index + a snapshot key with
    // NO revision fields (what pre-WP-16 builds wrote).
    localStorage.setItem(
      snapshotKey(legacyWorkspaceId),
      JSON.stringify(legacySnapshot)
    )
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        schema: "workspace_split_v1",
        splitVersion: 1,
        version: 1,
        state: {
          workspaceId: legacyWorkspaceId,
          savedWorkspaces: [],
          archivedWorkspaces: [],
          workspaceCollections: [],
          workspaceIds: [legacyWorkspaceId],
          workspaceSnapshots: {},
          workspaceChatSessions: {}
        }
      })
    )

    const storage = createWorkspaceStorage()
    const envelope = JSON.stringify({
      state: {
        workspaceId: legacyWorkspaceId,
        savedWorkspaces: [],
        archivedWorkspaces: [],
        workspaceCollections: [],
        workspaceSnapshots: { [legacyWorkspaceId]: legacySnapshot },
        workspaceChatSessions: {}
      },
      version: 1
    })

    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")

    // Identical content — but the differ has no revision for this workspace,
    // so the first pass must serialize and rewrite it revision-carrying.
    await storage.setItem(STORAGE_KEY, envelope)
    await settlePersistWrites()

    const rewriteCalls = setItemSpy.mock.calls.filter(
      ([name]) => name === snapshotKey(legacyWorkspaceId)
    )
    expect(rewriteCalls.length).toBeGreaterThanOrEqual(1)

    const stored = JSON.parse(
      localStorage.getItem(snapshotKey(legacyWorkspaceId)) || "{}"
    )
    expect(stored.notes).toBe("Legacy notes")
    expect(typeof stored.__tldwPersistRevision).toBe("number")
    expect(stored.__tldwPersistRevision).toBeGreaterThanOrEqual(1)

    // A second identical pass is now tracked: content AND revision are
    // unchanged, so the snapshot key must not be rewritten.
    setItemSpy.mockClear()
    await storage.setItem(STORAGE_KEY, envelope)
    await settlePersistWrites()
    expect(
      setItemSpy.mock.calls.filter(
        ([name]) => name === snapshotKey(legacyWorkspaceId)
      )
    ).toHaveLength(0)
  })

  it("hydration reads both legacy and revision-carrying snapshots and strips internal revision fields", async () => {
    const storage = createWorkspaceStorage()

    const seedIndex = (workspaceId: string) => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          schema: "workspace_split_v1",
          splitVersion: 1,
          version: 1,
          state: {
            workspaceId,
            savedWorkspaces: [],
            archivedWorkspaces: [],
            workspaceCollections: [],
            workspaceIds: [workspaceId],
            workspaceSnapshots: {},
            workspaceChatSessions: {}
          }
        })
      )
    }

    // Legacy shape: no internal fields.
    const legacyId = "read-legacy-ws"
    localStorage.setItem(
      snapshotKey(legacyId),
      JSON.stringify(makeSnapshot(legacyId, "Legacy read notes"))
    )
    seedIndex(legacyId)
    const legacyEnvelope = await Promise.resolve(storage.getItem(STORAGE_KEY))
    const legacyState = legacyEnvelope
      ? JSON.parse(legacyEnvelope).state
      : null
    expect(legacyState?.workspaceSnapshots?.[legacyId]?.notes).toBe(
      "Legacy read notes"
    )
    expect(
      legacyState?.workspaceSnapshots?.[legacyId]?.__tldwPersistRevision
    ).toBeUndefined()

    // Revision-carrying shape: internal fields must be stripped on read.
    const revisionedId = "read-revisioned-ws"
    const revisionedSnapshot = {
      ...makeSnapshot(revisionedId, "Revisioned read notes"),
      __tldwPersistRevision: 7,
      __tldwPersistUpdatedAt: 1770000000000
    }
    localStorage.setItem(
      snapshotKey(revisionedId),
      JSON.stringify(revisionedSnapshot)
    )
    seedIndex(revisionedId)
    const revisionedEnvelope = await Promise.resolve(
      storage.getItem(STORAGE_KEY)
    )
    const revisionedState = revisionedEnvelope
      ? JSON.parse(revisionedEnvelope).state
      : null
    expect(revisionedState?.workspaceSnapshots?.[revisionedId]?.notes).toBe(
      "Revisioned read notes"
    )
    expect(
      revisionedState?.workspaceSnapshots?.[revisionedId]
        ?.__tldwPersistRevision
    ).toBeUndefined()
    expect(
      revisionedState?.workspaceSnapshots?.[revisionedId]
        ?.__tldwPersistUpdatedAt
    ).toBeUndefined()
  })
})
