// Workspace persistence micro-benchmark — store persist path (TASK-13520, Batch W0 Stage 2).
//
// Measures what ONE trivial `set()` on the real `useWorkspaceStore` costs when the
// persisted state holds 10 workspaces x 50 generated artifacts (500 artifacts,
// 1 KiB content each, ~520 KiB of snapshot JSON):
//
//   - zustand persist serializes the ENTIRE partialized state via
//     `JSON.stringify` on every set (even though split-key storage then
//     re-serializes per-workspace snapshots and discards the monolithic
//     envelope),
//   - `partialize` -> `recordWorkspacePersistenceDiagnostics` stringifies the
//     payload again per section (dev/test builds only; NODE_ENV=test keeps this
//     on, so those calls are counted as part of the measured path),
//   - `writeSplitWorkspacePersistence` re-stringifies every workspace snapshot
//     to diff it against localStorage, then rewrites the changed snapshot key
//     plus the split index (which embeds the active workspace snapshot).
//
// IndexedDB offload is unavailable under jsdom (`indexedDB` is undefined), so
// the adapter reports unavailable and artifacts stay in localStorage; no
// fake-indexeddb dependency is added (module-level adapter is left untouched).
//
// Method:
//   - Seeds the real store once with the 10x50 fixture via `setState` and waits
//     for the persistence write to land (split index present in localStorage).
//   - Installs passthrough spies on `JSON.stringify` and
//     `Storage.prototype.setItem` (jsdom's Storage is a WebIDL proxy: spying on
//     the `localStorage` INSTANCE stores the spy as a named data entry instead
//     of intercepting — the prototype spy intercepts every real write).
//   - Runs one trivial `set()` (`notes` flip-flop) and drains persistence;
//     repeats 3x in-process for reproducibility (counts/bytes are deterministic,
//     duration is informational).
//
// This is a measurement harness, not a regression gate: assertions only
// sanity-check that counters are finite, non-negative and non-zero and that the
// harness drove the intended persistence path. Perf thresholds live in
// Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md; batch W2 re-runs this bench
// after touching the persistence paths and records the before/after delta.
//
// Run: cd apps/packages/ui && bun run vitest run src/store/__tests__/workspace-persist.bench.test.ts
import { describe, expect, it, vi } from "vitest"

import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import {
  DEFAULT_AUDIO_SETTINGS,
  DEFAULT_WORKSPACE_NOTE,
  type GeneratedArtifact,
  type SavedWorkspace
} from "@/types/workspace"
import { useWorkspaceStore } from "../workspace"

const WORKSPACE_COUNT = 10
const ARTIFACTS_PER_WORKSPACE = 50
const ARTIFACT_CONTENT_BYTES = 1024
const MEASUREMENT_RUNS = 3

const STORAGE_KEY = WORKSPACE_STORAGE_KEY
const ACTIVE_WORKSPACE_INDEX = 0
const workspaceIdFor = (index: number) => `bench-ws-${String(index).padStart(2, "0")}`
const snapshotKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`
const notesForRun = (run: number) => `bench trivial set ${run}`

const textEncoder = new TextEncoder()
const utf8Bytes = (value: string): number => textEncoder.encode(value).length

const FIXTURE_DATE = new Date("2026-10-06T00:00:00.000Z")

const artifactContent = (workspaceIndex: number, artifactIndex: number): string => {
  const prefix = `bench ${workspaceIndex}/${artifactIndex}: `
  return `${prefix}${"x".repeat(ARTIFACT_CONTENT_BYTES - prefix.length)}`
}

const makeArtifacts = (workspaceIndex: number): GeneratedArtifact[] =>
  Array.from({ length: ARTIFACTS_PER_WORKSPACE }, (_, artifactIndex) => ({
    id: `${workspaceIdFor(workspaceIndex)}-artifact-${artifactIndex}`,
    type: "summary" as const,
    title: `Bench artifact ${workspaceIndex}/${artifactIndex}`,
    status: "completed" as const,
    content: artifactContent(workspaceIndex, artifactIndex),
    createdAt: FIXTURE_DATE
  }))

const makeSnapshot = (workspaceIndex: number) => ({
  workspaceId: workspaceIdFor(workspaceIndex),
  workspaceName: `Bench Workspace ${workspaceIndex}`,
  workspaceTag: `workspace:bench-${workspaceIndex}`,
  studyMaterialsPolicy: null,
  assistantDefaults: null,
  workspaceCreatedAt: FIXTURE_DATE,
  workspaceChatReferenceId: workspaceIdFor(workspaceIndex),
  sources: [],
  selectedSourceIds: [],
  sourceFolders: [],
  sourceFolderMemberships: [],
  selectedSourceFolderIds: [],
  activeFolderId: null,
  generatedArtifacts: makeArtifacts(workspaceIndex),
  notes: "",
  currentNote: { ...DEFAULT_WORKSPACE_NOTE },
  workspaceBanner: { title: "", subtitle: "", image: null },
  leftPaneCollapsed: false,
  rightPaneCollapsed: false,
  audioSettings: { ...DEFAULT_AUDIO_SETTINGS }
})

const makeSavedWorkspaces = (): SavedWorkspace[] =>
  Array.from({ length: WORKSPACE_COUNT }, (_, index) => ({
    id: workspaceIdFor(index),
    name: `Bench Workspace ${index}`,
    tag: `workspace:bench-${index}`,
    collectionId: null,
    createdAt: FIXTURE_DATE,
    lastAccessedAt: FIXTURE_DATE,
    sourceCount: 0
  }))

const snapshots = Array.from({ length: WORKSPACE_COUNT }, (_, index) =>
  makeSnapshot(index)
)

const seedFixture = (): void => {
  const active = snapshots[ACTIVE_WORKSPACE_INDEX]
  const snapshotMap = Object.fromEntries(
    snapshots.map((snapshot) => [snapshot.workspaceId, snapshot])
  )

  // The active workspace snapshot is rebuilt by `partialize` from top-level
  // state (`buildWorkspaceSnapshot`), so the active fixture is seeded there too.
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
    generatedArtifacts: active.generatedArtifacts.map((artifact) => ({
      ...artifact
    })),
    notes: active.notes,
    currentNote: { ...active.currentNote },
    workspaceBanner: { title: "", subtitle: "", image: null },
    leftPaneCollapsed: false,
    rightPaneCollapsed: false,
    audioSettings: { ...DEFAULT_AUDIO_SETTINGS },
    savedWorkspaces: makeSavedWorkspaces(),
    archivedWorkspaces: [],
    workspaceCollections: [],
    workspaceSnapshots: snapshotMap,
    workspaceChatSessions: {}
  })
}

// The persist write is fire-and-forget from zustand's setState override. With
// the IndexedDB adapter unavailable the chain only awaits resolved promises, so
// a couple of macrotask ticks drain it; the landed-write verification below
// fails loudly if that ever stops being enough.
const flushPersistenceWrites = async (): Promise<void> => {
  await Promise.resolve()
  for (let tick = 0; tick < 5; tick += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

type RunMetrics = {
  run: number
  stringifyCalls: number
  stringifyBytes: number
  setItemCalls: number
  setItemBytes: number
  setItemKeys: string[]
  durationMs: number
}

const spreadPercent = (values: number[]): number => {
  const min = Math.min(...values)
  const max = Math.max(...values)
  return min === 0 ? Number.POSITIVE_INFINITY : ((max - min) / min) * 100
}

describe("workspace persist bench (one trivial set, 10x50 fixture)", () => {
  it("records JSON.stringify calls and serialized bytes for one trivial set()", async () => {
    localStorage.clear()

    const stringifySpy = vi.spyOn(JSON, "stringify")
    // See the file header: under jsdom the instance-level
    // `vi.spyOn(localStorage, "setItem")` pattern does NOT intercept (the spy
    // becomes a named storage entry), so the write counter spies the prototype.
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")

    try {
      // Seed (outside the measurement window).
      seedFixture()
      await flushPersistenceWrites()

      const indexedDbOffloadAvailable =
        typeof indexedDB !== "undefined" && typeof window.indexedDB !== "undefined"
      const fixtureSnapshotsBytes = utf8Bytes(JSON.stringify(snapshots))
      for (let index = 0; index < WORKSPACE_COUNT; index += 1) {
        expect(
          localStorage.getItem(snapshotKey(workspaceIdFor(index)))
        ).toBeTruthy()
      }
      const seededIndex = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")
      expect(seededIndex.schema).toBe("workspace_split_v1")

      const runs: RunMetrics[] = []
      for (let run = 1; run <= MEASUREMENT_RUNS; run += 1) {
        stringifySpy.mockClear()
        setItemSpy.mockClear()

        const startedAt = performance.now()
        useWorkspaceStore.setState({ notes: notesForRun(run) })
        await flushPersistenceWrites()
        const durationMs = performance.now() - startedAt

        const stringifyBytes = stringifySpy.mock.results.reduce<number>(
          (total, result) =>
            result.type === "return" && typeof result.value === "string"
              ? total + utf8Bytes(result.value)
              : total,
          0
        )
        const setItemKeys: string[] = []
        const setItemBytes = setItemSpy.mock.calls.reduce<number>((total, call) => {
          const [name, value] = call
          setItemKeys.push(name)
          return total + utf8Bytes(String(value))
        }, 0)

        runs.push({
          run,
          stringifyCalls: stringifySpy.mock.calls.length,
          stringifyBytes,
          setItemCalls: setItemSpy.mock.calls.length,
          setItemBytes,
          setItemKeys,
          durationMs: Number(durationMs.toFixed(2))
        })
        console.log(
          `[workspace-persist.bench] run ${run} ${JSON.stringify(runs[runs.length - 1])}`
        )

        // The harness drove the intended path: the trivial change landed in the
        // active snapshot key and in the split index (which embeds the active
        // snapshot) within the measured window.
        const activeId = workspaceIdFor(ACTIVE_WORKSPACE_INDEX)
        expect(setItemKeys).toContain(snapshotKey(activeId))
        expect(setItemKeys).toContain(STORAGE_KEY)
        const storedSnapshot = JSON.parse(
          localStorage.getItem(snapshotKey(activeId)) || "{}"
        )
        expect(storedSnapshot.notes).toBe(notesForRun(run))
        const indexEnvelope = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")
        expect(indexEnvelope.state?.workspaceSnapshots?.[activeId]?.notes).toBe(
          notesForRun(run)
        )

        // Measurement-recorded sanity assertions: finite, non-negative, non-zero.
        expect(Number.isFinite(runs[runs.length - 1].stringifyCalls)).toBe(true)
        expect(Number.isFinite(stringifyBytes)).toBe(true)
        expect(Number.isFinite(runs[runs.length - 1].setItemCalls)).toBe(true)
        expect(Number.isFinite(setItemBytes)).toBe(true)
        expect(Number.isFinite(durationMs)).toBe(true)
        expect(runs[runs.length - 1].stringifyCalls).toBeGreaterThan(0)
        expect(stringifyBytes).toBeGreaterThan(0)
        expect(runs[runs.length - 1].setItemCalls).toBeGreaterThan(0)
        expect(setItemBytes).toBeGreaterThan(0)
        expect(durationMs).toBeGreaterThanOrEqual(0)
      }

      const summary = {
        workspaces: WORKSPACE_COUNT,
        artifactsPerWorkspace: ARTIFACTS_PER_WORKSPACE,
        artifactContentBytes: ARTIFACT_CONTENT_BYTES,
        fixtureSnapshotsBytes,
        indexedDbOffloadAvailable,
        runs: MEASUREMENT_RUNS,
        stringifyCalls: runs.map((entry) => entry.stringifyCalls),
        stringifyBytes: runs.map((entry) => entry.stringifyBytes),
        setItemCalls: runs.map((entry) => entry.setItemCalls),
        setItemBytes: runs.map((entry) => entry.setItemBytes),
        setItemKeys: runs[0].setItemKeys,
        durationMs: runs.map((entry) => entry.durationMs),
        spreadPctStringifyBytes: Number(
          spreadPercent(runs.map((entry) => entry.stringifyBytes)).toFixed(2)
        ),
        spreadPctSetItemBytes: Number(
          spreadPercent(runs.map((entry) => entry.setItemBytes)).toFixed(2)
        ),
        spreadPctDurationMs: Number(
          spreadPercent(runs.map((entry) => entry.durationMs)).toFixed(2)
        )
      }
      console.log(`[workspace-persist.bench] summary ${JSON.stringify(summary)}`)

      // Reproducibility (informational, not a perf threshold): byte and call
      // counts are deterministic, so the spread across runs must be exactly 0.
      expect(summary.spreadPctStringifyBytes).toBe(0)
      expect(summary.spreadPctSetItemBytes).toBe(0)
      expect(new Set(summary.stringifyCalls).size).toBe(1)
      expect(new Set(summary.setItemCalls).size).toBe(1)
    } finally {
      stringifySpy.mockRestore()
      setItemSpy.mockRestore()
      localStorage.clear()
    }
  })
})
