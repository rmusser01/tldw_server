import { describe, expect, it } from "vitest"
import {
  buildResearchWorkspaceSnapshotStorageKey,
  WORKSPACE_STORAGE_KEY
} from "@/store/research-workspace-legacy-storage-inventory"
import { buildResearchWorkspaceMigrationTombstoneKey } from "@/store/workspace-migration"
import {
  buildWorkspaceReconciliationDryRun,
  buildWorkspaceReconciliationMarkerStorageKey,
  discoverLocalResearchWorkspaceEntries,
  readWorkspaceReconciliationMarker,
  writeWorkspaceReconciliationMarker,
  type WorkspaceReconciliationMarkerV1
} from "../workspace-local-reconciliation"

const splitIndex = (savedWorkspaces: unknown[]) =>
  JSON.stringify({
    schema: "workspace_split_v1",
    version: 12,
    state: {
      workspaceId: "local-ready",
      savedWorkspaces,
      archivedWorkspaces: [],
      workspaceIds: savedWorkspaces
        .filter(
          (workspace): workspace is { id: string } =>
            typeof workspace === "object" &&
            workspace !== null &&
            typeof (workspace as { id?: unknown }).id === "string"
        )
        .map((workspace) => workspace.id),
      workspaceSnapshots: {},
      workspaceChatSessions: {}
    }
  })

const snapshot = (workspaceId: string, workspaceName: string, sourceCount = 0) =>
  JSON.stringify({
    workspaceId,
    workspaceName,
    workspaceTag: `workspace:${workspaceId}`,
    sources: Array.from({ length: sourceCount }, (_, index) => ({
      id: `source-${index + 1}`
    })),
    selectedSourceIds: []
  })

const mapReader =
  (values: Record<string, string | null>) =>
  (key: string): string | null =>
    values[key] ?? null

describe("local Research Workspace reconciliation", () => {
  it("discovers local entries and assigns conservative dry-run states", () => {
    const localOnlySnapshotKey =
      buildResearchWorkspaceSnapshotStorageKey("snapshot-only")
    const unsupportedSnapshotKey =
      buildResearchWorkspaceSnapshotStorageKey("unsupported")
    const localStorageValues = {
      [WORKSPACE_STORAGE_KEY]: splitIndex([
        {
          id: "local-ready",
          name: "Ready Local",
          sourceCount: 3
        },
        {
          id: "server-same-id",
          name: "Already Server",
          sourceCount: 1
        },
        {
          id: "name-conflict",
          name: "Shared Name",
          sourceCount: 2
        },
        {
          id: "possible-duplicate",
          name: "Case Duplicate",
          sourceCount: 4
        }
      ]),
      [localOnlySnapshotKey]: snapshot("snapshot-only", "Snapshot Only", 2),
      [unsupportedSnapshotKey]: "{broken-json"
    }

    const localEntries = discoverLocalResearchWorkspaceEntries({
      discoveredLocalStorageKeys: [
        WORKSPACE_STORAGE_KEY,
        localOnlySnapshotKey,
        unsupportedSnapshotKey
      ],
      readLocalStorageValue: mapReader(localStorageValues)
    })
    const dryRun = buildWorkspaceReconciliationDryRun({
      localEntries,
      serverWorkspaces: [
        {
          id: "server-same-id",
          name: "Already Server",
          workspace_profile: "research"
        },
        {
          id: "server-conflict",
          name: "Shared Name",
          workspace_profile: "research"
        },
        {
          id: "server-duplicate",
          name: "case duplicate",
          workspace_profile: "research"
        }
      ]
    })

    expect(dryRun.items.map((item) => [item.localWorkspaceId, item.state])).toEqual(
      expect.arrayContaining([
        ["local-ready", "ready_to_create_metadata"],
        ["server-same-id", "server_row_exists"],
        ["name-conflict", "name_conflict"],
        ["possible-duplicate", "possible_duplicate"],
        ["snapshot-only", "local_only"],
        ["unsupported", "unsupported_local_payload"]
      ])
    )
    expect(
      dryRun.items.find((item) => item.localWorkspaceId === "snapshot-only")
        ?.sourceCount
    ).toBe(2)
  })

  it("writes and reads only the minimal reconciliation marker after confirmed actions", () => {
    const storageValues: Record<string, string> = {}
    const storage = {
      getItem: (key: string) => storageValues[key] ?? null,
      setItem: (key: string, value: string) => {
        storageValues[key] = value
      },
      removeItem: (key: string) => {
        delete storageValues[key]
      }
    }
    const marker: WorkspaceReconciliationMarkerV1 = {
      schemaVersion: 1,
      serverWorkspaceId: "server-1",
      serverName: "Server Workspace",
      serverProfile: "research",
      linkedAt: "2026-06-04T12:00:00.000Z",
      status: "metadata_promoted"
    }

    writeWorkspaceReconciliationMarker({
      storage,
      localWorkspaceId: "local-1",
      marker
    })

    const key = buildWorkspaceReconciliationMarkerStorageKey("local-1")
    expect(JSON.parse(storageValues[key] ?? "{}")).toEqual(marker)
    expect(storageValues[key]).not.toContain("sources")
    expect(storageValues[key]).not.toContain("chat")
    expect(
      readWorkspaceReconciliationMarker({ storage, localWorkspaceId: "local-1" })
    ).toEqual(marker)
  })

  it("preserves migration tombstones by making tombstoned entries non-actionable", () => {
    const localStorageValues = {
      [WORKSPACE_STORAGE_KEY]: splitIndex([
        {
          id: "local tombstone",
          name: "Tombstoned Workspace",
          sourceCount: 1
        }
      ]),
      [buildResearchWorkspaceMigrationTombstoneKey("local tombstone")]:
        JSON.stringify({
          legacyWorkspaceId: "local tombstone",
          serverWorkspaceId: "server-tombstone",
          migrationId: "research-workspace-local-tombstone-abc",
          deletedAt: "2026-06-04T12:00:00.000Z",
          contentRetained: false
        })
    }
    const storage = {
      getItem: mapReader(localStorageValues),
      setItem: () => undefined,
      removeItem: () => undefined
    }

    const localEntries = discoverLocalResearchWorkspaceEntries({
      discoveredLocalStorageKeys: [WORKSPACE_STORAGE_KEY],
      readLocalStorageValue: mapReader(localStorageValues),
      storage
    })
    const dryRun = buildWorkspaceReconciliationDryRun({
      localEntries,
      serverWorkspaces: []
    })

    expect(dryRun.items[0]).toMatchObject({
      localWorkspaceId: "local tombstone",
      state: "unsupported_local_payload",
      tombstoned: true,
      actionable: false,
      reason: "migration_tombstone_present"
    })
  })
})
