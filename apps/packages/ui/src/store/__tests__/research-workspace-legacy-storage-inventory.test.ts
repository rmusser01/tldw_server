import { describe, expect, it } from "vitest"
import {
  classifyResearchWorkspaceLegacyStorageSurface,
  evaluateResearchWorkspaceLegacyDeletionEligibility,
  RESEARCH_WORKSPACE_INDEXEDDB_ARTIFACT_STORE,
  RESEARCH_WORKSPACE_INDEXEDDB_CHAT_STORE,
  RESEARCH_WORKSPACE_INDEXEDDB_NAME,
  RESEARCH_WORKSPACE_LEGACY_STORAGE_INVENTORY
} from "../research-workspace-legacy-storage-inventory"

describe("Research Workspace legacy storage inventory", () => {
  it("classifies monolithic, split snapshot, split chat, and IndexedDB stores as content-bearing", () => {
    const monolithic = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw-workspace"
    })
    const snapshot = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw-workspace:workspace:workspace-a:snapshot"
    })
    const chat = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw-workspace:workspace:workspace-a:chat"
    })
    const chatStore = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "indexeddb_store",
      databaseName: RESEARCH_WORKSPACE_INDEXEDDB_NAME,
      storeName: RESEARCH_WORKSPACE_INDEXEDDB_CHAT_STORE
    })
    const artifactStore = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "indexeddb_store",
      databaseName: RESEARCH_WORKSPACE_INDEXEDDB_NAME,
      storeName: RESEARCH_WORKSPACE_INDEXEDDB_ARTIFACT_STORE
    })

    expect(monolithic?.classification).toBe("content")
    expect(snapshot).toMatchObject({
      classification: "content",
      workspaceId: "workspace-a",
      contentClasses: expect.arrayContaining(["sources", "notes", "artifacts"])
    })
    expect(chat).toMatchObject({
      classification: "content",
      workspaceId: "workspace-a",
      contentClasses: expect.arrayContaining(["chat_messages"])
    })
    expect(chatStore?.classification).toBe("content")
    expect(artifactStore?.classification).toBe("content")
  })

  it("classifies Research Workspace UI-only keys as retained local preferences", () => {
    const pinned = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw:research-workspace:pinned-workspaces:v1"
    })
    const recentOutputTypes = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw:research-workspace:recent-output-types:v1"
    })
    const addSourceTabUsage = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw:research-workspace:add-source-tab-usage:v1"
    })
    const onboardingDismissed = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw:research-workspace:onboarding-dismissed:v1"
    })

    for (const result of [
      pinned,
      recentOutputTypes,
      addSourceTabUsage,
      onboardingDismissed
    ]) {
      expect(result).toMatchObject({
        classification: "ui_only",
        deletionPolicy: "retain_local"
      })
    }
  })

  it("keeps obsolete flags and legacy telemetry non-content and non-authoritative", () => {
    const oldMigrationFlag = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "workspace_migrated"
    })
    const legacyTelemetry = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw:workspace:playground:telemetry"
    })

    expect(oldMigrationFlag).toMatchObject({
      classification: "obsolete",
      deletionPolicy: "retain_local",
      authoritativeForMigration: false
    })
    expect(legacyTelemetry).toMatchObject({
      classification: "metadata",
      deletionPolicy: "delete_after_import",
      authoritativeForMigration: false
    })
  })

  it("classifies reconciliation markers as retained metadata without content authority", () => {
    const marker = classifyResearchWorkspaceLegacyStorageSurface({
      kind: "local_storage",
      key: "tldw:research-workspace:reconciliation:v1:workspace%20one"
    })

    expect(marker).toMatchObject({
      classification: "metadata",
      deletionPolicy: "retain_local",
      authoritativeForMigration: false,
      workspaceId: "workspace one"
    })
  })

  it("exposes one inventory item for every known stable surface id", () => {
    expect(
      RESEARCH_WORKSPACE_LEGACY_STORAGE_INVENTORY.map((item) => item.id)
    ).toEqual(
      expect.arrayContaining([
        "localStorage:tldw-workspace",
        "localStorage:tldw-workspace:workspace:*:snapshot",
        "localStorage:tldw-workspace:workspace:*:chat",
        "indexedDB:tldw-workspace-storage/workspace-chat-sessions",
        "indexedDB:tldw-workspace-storage/workspace-artifact-payloads",
        "localStorage:tldw:research-workspace:pinned-workspaces:v1"
      ])
    )
  })
})

describe("Research Workspace legacy deletion eligibility", () => {
  it("blocks deletion for content-bearing surfaces until a manifest covers them", () => {
    const blocked = evaluateResearchWorkspaceLegacyDeletionEligibility({
      discoveredLocalStorageKeys: [
        "tldw-workspace",
        "tldw-workspace:workspace:workspace-a:snapshot",
        "tldw-workspace:workspace:workspace-a:chat"
      ],
      discoveredIndexedDbStores: [
        {
          databaseName: RESEARCH_WORKSPACE_INDEXEDDB_NAME,
          storeName: RESEARCH_WORKSPACE_INDEXEDDB_CHAT_STORE
        }
      ],
      manifestCoveredSurfaceIds: ["localStorage:tldw-workspace"]
    })

    expect(blocked.eligible).toBe(false)
    expect(blocked.blockingSurfaces.map((surface) => surface.id)).toEqual(
      expect.arrayContaining([
        "localStorage:tldw-workspace:workspace:workspace-a:snapshot",
        "localStorage:tldw-workspace:workspace:workspace-a:chat",
        "indexedDB:tldw-workspace-storage/workspace-chat-sessions"
      ])
    )

    const eligible = evaluateResearchWorkspaceLegacyDeletionEligibility({
      discoveredLocalStorageKeys: [
        "tldw-workspace",
        "tldw-workspace:workspace:workspace-a:snapshot",
        "tldw-workspace:workspace:workspace-a:chat"
      ],
      discoveredIndexedDbStores: [
        {
          databaseName: RESEARCH_WORKSPACE_INDEXEDDB_NAME,
          storeName: RESEARCH_WORKSPACE_INDEXEDDB_CHAT_STORE
        }
      ],
      manifestCoveredSurfaceIds: [
        "localStorage:tldw-workspace",
        "localStorage:tldw-workspace:workspace:workspace-a:snapshot",
        "localStorage:tldw-workspace:workspace:workspace-a:chat",
        "indexedDB:tldw-workspace-storage/workspace-chat-sessions"
      ]
    })

    expect(eligible.eligible).toBe(true)
    expect(eligible.blockingSurfaces).toEqual([])
  })

  it("blocks unknown workspace-prefixed localStorage keys", () => {
    const result = evaluateResearchWorkspaceLegacyDeletionEligibility({
      discoveredLocalStorageKeys: [
        "tldw-workspace:workspace:workspace-a:unmapped-content"
      ],
      manifestCoveredSurfaceIds: []
    })

    expect(result.eligible).toBe(false)
    expect(result.unknownSurfaces).toEqual([
      {
        id: "unknown:localStorage:tldw-workspace:workspace:workspace-a:unmapped-content",
        kind: "local_storage",
        key: "tldw-workspace:workspace:workspace-a:unmapped-content",
        deletionPolicy: "unknown_blocks_deletion"
      }
    ])
  })

  it("reports UI-only surfaces as retained without blocking content deletion", () => {
    const result = evaluateResearchWorkspaceLegacyDeletionEligibility({
      discoveredLocalStorageKeys: [
        "tldw:research-workspace:pinned-workspaces:v1",
        "tldw:research-workspace:recent-output-types:v1"
      ],
      manifestCoveredSurfaceIds: []
    })

    expect(result.eligible).toBe(true)
    expect(result.blockingSurfaces).toEqual([])
    expect(result.retainedLocalSurfaces.map((surface) => surface.id)).toEqual([
      "localStorage:tldw:research-workspace:pinned-workspaces:v1",
      "localStorage:tldw:research-workspace:recent-output-types:v1"
    ])
  })
})
