import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  AudiobookChapterAsset,
  AudiobookProject
} from "@/db/dexie/types"
import type { LegacyAudiobookMigrationMarker } from "../audiobook-projects"

const { projectRows, assetRows, projectTable, assetTable } = vi.hoisted(() => {
  type ProjectRow = AudiobookProject & {
    audioStudioMigration?: LegacyAudiobookMigrationMarker
  } & Record<string, unknown>

  const projectRows = new Map<string, ProjectRow>()
  const assetRows = new Map<string, AudiobookChapterAsset>()

  const projectTable = {
    orderBy: vi.fn((field: string) => ({
      reverse: vi.fn(() => ({
        toArray: vi.fn(async () => {
          const rows = Array.from(projectRows.values())
          rows.sort((a, b) => Number(b[field] ?? 0) - Number(a[field] ?? 0))
          return rows
        })
      }))
    })),
    get: vi.fn(async (id: string) => projectRows.get(id)),
    update: vi.fn(async (id: string, updates: Record<string, unknown>) => {
      const current = projectRows.get(id)
      if (!current) return 0
      projectRows.set(id, { ...current, ...updates })
      return 1
    }),
    delete: vi.fn(async (id: string) => {
      projectRows.delete(id)
    })
  }

  const assetTable = {
    where: vi.fn((field: string) => ({
      equals: vi.fn((value: string) => ({
        toArray: vi.fn(async () =>
          Array.from(assetRows.values()).filter((asset) => {
            const fieldValue = asset[field as keyof AudiobookChapterAsset]
            return fieldValue === value
          })
        )
      }))
    })),
    delete: vi.fn(async (id: string) => {
      assetRows.delete(id)
    })
  }

  return { projectRows, assetRows, projectTable, assetTable }
})

vi.mock("@/db/dexie/schema", () => ({
  db: {
    audiobookProjects: projectTable,
    audiobookChapterAssets: assetTable
  }
}))

import {
  listLegacyAudiobookProjectsForMigration,
  markLegacyAudiobookProjectMigrated,
  serializeLegacyAudiobookProjectForMigration
} from "../audiobook-projects"

const makeProject = (
  id: string,
  overrides: Partial<AudiobookProject> & Record<string, unknown> = {}
): AudiobookProject & Record<string, unknown> => ({
  id,
  title: `Project ${id}`,
  author: "Author",
  description: "Description",
  rawContent: "Chapter text",
  chapters: [
    {
      id: "chapter-1",
      title: "Chapter 1",
      content: "Chapter text",
      order: 0,
      voiceConfig: { voice: "Ava" },
      status: "completed",
      audioDuration: 12
    }
  ],
  chapterAudioAssetIds: { "chapter-1": "asset-1" },
  defaultVoiceConfig: { provider: "openai" },
  status: "completed",
  totalDuration: 12,
  createdAt: 100,
  updatedAt: 200,
  ...overrides
})

const makeAsset = (
  id: string,
  overrides: Partial<AudiobookChapterAsset> = {}
): AudiobookChapterAsset => ({
  id,
  projectId: "legacy-1",
  chapterId: "chapter-1",
  mimeType: "audio/mpeg",
  sizeBytes: 2048,
  blob: new Blob(["audio"], { type: "audio/mpeg" }),
  createdAt: 300,
  ...overrides
})

describe("audiobook migration Dexie helpers", () => {
  beforeEach(() => {
    projectRows.clear()
    assetRows.clear()
    vi.clearAllMocks()
  })

  it("lists unmigrated legacy projects without returning already migrated rows", async () => {
    projectRows.set("legacy-older", makeProject("legacy-older", { updatedAt: 50 }))
    projectRows.set("legacy-newer", makeProject("legacy-newer", { updatedAt: 300 }))
    projectRows.set(
      "legacy-migrated",
      makeProject("legacy-migrated", {
        updatedAt: 400,
        audioStudioMigration: { status: "migrated", projectId: "server-1" }
      })
    )

    const projects = await listLegacyAudiobookProjectsForMigration()

    expect(projects.map((project) => project.id)).toEqual([
      "legacy-newer",
      "legacy-older"
    ])
  })

  it("serializes a project and audio asset metadata without deleting local data", async () => {
    projectRows.set("legacy-1", makeProject("legacy-1"))
    assetRows.set("asset-1", makeAsset("asset-1"))

    const payload = await serializeLegacyAudiobookProjectForMigration("legacy-1")

    expect(payload).toMatchObject({
      migration_schema_version: 1,
      legacy_project_id: "legacy-1",
      title: "Project legacy-1",
      workflow: "narration",
      chapters: [
        {
          legacy_chapter_id: "chapter-1",
          title: "Chapter 1",
          body_text: "Chapter text",
          order: 0
        }
      ],
      audio_assets: [
        {
          legacy_asset_id: "asset-1",
          legacy_chapter_id: "chapter-1",
          mime_type: "audio/mpeg",
          size_bytes: 2048,
          has_blob: true
        }
      ]
    })
    expect(projectTable.delete).not.toHaveBeenCalled()
    expect(assetTable.delete).not.toHaveBeenCalled()
    expect(projectRows.has("legacy-1")).toBe(true)
    expect(assetRows.has("asset-1")).toBe(true)
  })

  it("marks a local project migrated without deleting the Dexie row", async () => {
    projectRows.set("legacy-1", makeProject("legacy-1"))

    await markLegacyAudiobookProjectMigrated("legacy-1", {
      projectId: "server-project-1",
      migrationId: "migration-1"
    })

    const row = projectRows.get("legacy-1")
    expect(row?.audioStudioMigration).toMatchObject({
      status: "migrated",
      projectId: "server-project-1",
      migrationId: "migration-1",
      schemaVersion: 1
    })
    expect(typeof row?.audioStudioMigration.completedAt).toBe("number")
    expect(projectTable.delete).not.toHaveBeenCalled()
  })
})
