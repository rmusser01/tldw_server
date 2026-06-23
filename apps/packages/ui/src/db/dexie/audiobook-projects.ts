import { db } from "./schema"
import type {
  AudiobookProject,
  AudiobookChapterAsset,
  SerializedAudioChapter
} from "./types"

// Storage cap for audiobook assets (200MB)
export const AUDIOBOOK_STORAGE_CAP_BYTES = 200 * 1024 * 1024
export const AUDIOBOOK_MIGRATION_SCHEMA_VERSION = 1

export type LegacyAudiobookMigrationChapter = {
  legacy_chapter_id: string
  title: string
  body_text: string
  order: number
  voice_config: Record<string, any>
  status: SerializedAudioChapter["status"]
  audio_duration?: number
  error_message?: string
}

export type LegacyAudiobookMigrationAudioAsset = {
  legacy_asset_id: string
  legacy_chapter_id: string
  mime_type: string
  size_bytes: number
  created_at: number
  has_blob: boolean
}

export type LegacyAudiobookProjectMigrationPayload = {
  migration_schema_version: typeof AUDIOBOOK_MIGRATION_SCHEMA_VERSION
  legacy_project_id: string
  workflow: "narration"
  title: string
  author: string
  description?: string
  cover_image_asset_id?: string
  cover_image_url?: string
  raw_content: string
  status: AudiobookProject["status"]
  total_duration?: number
  created_at: number
  updated_at: number
  default_voice_config: Record<string, any>
  chapter_audio_asset_ids: Record<string, string>
  chapters: LegacyAudiobookMigrationChapter[]
  audio_assets: LegacyAudiobookMigrationAudioAsset[]
}

export type LegacyAudiobookMigrationMarker = {
  schemaVersion: typeof AUDIOBOOK_MIGRATION_SCHEMA_VERSION
  status: "migrated"
  projectId: string
  migrationId?: string
  completedAt: number
}

type AudiobookProjectWithMigration = AudiobookProject & {
  audioStudioMigration?: LegacyAudiobookMigrationMarker
}

const isMigratedToAudioStudio = (project: AudiobookProject): boolean => {
  const marker = (project as AudiobookProjectWithMigration).audioStudioMigration
  return (
    marker?.status === "migrated" &&
    typeof marker.projectId === "string" &&
    marker.projectId.length > 0
  )
}

/**
 * Get all audiobook projects
 */
export async function getAudiobookProjects(): Promise<AudiobookProject[]> {
  return await db.audiobookProjects
    .orderBy("updatedAt")
    .reverse()
    .toArray()
}

/**
 * List local Audiobook Studio projects that have not been marked migrated.
 */
export async function listLegacyAudiobookProjectsForMigration(): Promise<
  AudiobookProject[]
> {
  const projects = await getAudiobookProjects()
  return projects.filter((project) => !isMigratedToAudioStudio(project))
}

/**
 * Get a single audiobook project by ID
 */
export async function getAudiobookProjectById(
  id: string
): Promise<AudiobookProject | undefined> {
  return await db.audiobookProjects.get(id)
}

/**
 * Serialize local Audiobook Studio project structure for server migration.
 *
 * Audio blobs are intentionally not included here; artifact upload is a later,
 * explicit migration step. The returned asset rows are metadata only.
 */
export async function serializeLegacyAudiobookProjectForMigration(
  id: string
): Promise<LegacyAudiobookProjectMigrationPayload> {
  const project = await getAudiobookProjectById(id)
  if (!project) {
    throw new Error(`Legacy audiobook project not found: ${id}`)
  }

  const assets = await getChapterAssetsByProject(id)

  return {
    migration_schema_version: AUDIOBOOK_MIGRATION_SCHEMA_VERSION,
    legacy_project_id: project.id,
    workflow: "narration",
    title: project.title,
    author: project.author,
    description: project.description,
    cover_image_asset_id: project.coverImageAssetId,
    cover_image_url: project.coverImageUrl,
    raw_content: project.rawContent,
    status: project.status,
    total_duration: project.totalDuration,
    created_at: project.createdAt,
    updated_at: project.updatedAt,
    default_voice_config: project.defaultVoiceConfig,
    chapter_audio_asset_ids: { ...project.chapterAudioAssetIds },
    chapters: project.chapters.map((chapter) => ({
      legacy_chapter_id: chapter.id,
      title: chapter.title,
      body_text: chapter.content,
      order: chapter.order,
      voice_config: chapter.voiceConfig,
      status: chapter.status,
      audio_duration: chapter.audioDuration,
      error_message: chapter.errorMessage
    })),
    audio_assets: assets.map((asset) => ({
      legacy_asset_id: asset.id,
      legacy_chapter_id: asset.chapterId,
      mime_type: asset.mimeType,
      size_bytes: asset.sizeBytes,
      created_at: asset.createdAt,
      has_blob: Boolean(asset.blob)
    }))
  }
}

/**
 * Create or update an audiobook project
 */
export async function upsertAudiobookProject(
  project: AudiobookProject
): Promise<void> {
  await db.audiobookProjects.put({
    ...project,
    updatedAt: Date.now()
  })
}

/**
 * Update specific fields of an audiobook project
 */
export async function updateAudiobookProject(
  id: string,
  updates: Partial<AudiobookProject>
): Promise<void> {
  await db.audiobookProjects.update(id, {
    ...updates,
    updatedAt: Date.now()
  })
}

/**
 * Mark a legacy local project as migrated without deleting the Dexie row.
 */
export async function markLegacyAudiobookProjectMigrated(
  id: string,
  migration: { projectId: string; migrationId?: string }
): Promise<void> {
  const marker: LegacyAudiobookMigrationMarker = {
    schemaVersion: AUDIOBOOK_MIGRATION_SCHEMA_VERSION,
    status: "migrated",
    projectId: migration.projectId,
    migrationId: migration.migrationId,
    completedAt: Date.now()
  }

  await db.audiobookProjects.update(id, {
    audioStudioMigration: marker,
    updatedAt: Date.now()
  } as Partial<AudiobookProject> & {
    audioStudioMigration: LegacyAudiobookMigrationMarker
  })
}

/**
 * Delete an audiobook project and all its chapter audio assets
 */
export async function deleteAudiobookProject(id: string): Promise<void> {
  await db.transaction(
    "rw",
    [db.audiobookProjects, db.audiobookChapterAssets],
    async () => {
      // Delete all chapter audio assets for this project
      await db.audiobookChapterAssets.where("projectId").equals(id).delete()
      // Delete the project itself
      await db.audiobookProjects.delete(id)
    }
  )
}

/**
 * Duplicate an audiobook project (without audio assets)
 */
export async function duplicateAudiobookProject(
  id: string,
  newTitle?: string
): Promise<string | null> {
  const project = await getAudiobookProjectById(id)
  if (!project) return null

  const now = Date.now()
  const newId = crypto.randomUUID()

  const duplicated: AudiobookProject = {
    ...project,
    id: newId,
    title: newTitle || `${project.title} (Copy)`,
    // Reset chapters to pending status without audio
    chapters: project.chapters.map((ch) => ({
      ...ch,
      id: crypto.randomUUID(),
      status: "pending" as const,
      audioDuration: undefined,
      errorMessage: undefined
    })),
    chapterAudioAssetIds: {},
    status: "draft",
    totalDuration: undefined,
    createdAt: now,
    updatedAt: now,
    lastOpenedAt: undefined
  }

  await upsertAudiobookProject(duplicated)
  return newId
}

/**
 * Mark a project as opened (updates lastOpenedAt)
 */
export async function markProjectOpened(id: string): Promise<void> {
  await db.audiobookProjects.update(id, {
    lastOpenedAt: Date.now()
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Chapter Audio Asset Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get a chapter audio asset by ID
 */
export async function getChapterAsset(
  id: string
): Promise<AudiobookChapterAsset | undefined> {
  return await db.audiobookChapterAssets.get(id)
}

/**
 * Get all chapter audio assets for a project
 */
export async function getChapterAssetsByProject(
  projectId: string
): Promise<AudiobookChapterAsset[]> {
  return await db.audiobookChapterAssets
    .where("projectId")
    .equals(projectId)
    .toArray()
}

/**
 * Get total bytes used by audiobook assets
 */
export async function getAudiobookAssetsTotalBytes(): Promise<number> {
  const assets = await db.audiobookChapterAssets.toArray()
  return assets.reduce((sum, asset) => sum + (asset.sizeBytes || 0), 0)
}

/**
 * Store chapter audio blob
 */
export async function storeChapterAudio(
  projectId: string,
  chapterId: string,
  blob: Blob
): Promise<{ assetId: string | null; stored: boolean }> {
  const sizeBytes = blob.size
  const totalBytes = await getAudiobookAssetsTotalBytes()

  if (totalBytes + sizeBytes > AUDIOBOOK_STORAGE_CAP_BYTES) {
    return { assetId: null, stored: false }
  }

  // Check if there's already an asset for this chapter and delete it
  const existing = await db.audiobookChapterAssets
    .where("projectId")
    .equals(projectId)
    .filter((asset) => asset.chapterId === chapterId)
    .first()

  if (existing) {
    await db.audiobookChapterAssets.delete(existing.id)
  }

  const asset: AudiobookChapterAsset = {
    id: crypto.randomUUID(),
    projectId,
    chapterId,
    mimeType: blob.type || "audio/mpeg",
    sizeBytes,
    blob,
    createdAt: Date.now()
  }

  await db.audiobookChapterAssets.put(asset)
  return { assetId: asset.id, stored: true }
}

/**
 * Get chapter audio blob by asset ID
 */
export async function getChapterAudioBlob(
  assetId: string
): Promise<Blob | null> {
  const asset = await db.audiobookChapterAssets.get(assetId)
  return asset?.blob || null
}

/**
 * Delete a specific chapter audio asset
 */
export async function deleteChapterAsset(id: string): Promise<void> {
  await db.audiobookChapterAssets.delete(id)
}

/**
 * Delete all chapter audio assets for a project
 */
export async function deleteChapterAssetsByProject(
  projectId: string
): Promise<void> {
  await db.audiobookChapterAssets.where("projectId").equals(projectId).delete()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions for Store Integration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a new empty project
 */
export function createEmptyProject(title?: string): AudiobookProject {
  const now = Date.now()
  return {
    id: crypto.randomUUID(),
    title: title || "Untitled Audiobook",
    author: "",
    description: "",
    coverImageUrl: "",
    rawContent: "",
    chapters: [],
    chapterAudioAssetIds: {},
    defaultVoiceConfig: {},
    status: "draft",
    createdAt: now,
    updatedAt: now
  }
}

/**
 * Serialize chapters from the store format (with Blobs) to DB format (without Blobs)
 */
export function serializeChapters(
  chapters: Array<{
    id: string
    title: string
    content: string
    order: number
    voiceConfig: Record<string, any>
    status: string
    audioDuration?: number
    errorMessage?: string
    audioBlob?: Blob
  }>
): SerializedAudioChapter[] {
  return chapters.map((ch) => ({
    id: ch.id,
    title: ch.title,
    content: ch.content,
    order: ch.order,
    voiceConfig: ch.voiceConfig,
    status: ch.status as SerializedAudioChapter["status"],
    audioDuration: ch.audioDuration,
    errorMessage: ch.errorMessage
  }))
}
