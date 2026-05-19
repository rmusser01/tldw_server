/**
 * Prompt Sync Service
 *
 * Provides sync operations between local IndexedDB prompts
 * and server-side Prompt Studio. Manual sync remains available, and
 * workspace prompt saves can auto-sync by default.
 */

import { db } from '@/db/dexie/schema'
import { PageAssistDatabase } from '@/db/dexie/chat'
import { generateID } from '@/db/dexie/helpers'
import {
  Prompt as LocalPrompt,
  PromptSyncStatus,
  FewShotExample,
  PromptModule
} from '@/db/dexie/types'
import {
  createProject,
  createPrompt as createServerPrompt,
  updatePrompt as updateServerPrompt,
  getPrompt as getServerPrompt,
  listProjects,
  Prompt as ServerPrompt,
  PromptCreatePayload,
  PromptUpdatePayload,
  Project,
  StandardResponse
} from '@/services/prompt-studio'
import type { ApiSendResponse } from '@/services/api-send'
import { unwrapApiResponseData } from '@/services/response-envelope'
import {
  getPromptStudioDefaults,
  setPromptStudioDefaults
} from '@/services/prompt-studio-settings'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type SyncResult = {
  success: boolean
  localId: string
  serverId?: number
  error?: string
  syncStatus: PromptSyncStatus
}

export type ConflictInfo = {
  localPrompt: LocalPrompt
  serverPrompt: ServerPrompt
  localUpdatedAt: number
  serverUpdatedAt: string
}

export type ConflictResolution = 'keep_local' | 'keep_server' | 'keep_both'

const AUTO_SYNC_PROJECT_NAME = 'Workspace Prompts'
const AUTO_SYNC_PROJECT_DESCRIPTION =
  'Auto-created project used to persist prompts saved from the Prompts workspace.'
const CURRENT_PROMPT_SYNC_PAYLOAD_VERSION = 1

type PromptFormat = 'legacy' | 'structured'

type ComparablePromptPayload = {
  promptFormat: PromptFormat
  promptSchemaVersion: number | null
  promptDefinition: Record<string, any> | null
  systemPrompt: string
  userPrompt: string
  fewShotExamples: FewShotExample[] | null
  modulesConfig: PromptModule[] | null
}

const isValidProjectId = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

/**
 * Unwrap the nested `ApiSendResponse<StandardResponse<T>>` envelope.
 *
 * Server endpoints return `{ ok, data: { success, data: T } }`.  Some
 * endpoints omit the inner `StandardResponse` wrapper and place the
 * payload directly in `data`.  This helper handles both shapes.
 */
const unwrapResponseData = <T>(
  response: ApiSendResponse<StandardResponse<T>> | ApiSendResponse<T>
): T | null => {
  return unwrapApiResponseData<T>(response?.data) ?? null
}

const toText = (value: unknown): string => (typeof value === 'string' ? value : '')
const toFiniteNumberOrNull = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null
const toPromptFormat = (value: unknown): PromptFormat =>
  value === 'structured' ? 'structured' : 'legacy'
const toRecordOrNull = (value: unknown): Record<string, any> | null =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, any>)
    : null
const toArrayOrNull = <T>(value: unknown): T[] | null =>
  Array.isArray(value) ? (value as T[]) : null

const getLocalPromptTextsForConflict = (
  local: LocalPrompt
): { systemPrompt: string; userPrompt: string } => {
  const explicitSystem = toText(local.system_prompt)
  const explicitUser = toText(local.user_prompt)
  const contentFallback = toText(local.content)

  return {
    systemPrompt:
      explicitSystem || (local.is_system ? contentFallback : ''),
    userPrompt:
      explicitUser || (!local.is_system ? contentFallback : '')
  }
}

const getServerPromptTextsForConflict = (
  server: ServerPrompt
): { systemPrompt: string; userPrompt: string } => ({
  systemPrompt: toText(server.system_prompt),
  userPrompt: toText(server.user_prompt)
})

const normalizeForStableHash = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeForStableHash(item))
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, item]) => item !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, normalizeForStableHash(item)])
    )
  }

  return value
}

const promptPayloadHash = (payload: ComparablePromptPayload): string => {
  const combined = JSON.stringify(normalizeForStableHash(payload))
  let hash = 0x811c9dc5
  for (let i = 0; i < combined.length; i += 1) {
    hash ^= combined.charCodeAt(i)
    hash = (hash * 0x01000193) >>> 0
  }
  return hash.toString(16).padStart(8, '0')
}

const getLocalPromptComparablePayload = (
  local: LocalPrompt
): ComparablePromptPayload => {
  const localText = getLocalPromptTextsForConflict(local)
  return {
    promptFormat: toPromptFormat(local.promptFormat),
    promptSchemaVersion: toFiniteNumberOrNull(local.promptSchemaVersion),
    promptDefinition: toRecordOrNull(local.structuredPromptDefinition),
    systemPrompt: localText.systemPrompt,
    userPrompt: localText.userPrompt,
    fewShotExamples: toArrayOrNull<FewShotExample>(local.fewShotExamples),
    modulesConfig: toArrayOrNull<PromptModule>(local.modulesConfig)
  }
}

const getServerPromptComparablePayload = (
  server: ServerPrompt
): ComparablePromptPayload => {
  const serverText = getServerPromptTextsForConflict(server)
  return {
    promptFormat: toPromptFormat(server.prompt_format),
    promptSchemaVersion: toFiniteNumberOrNull(server.prompt_schema_version),
    promptDefinition: toRecordOrNull(server.prompt_definition),
    systemPrompt: serverText.systemPrompt,
    userPrompt: serverText.userPrompt,
    fewShotExamples: toArrayOrNull<FewShotExample>(server.few_shot_examples),
    modulesConfig: toArrayOrNull<PromptModule>(server.modules_config)
  }
}

const hasPromptContentConflict = (
  local: LocalPrompt,
  server: ServerPrompt
): boolean => {
  const localPayload = getLocalPromptComparablePayload(local)
  const serverPayload = getServerPromptComparablePayload(server)
  return promptPayloadHash(localPayload) !== promptPayloadHash(serverPayload)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert a local prompt to server create payload.
 */
function localToServerPayload(
  local: LocalPrompt,
  projectId: number
): PromptCreatePayload {
  const promptFormat = toPromptFormat(local.promptFormat)
  return {
    project_id: projectId,
    name: local.name || local.title,
    system_prompt: local.system_prompt,
    user_prompt: local.user_prompt,
    prompt_format: promptFormat,
    prompt_schema_version:
      promptFormat === 'structured'
        ? toFiniteNumberOrNull(local.promptSchemaVersion)
        : null,
    prompt_definition:
      promptFormat === 'structured'
        ? toRecordOrNull(local.structuredPromptDefinition)
        : null,
    few_shot_examples: local.fewShotExamples,
    modules_config: local.modulesConfig,
    change_description: local.changeDescription || 'Initial sync from workspace'
  }
}

/**
 * Convert a local prompt to server update payload.
 */
function localToServerUpdatePayload(local: LocalPrompt): PromptUpdatePayload {
  const promptFormat = toPromptFormat(local.promptFormat)
  return {
    name: local.name || local.title,
    system_prompt: local.system_prompt,
    user_prompt: local.user_prompt,
    prompt_format: promptFormat,
    prompt_schema_version:
      promptFormat === 'structured'
        ? toFiniteNumberOrNull(local.promptSchemaVersion)
        : null,
    prompt_definition:
      promptFormat === 'structured'
        ? toRecordOrNull(local.structuredPromptDefinition)
        : null,
    few_shot_examples: local.fewShotExamples,
    modules_config: local.modulesConfig,
    change_description: local.changeDescription || 'Synced from workspace'
  }
}

/**
 * Convert server prompt to local prompt fields.
 */
function serverToLocalFields(server: ServerPrompt): Partial<LocalPrompt> {
  return {
    serverId: server.id,
    studioProjectId: server.project_id,
    studioPromptId: server.id,
    name: server.name,
    system_prompt: server.system_prompt,
    user_prompt: server.user_prompt,
    promptFormat: toPromptFormat(server.prompt_format),
    promptSchemaVersion: toFiniteNumberOrNull(server.prompt_schema_version),
    structuredPromptDefinition: toRecordOrNull(server.prompt_definition),
    syncPayloadVersion: CURRENT_PROMPT_SYNC_PAYLOAD_VERSION,
    fewShotExamples: toArrayOrNull<FewShotExample>(server.few_shot_examples),
    modulesConfig: toArrayOrNull<PromptModule>(server.modules_config),
    versionNumber: server.version_number,
    changeDescription: server.change_description,
    serverParentVersionId: server.parent_version_id,
    serverUpdatedAt: server.updated_at,
    syncStatus: 'synced' as PromptSyncStatus,
    lastSyncedAt: Date.now()
  }
}

/**
 * Create a new local prompt from server prompt.
 */
function serverToNewLocalPrompt(server: ServerPrompt): LocalPrompt {
  const now = Date.now()
  return {
    id: generateID(),
    title: server.name,
    name: server.name,
    content: server.system_prompt || server.user_prompt || '',
    is_system: !!server.system_prompt,
    system_prompt: server.system_prompt,
    user_prompt: server.user_prompt,
    promptFormat: toPromptFormat(server.prompt_format),
    promptSchemaVersion: toFiniteNumberOrNull(server.prompt_schema_version),
    structuredPromptDefinition: toRecordOrNull(server.prompt_definition),
    syncPayloadVersion: CURRENT_PROMPT_SYNC_PAYLOAD_VERSION,
    createdAt: now,
    updatedAt: now,
    usageCount: 0,
    lastUsedAt: null,
    // Server sync fields
    serverId: server.id,
    studioProjectId: server.project_id,
    studioPromptId: server.id,
    fewShotExamples: toArrayOrNull<FewShotExample>(server.few_shot_examples),
    modulesConfig: toArrayOrNull<PromptModule>(server.modules_config),
    versionNumber: server.version_number,
    changeDescription: server.change_description,
    serverParentVersionId: server.parent_version_id,
    serverUpdatedAt: server.updated_at,
    syncStatus: 'synced',
    sourceSystem: 'studio',
    lastSyncedAt: now
  }
}

export async function shouldAutoSyncWorkspacePrompts(): Promise<boolean> {
  try {
    const defaults = await getPromptStudioDefaults()
    return defaults.autoSyncWorkspacePrompts !== false
  } catch {
    return true
  }
}

export async function resolveAutoSyncProjectId(
  preferredProjectId?: number | null
): Promise<number | null> {
  if (isValidProjectId(preferredProjectId)) {
    return preferredProjectId
  }

  let defaults = await getPromptStudioDefaults()
  if (isValidProjectId(defaults.defaultProjectId)) {
    return defaults.defaultProjectId
  }

  const projects = await getAvailableProjects()
  const firstProjectId = projects.find((project) =>
    isValidProjectId(project.id)
  )?.id

  if (isValidProjectId(firstProjectId)) {
    await setPromptStudioDefaults({ defaultProjectId: firstProjectId })
    return firstProjectId
  }

  try {
    const created = unwrapResponseData<Project>(
      await createProject({
        name: AUTO_SYNC_PROJECT_NAME,
        description: AUTO_SYNC_PROJECT_DESCRIPTION
      })
    )
    const createdId = created?.id
    if (isValidProjectId(createdId)) {
      await setPromptStudioDefaults({ defaultProjectId: createdId })
      return createdId
    }
  } catch {
    // Fall through and return null. Caller decides whether to mark pending.
  }

  // Avoid repeated failed create attempts in the same session by caching "no default".
  defaults = await getPromptStudioDefaults()
  if (defaults.defaultProjectId !== null && defaults.defaultProjectId !== undefined) {
    await setPromptStudioDefaults({ defaultProjectId: null })
  }
  return null
}

export async function autoSyncPrompt(
  localId: string,
  preferredProjectId?: number | null
): Promise<SyncResult> {
  const local = await db.prompts.get(localId)
  if (!local) {
    return {
      success: false,
      localId,
      error: 'Local prompt not found',
      syncStatus: 'local'
    }
  }

  const projectId = await resolveAutoSyncProjectId(
    preferredProjectId ?? local.studioProjectId
  )

  if (!isValidProjectId(projectId)) {
    await db.prompts.update(localId, {
      syncStatus: 'pending',
      updatedAt: Date.now()
    })
    return {
      success: false,
      localId,
      error:
        'No Prompt Studio project available for auto-sync. Configure a default project in Prompt Studio settings.',
      syncStatus: 'pending'
    }
  }

  const result = await pushToStudio(localId, projectId)
  if (!result.success) {
    await db.prompts.update(localId, {
      syncStatus: 'pending',
      studioProjectId: projectId,
      updatedAt: Date.now()
    })
  }
  return result
}

// ─────────────────────────────────────────────────────────────────────────────
// Sync Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Push a local prompt to Prompt Studio (create or update).
 *
 * @param localId - Local prompt ID
 * @param projectId - Target Prompt Studio project ID
 * @returns Sync result
 */
export async function pushToStudio(
  localId: string,
  projectId: number
): Promise<SyncResult> {
  try {
    const local = await db.prompts.get(localId)
    if (!local) {
      return {
        success: false,
        localId,
        error: 'Local prompt not found',
        syncStatus: 'local'
      }
    }

    // If already linked, update existing server prompt
    if (local.serverId) {
      const updatePayload = localToServerUpdatePayload(local)
      const response = await updateServerPrompt(local.serverId, updatePayload)
      const serverPrompt = unwrapResponseData<ServerPrompt>(response)

      if (!serverPrompt) {
        return {
          success: false,
          localId,
          serverId: local.serverId,
          error: 'Failed to update server prompt',
          syncStatus: 'pending'
        }
      }

      const updateFields = serverToLocalFields(serverPrompt)
      await db.prompts.update(localId, updateFields)

      return {
        success: true,
        localId,
        serverId: serverPrompt.id,
        syncStatus: 'synced'
      }
    }

    // Create new server prompt
    const createPayload = localToServerPayload(local, projectId)
    const response = await createServerPrompt(createPayload)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      return {
        success: false,
        localId,
        error: 'Failed to create server prompt',
        syncStatus: 'pending'
      }
    }

    const updateFields = serverToLocalFields(serverPrompt)
    updateFields.studioProjectId = projectId
    await db.prompts.update(localId, updateFields)

    return {
      success: true,
      localId,
      serverId: serverPrompt.id,
      syncStatus: 'synced'
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId,
      error: error instanceof Error ? error.message : 'Push failed',
      syncStatus: 'pending'
    }
  }
}

/**
 * Pull a server prompt to local storage.
 *
 * @param serverId - Server prompt ID
 * @param existingLocalId - Optional existing local ID to update
 * @returns Sync result
 */
export async function pullFromStudio(
  serverId: number,
  existingLocalId?: string
): Promise<SyncResult> {
  try {
    const response = await getServerPrompt(serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      return {
        success: false,
        localId: existingLocalId || '',
        serverId,
        error: 'Server prompt not found',
        syncStatus: 'local'
      }
    }

    // Update existing local prompt
    if (existingLocalId) {
      const local = await db.prompts.get(existingLocalId)
      if (local) {
        const updateFields = serverToLocalFields(serverPrompt)
        await db.prompts.update(existingLocalId, updateFields)

        return {
          success: true,
          localId: existingLocalId,
          serverId,
          syncStatus: 'synced'
        }
      }
    }

    // Check if we already have this prompt locally by serverId
    const existing = await db.prompts.where('serverId').equals(serverId).first()
    if (existing) {
      const updateFields = serverToLocalFields(serverPrompt)
      await db.prompts.update(existing.id, updateFields)

      return {
        success: true,
        localId: existing.id,
        serverId,
        syncStatus: 'synced'
      }
    }

    // Create new local prompt
    const newLocal = serverToNewLocalPrompt(serverPrompt)
    await db.prompts.add(newLocal)

    return {
      success: true,
      localId: newLocal.id,
      serverId,
      syncStatus: 'synced'
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId: existingLocalId || '',
      serverId,
      error: error instanceof Error ? error.message : 'Pull failed',
      syncStatus: 'local'
    }
  }
}

/**
 * Link an existing local prompt to an existing server prompt.
 *
 * @param localId - Local prompt ID
 * @param serverId - Server prompt ID
 * @returns Sync result
 */
export async function linkPrompts(
  localId: string,
  serverId: number
): Promise<SyncResult> {
  try {
    const local = await db.prompts.get(localId)
    if (!local) {
      return {
        success: false,
        localId,
        serverId,
        error: 'Local prompt not found',
        syncStatus: 'local'
      }
    }

    const response = await getServerPrompt(serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      return {
        success: false,
        localId,
        serverId,
        error: 'Server prompt not found',
        syncStatus: 'local'
      }
    }

    // Link by updating local with server reference
    await db.prompts.update(localId, {
      serverId,
      studioProjectId: serverPrompt.project_id,
      studioPromptId: serverId,
      serverUpdatedAt: serverPrompt.updated_at,
      syncStatus: 'pending', // Pending because content may differ
      lastSyncedAt: Date.now()
    })

    return {
      success: true,
      localId,
      serverId,
      syncStatus: 'pending'
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId,
      serverId,
      error: error instanceof Error ? error.message : 'Link failed',
      syncStatus: 'local'
    }
  }
}

/**
 * Unlink a local prompt from server (keep local copy).
 */
export async function unlinkPrompt(localId: string): Promise<SyncResult> {
  let local: LocalPrompt | undefined
  try {
    local = await db.prompts.get(localId)
    if (!local) {
      return {
        success: false,
        localId,
        error: 'Local prompt not found',
        syncStatus: 'local'
      }
    }

    await db.prompts.update(localId, {
      serverId: null,
      studioProjectId: null,
      studioPromptId: null,
      serverUpdatedAt: null,
      syncStatus: 'local',
      sourceSystem: 'workspace',
      lastSyncedAt: null
    })

    return {
      success: true,
      localId,
      syncStatus: 'local'
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId,
      error: error instanceof Error ? error.message : 'Unlink failed',
      syncStatus: local?.syncStatus || 'local'
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Status & Conflict Detection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get sync status for a local prompt.
 */
export async function getSyncStatus(localId: string): Promise<{
  status: PromptSyncStatus
  serverId?: number
  lastSyncedAt?: number
  hasConflict: boolean
}> {
  const local = await db.prompts.get(localId)
  if (!local) {
    return { status: 'local', hasConflict: false }
  }

  if (!local.serverId) {
    return { status: 'local', hasConflict: false }
  }

  // Check for conflict by comparing timestamps and content fingerprints.
  try {
    const response = await getServerPrompt(local.serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      // Server prompt deleted
      return {
        status: 'conflict',
        serverId: local.serverId,
        lastSyncedAt: local.lastSyncedAt || undefined,
        hasConflict: true
      }
    }

    const serverUpdatedAt = serverPrompt.updated_at
    const serverVersionChanged = local.serverUpdatedAt !== serverUpdatedAt
    const localHasUnsyncedChanges = (local.updatedAt || 0) > (local.lastSyncedAt || 0)
    const contentChanged = hasPromptContentConflict(local, serverPrompt)
    const hasConflict = serverVersionChanged && localHasUnsyncedChanges && contentChanged

    return {
      status: hasConflict ? 'conflict' : local.syncStatus || 'synced',
      serverId: local.serverId,
      lastSyncedAt: local.lastSyncedAt || undefined,
      hasConflict
    }
  } catch {
    return {
      status: local.syncStatus || 'local',
      serverId: local.serverId,
      lastSyncedAt: local.lastSyncedAt || undefined,
      hasConflict: false
    }
  }
}

/**
 * Get detailed conflict information.
 */
export async function getConflictInfo(localId: string): Promise<ConflictInfo | null> {
  const local = await db.prompts.get(localId)
  if (!local || !local.serverId) return null

  try {
    const response = await getServerPrompt(local.serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)
    if (!serverPrompt) return null

    return {
      localPrompt: local,
      serverPrompt,
      localUpdatedAt: local.updatedAt || local.createdAt,
      serverUpdatedAt: serverPrompt.updated_at
    }
  } catch {
    return null
  }
}

/**
 * Resolve a sync conflict.
 */
export async function resolveConflict(
  localId: string,
  resolution: ConflictResolution
): Promise<SyncResult> {
  const local = await db.prompts.get(localId)
  if (!local || !local.serverId) {
    return {
      success: false,
      localId,
      error: 'No conflict to resolve',
      syncStatus: 'local'
    }
  }

  switch (resolution) {
    case 'keep_local':
      // Push local to server (overwrite server)
      return await pushToStudio(localId, local.studioProjectId!)

    case 'keep_server':
      // Pull server to local (overwrite local)
      return await pullFromStudio(local.serverId, localId)

    case 'keep_both':
      // Unlink and create a new server version
      await unlinkPrompt(localId)
      if (local.studioProjectId) {
        return await pushToStudio(localId, local.studioProjectId)
      }
      return {
        success: true,
        localId,
        syncStatus: 'local'
      }

    default:
      return {
        success: false,
        localId,
        error: 'Invalid resolution',
        syncStatus: 'conflict'
      }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get all prompts with their sync status.
 */
export async function getAllPromptsWithSyncStatus(): Promise<Array<{
  prompt: LocalPrompt
  syncStatus: PromptSyncStatus
  isSynced: boolean
}>> {
  const dbInstance = new PageAssistDatabase()
  const prompts = await dbInstance.getAllPrompts()

  return prompts.map(prompt => ({
    prompt,
    syncStatus: prompt.syncStatus || 'local',
    isSynced: prompt.syncStatus === 'synced'
  }))
}

/**
 * Get all prompts linked to a specific project.
 */
export async function getPromptsByProject(projectId: number): Promise<LocalPrompt[]> {
  return await db.prompts
    .where('studioProjectId')
    .equals(projectId)
    .filter(p => !p.deletedAt)
    .toArray()
}

/**
 * Get available projects for syncing.
 */
export async function getAvailableProjects(): Promise<Project[]> {
  try {
    const response = await listProjects({ per_page: 100 })
    return unwrapResponseData<Project[]>(response) ?? []
  } catch {
    return []
  }
}
