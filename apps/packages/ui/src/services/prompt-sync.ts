/**
 * Prompt Sync Service
 *
 * Provides sync operations between local IndexedDB prompts
 * and server-side Prompt Studio. Manual sync remains available, and
 * workspace prompt saves can auto-sync by default.
 */
import { PageAssistDatabase } from "@/db/dexie/chat"
import { generateID } from "@/db/dexie/helpers"
import { db } from "@/db/dexie/schema"
import {
  FewShotExample,
  Prompt as LocalPrompt,
  PromptModule,
  PromptSyncStatus
} from "@/db/dexie/types"
import type { ApiSendResponse } from "@/services/api-send"
import {
  Project,
  PromptCreatePayload,
  PromptUpdatePayload,
  Prompt as ServerPrompt,
  StandardResponse,
  createProject,
  createPrompt as createServerPrompt,
  getPrompt as getServerPrompt,
  listProjects,
  updatePrompt as updateServerPrompt
} from "@/services/prompt-studio"
import {
  getPromptStudioDefaults,
  setPromptStudioDefaults
} from "@/services/prompt-studio-settings"
import {
  clearRecipePersistenceScoped,
  markRecipePersistenceUnknown,
  readRecipePersistenceUncertainty
} from "@/services/recipe-persistence-uncertainty"
import { unwrapApiResponseData } from "@/services/response-envelope"
import {
  type ParsedStructuredPromptDefinition,
  parseStructuredPromptDefinitionForTransport
} from "@/services/structured-prompt-transport"
import type {
  RecipePersistenceDispatch,
  RecipePersistenceRequestPolicy
} from "@/services/tldw/recipe-request-snapshot"

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type SyncResult = {
  success: boolean
  localId: string
  recipeOwnership: RecipeSyncOwnership | null
  serverId?: number
  error?: string
  syncStatus: PromptSyncStatus
  failureKind?: "validation" | "invalid_server_payload" | "transient"
  /** Shared recovery state prevents retry/rollback, independently of this dispatch. */
  recipeWriteBlocked?: true
}

export type RecipeSyncOwnership = Readonly<{
  dispatch: RecipePersistenceDispatch
  localId: string
}>

export type RecipePersistenceInput = Readonly<{ expectedOwnerId: string }>

/** Dispatch evidence alone determines whether a failed write could have mutated. */
export function classifyRecipeDispatch(dispatch: RecipePersistenceDispatch) {
  if (dispatch.state === "not_dispatched") return "known_rejection"
  return dispatch.state === "dispatched" && dispatch.actualOwnerId
    ? "scoped_uncertain"
    : "unknown_owner"
}

const notDispatched = (localId: string): RecipeSyncOwnership => ({
  localId,
  dispatch: { state: "not_dispatched", actualOwnerId: null }
})

const durableRecipeBlock = (localId: string): SyncResult => ({
  success: false,
  localId,
  recipeOwnership: notDispatched(localId),
  recipeWriteBlocked: true,
  error: "Recipe has an unresolved durable error",
  syncStatus: "error",
  failureKind: "validation"
})

/** A blocked authority read is not evidence that this attempt dispatched. */
const readRecipeWriteBlock = async (
  localId: string,
  expectedOwnerId: string,
  recipeOwnership = notDispatched(localId)
): Promise<SyncResult | null> => {
  let error = "Recipe has an unresolved operation"
  try {
    if (
      (await readRecipePersistenceUncertainty(localId, expectedOwnerId)) ===
      "clear"
    )
      return null
  } catch {
    error = "Recipe uncertainty authority is unavailable"
  }
  // The authority await may have overlapped another writer. Status is display
  // data; the explicit block remains authoritative even if this read fails.
  const current = await db.prompts.get(localId).catch(() => undefined)
  return {
    ...durableRecipeBlock(localId),
    recipeOwnership,
    ...(current?.serverId ? { serverId: current.serverId } : {}),
    error,
    syncStatus: current?.syncStatus || "error"
  }
}

/** Every v2 pending transition checks the current lock inside its write transaction. */
const markLocalPending = async (
  localId: string,
  isRecipe: boolean,
  fields: Partial<LocalPrompt>
): Promise<boolean> => {
  let durableError = false
  await db.prompts.update(
    localId,
    isRecipe
      ? (current) => {
          if (current.syncStatus === "error") {
            durableError = true
            return false
          }
          Object.assign(current, fields, { syncStatus: "pending" })
        }
      : { ...fields, syncStatus: "pending" }
  )
  return durableError
}

const clearReconciledOwner = async (ownership: RecipeSyncOwnership | null) => {
  if (
    ownership?.dispatch.state === "dispatched" &&
    ownership.dispatch.actualOwnerId
  ) {
    await clearRecipePersistenceScoped(
      ownership.localId,
      ownership.dispatch.actualOwnerId
    )
  }
}

/** Only endpoint-contract rejections known to happen without a mutation qualify.
 * Bare status codes (including proxy errors) are not sufficient evidence.
 */
const isKnownMutationRejection = (
  response: ApiSendResponse<unknown>
): boolean => {
  if (response.ok !== false) return false
  const detail = (response.data as { detail?: unknown } | null)?.detail
  if (response.status === 422 && Array.isArray(detail) && detail.length > 0) {
    return detail.every(
      (item) =>
        item &&
        Array.isArray(item.loc) &&
        typeof item.msg === "string" &&
        typeof item.type === "string"
    )
  }
  if (response.status === 403) return detail === "Access denied to this prompt"
  if (response.status === 401)
    return (
      detail === "Not authenticated" ||
      detail === "Invalid authentication credentials"
    )
  if (response.status === 409)
    return detail === "Prompt with this name already exists in the project"
  return false
}

export type ConflictInfo = {
  localPrompt: LocalPrompt
  serverPrompt: ServerPrompt
  localUpdatedAt: number
  serverUpdatedAt: string
}

export type ConflictResolution = "keep_local" | "keep_server" | "keep_both"

const AUTO_SYNC_PROJECT_NAME = "Workspace Prompts"
const AUTO_SYNC_PROJECT_DESCRIPTION =
  "Auto-created project used to persist prompts saved from the Prompts workspace."
const CURRENT_PROMPT_SYNC_PAYLOAD_VERSION = 1

type PromptFormat = "legacy" | "structured"

type ComparablePromptPayload = {
  promptFormat: PromptFormat
  promptSchemaVersion: number | null
  definitionKind: string | null
  promptDefinition: ParsedStructuredPromptDefinition | null
  systemPrompt: string
  userPrompt: string
  fewShotExamples: FewShotExample[] | null
  modulesConfig: PromptModule[] | null
}

const isValidProjectId = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  Number.isInteger(value) &&
  value > 0

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

const toText = (value: unknown): string =>
  typeof value === "string" ? value : ""
const toArrayOrNull = <T>(value: unknown): T[] | null =>
  Array.isArray(value) ? (value as T[]) : null

const parsePromptIdentity = (
  value: unknown,
  promptFormat: unknown,
  promptSchemaVersion: unknown
): {
  promptFormat: PromptFormat
  promptSchemaVersion: number | null
  promptDefinition: ParsedStructuredPromptDefinition | null
} => {
  const promptDefinition = parseStructuredPromptDefinitionForTransport(
    value,
    promptFormat,
    promptSchemaVersion
  )
  if (promptDefinition === null) {
    return {
      promptFormat: "legacy",
      promptSchemaVersion: null,
      promptDefinition: null
    }
  }
  return {
    promptFormat: "structured",
    promptSchemaVersion: promptDefinition.schema_version,
    promptDefinition
  }
}

const getLocalPromptTextsForConflict = (
  local: LocalPrompt
): { systemPrompt: string; userPrompt: string } => {
  const explicitSystem = toText(local.system_prompt)
  const explicitUser = toText(local.user_prompt)
  const contentFallback = toText(local.content)

  return {
    systemPrompt: explicitSystem || (local.is_system ? contentFallback : ""),
    userPrompt: explicitUser || (!local.is_system ? contentFallback : "")
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

  if (value && typeof value === "object") {
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
  return hash.toString(16).padStart(8, "0")
}

const getLocalPromptComparablePayload = (
  local: LocalPrompt
): ComparablePromptPayload => {
  const localText = getLocalPromptTextsForConflict(local)
  const identity = parsePromptIdentity(
    local.structuredPromptDefinition,
    local.promptFormat,
    local.promptSchemaVersion
  )
  return {
    promptFormat: identity.promptFormat,
    promptSchemaVersion: identity.promptSchemaVersion,
    definitionKind:
      identity.promptDefinition?.schema_version === 2
        ? identity.promptDefinition.definition_kind
        : null,
    promptDefinition: identity.promptDefinition,
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
  const identity = parsePromptIdentity(
    server.prompt_definition,
    server.prompt_format,
    server.prompt_schema_version
  )
  return {
    promptFormat: identity.promptFormat,
    promptSchemaVersion: identity.promptSchemaVersion,
    definitionKind:
      identity.promptDefinition?.schema_version === 2
        ? identity.promptDefinition.definition_kind
        : null,
    promptDefinition: identity.promptDefinition,
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
  const identity = parsePromptIdentity(
    local.structuredPromptDefinition,
    local.promptFormat,
    local.promptSchemaVersion
  )
  return {
    project_id: projectId,
    name: local.name || local.title,
    system_prompt: local.system_prompt,
    user_prompt: local.user_prompt,
    prompt_format: identity.promptFormat,
    prompt_schema_version: identity.promptSchemaVersion,
    prompt_definition: identity.promptDefinition,
    few_shot_examples: local.fewShotExamples,
    modules_config: local.modulesConfig,
    change_description: local.changeDescription || "Initial sync from workspace"
  }
}

/**
 * Convert a local prompt to server update payload.
 */
function localToServerUpdatePayload(local: LocalPrompt): PromptUpdatePayload {
  const identity = parsePromptIdentity(
    local.structuredPromptDefinition,
    local.promptFormat,
    local.promptSchemaVersion
  )
  return {
    name: local.name || local.title,
    system_prompt: local.system_prompt,
    user_prompt: local.user_prompt,
    prompt_format: identity.promptFormat,
    prompt_schema_version: identity.promptSchemaVersion,
    prompt_definition: identity.promptDefinition,
    few_shot_examples: local.fewShotExamples,
    modules_config: local.modulesConfig,
    change_description: local.changeDescription || "Synced from workspace"
  }
}

/**
 * Convert server prompt to local prompt fields.
 */
function serverToLocalFields(server: ServerPrompt): Partial<LocalPrompt> {
  const identity = parsePromptIdentity(
    server.prompt_definition,
    server.prompt_format,
    server.prompt_schema_version
  )
  return {
    serverId: server.id,
    studioProjectId: server.project_id,
    studioPromptId: server.id,
    name: server.name,
    system_prompt: server.system_prompt,
    user_prompt: server.user_prompt,
    promptFormat: identity.promptFormat,
    promptSchemaVersion: identity.promptSchemaVersion,
    structuredPromptDefinition: identity.promptDefinition,
    syncPayloadVersion: CURRENT_PROMPT_SYNC_PAYLOAD_VERSION,
    fewShotExamples: toArrayOrNull<FewShotExample>(server.few_shot_examples),
    modulesConfig: toArrayOrNull<PromptModule>(server.modules_config),
    versionNumber: server.version_number,
    changeDescription: server.change_description,
    serverParentVersionId: server.parent_version_id,
    serverUpdatedAt: server.updated_at,
    syncStatus: "synced" as PromptSyncStatus,
    lastSyncedAt: Date.now()
  }
}

/**
 * Create a new local prompt from server prompt.
 */
function serverToNewLocalPrompt(server: ServerPrompt): LocalPrompt {
  const now = Date.now()
  const identity = parsePromptIdentity(
    server.prompt_definition,
    server.prompt_format,
    server.prompt_schema_version
  )
  return {
    id: generateID(),
    title: server.name,
    name: server.name,
    content: server.system_prompt || server.user_prompt || "",
    is_system: !!server.system_prompt,
    system_prompt: server.system_prompt,
    user_prompt: server.user_prompt,
    promptFormat: identity.promptFormat,
    promptSchemaVersion: identity.promptSchemaVersion,
    structuredPromptDefinition: identity.promptDefinition,
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
    syncStatus: "synced",
    sourceSystem: "studio",
    lastSyncedAt: now
  }
}

async function persistServerPrompt(
  localId: string,
  local: LocalPrompt,
  projectId: number,
  input: RecipePersistenceInput | undefined,
  createCopy: boolean
): Promise<SyncResult> {
  let recipeOwnership: RecipeSyncOwnership | null = notDispatched(localId)
  let isRecipe = false
  const originalStatus = local.syncStatus || "local"
  const failure = async (
    error: unknown,
    failureKind: SyncResult["failureKind"]
  ): Promise<SyncResult> => {
    const uncertain =
      isRecipe &&
      recipeOwnership?.dispatch.state !== "not_dispatched" &&
      failureKind !== "validation"
    if (uncertain) {
      // Every caller (including outbox/batch) must replace retryable pending state.
      // Storage failure cannot erase the authority's pre-dispatch marker.
      try {
        await db.prompts.update(localId, { syncStatus: "error" })
      } catch {
        /* Preserve dispatch evidence. */
      }
    }
    return {
      success: false,
      localId,
      recipeOwnership,
      ...(local.serverId ? { serverId: local.serverId } : {}),
      error: error instanceof Error ? error.message : String(error),
      syncStatus: uncertain
        ? "error"
        : failureKind === "validation"
          ? originalStatus
          : "pending",
      failureKind
    }
  }
  let payload: PromptCreatePayload | PromptUpdatePayload
  let policy: RecipePersistenceRequestPolicy
  try {
    payload = createCopy
      ? localToServerPayload(local, projectId)
      : localToServerUpdatePayload(local)
    isRecipe = payload.prompt_schema_version === 2
    if (isRecipe) {
      if (local.syncStatus === "error") return durableRecipeBlock(localId)
      if (!input?.expectedOwnerId)
        return failure("Recipe persistence owner is required", "validation")
      policy = {
        mode: "require",
        expectedOwnerId: input.expectedOwnerId,
        localId
      }
      const blocked = await readRecipeWriteBlock(localId, input.expectedOwnerId)
      if (blocked) return blocked
    } else {
      policy = { mode: "capture" }
    }
  } catch (error) {
    return failure(error, "validation")
  }

  // Any exception crossing the transport boundary without dispatch metadata is ambiguous.
  recipeOwnership = {
    localId,
    dispatch: { state: "unknown", actualOwnerId: null }
  }
  try {
    const response = createCopy
      ? await createServerPrompt(payload as PromptCreatePayload, {
          recipePersistence: policy
        })
      : await updateServerPrompt(
          local.serverId!,
          payload as PromptUpdatePayload,
          { recipePersistence: policy }
        )
    recipeOwnership = {
      localId,
      dispatch: response.recipePersistence ?? {
        state: "unknown",
        actualOwnerId: null
      }
    }
    if (recipeOwnership.dispatch.state === "not_dispatched") {
      if (isRecipe && input) {
        // A late authority reservation can reject after sync's clear preflight.
        const blocked = await readRecipeWriteBlock(
          localId,
          input.expectedOwnerId,
          recipeOwnership
        )
        if (blocked) return blocked
      }
      return failure(
        response.error || "Prompt was not dispatched",
        "validation"
      )
    }
    if (
      policy.mode === "require" &&
      classifyRecipeDispatch(recipeOwnership.dispatch) === "unknown_owner"
    ) {
      await markRecipePersistenceUnknown(localId)
      return failure(
        "Recipe dispatch owner is unknown",
        "invalid_server_payload"
      )
    }
    if (isKnownMutationRejection(response)) {
      await clearReconciledOwner(recipeOwnership)
      return failure(response.error || "Prompt rejected", "validation")
    }
    const serverPrompt =
      response.ok === false ? null : unwrapResponseData<ServerPrompt>(response)
    if (!serverPrompt || !isValidProjectId(serverPrompt.id)) {
      return failure(
        response.error || "Invalid server prompt response",
        "invalid_server_payload"
      )
    }
    let fields: Partial<LocalPrompt>
    try {
      fields = serverToLocalFields(serverPrompt)
    } catch (error) {
      return failure(error, "invalid_server_payload")
    }
    if (createCopy) fields.studioProjectId = projectId
    const updated = await db.prompts.update(localId, fields)
    if (updated === 0)
      return failure(
        "Local prompt disappeared during reconciliation",
        "transient"
      )
    await clearReconciledOwner(recipeOwnership)
    return {
      success: true,
      localId,
      recipeOwnership,
      serverId: serverPrompt.id,
      syncStatus: "synced"
    }
  } catch (error) {
    if (
      policy.mode === "require" &&
      classifyRecipeDispatch(recipeOwnership.dispatch) === "unknown_owner"
    ) {
      // A failed background message can coexist with its retained scoped dispatch marker.
      try {
        await markRecipePersistenceUnknown(localId)
      } catch {
        /* Fail closed at the consumer too. */
      }
    }
    return failure(error, "transient")
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
  if (
    defaults.defaultProjectId !== null &&
    defaults.defaultProjectId !== undefined
  ) {
    await setPromptStudioDefaults({ defaultProjectId: null })
  }
  return null
}

export async function autoSyncPrompt(
  localId: string,
  preferredProjectId?: number | null,
  input?: RecipePersistenceInput
): Promise<SyncResult> {
  const recipeOwnership = notDispatched(localId)
  const local = await db.prompts.get(localId)
  if (!local) {
    return {
      success: false,
      localId,
      recipeOwnership,
      error: "Local prompt not found",
      syncStatus: "local"
    }
  }

  let identity: ReturnType<typeof parsePromptIdentity>
  try {
    identity = parsePromptIdentity(
      local.structuredPromptDefinition,
      local.promptFormat,
      local.promptSchemaVersion
    )
  } catch (error: unknown) {
    return {
      success: false,
      localId,
      recipeOwnership,
      ...(local.serverId ? { serverId: local.serverId } : {}),
      error:
        error instanceof Error ? error.message : "invalid_prompt_definition",
      syncStatus: local.syncStatus || (local.serverId ? "conflict" : "local"),
      failureKind: "validation"
    }
  }

  const isRecipe = identity.promptSchemaVersion === 2
  if (isRecipe && local.syncStatus === "error") {
    return durableRecipeBlock(localId)
  }
  if (isRecipe && !input?.expectedOwnerId) {
    return {
      success: false,
      localId,
      recipeOwnership,
      error: "Recipe persistence owner is required",
      syncStatus: local.syncStatus || "local",
      failureKind: "validation"
    }
  }

  if (isRecipe && input) {
    const blocked = await readRecipeWriteBlock(localId, input.expectedOwnerId)
    if (blocked) return blocked
  }
  const projectId = isRecipe
    ? isValidProjectId(preferredProjectId)
      ? preferredProjectId
      : isValidProjectId(local.studioProjectId)
        ? local.studioProjectId
        : (await getPromptStudioDefaults()).defaultProjectId
    : await resolveAutoSyncProjectId(
        preferredProjectId ?? local.studioProjectId
      )

  if (!isValidProjectId(projectId)) {
    if (await markLocalPending(localId, isRecipe, { updatedAt: Date.now() }))
      return durableRecipeBlock(localId)
    return {
      success: false,
      localId,
      recipeOwnership,
      error:
        "No Prompt Studio project available for auto-sync. Configure a default project in Prompt Studio settings.",
      syncStatus: "pending",
      failureKind: "transient"
    }
  }

  const result = await pushToStudio(localId, projectId, input)
  if (
    !result.success &&
    result.failureKind === "transient" &&
    (!isRecipe || result.recipeOwnership?.dispatch.state === "not_dispatched")
  ) {
    if (
      await markLocalPending(localId, isRecipe, {
        studioProjectId: projectId,
        updatedAt: Date.now()
      })
    )
      return durableRecipeBlock(localId)
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
  projectId: number,
  input?: RecipePersistenceInput
): Promise<SyncResult> {
  const recipeOwnership = notDispatched(localId)
  let local: LocalPrompt | undefined
  try {
    local = await db.prompts.get(localId)
  } catch (error) {
    return {
      success: false,
      localId,
      recipeOwnership,
      error: error instanceof Error ? error.message : "Push failed",
      syncStatus: "pending",
      failureKind: "transient"
    }
  }
  if (!local)
    return {
      success: false,
      localId,
      recipeOwnership,
      error: "Local prompt not found",
      syncStatus: "local"
    }
  return persistServerPrompt(localId, local, projectId, input, !local.serverId)
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
  let recipeOwnership: RecipeSyncOwnership | null = null
  try {
    const response = await getServerPrompt(serverId, {
      recipePersistence: { mode: "capture" }
    })
    const dispatch = response.recipePersistence
    recipeOwnership = dispatch
      ? { localId: existingLocalId || "", dispatch }
      : null
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      return {
        success: false,
        localId: existingLocalId || "",
        recipeOwnership,
        serverId,
        error: "Server prompt not found",
        syncStatus: "local"
      }
    }

    // Update existing local prompt
    if (existingLocalId) {
      const local = await db.prompts.get(existingLocalId)
      if (local) {
        const updateFields = serverToLocalFields(serverPrompt)
        if ((await db.prompts.update(existingLocalId, updateFields)) === 0) {
          throw new Error("Local prompt disappeared during reconciliation")
        }
        recipeOwnership = dispatch
          ? { localId: existingLocalId, dispatch }
          : null
        await clearReconciledOwner(recipeOwnership)

        return {
          success: true,
          localId: existingLocalId,
          recipeOwnership,
          serverId,
          syncStatus: "synced"
        }
      }
    }

    // Check if we already have this prompt locally by serverId
    const existing = await db.prompts.where("serverId").equals(serverId).first()
    if (existing) {
      const updateFields = serverToLocalFields(serverPrompt)
      if ((await db.prompts.update(existing.id, updateFields)) === 0) {
        throw new Error("Local prompt disappeared during reconciliation")
      }
      recipeOwnership = dispatch ? { localId: existing.id, dispatch } : null
      await clearReconciledOwner(recipeOwnership)

      return {
        success: true,
        localId: existing.id,
        recipeOwnership,
        serverId,
        syncStatus: "synced"
      }
    }

    // Create new local prompt
    const newLocal = serverToNewLocalPrompt(serverPrompt)
    await db.prompts.add(newLocal)
    recipeOwnership = dispatch ? { localId: newLocal.id, dispatch } : null
    await clearReconciledOwner(recipeOwnership)

    return {
      success: true,
      localId: newLocal.id,
      recipeOwnership,
      serverId,
      syncStatus: "synced"
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId: existingLocalId || "",
      recipeOwnership,
      serverId,
      error: error instanceof Error ? error.message : "Pull failed",
      syncStatus: "local"
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
  const recipeOwnership = null
  try {
    const local = await db.prompts.get(localId)
    if (!local) {
      return {
        success: false,
        localId,
        recipeOwnership,
        serverId,
        error: "Local prompt not found",
        syncStatus: "local"
      }
    }
    const identity = getLocalPromptComparablePayload(local)

    const response = await getServerPrompt(serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      return {
        success: false,
        localId,
        recipeOwnership,
        serverId,
        error: "Server prompt not found",
        syncStatus: "local"
      }
    }

    getServerPromptComparablePayload(serverPrompt)

    // Link by updating local with server reference
    if (
      await markLocalPending(localId, identity.promptSchemaVersion === 2, {
        serverId,
        studioProjectId: serverPrompt.project_id,
        studioPromptId: serverId,
        serverUpdatedAt: serverPrompt.updated_at,
        lastSyncedAt: Date.now()
      })
    )
      return durableRecipeBlock(localId)

    return {
      success: true,
      localId,
      recipeOwnership,
      serverId,
      syncStatus: "pending"
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId,
      recipeOwnership,
      serverId,
      error: error instanceof Error ? error.message : "Link failed",
      syncStatus: "local"
    }
  }
}

/**
 * Unlink a local prompt from server (keep local copy).
 */
export async function unlinkPrompt(localId: string): Promise<SyncResult> {
  const recipeOwnership = null
  let local: LocalPrompt | undefined
  try {
    local = await db.prompts.get(localId)
    if (!local) {
      return {
        success: false,
        localId,
        recipeOwnership,
        error: "Local prompt not found",
        syncStatus: "local"
      }
    }

    await db.prompts.update(localId, {
      serverId: null,
      studioProjectId: null,
      studioPromptId: null,
      serverUpdatedAt: null,
      syncStatus: "local",
      sourceSystem: "workspace",
      lastSyncedAt: null
    })

    return {
      success: true,
      localId,
      recipeOwnership,
      syncStatus: "local"
    }
  } catch (error: unknown) {
    return {
      success: false,
      localId,
      recipeOwnership,
      error: error instanceof Error ? error.message : "Unlink failed",
      syncStatus: local?.syncStatus || "local"
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
    return { status: "local", hasConflict: false }
  }

  if (!local.serverId) {
    return { status: "local", hasConflict: false }
  }

  // Check for conflict by comparing timestamps and content fingerprints.
  try {
    getLocalPromptComparablePayload(local)
    const response = await getServerPrompt(local.serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)

    if (!serverPrompt) {
      // Server prompt deleted
      return {
        status: "conflict",
        serverId: local.serverId,
        lastSyncedAt: local.lastSyncedAt || undefined,
        hasConflict: true
      }
    }

    const serverUpdatedAt = serverPrompt.updated_at
    const serverVersionChanged = local.serverUpdatedAt !== serverUpdatedAt
    const localHasUnsyncedChanges =
      (local.updatedAt || 0) > (local.lastSyncedAt || 0)
    const contentChanged = hasPromptContentConflict(local, serverPrompt)
    const hasConflict =
      serverVersionChanged && localHasUnsyncedChanges && contentChanged

    return {
      status: hasConflict ? "conflict" : local.syncStatus || "synced",
      serverId: local.serverId,
      lastSyncedAt: local.lastSyncedAt || undefined,
      hasConflict
    }
  } catch {
    return {
      status: local.syncStatus || "local",
      serverId: local.serverId,
      lastSyncedAt: local.lastSyncedAt || undefined,
      hasConflict: false
    }
  }
}

/**
 * Get detailed conflict information.
 */
export async function getConflictInfo(
  localId: string
): Promise<ConflictInfo | null> {
  const local = await db.prompts.get(localId)
  if (!local || !local.serverId) return null

  try {
    getLocalPromptComparablePayload(local)
    const response = await getServerPrompt(local.serverId)
    const serverPrompt = unwrapResponseData<ServerPrompt>(response)
    if (!serverPrompt) return null
    getServerPromptComparablePayload(serverPrompt)

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
  resolution: ConflictResolution,
  input?: RecipePersistenceInput
): Promise<SyncResult> {
  const recipeOwnership = null
  const local = await db.prompts.get(localId)
  if (!local || !local.serverId) {
    return {
      success: false,
      localId,
      recipeOwnership,
      error: "No conflict to resolve",
      syncStatus: "local"
    }
  }

  switch (resolution) {
    case "keep_local":
      // Push local to server (overwrite server)
      return await pushToStudio(localId, local.studioProjectId!, input)

    case "keep_server":
      // Pull server to local (overwrite local)
      return await pullFromStudio(local.serverId, localId)

    case "keep_both":
      // Validate/create first; preserve the existing link until one final update.
      if (isValidProjectId(local.studioProjectId)) {
        return await persistServerPrompt(
          localId,
          local,
          local.studioProjectId,
          input,
          true
        )
      }
      return {
        success: false,
        localId,
        recipeOwnership,
        error:
          "No valid Prompt Studio project available for keep-both resolution",
        syncStatus: local.syncStatus || "conflict"
      }

    default:
      return {
        success: false,
        localId,
        recipeOwnership,
        error: "Invalid resolution",
        syncStatus: "conflict"
      }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Get all prompts with their sync status.
 */
export async function getAllPromptsWithSyncStatus(): Promise<
  Array<{
    prompt: LocalPrompt
    syncStatus: PromptSyncStatus
    isSynced: boolean
  }>
> {
  const dbInstance = new PageAssistDatabase()
  const prompts = await dbInstance.getAllPrompts()

  return prompts.map((prompt) => ({
    prompt,
    syncStatus: prompt.syncStatus || "local",
    isSynced: prompt.syncStatus === "synced"
  }))
}

/**
 * Get all prompts linked to a specific project.
 */
export async function getPromptsByProject(
  projectId: number
): Promise<LocalPrompt[]> {
  return await db.prompts
    .where("studioProjectId")
    .equals(projectId)
    .filter((p) => !p.deletedAt)
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
