import { z } from "zod"
import {
  loadOwnedWorkspace,
  validateOwnedWorkspaceMetadata,
  validateOwnedWorkspaceRecord,
  validateOwnedWorkspaceNotes,
  type OwnedWorkspaceBundle,
  type OwnedWorkspaceReader
} from "@/store/workspace-api"
import type { OwnedWorkspaceScope } from "@/store/owned-workspace-state"
import { ownedWorkspaceDraftKey } from "@/store/owned-workspace-state"
import type {
  WorkspaceApiResponse,
  WorkspacePatchRequest,
  WorkspaceContextResponse,
  WorkspaceNoteApiResponse,
  WorkspaceNoteCreateRequest,
  WorkspaceNoteUpdateRequest
} from "./tldw/domains/workspace-api"
import {
  normalizeWorkspaceApiResponse,
  serializeWorkspacePatchRequest
} from "./tldw/domains/workspace-api"
import {
  normalizeEffectiveWorkspaceAssistantDefault,
  normalizeWorkspaceAssistantDefaults
} from "@/types/workspace-assistant-defaults"
import { createSafeStorage } from "@/utils/safe-storage"
import { isWorkspaceTargetId } from "@/utils/workspace-target"
import { TldwApiError } from "./tldw/api-error"
import { isCookieSessionBrowserTransport } from "./tldw/browser-networking"
import { resolveDirectBrowserConfig } from "./tldw/direct-browser-config"
import { isHostedTldwDeployment } from "./tldw/deployment-mode"
import type { PathOrUrl } from "./tldw/openapi-guard"
import {
  resolveBrowserRequestTransport,
  tldwRequest
} from "./tldw/request-core"
import { getRuntimeSingleUserApiKeyOverride } from "./tldw/runtime-auth-override"

export class OwnedWorkspaceOpeningError extends Error {
  constructor(
    public readonly reason: "denied" | "connection" | "invalid-response"
  ) {
    super(`Workspace opening ${reason}`)
    this.name = "OwnedWorkspaceOpeningError"
  }
}

export type OwnedWorkspaceReadContext = {
  readonly scope: OwnedWorkspaceScope
  load(): Promise<OwnedWorkspaceBundle>
}

const PRINCIPAL_PATH = "/api/v1/users/me/profile?sections=identity"

/** Capture one connection without retaining mutable credentials or retrying writes. */
async function captureOwnedWorkspaceConnection(
  workspaceId: string | null,
  signal: AbortSignal
) {
  signal.throwIfAborted()
  if (workspaceId !== null && !isWorkspaceTargetId(workspaceId))
    throw new OwnedWorkspaceOpeningError("invalid-response")
  const resolved = await resolveDirectBrowserConfig(createSafeStorage())
  signal.throwIfAborted()
  const config = { ...resolved }
  const hosted = isHostedTldwDeployment()
  if (!hosted) {
    try {
      const url = new URL(config.serverUrl || "")
      if (
        !/^https?:$/.test(url.protocol) ||
        url.username ||
        url.password ||
        url.search ||
        url.hash
      )
        throw new Error("Invalid server URL")
    } catch {
      throw new OwnedWorkspaceOpeningError("connection")
    }
  }
  if (
    config.orgId != null &&
    (!Number.isSafeInteger(config.orgId) || config.orgId < 1)
  )
    throw new OwnedWorkspaceOpeningError("connection")

  const transport = resolveBrowserRequestTransport({
    config,
    path: PRINCIPAL_PATH
  })
  const cookieSession = isCookieSessionBrowserTransport({
    authMode: config.authMode,
    authSource: config.authSource,
    transportMode: transport.mode,
    transportKind: transport.kind,
    pageOrigin: window.location.origin
  })
  const runtimeKey = getRuntimeSingleUserApiKeyOverride()
  if (runtimeKey && !hosted && !cookieSession) {
    config.authMode = "single-user"
    config.apiKey = runtimeKey
  }
  Object.freeze(config)
  const serverBase =
    transport.kind === "same-origin"
      ? window.location.origin
      : new URL(config.serverUrl!).toString().replace(/\/+$/, "")
  const fetchAtOpening = fetch
  const request = async <T>(
    path: string,
    method: "GET" | "POST" | "PUT" | "PATCH" = "GET",
    body?:
      | WorkspaceNoteCreateRequest
      | WorkspaceNoteUpdateRequest
      | WorkspacePatchRequest,
    expectedUserId?: string
  ): Promise<T> => {
    signal.throwIfAborted()
    const response = await tldwRequest(
      {
        // The browser-origin resolver drops deployment subpaths. An absolute
        // configured-server URL preserves them through request-core's allowlist.
        path: (transport.kind === "absolute"
          ? `${serverBase}${path}`
          : path) as PathOrUrl,
        method,
        body,
        headers: expectedUserId
          ? { "X-TLDW-Expected-User-ID": expectedUserId }
          : undefined,
        abortSignal: signal,
        timeoutMs: 30_000
      },
      {
        getConfig: async () => config,
        useRuntimeAuthOverride: false,
        fetchFn: (url, init) =>
          fetchAtOpening(url, {
            ...init,
            credentials: hosted || cookieSession ? "same-origin" : "omit",
            cache: "no-store",
            redirect: "error"
          })
      }
    )
    signal.throwIfAborted()
    if (!response.ok)
      throw new TldwApiError("Workspace request failed", response.status, null)
    return response.data as T
  }
  const read = <T>(path: string) => request<T>(path)
  const readPrincipal = async () => {
    const profile = await read<{
      user?: { id?: unknown } | null
      section_errors?: { identity?: unknown }
    } | null>(PRINCIPAL_PATH)
    const user = profile?.user
    if (
      profile?.section_errors?.identity ||
      typeof user?.id !== "number" ||
      !Number.isSafeInteger(user.id) ||
      user.id < 1
    )
      throw new OwnedWorkspaceOpeningError("denied")
    return String(user.id)
  }
  const principalId = await readPrincipal()
  const scope = Object.freeze({
    serverBase,
    principalId,
    organizationId: cookieSession
      ? null
      : config.orgId == null
        ? null
        : String(config.orgId)
  })
  return { scope, read, request, readPrincipal }
}

/** The opening interface exposes reads only; opening never mutates a workspace. */
export async function createOwnedWorkspaceReadContext(
  workspaceId: string,
  signal: AbortSignal
): Promise<OwnedWorkspaceReadContext> {
  const { scope, request, readPrincipal } = await captureOwnedWorkspaceConnection(
    workspaceId,
    signal
  )
  const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`
  const read = <T>(resourcePath: string) =>
    request<T>(resourcePath, "GET", undefined, scope.principalId)
  const reader: OwnedWorkspaceReader = {
    getWorkspace: () => read(path),
    getWorkspaceSources: () => read(`${path}/sources`),
    getWorkspaceArtifacts: () => read(`${path}/artifacts`),
    getWorkspaceNotes: () => read(`${path}/notes`)
  }
  return {
    scope,
    async load() {
      const bundle = await loadOwnedWorkspace(workspaceId, reader, signal)
      // Each bundle read asserts the captured principal because cookies can change
      // between requests. This final recheck is an additional activation guard.
      if ((await readPrincipal()) !== scope.principalId)
        throw new OwnedWorkspaceOpeningError("denied")
      const workspace = bundle.workspace
      return {
        ...bundle,
        workspace: {
          ...workspace,
          assistantDefaults: normalizeWorkspaceAssistantDefaults(
            workspace.assistant_defaults ?? workspace.assistantDefaults ?? null
          ),
          effectiveAssistantDefault:
            normalizeEffectiveWorkspaceAssistantDefault(
              workspace.effective_assistant_default ??
                workspace.effectiveAssistantDefault ??
                null
            )
        }
      }
    }
  }
}

/** Lifecycle writes cannot carry unrelated metadata or create a missing row. */
export async function createOwnedWorkspaceLifecycleContext(
  workspaceId: string,
  expectedScope: OwnedWorkspaceScope,
  signal: AbortSignal
) {
  const expectedKey = ownedWorkspaceDraftKey(expectedScope, workspaceId)
  const { scope, request } = await captureOwnedWorkspaceConnection(
    workspaceId,
    signal
  )
  if (ownedWorkspaceDraftKey(scope, workspaceId) !== expectedKey)
    throw new TldwApiError("Workspace account changed", 412, null)
  const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`
  const validate = (response: unknown) => {
    validateOwnedWorkspaceRecord(response, workspaceId)
    return normalizeWorkspaceApiResponse(response)
  }
  return {
    scope,
    async get() {
      return validate(await request(path, "GET", undefined, scope.principalId))
    },
    async setArchived(archived: boolean, version: number) {
      if (
        typeof archived !== "boolean" ||
        !Number.isSafeInteger(version) ||
        version < 1
      )
        throw new OwnedWorkspaceOpeningError("invalid-response")
      const workspace = validate(
        await request(path, "PATCH", { archived, version }, scope.principalId)
      )
      if (workspace.archived !== archived || workspace.version <= version)
        throw new OwnedWorkspaceOpeningError("invalid-response")
      return workspace
    }
  }
}

export type OwnedWorkspaceDirectoryDetails = Pick<
  WorkspaceContextResponse,
  "attention_state" | "project_root" | "active_operations"
> & {
  sources: { summary: { total: number; selected: number } }
}

const directoryDetailsSchema = z.object({
  attention_state: z.enum([
    "ready",
    "setup_pending",
    "working",
    "needs_attention",
    "blocked",
    "archived"
  ]),
  project_root: z.object({
    state: z.enum([
      "not_configured",
      "provisioning",
      "attached",
      "unavailable",
      "missing",
      "detached",
      "failed",
      "cleanup_pending",
      "archived"
    ]),
    root_id: z.string().nullable(),
    backend: z.enum(["host_local", "sandbox_volume"]).nullable(),
    display_name: z.string().nullable(),
    path_hint: z.string().nullable(),
    git_state: z.string().nullable(),
    file_inventory_state: z.string().nullable(),
    file_inventory: z.object({
      state: z.string().nullable(),
      indexed_file_count: z.number().int().nonnegative().nullable(),
      total_file_count: z.number().int().nonnegative().nullable(),
      updated_at: z.string().nullable(),
      available: z.boolean()
    }),
    indexing_state: z.string().nullable(),
    sandbox_mount_state: z.string().nullable(),
    mcp_trust_state: z.string().nullable()
  }),
  sources: z.object({
    summary: z
      .object({
        total: z.number().int().nonnegative(),
        selected: z.number().int().nonnegative()
      })
      .refine(({ total, selected }) => selected <= total)
  }),
  // Older context responses omit operations; retain that documented compatibility.
  active_operations: z
    .array(
      z.object({
        operation_id: z.string().min(1),
        workspace_id: z.string().min(1),
        command: z.string(),
        status: z.enum([
          "queued",
          "running",
          "succeeded",
          "failed",
          "conflicted",
          "expired"
        ]),
        started_at: z.string(),
        updated_at: z.string(),
        retryable: z.boolean(),
        diagnostics: z.record(z.string(), z.unknown()),
        poll_href: z.string()
      })
    )
    .default([])
})

/** Directory rows and lifecycle commands share one verified account identity. */
export async function createOwnedWorkspaceDirectoryContext(
  signal: AbortSignal
) {
  const { scope, request } = await captureOwnedWorkspaceConnection(null, signal)
  return {
    scope,
    async list(): Promise<{ items: WorkspaceApiResponse[] }> {
      const response = await request<{ items?: unknown } | null>(
        "/api/v1/workspaces/",
        "GET",
        undefined,
        scope.principalId
      )
      if (!Array.isArray(response?.items))
        throw new OwnedWorkspaceOpeningError("invalid-response")
      const seen = new Set<string>()
      const items = response.items.map((row: unknown) => {
        const id = (row as { id?: unknown } | null)?.id
        if (!isWorkspaceTargetId(id) || seen.has(id))
          throw new OwnedWorkspaceOpeningError("invalid-response")
        validateOwnedWorkspaceRecord(row, id)
        seen.add(id)
        return normalizeWorkspaceApiResponse(row)
      })
      return { items }
    },
    async getContext(id: string): Promise<OwnedWorkspaceDirectoryDetails> {
      if (!isWorkspaceTargetId(id))
        throw new OwnedWorkspaceOpeningError("invalid-response")
      const response = await request<WorkspaceContextResponse | null>(
        `/api/v1/workspaces/${encodeURIComponent(id)}/context`,
        "GET",
        undefined,
        scope.principalId
      )
      if (!response || response.workspace_id !== id)
        throw new OwnedWorkspaceOpeningError("invalid-response")
      validateOwnedWorkspaceRecord(response.workspace, id)
      // With strictNullChecks off, Zod infers nullable required keys as optional.
      const details = directoryDetailsSchema.parse(
        response
      ) as OwnedWorkspaceDirectoryDetails
      if (
        details.active_operations.some(
          (operation) => operation.workspace_id !== id
        )
      )
        throw new OwnedWorkspaceOpeningError("invalid-response")
      return details
    }
  }
}

export type OwnedWorkspaceMetadataPatch = Omit<
  WorkspacePatchRequest,
  "archived" | "workspace_profile"
>

export type OwnedWorkspaceMetadataContext = {
  get(): Promise<WorkspaceApiResponse>
  patch(body: OwnedWorkspaceMetadataPatch): Promise<WorkspaceApiResponse>
  listPersonas(): Promise<OwnedWorkspacePersonaOption[]>
}

const personaOptionsSchema = z
  .array(
    z.object({
      id: z
        .string()
        .min(1)
        .refine((id) => id === id.trim()),
      name: z.string()
    })
  )
  .refine((rows) => new Set(rows.map((row) => row.id)).size === rows.length)
export type OwnedWorkspacePersonaOption = z.infer<
  typeof personaOptionsSchema
>[number]

/** Metadata editing never creates an absent workspace or retries a conflict. */
export async function createOwnedWorkspaceMetadataContext(
  workspaceId: string,
  expectedScope: OwnedWorkspaceScope,
  signal: AbortSignal
): Promise<OwnedWorkspaceMetadataContext> {
  const expectedKey = ownedWorkspaceDraftKey(expectedScope, workspaceId)
  const { scope, request } = await captureOwnedWorkspaceConnection(
    workspaceId,
    signal
  )
  if (ownedWorkspaceDraftKey(scope, workspaceId) !== expectedKey)
    throw new TldwApiError("Workspace account changed", 412, null)
  const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`
  const validate = (response: unknown) => {
    validateOwnedWorkspaceMetadata(response, workspaceId)
    return normalizeWorkspaceApiResponse(response)
  }
  return {
    async listPersonas() {
      return personaOptionsSchema.parse(
        await request(
          "/api/v1/persona/catalog?ensure_default=false",
          "GET",
          undefined,
          scope.principalId
        )
      )
    },
    async get() {
      return validate(await request(path, "GET", undefined, scope.principalId))
    },
    async patch(body) {
      if ("archived" in body || "workspace_profile" in body)
        throw new OwnedWorkspaceOpeningError("invalid-response")
      const submitted = serializeWorkspacePatchRequest(structuredClone(body))
      if (
        !Number.isSafeInteger(submitted.version) ||
        submitted.version < 1 ||
        !Object.entries(submitted).some(
          ([key, value]) => key !== "version" && value !== undefined
        )
      )
        throw new OwnedWorkspaceOpeningError("invalid-response")
      const receipt = validate(
        await request(path, "PATCH", submitted, scope.principalId)
      )
      if (receipt.version <= submitted.version)
        throw new OwnedWorkspaceOpeningError("invalid-response")
      return receipt
    }
  }
}

export type OwnedWorkspaceNotesContext = {
  list(): Promise<WorkspaceNoteApiResponse[]>
  create(body: WorkspaceNoteCreateRequest): Promise<WorkspaceNoteApiResponse>
  update(
    noteId: number,
    body: WorkspaceNoteUpdateRequest
  ): Promise<WorkspaceNoteApiResponse>
}

/** Explicit note operations must match the already activated workspace identity. */
export async function createOwnedWorkspaceNotesContext(
  workspaceId: string,
  expectedScope: OwnedWorkspaceScope,
  signal: AbortSignal
): Promise<OwnedWorkspaceNotesContext> {
  const expectedKey = ownedWorkspaceDraftKey(expectedScope, workspaceId)
  const { scope, request } = await captureOwnedWorkspaceConnection(
    workspaceId,
    signal
  )
  if (ownedWorkspaceDraftKey(scope, workspaceId) !== expectedKey)
    throw new TldwApiError("Workspace account changed", 412, null)
  const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/notes`
  const validateReceipt = (receipt: unknown): WorkspaceNoteApiResponse => {
    const rows = [receipt]
    validateOwnedWorkspaceNotes(rows, workspaceId)
    return rows[0]
  }
  return {
    async list() {
      const rows = await request<unknown>(
        path,
        "GET",
        undefined,
        scope.principalId
      )
      validateOwnedWorkspaceNotes(rows, workspaceId)
      return rows
    },
    async create(body) {
      const submitted = {
        ...body,
        ...(body.keywords ? { keywords: [...body.keywords] } : {})
      }
      return validateReceipt(
        await request(path, "POST", submitted, scope.principalId)
      )
    },
    async update(noteId, body) {
      const submitted = { ...body }
      if (
        !Number.isSafeInteger(noteId) ||
        noteId < 1 ||
        !Number.isSafeInteger(submitted.version) ||
        submitted.version < 1
      )
        throw new OwnedWorkspaceOpeningError("invalid-response")
      const receipt = validateReceipt(
        await request(`${path}/${noteId}`, "PUT", submitted, scope.principalId)
      )
      if (receipt.id !== noteId || receipt.version <= submitted.version)
        throw new OwnedWorkspaceOpeningError("invalid-response")
      return receipt
    }
  }
}
