import { normalizeNoteKeyword } from "@/services/note-keywords";
import { createSlug } from "@/store/workspace";
import { readKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance";
import { bgRequest } from "@/services/background-proxy";
import { loadServicePromptSnapshot } from "@/services/service-prompts";
import { requestScopeFields } from "@/services/tldw/domains/service-prompts";
import type {
  WorkspaceArtifactApiResponse,
  WorkspaceContextResponse,
  WorkspaceListApiResponse,
  WorkspaceNoteApiResponse,
} from "@/services/tldw/domains/workspace-api";
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error";
import { hydrateWorkspaceFromServer, type LocalWorkspaceState } from "@/store/workspace-api";
import { RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX } from "@/store/workspace-migration";
import type { WorkspaceSourceStatus } from "@/types/workspace";

const INFORMATIONAL_CONTEXT_ERRORS = new Set([
  "jobs_unavailable",
  "media_db_unavailable",
  "membership_summary_unavailable",
]);

type MigrationReceipt = {
  id: string;
  deletedAt: number;
  key: string;
  scopeKey?: string;
  value: Record<string, unknown>;
};

const readMigrationReceipts = (storage: Storage): MigrationReceipt[] => {
  const receipts: MigrationReceipt[] = [];
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    if (!key?.startsWith(`${RESEARCH_WORKSPACE_MIGRATION_TOMBSTONE_PREFIX}:`))
      continue;
    try {
      const receipt = JSON.parse(storage.getItem(key) || "null");
      if (
        receipt?.contentRetained !== false ||
        receipt.restoreInvalidated === true ||
        typeof receipt.serverWorkspaceId !== "string" ||
        !receipt.serverWorkspaceId.trim() ||
        typeof receipt.migrationId !== "string" ||
        !receipt.migrationId.trim()
      )
        continue;
      const deletedAt = Date.parse(receipt.deletedAt);
      if (Number.isFinite(deletedAt))
        receipts.push({
          id: receipt.serverWorkspaceId,
          deletedAt,
          key,
          value: receipt,
          scopeKey:
            typeof receipt.serverScopeKey === "string"
              ? receipt.serverScopeKey
              : undefined,
        });
    } catch {
      // A malformed local receipt grants no authority and cannot identify a server workspace.
    }
  }
  return receipts.sort((a, b) => b.deletedAt - a.deletedAt);
};

/** Untrusted location hint only; callers must resolve it within their captured account. */
export const readMigratedResearchWorkspaceId = (
  storage: Storage,
): string | null => readMigrationReceipts(storage)[0]?.id ?? null;

/** Restore one migrated workspace atomically while its captured account lease remains live. */
export const restoreMigratedResearchWorkspace = async (options: {
  signal: AbortSignal;
  apply: (workspace: LocalWorkspaceState, scopeKey: string) => boolean;
  storage?: Storage;
}): Promise<boolean> => {
  const storage = options.storage ?? window.localStorage;
  const receipts = readMigrationReceipts(storage);
  if (!receipts.length) return false;
  const scope = await loadServicePromptSnapshot([], { signal: options.signal });
  const assertCurrent = () => {
    if (scope.scopeInvalidatedSignal.aborted)
      throw createServicePromptScopeChangedError();
    if (options.signal.aborted || scope.scopeSignal.aborted)
      throw new DOMException("Workspace restoration cancelled", "AbortError");
  };
  const request = {
    method: "GET" as const,
    abortSignal: scope.scopeSignal,
    ...requestScopeFields(scope.requestScope),
  };
  try {
    assertCurrent();
    let receipt = receipts.find((value) => value.scopeKey === scope.scopeKey);
    if (!receipt) {
      const unbound = receipts.filter((value) => !value.scopeKey);
      if (!unbound.length) return false;
      const listed = await bgRequest<WorkspaceListApiResponse>({
        ...request,
        path: "/api/v1/workspaces/",
      });
      assertCurrent();
      if (!Array.isArray(listed.items))
        throw new Error("The current account's workspaces could not be listed");
      receipt = unbound.find((value) =>
        listed.items.some(
          (workspace) =>
            workspace.id === value.id &&
            !workspace.deleted &&
            !workspace.archived,
        ),
      );
      if (!receipt) return false;
    }
    const workspaceId = receipt.id;
    const invalidateReceipt = () => {
      // Retain the migration tombstone so deleted legacy content cannot reappear.
      storage.setItem(
        receipt.key,
        JSON.stringify({ ...receipt.value, restoreInvalidated: true }),
      );
      return false;
    };
    const path =
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}` as const;
    let context: WorkspaceContextResponse;
    try {
      context = await bgRequest<WorkspaceContextResponse>({
        ...request,
        path: `${path}/context`,
      });
    } catch (error) {
      assertCurrent();
      if ((error as { status?: number } | null)?.status === 404)
        return invalidateReceipt();
      throw error;
    }
    assertCurrent();
    if (
      context.workspace_id !== workspaceId ||
      context.workspace.id !== workspaceId
    ) {
      throw new Error("Workspace restoration returned a different workspace");
    }
    if (context.workspace.deleted || context.workspace.archived)
      return invalidateReceipt();
    if (
      context.partial_errors?.some(
        (error) => !INFORMATIONAL_CONTEXT_ERRORS.has(error.code),
      ) ||
      !Array.isArray(context.sources?.items) ||
      context.sources.items.some(
        (source) => source.workspace_id !== workspaceId,
      )
    ) {
      throw new Error("The server workspace could not be restored completely");
    }
    const [artifacts, notes] = await Promise.all([
      bgRequest<WorkspaceArtifactApiResponse[]>({
        ...request,
        path: `${path}/artifacts`,
      }),
      bgRequest<WorkspaceNoteApiResponse[]>({
        ...request,
        path: `${path}/notes`,
      }),
    ]);
    assertCurrent();
    if (
      artifacts.some((artifact) => artifact.workspace_id !== workspaceId) ||
      notes.some((note) => note.workspace_id !== workspaceId)
    ) {
      throw new Error("Workspace restoration returned a different workspace");
    }
    const local = await hydrateWorkspaceFromServer(workspaceId, {
      requireComplete: true,
      fetch: async () => ({
        ...context.workspace,
        metadata: context.workspace,
        sources: context.sources.items,
        artifacts,
        notes,
      }),
    });
    assertCurrent();
    local.sources = local.sources.map((source, index) => {
      const authoritative = context.sources.items[index];
      const status: WorkspaceSourceStatus =
        authoritative.state === "queryable" ||
        authoritative.state === "partially_queryable"
          ? "ready"
          : ["failed", "missing_media", "blocked_by_permissions"].includes(
                authoritative.state,
              )
            ? "error"
            : "processing";
      return { ...source, status, readiness: authoritative.readiness };
    });
    // The canonical installer retains workspace notes separately from UUID Notes.
    const workspaceTag = `workspace:${createSlug(local.name) || workspaceId.slice(0, 8)}`;
    // Canonical Quick Notes survive the legacy snapshot tombstone. The marker is
    // descriptive evidence only; server membership and selection stay authoritative.
    const found = await bgRequest<{
      notes?: Array<{
        id: string;
        title: string;
        content: string;
        keywords?: unknown[];
        version: number;
      }>;
    }>({
      ...request,
      path: `/api/v1/notes/search/?tokens=${encodeURIComponent(`workspace:${workspaceId}`)}&limit=100&include_keywords=true`,
    });
    assertCurrent();
    const canonical = (found.notes || []).find(
      (note) =>
        typeof note.id === "string" &&
        typeof note.content === "string" &&
        readKnowledgeNoteProvenance(note.content)?.research?.workspace_id ===
          workspaceId,
    );
    if (canonical) {
      const provenance = readKnowledgeNoteProvenance(canonical.content)!;
      local.currentNote = {
        id: canonical.id,
        title: canonical.title,
        content: canonical.content,
        keywords: (canonical.keywords || [])
          .map(normalizeNoteKeyword)
          .filter(
            (keyword): keyword is string =>
              keyword !== null &&
              keyword !== workspaceTag &&
              keyword !== `workspace:${workspaceId}`,
          ),
        version: canonical.version,
        isDirty: false,
        serverWorkspaceId: workspaceId,
        serverScopeKey: scope.scopeKey,
      };
      local.sources = local.sources.map((source) => {
        const retained = provenance.research!.sources.find(
          (item) => item.mediaId === source.mediaId,
        );
        return retained
          ? { ...source, knowledgeQaEvidence: retained.evidence }
          : source;
      });
    }
    assertCurrent();
    if (!options.apply(local, scope.scopeKey))
      throw new Error("Workspace restoration could not be installed without replacing retained content");
    storage.setItem(
      receipt.key,
      JSON.stringify({ ...receipt.value, serverScopeKey: scope.scopeKey }),
    );
    return true;
  } finally {
    scope.release();
  }
};
