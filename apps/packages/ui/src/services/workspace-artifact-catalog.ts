import type { GeneratedArtifact } from "@/types/workspace";
import { useWorkspaceStore } from "@/store/workspace";
import { bindWorkspaceArtifactOrigin } from "@/store/workspace-artifact-codec";
import { mapServerArtifactToLocal } from "@/store/workspace-api";
import { bgRequest, type BgRequestInit } from "@/services/background-proxy";
import { loadServicePromptSnapshot } from "@/services/service-prompts";
import { isServicePromptRequestPath } from "@/services/tldw/service-prompt-scope-error";
import { requestScopeFields } from "@/services/tldw/domains/service-prompts";
import type { WorkspaceArtifactApiResponse, WorkspaceArtifactRevision } from "@/services/tldw/domains/workspace-api";

type RequestScope = Pick<BgRequestInit, "servicePromptConfig" | "abortSignal" | "timeoutMs" | "headers">;
const catalogOwners = new Map<string, string>();
const pathFor = (id: string) => `/api/v1/workspaces/${encodeURIComponent(id)}` as const;

const detailPath = (workspaceId: string, id: string) => `${pathFor(workspaceId)}/artifacts/${encodeURIComponent(id)}` as const;

async function readDetail(workspaceId: string, id: string, request: RequestScope) {
  const row = await bgRequest<WorkspaceArtifactApiResponse>({
    ...request, method: "GET", path: detailPath(workspaceId, id), timeoutMs: request.timeoutMs ?? 120_000,
  });
  if (row.id !== id || row.workspace_id !== workspaceId) throw new Error("The server returned a different saved output.");
  return row;
}

/** Older servers expose only the full list. A successful catalog read proves detail support. */
export async function findSavedWorkspaceArtifact(workspaceId: string, id: string, scopeKey: string, request: RequestScope) {
  if (catalogOwners.get(workspaceId) === scopeKey && isServicePromptRequestPath(detailPath(workspaceId, id), "GET")) {
    try {
      return await readDetail(workspaceId, id, request);
    } catch (error) {
      if ((error as {status?: number})?.status === 404) return undefined;
      throw error;
    }
  }
  const rows = await bgRequest<WorkspaceArtifactApiResponse[]>({
    ...request, method: "GET", path: `${pathFor(workspaceId)}/artifacts`,
  });
  if (!Array.isArray(rows) || rows.some(row => row.workspace_id !== workspaceId))
    throw new Error("Saved artifacts belong to another workspace.");
  const matches = rows.filter(row => row.id === id);
  if (matches.length > 1) throw new Error("Saved artifact identity is ambiguous.");
  return matches[0];
}

export async function readWorkspaceArtifactCatalog(workspaceId: string, scopeKey: string, request: RequestScope) {
  const readAll = async () => {
    const rows = await bgRequest<WorkspaceArtifactApiResponse[]>({
      ...request, method: "GET", path: `${pathFor(workspaceId)}/artifacts`, timeoutMs: 120_000,
    });
    if (rows.some(row => row.workspace_id !== workspaceId)) throw new Error("Workspace restoration returned a different workspace");
    return rows.map(mapServerArtifactToLocal);
  };
  let inventory: WorkspaceArtifactRevision[];
  try {
    inventory = await bgRequest<WorkspaceArtifactRevision[]>({
      ...request, method: "GET", path: `${pathFor(workspaceId)}/artifact-revisions`, expectedStatuses: [404],
    });
  } catch (error) {
    if ((error as {status?: number})?.status === 404) {
      catalogOwners.delete(workspaceId);
      return readAll();
    }
    throw error;
  }
  if (!Array.isArray(inventory) || inventory.some(row => !row.id || !row.display_id || row.workspace_id !== workspaceId ||
      !Number.isSafeInteger(row.version) || row.version < 1) ||
      new Set(inventory.map(row => row.id)).size !== inventory.length)
    throw new Error("The saved output list could not be verified.");
  catalogOwners.set(workspaceId, scopeKey);
  // Legacy IDs containing a slash or a dot segment are readable only through the full list.
  if ((inventory.length && inventory.every(row => !row.defer_detail)) ||
      inventory.some(row => !isServicePromptRequestPath(detailPath(workspaceId, row.id), "GET"))) return readAll();
  const artifacts: GeneratedArtifact[] = [];
  for (const row of inventory) {
    if (row.defer_detail && (row.artifact_type === "summary" || row.artifact_type === "compare_sources" || row.artifact_type === "report") &&
        ["complete", "completed", "failed"].includes(row.status)) {
      artifacts.push({
        id: row.display_id, workspaceArtifactId: row.id, workspaceArtifactVersion: row.version,
        detailDeferred: { workspaceId, scopeKey, imported: row.imported, model: row.model ?? undefined, provider: row.provider ?? undefined },
        title: row.title, type: row.artifact_type,
        status: row.status === "failed" ? "failed" : "completed", createdAt: new Date(row.created_at),
        reviewStatus: row.review_state, version: row.display_version,
      });
    } else {
      artifacts.push(mapServerArtifactToLocal(await readDetail(workspaceId, row.id, request)));
    }
  }
  const visible = artifacts.filter(a => !(a.reviewStatus === "archived" && a.removedFromStudio));
  if (new Set(visible.map(a => a.id)).size !== visible.length) throw new Error("Saved outputs have conflicting display identities.");
  return artifacts;
}

/** Hydrate from the current account before viewing, copying or exporting a deferred output. */
export async function materializeWorkspaceOutputs(workspaceId: string, target?: GeneratedArtifact, signal?: AbortSignal) {
  const state = useWorkspaceStore.getState();
  if (state.workspaceId !== workspaceId) throw new Error("The active workspace changed.");
  const pending = target ? [target] : state.generatedArtifacts.filter(a => a.detailDeferred);
  if (!pending.length) return;
  const scope = await loadServicePromptSnapshot([], { signal });
  try {
    const checkCurrent = () => {
      const current = useWorkspaceStore.getState();
      if (scope.scopeSignal.aborted || scope.scopeInvalidatedSignal.aborted || current.workspaceId !== workspaceId ||
          pending.some(a => a.detailDeferred?.scopeKey !== scope.scopeKey || a.detailDeferred.workspaceId !== workspaceId ||
            !current.generatedArtifacts.includes(a)))
        throw new Error("The saved output changed. Reopen the workspace and try again.");
    };
    checkCurrent();
    const request = { ...requestScopeFields(scope.requestScope), abortSignal: AbortSignal.any([scope.scopeSignal, scope.scopeInvalidatedSignal]) };
    const details = new Map<GeneratedArtifact, GeneratedArtifact>();
    for (const card of pending) {
      checkCurrent();
      if (!card.workspaceArtifactId) throw new Error("Missing saved output identity.");
      const row = await readDetail(workspaceId, card.workspaceArtifactId, request);
      const detail = mapServerArtifactToLocal(row);
      if (row.version !== card.workspaceArtifactVersion || detail.id !== card.id || detail.version !== card.version ||
          detail.createdAt.getTime() !== card.createdAt.getTime())
        throw new Error("This saved output has changed. Reopen the workspace to load the latest version.");
      details.set(card, bindWorkspaceArtifactOrigin(detail, scope.scopeKey, workspaceId));
    }
    checkCurrent();
    useWorkspaceStore.setState(s => ({generatedArtifacts: s.generatedArtifacts.map(a => details.get(a) ?? a)}));
    return target ? details.get(target) : undefined;
  } finally {
    scope.release();
  }
}
