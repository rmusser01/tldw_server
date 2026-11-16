import { mapServerArtifactToLocal } from '@/store/workspace-api';
import { bgRequest, type BgRequestInit } from '@/services/background-proxy';
import type { ServicePromptSnapshot } from '@/services/service-prompts';
import { requestScopeFields } from '@/services/tldw/domains/service-prompts';
import type {
  WorkspaceArtifactApiResponse,
  WorkspaceArtifactCreateRequest,
} from '@/services/tldw/domains/workspace-api';
import { createServicePromptScopeChangedError } from '@/services/tldw/service-prompt-scope-error';
import { hasResearchWorkspaceMigrationTombstone } from '@/store/workspace';
import {
  workspaceArtifactDraftPayload,
  sameWorkspaceArtifactPayload,
  hasWorkspaceArtifactOrigin,
  hasWorkspaceArtifactOriginEvidence,
  readWorkspaceArtifactClientPayload,
  bindWorkspaceArtifactOrigin,
} from '@/store/workspace-artifact-codec';
import type { GeneratedArtifact } from '@/types/workspace';

export type WorkspaceArtifactSaveScope = {
  workspaceId: string;
  scope: ServicePromptSnapshot;
  isCurrent: () => boolean;
  deadline?: number;
  reviewChangeOnly?: boolean;
  updateOnly?: "content" | "client" | "review";
};

const canonicalWorkspaces = new Map<string, string>();

export const rememberCanonicalArtifactWorkspace = (workspaceId: string, scopeKey: string): void => {
  // Cached workspace switches must retain enrollment for this page lifetime.
  canonicalWorkspaces.set(workspaceId, scopeKey);
};
export const hasCanonicalArtifactWorkspace = (workspaceId: string, scopeKey?: string): boolean =>
  canonicalWorkspaces.has(workspaceId) &&
  (scopeKey === undefined || canonicalWorkspaces.get(workspaceId) === scopeKey);

const pending = new Map<string, Promise<unknown>>();

/** Save a review draft in the already-migrated canonical workspace, never a source/publication. */
export const persistWorkspaceArtifact = async (
  artifact: GeneratedArtifact,
  options: WorkspaceArtifactSaveScope
): Promise<GeneratedArtifact> => {
  const { workspaceId, scope, isCurrent } = options;
  const deadline = options.deadline ?? performance.now() + 15000;
  const signal = AbortSignal.any([
    scope.scopeSignal,
    scope.scopeInvalidatedSignal,
    AbortSignal.timeout(Math.max(0, Math.ceil(deadline - performance.now()))),
  ]);
  const assertCurrent = () => {
    if (scope.scopeSignal.aborted || scope.scopeInvalidatedSignal.aborted || !isCurrent())
      throw createServicePromptScopeChangedError();
    if (signal.aborted || performance.now() >= deadline)
      throw new DOMException('Saving the draft exceeded 15 seconds. Retry saving.', 'TimeoutError');
  };
  assertCurrent();
  if (options.reviewChangeOnly && options.updateOnly)
    throw new Error("Choose one artifact update mode.");
  if (!workspaceId.trim() || workspaceId !== workspaceId.trim() || workspaceId.length > 128)
    throw new Error('Invalid artifact workspace.');
  // Legacy/offline workspaces keep ADR-008's local cache; enrollment survives a missing tombstone.
  if (!artifact.workspaceArtifactId && !hasResearchWorkspaceMigrationTombstone(workspaceId) &&
    !hasCanonicalArtifactWorkspace(workspaceId))
    return artifact;
  if (hasCanonicalArtifactWorkspace(workspaceId) &&
    !hasCanonicalArtifactWorkspace(workspaceId, scope.scopeKey))
    throw createServicePromptScopeChangedError();
  if (!hasWorkspaceArtifactOrigin(artifact, scope.scopeKey, workspaceId))
    throw new Error('This draft has no verified origin in the current account and workspace.');
  const key = JSON.stringify([
    scope.scopeKey,
    workspaceId,
    artifact.workspaceArtifactId ?? artifact.id,
  ]);
  const prior = pending.get(key);
  const operation = (async () => {
    if (prior) await prior.catch(() => {});
    assertCurrent();
    let payload: Partial<WorkspaceArtifactCreateRequest> & { id: string } = options.reviewChangeOnly || options.updateOnly === "review"
      ? { id: artifact.workspaceArtifactId ?? artifact.id }
      : workspaceArtifactDraftPayload(artifact);
    const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/artifacts` as const;
    const request = () => {
      assertCurrent();
      return {
        ...requestScopeFields(scope.requestScope),
        abortSignal: signal,
        timeoutMs: Math.max(1, Math.ceil(deadline - performance.now())),
      };
    };
    const read = async () => {
      const rows = await bgRequest<WorkspaceArtifactApiResponse[]>({
        ...request(),
        path,
        method: 'GET',
      });
      assertCurrent();
      if (!Array.isArray(rows) || rows.some((row) => row.workspace_id !== workspaceId))
        throw new Error('Saved artifacts belong to another workspace.');
      const found = rows.filter((row) => row.id === payload.id);
      if (found.length > 1) throw new Error('Saved artifact identity is ambiguous.');
      return found[0];
    };
    const matches = (row: WorkspaceArtifactApiResponse) =>
      Object.entries(payload).every(([field, value]) =>
        sameWorkspaceArtifactPayload(row[field as keyof WorkspaceArtifactApiResponse], value)
      );
    const accepted = (row: WorkspaceArtifactApiResponse) => {
      assertCurrent();
      if (
        row.id !== payload.id ||
        row.workspace_id !== workspaceId ||
        !Number.isSafeInteger(row.version) ||
        row.version < 1 ||
        !matches(row)
      )
        throw new Error('Saved artifact acknowledgement does not match the draft.');
      const restored = mapServerArtifactToLocal(row);
      return bindWorkspaceArtifactOrigin(
        options.reviewChangeOnly || options.updateOnly ? restored : { ...artifact, ...restored },
        scope.scopeKey,
        workspaceId
      );
    };
    const existing = await read();
    if (options.updateOnly === "review") {
      if (!existing) throw new Error('The saved draft is no longer available.');
      payload = { id: existing.id, review_state: artifact.reviewStatus ?? "draft", review_metadata: artifact.reviewMetadata ?? {} };
    } else if (options.updateOnly) {
      if (!existing) throw new Error('The saved draft is no longer available.');
      readWorkspaceArtifactClientPayload(existing.version_metadata);
      const client = payload.version_metadata?.workspaceClient;
      payload = {
        id: existing.id,
        version_metadata: {
          ...existing.version_metadata,
          workspaceClient: {
            ...(existing.version_metadata?.workspaceClient as Record<string, unknown> | undefined),
            ...(client as Record<string, unknown>),
          },
        },
        ...(options.updateOnly === 'content' ? { content: artifact.content ?? null } : {}),
      };
    } else if (options.reviewChangeOnly) {
      if (!existing) throw new Error('The saved draft is no longer available.');
      readWorkspaceArtifactClientPayload(existing.version_metadata);
      const previousMetadata = existing.version_metadata ?? {};
      const previousClient = previousMetadata.workspaceClient as
        | Record<string, unknown>
        | undefined;
      payload = {
        id: existing.id,
        review_state: artifact.reviewStatus ?? 'draft',
        version_metadata: {
          ...previousMetadata,
          workspaceClient: {
            ...previousClient,
            schemaVersion: 1,
            removedFromStudio: artifact.removedFromStudio === true,
          },
        },
      };
    } else if (existing) {
      for (const [local, remote] of [
        ['ownerScope', 'owner_scope'],
        ['ownerId', 'owner_id'],
        ['projectId', 'project_id'],
        ['taskId', 'task_id'],
        ['sourceCollectionId', 'source_collection_id'],
        ['reviewMetadata', 'review_metadata'],
        ['redaction', 'redaction'],
      ] as const)
        if (artifact[local] === undefined && existing[remote] != null)
          Object.assign(payload, { [remote]: existing[remote] });
      payload.version_metadata = { ...existing.version_metadata, ...payload.version_metadata };
    }
    if (existing && matches(existing)) return accepted(existing);
    if (existing && existing.version !== artifact.workspaceArtifactVersion)
      throw new Error('This saved draft changed. Reload it before saving another revision.');
    if (!existing && artifact.workspaceArtifactId)
      throw new Error('The saved draft is no longer available.');
    const { id: _id, ...updates } = payload;
    try {
      const saved = existing
        ? await bgRequest<WorkspaceArtifactApiResponse>({
            ...request(),
            path: `${path}/${encodeURIComponent(existing.id)}`,
            method: 'PUT',
            body: { ...updates, version: artifact.workspaceArtifactVersion },
          })
        : await bgRequest<WorkspaceArtifactApiResponse>({
            ...request(),
            path,
            method: 'POST',
            body: payload,
          });
      return accepted(saved);
    } catch (error) {
      assertCurrent();
      // One exact readback resolves an uncertain acknowledgement; never resubmit blindly.
      const saved = await read();
      if (saved && matches(saved)) return accepted(saved);
      throw error;
    }
  })();
  pending.set(key, operation);
  try {
    return await operation;
  } finally {
    if (pending.get(key) === operation) pending.delete(key);
  }
};

export const loadWorkspaceArtifacts = (
  workspaceId: string,
  options: Pick<BgRequestInit, "headers" | "servicePromptConfig" | "abortSignal" | "timeoutMs">
): Promise<WorkspaceArtifactApiResponse[]> => bgRequest<WorkspaceArtifactApiResponse[]>({
  path: `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/artifacts`,
  method: "GET", ...options,
});


/** Revalidate a cold reader's canonical target before writing only review metadata. */
export const persistWorkspaceArtifactReview = async (
  artifact: GeneratedArtifact,
  patch: Pick<GeneratedArtifact, "reviewStatus" | "reviewMetadata">,
  options: WorkspaceArtifactSaveScope
): Promise<GeneratedArtifact> => {
  const { scope, workspaceId } = options;
  const deadline = options.deadline ?? performance.now() + 15000;
  const assertCurrent = () => {
    if (!options.isCurrent() || scope.scopeSignal.aborted || scope.scopeInvalidatedSignal.aborted)
      throw createServicePromptScopeChangedError();
    if (performance.now() >= deadline) throw new DOMException("Saving the review timed out. Your draft is retained.", "TimeoutError");
  };
  assertCurrent();
  if (!artifact.workspaceArtifactId) throw new Error("Save this output before recording a review.");
  if (hasWorkspaceArtifactOriginEvidence(artifact) && !hasWorkspaceArtifactOrigin(artifact, scope.scopeKey, workspaceId))
    throw new Error("This output belongs to another account or workspace.");
  const rows = await loadWorkspaceArtifacts(workspaceId, {
    ...requestScopeFields(scope.requestScope),
    abortSignal: AbortSignal.any([scope.scopeSignal, scope.scopeInvalidatedSignal, AbortSignal.timeout(Math.max(1, Math.ceil(deadline - performance.now())))]),
    timeoutMs: Math.max(1, Math.ceil(deadline - performance.now())),
  });
  assertCurrent();
  if (!Array.isArray(rows) || rows.some(row => row.workspace_id !== workspaceId))
    throw new Error("Saved outputs belong to another workspace.");
  const matches = rows.filter(row => row.id === artifact.workspaceArtifactId);
  if (matches.length !== 1) throw new Error("The original output is no longer available.");
  const canonical = mapServerArtifactToLocal(matches[0]);
  if (canonical.workspaceArtifactVersion !== artifact.workspaceArtifactVersion || canonical.type !== artifact.type ||
      !sameWorkspaceArtifactPayload([canonical.content, canonical.data, canonical.versionMetadata, canonical.reviewMetadata],
        [artifact.content, artifact.data, artifact.versionMetadata, artifact.reviewMetadata]))
    throw new Error("This output changed. Your review draft is retained; reopen the latest output before saving.");
  const verified = bindWorkspaceArtifactOrigin({ ...canonical, id: artifact.id, ...patch }, scope.scopeKey, workspaceId);
  return persistWorkspaceArtifact(verified, { ...options, deadline, updateOnly: "review" });
};
