import { mapServerArtifactToLocal } from '@/store/workspace-api';
import { bgRequest } from '@/services/background-proxy';
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
};

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
  if (!workspaceId.trim() || workspaceId !== workspaceId.trim() || workspaceId.length > 128)
    throw new Error('Invalid artifact workspace.');
  // Legacy/offline workspaces keep ADR-008's local cache; enrollment survives a missing tombstone.
  if (!artifact.workspaceArtifactId && !hasResearchWorkspaceMigrationTombstone(workspaceId))
    return artifact;
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
    let payload: Partial<WorkspaceArtifactCreateRequest> & { id: string } = options.reviewChangeOnly
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
        options.reviewChangeOnly ? restored : { ...artifact, ...restored },
        scope.scopeKey,
        workspaceId
      );
    };
    const existing = await read();
    if (options.reviewChangeOnly) {
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
