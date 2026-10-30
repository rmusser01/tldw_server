import type { GeneratedArtifact, ArtifactSourceCoverage } from '@/types/workspace';
import type { WorkspaceArtifactCreateRequest } from '@/services/tldw/domains/workspace-api';
import { WORK_PRODUCT_TEMPLATES } from '@/workspace-templates/work-product-templates';

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');
const coverageEntry = (value: unknown) =>
  record(value) &&
  typeof value.sourceId === 'string' &&
  (value.mediaId === undefined || finite(value.mediaId)) &&
  (value.title === undefined || typeof value.title === 'string');
const coverage = (value: unknown): value is ArtifactSourceCoverage =>
  record(value) &&
  strings(value.selectedSourceIds) &&
  typeof value.minimumUsableSourcesMet === 'boolean' &&
  ['usableSources', 'skippedSources', 'truncatedSources'].every(
    (key) =>
      Array.isArray(value[key]) &&
      value[key].every(
        (item) =>
          coverageEntry(item) &&
          (key !== 'skippedSources' ||
            (record(item) &&
              ['missing_text', 'unready', 'context_limit', 'unknown'].includes(
                String(item.reason)
              )))
      )
  ) &&
  (value.sourceContextCharLimit === undefined ||
    (record(value.sourceContextCharLimit) &&
      finite(value.sourceContextCharLimit.perSource) &&
      finite(value.sourceContextCharLimit.total)));

const CLIENT_FIELDS = [
  'removedFromStudio',
  'data',
  'templateId',
  'sourceCoverage',
  'reviewChecklist',
  'serverId',
  'presentationId',
  'presentationVersion',
  'audioFormat',
  'estimatedTokens',
  'estimatedCostUsd',
  'totalTokens',
  'totalCostUsd',
  'rootArtifactId',
  'artifactVersionId',
  'previousVersionId',
  'version',
] as const;

export const readWorkspaceArtifactClientPayload = (
  metadata: unknown
): Partial<GeneratedArtifact> => {
  if (!record(metadata) || metadata.workspaceClient === undefined) return {};
  if (!record(metadata.workspaceClient) || metadata.workspaceClient.schemaVersion !== 1)
    throw new Error('Saved artifact client metadata version is unsupported.');
  const payload = metadata.workspaceClient;
  const output: Record<string, unknown> = {};
  for (const key of CLIENT_FIELDS) {
    const value = payload[key];
    if (value === undefined) continue;
    const valid =
      key === 'removedFromStudio'
        ? typeof value === 'boolean'
        : key === 'data'
          ? record(value)
          : key === 'templateId'
            ? WORK_PRODUCT_TEMPLATES.some((template) => template.id === value)
            : key === 'sourceCoverage'
              ? coverage(value)
              : key === 'reviewChecklist'
                ? Array.isArray(value) &&
                  value.every(
                    (item) =>
                      record(item) &&
                      typeof item.id === 'string' &&
                      typeof item.label === 'string' &&
                      typeof item.checked === 'boolean'
                  )
                : key === 'serverId'
                  ? typeof value === 'string' || finite(value)
                  : [
                        'presentationVersion',
                        'estimatedTokens',
                        'estimatedCostUsd',
                        'totalTokens',
                        'totalCostUsd',
                        'version',
                      ].includes(key)
                    ? finite(value) && value >= 0
                    : typeof value === 'string';
    if (!valid) throw new Error('Saved artifact client metadata is malformed.');
    output[key] = value;
  }
  // The local card ID is display identity; requests always use the separate canonical ID.
  if (payload.id !== undefined) {
    if (typeof payload.id !== 'string' || !/^[A-Za-z0-9._:-]{1,128}$/.test(payload.id))
      throw new Error('Saved artifact display identity is malformed.');
    output.id = payload.id;
  }
  return output as Partial<GeneratedArtifact>;
};

export const workspaceArtifactDraftPayload = (
  artifact: GeneratedArtifact
): WorkspaceArtifactCreateRequest => {
  const client: Record<string, unknown> = { schemaVersion: 1, id: artifact.id };
  for (const key of CLIENT_FIELDS) if (artifact[key] !== undefined) client[key] = artifact[key];
  const payload: WorkspaceArtifactCreateRequest = {
    id: artifact.workspaceArtifactId ?? artifact.id,
    artifact_type: artifact.type,
    title: artifact.title,
    status: artifact.status,
    content: artifact.content ?? null,
    content_type: artifact.contentType ?? 'text/markdown',
    preview_text: artifact.previewText ?? null,
    summary: artifact.summary ?? null,
    review_state: artifact.reviewStatus ?? 'draft',
    owner_scope: artifact.ownerScope ?? 'user',
    owner_id: artifact.ownerId ?? null,
    project_id: artifact.projectId ?? null,
    task_id: artifact.taskId ?? null,
    source_collection_id: artifact.sourceCollectionId ?? null,
    producer_metadata: artifact.producerMetadata ?? {},
    source_lineage: { sources: artifact.sourceLineage ?? [] },
    review_metadata: artifact.reviewMetadata ?? {},
    version_metadata: { ...artifact.versionMetadata, workspaceClient: client },
    export_refs: artifact.exportRefs ?? [],
    redaction: {
      ...artifact.redaction,
      support_safe: artifact.redaction?.supportSafe ?? true,
      redacted: artifact.redaction?.redacted ?? false,
      retention_class: artifact.redaction?.retentionClass ?? null,
      redacted_fields: artifact.redaction?.redactedFields ?? [],
    },
    schema_version: artifact.schemaVersion ?? 1,
  };
  // Refuse values JSON would silently erase or corrupt before submitting a durable draft.
  const serialized = JSON.stringify(payload, (_key, value) => {
    if (
      ['function', 'symbol', 'bigint'].includes(typeof value) ||
      (typeof value === 'number' && !Number.isFinite(value))
    )
      throw new Error('Generated artifact is not JSON serializable.');
    return value;
  });
  const result = JSON.parse(serialized) as WorkspaceArtifactCreateRequest;
  readWorkspaceArtifactClientPayload(result.version_metadata);
  return result;
};

export const sameWorkspaceArtifactPayload = (left: unknown, right: unknown): boolean => {
  const canonical = (value: unknown): unknown =>
    Array.isArray(value)
      ? value.map(canonical)
      : record(value)
        ? Object.fromEntries(
            Object.keys(value)
              .sort()
              .map((key) => [key, canonical(value[key])])
          )
        : value;
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
};

// Runtime-only evidence: JSON/localStorage cannot mint or carry an authenticated draft origin.
const ARTIFACT_ORIGIN = Symbol('workspace-artifact-origin');
type BoundArtifact = GeneratedArtifact & {
  [ARTIFACT_ORIGIN]?: Readonly<{ scopeKey: string; workspaceId: string }>;
};
export const hasWorkspaceArtifactOriginEvidence = (artifact: GeneratedArtifact): boolean =>
  Boolean((artifact as BoundArtifact)[ARTIFACT_ORIGIN]);
export const hasWorkspaceArtifactOrigin = (
  artifact: GeneratedArtifact,
  scopeKey: string,
  workspaceId: string
): boolean => {
  const origin = (artifact as BoundArtifact)[ARTIFACT_ORIGIN];
  return origin?.scopeKey === scopeKey && origin.workspaceId === workspaceId;
};
export const bindWorkspaceArtifactOrigin = <T extends GeneratedArtifact>(
  artifact: T,
  scopeKey: string,
  workspaceId: string
): T => {
  const prior = (artifact as BoundArtifact)[ARTIFACT_ORIGIN];
  if (prior && !hasWorkspaceArtifactOrigin(artifact, scopeKey, workspaceId))
    throw new Error('This draft belongs to another account or workspace.');
  if (!prior)
    Object.defineProperty(artifact, ARTIFACT_ORIGIN, {
      enumerable: true,
      value: Object.freeze({ scopeKey, workspaceId }),
    });
  return artifact;
};
