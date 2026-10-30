import type { WorkspaceArtifactApiResponse } from '@/services/tldw/domains/workspace-api';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GeneratedArtifact } from '@/types/workspace';
import type { ServicePromptSnapshot } from '@/services/service-prompts';
import {
  workspaceArtifactDraftPayload,
  readWorkspaceArtifactClientPayload,
  bindWorkspaceArtifactOrigin,
} from '@/store/workspace-artifact-codec';
import { hydrateWorkspaceFromServer } from '@/store/workspace-api';
import { isServicePromptRequestPath } from '@/services/tldw/service-prompt-scope-error';

const io = vi.hoisted(() => ({ request: vi.fn(), migrated: true }));
vi.mock('@/services/background-proxy', () => ({ bgRequest: io.request }));
vi.mock('@/store/workspace', () => ({ hasResearchWorkspaceMigrationTombstone: () => io.migrated }));
import { persistWorkspaceArtifact } from '../workspace-artifact-persistence';

const artifact = (): GeneratedArtifact =>
  bindWorkspaceArtifactOrigin(
    {
      id: 'draft-owned',
      type: 'report',
      title: 'Rowan',
      status: 'completed',
      content: 'Selene, Friday 18:00.',
      createdAt: new Date('2026-10-30T00:00:00Z'),
      producerMetadata: { producerType: 'deep_research_bundle_import', runId: 'rs_owned' },
      data: {
        deepResearch: { runId: 'rs_owned', selectedImportedSources: [{ sourceId: 'src_1' }] },
      },
      sourceLineage: [{ sourceId: 'src_1', mediaId: 1 }],
      sourceCoverage: {
        selectedSourceIds: ['src_1'],
        usableSources: [{ sourceId: 'src_1' }],
        skippedSources: [],
        truncatedSources: [],
        minimumUsableSourcesMet: true,
      },
      reviewChecklist: [{ id: 'citations', label: 'Verify citations', checked: false }],
      templateId: 'corpus_gap_finder',
      serverId: 82,
      rootArtifactId: 'acp-run-7',
      artifactVersionId: 'acp-run-7-v3',
      previousVersionId: 'acp-run-7-v2',
      version: 3,
    },
    'server:alice',
    'workspace-owned'
  );
const row = (draft = artifact(), version = 1): WorkspaceArtifactApiResponse => ({
  ...workspaceArtifactDraftPayload(draft),
  status: draft.status,
  content: draft.content ?? null,
  workspace_id: 'workspace-owned',
  version,
  created_at: draft.createdAt.toISOString(),
  completed_at: null,
  total_tokens: null,
  total_cost_usd: null,
});
const lease = () => {
  const controller = new AbortController();
  const scope = {
    scopeKey: 'server:alice',
    requestScope: { config: { serverUrl: 'https://owned.test' }, userId: 'alice' },
    scopeSignal: controller.signal,
    scopeInvalidatedSignal: controller.signal,
    release: vi.fn(),
  } as unknown as ServicePromptSnapshot;
  return { controller, options: { workspaceId: 'workspace-owned', scope, isCurrent: () => true } };
};
beforeEach(() => {
  io.request.mockReset();
  io.migrated = true;
});

describe('canonical Workspace draft persistence', () => {
  it('scopes only exact Deep bundle reads without admitting run mutations', () => {
    expect(isServicePromptRequestPath('/api/v1/research/runs/rs_owned/bundle', 'GET')).toBe(true);
    expect(isServicePromptRequestPath('/api/v1/research/runs/rs_owned/bundle', 'POST')).toBe(false);
    expect(isServicePromptRequestPath('/api/v1/research/runs/rs_owned%2Fother/bundle', 'GET')).toBe(
      false
    );
  });
  it('removes and undoes an enrolled draft without losing version history or resurrecting it on reload', async () => {
    const first = {
      ...artifact(),
      workspaceArtifactId: 'draft-owned',
      workspaceArtifactVersion: 1,
    };
    let stored = {
      ...row(first),
      source_lineage: {
        media_ids: [1, 2],
        selected_source_ids: ['selected-1'],
        sources: [{ sourceId: 'src_1' }],
        canonicalExtra: { retained: true },
      },
    };
    const originalLineage = structuredClone(stored.source_lineage);
    const versions = [stored];
    io.request.mockImplementation(async ({ method, body }) => {
      if (method === 'GET') return [stored];
      expect(method).toBe('PUT');
      stored = { ...stored, ...body, version: stored.version + 1 };
      versions.push(stored);
      return stored;
    });
    const archived = await persistWorkspaceArtifact(
      { ...first, reviewStatus: 'archived', removedFromStudio: true },
      { ...lease().options, reviewChangeOnly: true }
    );
    const fetch = async () => ({
      id: 'workspace-owned',
      name: 'Owned',
      version: 1,
      artifacts: [stored],
    });
    expect((await hydrateWorkspaceFromServer('workspace-owned', { fetch })).artifacts).toEqual([]);
    const restored = await persistWorkspaceArtifact(
      { ...archived, reviewStatus: 'draft', removedFromStudio: false },
      { ...lease().options, reviewChangeOnly: true }
    );
    expect(restored.workspaceArtifactVersion).toBe(3);
    expect(
      (await hydrateWorkspaceFromServer('workspace-owned', { fetch })).artifacts[0]
    ).toMatchObject({
      id: first.id,
      content: first.content,
      producerMetadata: first.producerMetadata,
      data: first.data,
    });
    expect(stored.source_lineage).toEqual(originalLineage);
    for (const [input] of io.request.mock.calls.filter(([input]) => input.method === 'PUT'))
      expect(Object.keys(input.body).sort()).toEqual([
        'review_state',
        'version',
        'version_metadata',
      ]);
    expect(versions.map((value) => value.version)).toEqual([1, 2, 3]);
    expect(io.request.mock.calls.some(([input]) => input.method === 'DELETE')).toBe(false);
  });
  it('preserves unspecified canonical ownership, redaction and worker metadata when enrolling an existing output', async () => {
    const first = {
      ...artifact(),
      workspaceArtifactId: 'draft-owned',
      workspaceArtifactVersion: 1,
    };
    const existing = {
      ...row(first),
      owner_scope: 'workspace',
      owner_id: 'workspace-owned',
      redaction: { support_safe: false, redacted: true },
      review_metadata: { reviewer: 'retained' },
      version_metadata: { workerReceipt: { completed: true } },
    };
    io.request.mockImplementation(async ({ method, body }) =>
      method === 'GET' ? [existing] : { ...existing, ...body, version: 2 }
    );
    await expect(persistWorkspaceArtifact(first, lease().options)).resolves.toMatchObject({
      workspaceArtifactVersion: 2,
    });
    expect(io.request.mock.calls[1][0].body).toMatchObject({
      owner_scope: 'workspace',
      owner_id: 'workspace-owned',
      redaction: { support_safe: false, redacted: true },
      review_metadata: { reviewer: 'retained' },
      version_metadata: {
        workerReceipt: { completed: true },
        workspaceClient: { schemaVersion: 1 },
      },
    });
  });
  it('refuses an expired operation before I/O and rejects unsupported client metadata', async () => {
    await expect(
      persistWorkspaceArtifact(artifact(), { ...lease().options, deadline: performance.now() - 1 })
    ).rejects.toMatchObject({ name: 'TimeoutError' });
    expect(io.request).not.toHaveBeenCalled();
    expect(() =>
      readWorkspaceArtifactClientPayload({ workspaceClient: { schemaVersion: 2, data: {} } })
    ).toThrow('unsupported');
  });
  it('refuses a same-ID draft from another account, server or serialized cache', async () => {
    const foreign = artifact();
    const bob = lease().options;
    bob.scope = { ...bob.scope, scopeKey: 'server:bob' };
    await expect(persistWorkspaceArtifact(foreign, bob)).rejects.toThrow('verified origin');
    const serialized = JSON.parse(JSON.stringify(foreign));
    await expect(persistWorkspaceArtifact(serialized, lease().options)).rejects.toThrow(
      'verified origin'
    );
    expect(io.request).not.toHaveBeenCalled();
  });
  it('round trips structured draft metadata without confusing generated-resource IDs and canonical versions', async () => {
    const draft = artifact();
    const local = await hydrateWorkspaceFromServer('workspace-owned', {
      fetch: async () => ({
        id: 'workspace-owned',
        name: 'Owned',
        version: 1,
        artifacts: [row(draft, 12)],
      }),
    });
    expect(local.artifacts[0]).toMatchObject({
      ...draft,
      workspaceArtifactId: draft.id,
      workspaceArtifactVersion: 12,
    });
    expect(local.artifacts[0].version).toBe(3);
    expect(local.artifacts[0].serverId).toBe(82);
  });
  it('reconciles an unknown acknowledgement once, with no duplicate submission', async () => {
    let stored: ReturnType<typeof row> | undefined;
    io.request.mockImplementation(async ({ method }) => {
      if (method === 'GET') return stored ? [stored] : [];
      stored = row();
      throw new Error('connection ended after server commit');
    });
    const saved = await persistWorkspaceArtifact(artifact(), lease().options);
    expect(saved).toMatchObject({
      workspaceArtifactId: 'draft-owned',
      workspaceArtifactVersion: 1,
    });
    expect(io.request.mock.calls.map(([input]) => input.method)).toEqual(['GET', 'POST', 'GET']);
    expect(io.request.mock.calls[1][0]).toMatchObject({
      servicePromptConfig: { serverUrl: 'https://owned.test', expectedUserId: 'alice' },
      headers: { 'X-TLDW-Expected-User-ID': 'alice' },
    });
  });
  it.each(['missing', 'different-content', 'different-provenance'])(
    'keeps an uncertain %s draft unsaved',
    async (kind) => {
      let attempted = false;
      io.request.mockImplementation(async ({ method }) => {
        if (method !== 'GET') {
          attempted = true;
          throw new Error('unknown save');
        }
        if (!attempted || kind === 'missing') return [];
        const saved = row();
        return [
          kind === 'different-content'
            ? { ...saved, content: 'another draft' }
            : { ...saved, producer_metadata: { producerType: 'foreign' } },
        ];
      });
      await expect(persistWorkspaceArtifact(artifact(), lease().options)).rejects.toThrow(
        'unknown save'
      );
      expect(io.request.mock.calls.filter(([input]) => input.method === 'POST')).toHaveLength(1);
    }
  );
  it('refuses retired authority before writing or reconciling', async () => {
    const { controller, options } = lease();
    io.request.mockImplementation(async () => {
      controller.abort();
      return [];
    });
    await expect(persistWorkspaceArtifact(artifact(), options)).rejects.toMatchObject({
      status: 412,
    });
    expect(io.request).toHaveBeenCalledTimes(1);
  });
  it('rejects another workspace and a stale optimistic revision', async () => {
    const draft = {
      ...artifact(),
      workspaceArtifactId: 'draft-owned',
      workspaceArtifactVersion: 1,
    };
    io.request.mockResolvedValue([{ ...row(), workspace_id: 'another' }]);
    await expect(persistWorkspaceArtifact(draft, lease().options)).rejects.toThrow(
      'another workspace'
    );
    io.request.mockResolvedValue([row({ ...draft, content: 'newer saved content' }, 2)]);
    await expect(persistWorkspaceArtifact(draft, lease().options)).rejects.toThrow('changed');
    expect(io.request.mock.calls.every(([input]) => input.method === 'GET')).toBe(true);
  });
  it('preserves legacy/offline caching and fails closed on malformed client payloads', async () => {
    io.migrated = false;
    const draft = artifact();
    await expect(persistWorkspaceArtifact(draft, lease().options)).resolves.toBe(draft);
    expect(io.request).not.toHaveBeenCalled();
    expect(() =>
      readWorkspaceArtifactClientPayload({
        workspaceClient: { schemaVersion: 1, reviewChecklist: [{}] },
      })
    ).toThrow('malformed');
    expect(() => workspaceArtifactDraftPayload({ ...draft, data: { value: Number.NaN } })).toThrow(
      'JSON'
    );
  });
  it.each([
    ['POST', '/artifacts'],
    ['PUT', '/artifacts/draft-owned'],
  ])('permits only the exact enrolled artifact %s route', (method, suffix) => {
    expect(isServicePromptRequestPath('/api/v1/workspaces/workspace-owned' + suffix, method)).toBe(
      true
    );
    expect(
      isServicePromptRequestPath('/api/v1/workspaces/workspace-owned%2Fother' + suffix, method)
    ).toBe(false);
    expect(isServicePromptRequestPath('/api/v1/workspaces/workspace-owned/context', method)).toBe(
      false
    );
  });
});
