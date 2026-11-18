import type { WorkspaceArtifactApiResponse } from '@/services/tldw/domains/workspace-api';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GeneratedArtifact } from '@/types/workspace';
import type { ServicePromptSnapshot } from '@/services/service-prompts';
import {
  workspaceArtifactDraftPayload,
  readWorkspaceArtifactClientPayload,
  bindWorkspaceArtifactOrigin,
} from '@/store/workspace-artifact-codec';
import { mapServerArtifactToLocal, hydrateWorkspaceFromServer } from '@/store/workspace-api';
import { isServicePromptRequestPath } from '@/services/tldw/service-prompt-scope-error';
import { tldwRequest } from '@/services/tldw/request-core';

const io = vi.hoisted(() => ({ request: vi.fn(), migrated: true }));
vi.mock('@/services/background-proxy', () => ({ bgRequest: io.request }));
vi.mock('@/store/workspace', () => ({ hasResearchWorkspaceMigrationTombstone: () => io.migrated }));
import { persistWorkspaceArtifact, persistWorkspaceArtifactReview } from '../workspace-artifact-persistence';

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
  it.each([
    { version: 311, root: 'native-output', current: 'native-output:v311', previous: 'native-output:v310' },
    { version: 1, root: 'native-output', current: 'native-output:v1', previous: null },
    { version: 1, root: undefined, current: undefined, previous: undefined },
  ])('uses native job revision $version instead of stale client lineage', ({ version, root, current, previous }) => {
    const draft = artifact();
    const original = row({ ...draft, rootArtifactId: 'native-output', artifactVersionId: 'native-output:v285',
      previousVersionId: 'native-output:v284' }, version);
    original.id = 'native-output';
    original.producer_metadata = { origin: 'research_workspace_output_job', job_id: 61 };
    original.root_artifact_id = root;
    original.artifact_version_id = current;
    original.previous_version_id = previous;
    const before = structuredClone(original);

    const restored = mapServerArtifactToLocal(original);

    expect(restored).toMatchObject({ version, workspaceArtifactVersion: version,
      rootArtifactId: root, artifactVersionId: current, previousVersionId: previous ?? undefined,
      id: draft.id, serverId: 82, content: draft.content, data: draft.data });
    expect(restored.versionMetadata?.workspaceClient).toEqual(original.version_metadata?.workspaceClient);
    expect(original).toEqual(before);
  });
  it.each(['research_pack_import', 'workspace_snapshot_import'])(
    'retains logical versions under the outer %s provenance wrapper', origin => {
      const original = row(artifact(), 12);
      original.root_artifact_id = 'destination-artifact';
      original.artifact_version_id = 'destination-artifact:v12';
      original.previous_version_id = 'destination-artifact:v11';
      original.producer_metadata = { origin,
        importedProducer: { origin: 'research_workspace_output_job', job_id: 61 } };
      const restored = mapServerArtifactToLocal(original);
      expect(restored).toMatchObject({ version: 3, rootArtifactId: 'acp-run-7',
        artifactVersionId: 'acp-run-7-v3', previousVersionId: 'acp-run-7-v2',
        workspaceArtifactVersion: 12, serverId: 82, producerMetadata: original.producer_metadata });
      expect(restored.versionMetadata?.workspaceClient).toEqual(original.version_metadata?.workspaceClient);
    }
  );
  it.each(['native-parent:v7', 'native-child:variant'])(
    'retains logical native parent %s while refreshing the current artifact revision', previousVersionId => {
    const original = row({ ...artifact(), rootArtifactId: 'native-parent',
      previousVersionId }, 4);
    original.id = 'native-child';
    original.producer_metadata = { origin: 'research_workspace_output_job', job_id: 62 };
    original.root_artifact_id = 'native-child';
    original.artifact_version_id = 'native-child:v4';
    original.previous_version_id = 'native-child:v3';
    expect(mapServerArtifactToLocal(original)).toMatchObject({ version: 4, workspaceArtifactVersion: 4,
      rootArtifactId: 'native-parent', previousVersionId, artifactVersionId: 'native-child:v4' });
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
  it.each(['content', 'client'] as const)('preserves raw canonical lineage, ownership and producer fields during a %s-only update', async (updateOnly) => {
    const original = {
      ...row(),
      source_lineage: {
        media_ids: [1, 2], selected_source_ids: ['src_1'],
        sources: [{ source_id: 'src_1', media_id: 1, opaque: { retained: true } }],
        canonicalExtra: { policy: 'retain' }
      },
      producer_metadata: { producerType: 'worker', opaque: { retained: true } },
      owner_scope: 'research_workspace', owner_id: 'workspace-owned',
      project_id: 'project-1', task_id: 'task-2', source_collection_id: 'collection-3',
      review_metadata: { reviewer: 'retained' }, redaction: { support_safe: false, opaque: true },
    };
    const draft = bindWorkspaceArtifactOrigin(
      { ...mapServerArtifactToLocal(original), content: 'Revised card content', data: { questions: [{ question: 'Revised', options: ['yes'], answer: 'yes' }] } },
      'server:alice', 'workspace-owned'
    );
    let stored = original;
    io.request.mockImplementation(async (request) => {
      if (request.method === 'GET') return [stored];
      stored = { ...stored, ...request.body, version: stored.version + 1 };
      return stored;
    });
    const saved = await persistWorkspaceArtifact(draft, { ...lease().options, updateOnly });
    for (const key of ['source_lineage', 'producer_metadata', 'owner_scope', 'owner_id', 'project_id',
      'task_id', 'source_collection_id', 'review_metadata', 'redaction'] as const)
      expect(stored[key]).toEqual(original[key]);
    expect(stored.content).toBe(updateOnly === 'content' ? 'Revised card content' : original.content);
    expect(saved.data).toEqual({ questions: [{ question: 'Revised', options: ['yes'], answer: 'yes' }] });
    expect(saved.workspaceArtifactVersion).toBe(2);
  });

});


describe("section reviewer metadata saves", () => {
  it("updates only review fields with an optimistic version while preserving the raw draft and evidence", async () => {
    const first = { ...artifact(), workspaceArtifactId: "draft-owned", workspaceArtifactVersion: 1 };
    let stored = row(first);
    io.request.mockImplementation(async ({ method, body }) => {
      if (method === "GET") return [stored];
      expect(Object.keys(body).sort()).toEqual(["review_metadata", "review_state", "version"]);
      stored = { ...stored, ...body, version: 2 };
      return stored;
    });
    const saved = await persistWorkspaceArtifact({ ...first, reviewStatus: "needs_revision", reviewMetadata: { reason: "Attribution correction" } }, { ...lease().options, updateOnly: "review" });
    expect(saved.reviewMetadata).toMatchObject({ reason: "Attribution correction" });
    expect(stored.content).toBe(first.content);
    expect(stored.version_metadata).toEqual(row(first).version_metadata);
  });
});


describe("cold artifact review authority", () => {
  it.each(["slow reads", "slow write headers", "slow write body", "deadline", "account cancellation", "workspace change"])(
    "retains the exact review save boundary with %s", async outcome => {
      vi.useFakeTimers();
      const { controller, options } = lease();
      let current = true;
      options.isCurrent = () => current;
      let stored = row();
      const original = structuredClone(stored);
      const cold = mapServerArtifactToLocal(stored);
      const wait = (ms: number, signal?: AbortSignal | null) => new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, ms);
        signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); }, { once: true });
      });
      io.request.mockImplementation(async input => {
        const response = await tldwRequest(input, {
          getConfig: async () => ({ serverUrl: "https://owned.test", authMode: "single-user", apiKey: "test-key" }),
          fetchFn: async (_url, init) => {
            if (input.method === "PUT" && outcome === "slow write headers") await wait(20_000, init?.signal);
            if (input.method === "PUT") stored = { ...stored, ...JSON.parse(String(init?.body)), version: stored.version + 1 };
            const data = input.method === "GET" ? [stored] : stored;
            const response = Response.json(data);
            vi.spyOn(response, "json").mockImplementation(async () => {
              if (outcome === "deadline") await wait(180_000, init?.signal);
              else if (outcome === "slow reads" || outcome === "account cancellation" || outcome === "workspace change" ||
                (outcome === "slow write body" && input.method === "PUT")) await wait(20_000, init?.signal);
              return data;
            });
            return response;
          },
        });
        if (!response.ok) throw Object.assign(new Error(response.error), { code: response.code });
        return response.data;
      });
      const pending = persistWorkspaceArtifactReview(cold, { reviewStatus: "needs_revision", reviewMetadata: { reason: "Exact retained review" } }, options)
        .then(value => ({ value }), error => ({ error }));
      let settled = false;
      void pending.then(() => { settled = true; });
      try {
        await vi.advanceTimersByTimeAsync(0);
        if (outcome.startsWith("slow")) {
          await vi.advanceTimersByTimeAsync(60_000);
          expect(await pending).toMatchObject({ value: { reviewStatus: "needs_revision", reviewMetadata: { reason: "Exact retained review" }, workspaceArtifactVersion: 2 } });
          const writes = io.request.mock.calls.map(([input]) => input).filter(input => input.method === "PUT");
          expect(writes).toHaveLength(1);
          expect(writes[0].body).toEqual({ review_state: "needs_revision", review_metadata: { reason: "Exact retained review" }, version: 1 });
        } else {
          await vi.advanceTimersByTimeAsync(outcome === "deadline" ? 119_999 : 12_000);
          expect(settled).toBe(false);
          if (outcome === "account cancellation") controller.abort();
          if (outcome === "workspace change") current = false;
          await vi.advanceTimersByTimeAsync(outcome === "workspace change" ? 8000 : 2);
          const result = await pending;
          expect(result).toHaveProperty("error");
          if ("error" in result) expect(result.error).toMatchObject(outcome === "deadline" ? { code: "REQUEST_TIMEOUT" } : outcome === "account cancellation" ? { code: "REQUEST_ABORTED" } : { status: 412, details: { detail: { code: "request_config_scope_changed" } } });
          expect(io.request.mock.calls.some(([input]) => input.method === "PUT")).toBe(false);
        }
        expect(stored.content).toBe(original.content);
        expect(stored.version_metadata).toEqual(original.version_metadata);
        expect(cold.reviewMetadata).toEqual(mapServerArtifactToLocal(original).reviewMetadata);
      } finally {
        controller.abort();
        await pending;
        vi.useRealTimers();
      }
    },
  );

  it("revalidates a cold canonical artifact before saving review metadata", async () => {
    const first = { ...artifact(), workspaceArtifactId: "draft-owned", workspaceArtifactVersion: 1 };
    const cold = JSON.parse(JSON.stringify(mapServerArtifactToLocal(row(first))));
    let stored = row(first);
    io.request.mockImplementation(async ({ method, body }) => {
      if (method === "GET") return [stored];
      stored = { ...stored, ...body, version: 2 }; return stored;
    });
    const saved = await persistWorkspaceArtifactReview(cold, { reviewStatus: "needs_revision", reviewMetadata: { reason: "Source checked" } }, lease().options);
    expect(saved.reviewStatus).toBe("needs_revision");
    expect(saved.content).toBe(first.content);
    expect(io.request.mock.calls.filter(([request]) => request.method === "PUT")).toHaveLength(1);
  });
  it.each(["foreign", "changed", "retired"])("refuses %s review authority without writing", async kind => {
    const first = { ...artifact(), workspaceArtifactId: "draft-owned", workspaceArtifactVersion: 1 };
    const options = lease().options;
    if (kind === "foreign") options.scope = { ...options.scope, scopeKey: "server:bob" };
    if (kind === "retired") { let calls = 0; options.isCurrent = () => ++calls < 2; }
    io.request.mockResolvedValue([row(first, kind === "changed" ? 2 : 1)]);
    await expect(persistWorkspaceArtifactReview(first, { reviewStatus: "accepted" }, options)).rejects.toThrow();
    expect(io.request.mock.calls.some(([request]) => request.method === "PUT")).toBe(false);
  });
  it.each(["content", "data", "versionMetadata", "reviewMetadata"])("refuses same-version %s drift without writing", async kind => {
    const stored = row();
    const cold = JSON.parse(JSON.stringify(mapServerArtifactToLocal(stored))) as GeneratedArtifact;
    if (kind === "content") stored.content = "Changed source claim";
    else if (kind === "data") stored.version_metadata = {
      ...stored.version_metadata, workspaceClient: { schemaVersion: 1, data: { changedEvidence: true } }
    };
    else if (kind === "versionMetadata") stored.version_metadata = { ...stored.version_metadata, revision_reason: "Changed revision" };
    else stored.review_metadata = { reason: "A different reviewer changed the assessment" };
    io.request.mockResolvedValue([stored]);

    await expect(persistWorkspaceArtifactReview(cold, { reviewStatus: "accepted" }, lease().options)).rejects.toThrow("output changed");

    expect(io.request.mock.calls.some(([request]) => request.method === "PUT")).toBe(false);
  });
});
