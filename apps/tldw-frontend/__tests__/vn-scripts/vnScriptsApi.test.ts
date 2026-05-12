import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiClient: {
    delete: vi.fn(),
    get: vi.fn(),
    patch: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}));

vi.mock('@web/lib/api', () => ({
  apiClient: mocks.apiClient,
}));

import {
  createVNScript,
  deleteVNScript,
  evaluateVNScriptVersionPolicy,
  getVNScript,
  getVNScriptDiagnostics,
  getVNScriptDraft,
  getVNScriptManifestSnapshot,
  getVNScriptVersion,
  listVNScripts,
  listVNScriptVersions,
  patchVNScript,
  publishVNScript,
  putVNScriptDraft,
  validateVNScriptDraft,
} from '@web/lib/api/vnScripts';

describe('vnScripts api client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.apiClient.delete.mockResolvedValue(undefined);
    mocks.apiClient.get.mockResolvedValue({});
    mocks.apiClient.patch.mockResolvedValue({});
    mocks.apiClient.post.mockResolvedValue({});
    mocks.apiClient.put.mockResolvedValue({});
  });

  it('calls script CRUD endpoints with expected paths and payloads', async () => {
    mocks.apiClient.post.mockResolvedValueOnce({
      id: 1,
      title: 'Opening Route',
      status: 'draft',
      primary_asset_pack_id: 7,
      policy_profile_id: 'local_default',
      generation_profile_id: 'story_default',
      generation_profiles: {},
      content_rating: 'teen',
    });

    const created = await createVNScript({
      title: 'Opening Route',
      primary_asset_pack_id: 7,
      content_rating: 'teen',
    });
    await listVNScripts({ limit: 10, offset: 20 });
    await getVNScript(1);
    await patchVNScript(1, { title: 'Updated Route', status: 'ready' });
    await deleteVNScript(1);

    expect(created.id).toBe(1);
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-scripts/scripts', {
      title: 'Opening Route',
      primary_asset_pack_id: 7,
      content_rating: 'teen',
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts', {
      params: { limit: 10, offset: 20 },
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1');
    expect(mocks.apiClient.patch).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1', {
      title: 'Updated Route',
      status: 'ready',
    });
    expect(mocks.apiClient.delete).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1');
  });

  it('calls draft, validation, diagnostics, and publish endpoints', async () => {
    const draft = { nodes: [{ id: 'start' }] };

    await getVNScriptDraft(1);
    await putVNScriptDraft(1, { if_revision: 2, draft });
    await validateVNScriptDraft(1, { draft });
    await getVNScriptDiagnostics(1);
    await publishVNScript(1, {
      draft_revision: 3,
      label: 'First pass',
      idempotency_key: 'publish-1',
      acknowledgements: ['ack-1'],
    });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/draft');
    expect(mocks.apiClient.put).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/draft', {
      if_revision: 2,
      draft,
    });
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/draft/validate', {
      draft,
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/draft/diagnostics');
    expect(mocks.apiClient.post).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/publish', {
      draft_revision: 3,
      label: 'First pass',
      idempotency_key: 'publish-1',
      acknowledgements: ['ack-1'],
    });
  });

  it('calls version, manifest snapshot, and policy evaluation endpoints', async () => {
    await listVNScriptVersions(1, { limit: 5, offset: 10 });
    await getVNScriptVersion(1, 4);
    await getVNScriptManifestSnapshot(1, 4);
    await evaluateVNScriptVersionPolicy(1, 4, { context: { mode: 'preview' } });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/versions', {
      params: { limit: 5, offset: 10 },
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/versions/4');
    expect(mocks.apiClient.get).toHaveBeenCalledWith(
      '/vn/vn-scripts/scripts/1/versions/4/manifest-snapshot'
    );
    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/scripts/1/versions/4/policy/evaluate',
      { context: { mode: 'preview' } }
    );
  });

  it('omits nullish query values before passing params to the API client', async () => {
    await listVNScripts({ limit: 10, offset: undefined } as Parameters<typeof listVNScripts>[0]);
    await listVNScriptVersions(1, {
      limit: undefined,
      offset: undefined,
    } as Parameters<typeof listVNScriptVersions>[1]);

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts', {
      params: { limit: 10 },
    });
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/1/versions', {
      params: {},
    });
  });
});
