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
  applyVNScriptSnippet,
  createVNScriptFromTemplate,
  createVNScript,
  deleteVNScript,
  evaluateVNScriptVersionPolicy,
  getVNScriptAuthoringCatalog,
  getVNScript,
  getVNScriptDiagnostics,
  getVNScriptDraft,
  getVNScriptDraftGraph,
  getVNScriptManifestSnapshot,
  getVNScriptVersion,
  getVNScriptVersionGraph,
  listVNScriptTemplates,
  listVNScripts,
  listVNScriptVersions,
  patchVNScript,
  previewVNScriptDraftGraph,
  previewVNScriptSnippet,
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

  it('calls the VN script template catalog endpoint', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      items: [
        {
          id: 'linear_scene',
          label: 'Linear scene',
          description: 'Simple intro scene',
          category: 'starter',
          recommended_content_rating: 'general',
          required_capabilities: [],
          preview: { scenes: 1 },
          default_title: 'Linear scene',
          default_description: 'A simple starter scene',
        },
      ],
    });

    const response = await listVNScriptTemplates();

    expect(response.items[0].id).toBe('linear_scene');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/templates');
  });

  it('creates a VN script from a template endpoint with the request payload', async () => {
    const request = {
      title: 'Template Route',
      description: 'Created from a starter',
      primary_asset_pack_id: 7,
      policy_profile_id: 'teen-policy',
      generation_profile_id: 'story-default',
      content_rating: 'teen' as const,
    };

    await createVNScriptFromTemplate('linear_scene', request);

    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/templates/linear_scene/scripts',
      request
    );
  });

  it('encodes VN script template ids before building the create endpoint', async () => {
    const request = {
      title: 'Template Route',
      primary_asset_pack_id: 7,
      content_rating: 'general' as const,
    };

    await createVNScriptFromTemplate('linear scene/α', request);

    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/templates/linear%20scene%2F%CE%B1/scripts',
      request
    );
  });

  it('allows create-from-template requests to rely on server template title defaults', async () => {
    const request = {
      primary_asset_pack_id: 7,
      content_rating: 'general' as const,
    };

    await createVNScriptFromTemplate('linear_scene', request);

    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/templates/linear_scene/scripts',
      request
    );
  });

  it('calls the VN script authoring catalog endpoint', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      schema_version: 'vn_script_authoring_catalog.v1',
      program_schema_version: 'vn_script_program.v1',
      capability_tokens: ['script_authoring_catalog'],
      generation_output_schemas: ['choice_set', 'narrative_dialogue', 'scene_update'],
      operation_categories: { narration: ['narrate'] },
      operations: [],
      snippets: [],
      limits: { max_operations: 100 },
    });

    const response = await getVNScriptAuthoringCatalog();

    expect(response.schema_version).toBe('vn_script_authoring_catalog.v1');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/vn-authoring-catalog');
  });

  it('calls the VN script snippet preview endpoint with the request payload', async () => {
    const request = {
      snippet_id: 'generated_choice_set',
      anchor: { label: 'start', op_index: 1, mode: 'after' },
      parameters: { prompt: 'Offer three routes.' },
      draft: { labels: { start: [] } },
      draft_revision: 4,
    };

    await previewVNScriptSnippet(17, request);

    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/scripts/17/draft/snippet-preview',
      request
    );
  });

  it('calls the VN script snippet apply endpoint with the request payload', async () => {
    const request = {
      if_revision: 4,
      snippet_id: 'generated_choice_set',
      anchor: { label: 'start', op_index: 1, mode: 'after' },
      parameters: { prompt: 'Offer three routes.' },
    };

    await applyVNScriptSnippet(17, request);

    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/scripts/17/draft/snippet-apply',
      request
    );
  });

  it('calls the VN script stored draft graph endpoint', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      schema_version: 'vn_script_authoring_graph.v1',
      graph_semantics_version: 'vn_script_authoring_graph_edges.v1',
      program_schema_version: 'vn_script_program.v1',
      script_id: 17,
      source: 'stored_draft',
      base_revision: 4,
      version_id: null,
      content_hash: 'sha256:stored',
      validation_context_source: 'current_draft_context',
      truncated: false,
      limits: { max_labels: 500 },
      outline: { entry_label: 'start', labels: [] },
      graph: { nodes: [], edges: [] },
      diagnostics: { errors: [], warnings: [] },
      validation_diagnostics: { valid: true, errors: [], warnings: [] },
    });

    const response = await getVNScriptDraftGraph(17);

    expect(response.source).toBe('stored_draft');
    expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-scripts/scripts/17/draft/graph');
  });

  it('calls the VN script draft graph preview endpoint with the supplied draft payload', async () => {
    const request = {
      draft_revision: 4,
      draft: {
        schema_version: 'vn_script_program.v1',
        entry_label: 'start',
        labels: {
          start: [{ op: 'end' }],
        },
      },
    };

    await previewVNScriptDraftGraph(17, request);

    expect(mocks.apiClient.post).toHaveBeenCalledWith(
      '/vn/vn-scripts/scripts/17/draft/graph-preview',
      request
    );
  });

  it('calls the VN script published version graph endpoint', async () => {
    await getVNScriptVersionGraph(17, 9);

    expect(mocks.apiClient.get).toHaveBeenCalledWith(
      '/vn/vn-scripts/scripts/17/versions/9/graph'
    );
  });
});
