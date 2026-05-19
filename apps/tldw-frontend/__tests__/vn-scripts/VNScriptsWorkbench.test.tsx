import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mocks = vi.hoisted(() => ({
  apiClient: {
    get: vi.fn(),
  },
  applyVNScriptSnippet: vi.fn(),
  createVNScriptFromTemplate: vi.fn(),
  createVNScript: vi.fn(),
  evaluateVNScriptVersionPolicy: vi.fn(),
  getVNScriptAuthoringCatalog: vi.fn(),
  getVNScriptDiagnostics: vi.fn(),
  getVNScriptDraft: vi.fn(),
  getVNScriptDraftGraph: vi.fn(),
  getVNScriptManifestSnapshot: vi.fn(),
  getVNScriptVersionGraph: vi.fn(),
  listVNScriptTemplates: vi.fn(),
  listVNScripts: vi.fn(),
  listVNScriptVersions: vi.fn(),
  previewVNScriptDraftGraph: vi.fn(),
  previewVNScriptSnippet: vi.fn(),
  publishVNScript: vi.fn(),
  putVNScriptDraft: vi.fn(),
  validateVNScriptDraft: vi.fn(),
}));

vi.mock('@web/lib/api', () => ({
  apiClient: mocks.apiClient,
}));

vi.mock('@web/lib/api/vnScripts', () => ({
  applyVNScriptSnippet: (...args: unknown[]) => mocks.applyVNScriptSnippet(...args),
  createVNScriptFromTemplate: (...args: unknown[]) => mocks.createVNScriptFromTemplate(...args),
  createVNScript: (...args: unknown[]) => mocks.createVNScript(...args),
  evaluateVNScriptVersionPolicy: (...args: unknown[]) => mocks.evaluateVNScriptVersionPolicy(...args),
  getVNScriptAuthoringCatalog: (...args: unknown[]) => mocks.getVNScriptAuthoringCatalog(...args),
  getVNScriptDiagnostics: (...args: unknown[]) => mocks.getVNScriptDiagnostics(...args),
  getVNScriptDraft: (...args: unknown[]) => mocks.getVNScriptDraft(...args),
  getVNScriptDraftGraph: (...args: unknown[]) => mocks.getVNScriptDraftGraph(...args),
  getVNScriptManifestSnapshot: (...args: unknown[]) => mocks.getVNScriptManifestSnapshot(...args),
  getVNScriptVersionGraph: (...args: unknown[]) => mocks.getVNScriptVersionGraph(...args),
  listVNScriptTemplates: (...args: unknown[]) => mocks.listVNScriptTemplates(...args),
  listVNScripts: (...args: unknown[]) => mocks.listVNScripts(...args),
  listVNScriptVersions: (...args: unknown[]) => mocks.listVNScriptVersions(...args),
  previewVNScriptDraftGraph: (...args: unknown[]) => mocks.previewVNScriptDraftGraph(...args),
  previewVNScriptSnippet: (...args: unknown[]) => mocks.previewVNScriptSnippet(...args),
  publishVNScript: (...args: unknown[]) => mocks.publishVNScript(...args),
  putVNScriptDraft: (...args: unknown[]) => mocks.putVNScriptDraft(...args),
  validateVNScriptDraft: (...args: unknown[]) => mocks.validateVNScriptDraft(...args),
}));

vi.mock('@web/components/ui/JsonEditor', () => ({
  JsonEditor: ({
    value,
    onChange,
    readOnly,
  }: {
    value: string;
    onChange: (nextValue: string) => void;
    readOnly?: boolean;
  }) => (
    <textarea
      aria-label="Draft JSON"
      readOnly={readOnly}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
  default: ({
    value,
    onChange,
    readOnly,
  }: {
    value: string;
    onChange: (nextValue: string) => void;
    readOnly?: boolean;
  }) => (
    <textarea
      aria-label="Draft JSON"
      readOnly={readOnly}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

import VNScriptsWorkbench from '@web/components/vn-scripts/VNScriptsWorkbench';

const openingScript = {
  id: 1,
  title: 'Opening Route',
  status: 'draft',
  primary_asset_pack_id: 7,
  policy_profile_id: 'teen-policy',
  generation_profile_id: 'story-default',
  generation_profiles: {},
  content_rating: 'teen',
};

const secondScript = {
  id: 2,
  title: 'Second Route',
  status: 'ready',
  primary_asset_pack_id: 8,
  policy_profile_id: 'general-policy',
  generation_profile_id: 'fast-default',
  generation_profiles: {},
  content_rating: 'general',
};

const draftResponse = {
  script_id: 1,
  revision: 3,
  draft: { scenes: [{ id: 'start', text: 'Wake up.' }] },
  diagnostics: { node_count: 1 },
};

const versionResponse = {
  id: 12,
  script_id: 1,
  version_number: 2,
  label: 'Launch candidate',
  draft_revision: 3,
  program: { scenes: [] },
  asset_pack_id: 7,
  manifest_snapshot_id: 101,
  policy_snapshot_id: 202,
  generation_profile_snapshot_id: 303,
  generation_profile_snapshots: { narrator: 404 },
  script_defaults: {},
  validation: { valid: true, warnings: 0 },
  created_at: '2026-05-12T10:15:00Z',
};

const secondVersionResponse = {
  ...versionResponse,
  id: 22,
  script_id: 2,
  version_number: 4,
  label: 'Second launch',
  asset_pack_id: 8,
  manifest_snapshot_id: 121,
  policy_snapshot_id: 222,
  generation_profile_snapshot_id: 323,
};

const linearTemplate = {
  id: 'linear_scene',
  label: 'Linear scene',
  description: 'Start with one authored scene.',
  category: 'starter',
  recommended_content_rating: 'general',
  required_capabilities: ['dialogue'],
  preview: { scenes: 1, choices: 0 },
  default_title: 'Linear scene',
  default_description: 'A simple authored opener.',
};

const choiceTemplate = {
  id: 'authored_choices',
  label: 'Authored choices',
  description: 'Start with player choices.',
  category: 'starter',
  recommended_content_rating: 'teen',
  required_capabilities: ['dialogue', 'choices'],
  preview: { scenes: 2, choices: 2 },
  default_title: 'Choice scene',
  default_description: 'An opener with two choices.',
};

const templateScript = {
  id: 21,
  title: 'Template Route',
  description: 'A simple authored opener.',
  status: 'draft',
  primary_asset_pack_id: 55,
  policy_profile_id: 'teen-policy',
  generation_profile_id: 'story-default',
  generation_profiles: {},
  content_rating: 'general',
};

const templateDraftResponse = {
  script_id: 21,
  revision: 1,
  draft: {
    version: 'vn_script_program.v1',
    primary_asset_pack_id: 55,
    scenes: [{ id: 'template_start', text: 'Template opening.' }],
  },
  diagnostics: { valid: true },
};

const vnCapabilitiesResponse = {
  schema_version: 'vn_capabilities.v1',
  generated_at: '2026-05-12T10:00:00Z',
  base_path: '/api/v1/vn',
  resources: {},
  enabled_modules: { scripts: true, play: true },
  features: {
    script_authoring_catalog: true,
    script_authoring_graph: true,
    scripted_generation: true,
  },
  limits: {},
  supported_content_ratings: ['general', 'teen', 'suggestive', 'mature'],
  visible_policy_profiles: [],
  visible_generation_profiles: [],
  supported_media_types: { image: [], audio: [] },
  scripted_generation: {
    enabled: true,
    output_schemas: ['choice_set'],
    confirmation_supported: true,
    revision_activation_supported: true,
    history_supported: true,
    debug_detail_supported: true,
    dynamic_choice_supported: true,
    scene_update_supported: true,
    max_automatic_generation_batch_count: 1,
    moderation_blocked_raw_reveal_supported: true,
  },
  route_migration: { canonical: '/api/v1/vn/vn-*', supersedes: [] },
  docs: {},
  openapi: '/openapi.json',
};

const choiceSnippet = {
  id: 'generated_choice_set',
  schema_version: 'vn_script_program.v1',
  label: 'Generated choice set',
  operation_sequence: ['insert_choice_set'],
  required_capability_tokens: ['script_authoring_catalog', 'choice_set'],
  parameters_schema: {
    type: 'object',
    properties: {
      prompt: { type: 'string', title: 'Prompt' },
      count: { type: 'number', title: 'Choice count' },
      shuffle: { type: 'boolean', title: 'Shuffle choices' },
      tone: { type: 'string', title: 'Tone', enum: ['dramatic', 'quiet'] },
    },
  },
  default_parameters: {
    prompt: 'Offer three routes.',
    count: 3,
    shuffle: false,
    tone: 'dramatic',
  },
  preview: [{ op: 'choice_set' }],
};

const unsupportedSnippet = {
  ...choiceSnippet,
  id: 'live_asset_generation',
  label: 'Live asset generation',
  operation_sequence: ['generate_asset'],
  required_capability_tokens: ['realtime_image_generation'],
};

const typedEnumSnippet = {
  ...choiceSnippet,
  id: 'typed_enum_snippet',
  label: 'Typed enum snippet',
  parameters_schema: {
    type: 'object',
    properties: {
      difficulty: { type: 'number', title: 'Difficulty', enum: [1, 2, 3] },
      includeRecap: { type: 'boolean', title: 'Include recap', enum: [true, false] },
    },
  },
  default_parameters: {
    difficulty: 2,
    includeRecap: true,
  },
};

const authoringCatalogResponse = {
  schema_version: 'vn_script_authoring_catalog.v1',
  program_schema_version: 'vn_script_program.v1',
  capability_tokens: ['script_authoring_catalog', 'choice_set'],
  generation_output_schemas: ['choice_set'],
  operation_categories: { choices: ['insert_choice_set'], assets: ['generate_asset'] },
  operations: [
    {
      op: 'insert_choice_set',
      label: 'Insert choice set',
      category: 'choices',
      capability_tokens: ['choice_set'],
    },
    {
      op: 'generate_asset',
      label: 'Generate asset',
      category: 'assets',
      capability_tokens: ['realtime_image_generation'],
    },
  ],
  snippets: [choiceSnippet, typedEnumSnippet, unsupportedSnippet],
  limits: { max_operations: 100 },
};

const graphResponse = {
  schema_version: 'vn_script_authoring_graph.v1',
  graph_semantics_version: 'vn_script_authoring_graph_edges.v1',
  program_schema_version: 'vn_script_program.v1',
  script_id: 1,
  source: 'stored_draft',
  base_revision: 3,
  version_id: null,
  content_hash: 'sha256:opening',
  validation_context_source: 'current_draft_context',
  truncated: false,
  limits: {
    max_labels: 500,
    max_ops: 5000,
    max_edges: 10000,
    max_supplied_draft_bytes: 1048576,
  },
  outline: {
    entry_label: 'start',
    labels: [
      {
        id: 'label:start',
        label: 'start',
        source_path: "$.labels['start']",
        op_count: 2,
        incoming_edge_count: 0,
        outgoing_edge_count: 1,
        reachable: true,
        terminal: 'continues',
        summary: 'Opening label',
      },
    ],
  },
  graph: {
    nodes: [
      {
        id: 'label:start',
        type: 'label',
        label: 'start',
        source_path: "$.labels['start']",
        reachable: true,
        terminal: 'continues',
        summary: 'Opening label',
      },
    ],
    edges: [],
  },
  diagnostics: {
    errors: [],
    warnings: [
      {
        code: 'graph_fallthrough_not_inferred',
        severity: 'warning',
        message: 'Fallthrough is not inferred.',
        path: "$.labels['start'][1]",
        details: { next_label: 'end' },
      },
    ],
  },
  validation_diagnostics: {
    valid: false,
    errors: [{ code: 'missing_target', message: 'Missing target.' }],
    warnings: [{ code: 'unused_label', message: 'Unused label.' }],
  },
};

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, reject, resolve };
}

function mockList(items = [openingScript, secondScript]) {
  mocks.listVNScripts.mockResolvedValue({
    items,
    limit: 25,
    offset: 0,
    total: items.length,
    has_more: false,
    pagination: { limit: 25, offset: 0, total: items.length, has_more: false },
  });
}

function mockTemplates(items = [linearTemplate, choiceTemplate]) {
  mocks.listVNScriptTemplates.mockResolvedValue({ items });
}

describe('VNScriptsWorkbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.apiClient.get.mockResolvedValue(vnCapabilitiesResponse);
    mockList();
    mockTemplates();
    mocks.getVNScriptAuthoringCatalog.mockResolvedValue(authoringCatalogResponse);
    mocks.previewVNScriptSnippet.mockResolvedValue({
      script_id: 1,
      base_revision: 3,
      snippet_id: 'generated_choice_set',
      draft: {
        scenes: [{ id: 'start', text: 'Wake up.' }],
        choices: [{ prompt: 'Offer three routes.' }],
      },
      diagnostics: { valid: true, preview_nodes: 2 },
      patch_summary: {
        inserted_ops: 2,
        created_labels: ['choice_intro'],
        changed_paths: ['labels.start[1]', 'labels.choice_intro'],
      },
      warnings: [{ code: 'preview_only', message: 'Preview only.' }],
    });
    mocks.applyVNScriptSnippet.mockResolvedValue({
      script_id: 1,
      revision: 4,
      snippet_id: 'generated_choice_set',
      draft: {
        scenes: [{ id: 'start', text: 'Wake up.' }],
        choices: [{ prompt: 'Offer three routes.' }],
      },
      diagnostics: { valid: true, applied_nodes: 2 },
      patch_summary: {
        inserted_ops: 2,
        created_labels: ['choice_intro'],
        changed_paths: ['labels.start[1]', 'labels.choice_intro'],
      },
    });
    mocks.createVNScript.mockResolvedValue({
      ...secondScript,
      id: 9,
      title: 'Created Route',
      primary_asset_pack_id: 44,
      policy_profile_id: 'policy-explicit',
      generation_profile_id: 'gen-explicit',
      content_rating: 'mature',
    });
    mocks.createVNScriptFromTemplate.mockResolvedValue({
      script: templateScript,
      draft: templateDraftResponse,
    });
    mocks.getVNScriptDraft.mockResolvedValue(draftResponse);
    mocks.getVNScriptDraftGraph.mockResolvedValue(graphResponse);
    mocks.previewVNScriptDraftGraph.mockResolvedValue({
      ...graphResponse,
      source: 'supplied_draft',
      content_hash: 'sha256:preview',
    });
    mocks.getVNScriptVersionGraph.mockResolvedValue({
      ...graphResponse,
      source: 'published_version',
      version_id: 12,
      validation_context_source: 'published_version_snapshot',
      content_hash: 'sha256:version',
    });
    mocks.putVNScriptDraft.mockResolvedValue({
      ...draftResponse,
      revision: 4,
      draft: { scenes: [{ id: 'edited' }] },
    });
    mocks.validateVNScriptDraft.mockResolvedValue({
      valid: false,
      errors: [{ code: 'missing_start', message: 'Missing start node' }],
      warnings: [{ code: 'unused_scene', message: 'Unused scene' }],
    });
    mocks.getVNScriptDiagnostics.mockResolvedValue({
      script_id: 1,
      revision: 3,
      diagnostics: {
        graph: { nodes: 1 },
        raw_debug_payload: 'secret raw payload',
        unreachable: ['unused'],
      },
    });
    mocks.publishVNScript.mockResolvedValue({
      script_id: 1,
      version_id: 13,
      version_number: 3,
      status: 'published',
      asset_pack_id: 7,
      manifest_snapshot_id: 111,
      policy_snapshot_id: 222,
      generation_profile_snapshot_id: 333,
      generation_profile_snapshots: {},
      validation: { valid: true },
      created_at: '2026-05-12T10:20:00Z',
    });
    mocks.listVNScriptVersions.mockResolvedValue({
      items: [versionResponse],
      limit: 25,
      offset: 0,
      total: 1,
      has_more: false,
      pagination: { limit: 25, offset: 0, total: 1, has_more: false },
    });
    mocks.getVNScriptManifestSnapshot.mockResolvedValue({
      id: 101,
      script_id: 1,
      version_id: 12,
      asset_pack_id: 7,
      manifest: { slots: ['background.interior'], raw_prompt: 'hidden prompt text' },
      manifest_hash: 'hash-101',
      created_at: '2026-05-12T10:15:00Z',
    });
    mocks.evaluateVNScriptVersionPolicy.mockResolvedValue({
      decision: 'allow',
      profile_id: 'teen-policy',
      reasons: [{ code: 'ok', internal_notes: 'hidden policy details' }],
      blocked: false,
      requires_acknowledgement: false,
      remediation: [],
    });
  });

  it('loads scripts on mount, selects the first script, and renders its draft and versions', async () => {
    render(<VNScriptsWorkbench />);

    expect(await screen.findByRole('button', { name: /Opening Route/ })).toBeInTheDocument();
    expect(mocks.listVNScripts).toHaveBeenCalledWith({ limit: 25, offset: 0 });
    await waitFor(() => expect(mocks.getVNScriptDraft).toHaveBeenCalledWith(1));
    await waitFor(() => expect(mocks.listVNScriptVersions).toHaveBeenCalledWith(1));

    expect(screen.getByText('Second Route')).toBeInTheDocument();
    expect(screen.getByText('Script #1')).toBeInTheDocument();
    expect(screen.getByDisplayValue(/Wake up/)).toBeInTheDocument();
    expect(screen.getByText('Version 2')).toBeInTheDocument();
    expect(screen.getByText('Launch candidate')).toBeInTheDocument();
    expect(screen.getByText('Manifest 101')).toBeInTheDocument();
    expect(screen.getByText('Policy 202')).toBeInTheDocument();
    expect(screen.getByText('Generation 303')).toBeInTheDocument();
    expect(screen.getByText(/2026-05-12T10:15:00Z/)).toBeInTheDocument();
  });

  it('loads and renders the saved draft graph outline with graph diagnostics separated from validation diagnostics', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Load saved graph' }));

    await waitFor(() => expect(mocks.getVNScriptDraftGraph).toHaveBeenCalledWith(1));
    expect(screen.getByText('Script graph')).toBeInTheDocument();
    expect(screen.getByText('stored_draft')).toBeInTheDocument();
    expect(screen.getByText('vn_script_authoring_graph.v1')).toBeInTheDocument();
    expect(screen.getByText('vn_script_program.v1')).toBeInTheDocument();
    expect(screen.getByText('sha256:opening')).toBeInTheDocument();
    expect(screen.getByText('Opening label')).toBeInTheDocument();
    expect(screen.getByText(/graph_fallthrough_not_inferred/)).toBeInTheDocument();
    expect(screen.getByText('Graph diagnostics')).toBeInTheDocument();
    expect(screen.getByText('Validation diagnostics')).toBeInTheDocument();
    expect(screen.getByText(/missing_target/)).toBeInTheDocument();
  });

  it('lets authors select an outline source path and highlights it near the draft editor', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Load saved graph' }));
    await user.click(await screen.findByRole('button', { name: /Select source path for start/ }));

    expect(screen.getByText(/Selected graph path/)).toBeInTheDocument();
    expect(screen.getAllByText("$.labels['start']").length).toBeGreaterThan(0);
    expect(screen.getByText(/Selected graph path/).closest('div')).toHaveClass('bg-primary/10');
  });

  it('shows graph loading and error states for failed graph requests', async () => {
    const user = userEvent.setup();
    const graphRequest = createDeferred<typeof graphResponse>();
    mocks.getVNScriptDraftGraph.mockReturnValueOnce(graphRequest.promise);
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Load saved graph' }));
    expect(screen.getByRole('button', { name: 'Load saved graph' })).toBeDisabled();

    graphRequest.reject(new Error('graph_backend_down'));

    expect(await screen.findByText('graph_backend_down')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Load saved graph' })).toBeEnabled();
  });

  it('clears graph loading state when switching scripts while a graph request is in flight', async () => {
    const user = userEvent.setup();
    const graphRequest = createDeferred<typeof graphResponse>();
    mocks.getVNScriptDraftGraph.mockReturnValueOnce(graphRequest.promise);
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Load saved graph' }));
    expect(screen.getByRole('button', { name: 'Load saved graph' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: /Second Route/ }));

    await waitFor(() => expect(mocks.getVNScriptDraft).toHaveBeenCalledWith(2));
    expect(screen.getByRole('button', { name: 'Load saved graph' })).toBeEnabled();
  });

  it('ignores stale saved graph responses after a newer preview graph response wins', async () => {
    const user = userEvent.setup();
    const savedRequest = createDeferred<typeof graphResponse>();
    const previewRequest = createDeferred<typeof graphResponse>();
    mocks.getVNScriptDraftGraph.mockReturnValueOnce(savedRequest.promise);
    mocks.previewVNScriptDraftGraph.mockReturnValueOnce(previewRequest.promise);
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Load saved graph' }));
    fireEvent.change(screen.getByLabelText('Draft JSON'), {
      target: { value: '{"labels":{"start":[{"op":"narrate","text":"Changed."}]}}' },
    });
    await user.click(screen.getByRole('button', { name: 'Preview current JSON graph' }));

    previewRequest.resolve({
      ...graphResponse,
      source: 'supplied_draft',
      content_hash: 'sha256:newer-preview',
      outline: {
        ...graphResponse.outline,
        labels: [{ ...graphResponse.outline.labels[0], summary: 'Newer preview label' }],
      },
    });
    await screen.findByText('sha256:newer-preview');

    savedRequest.resolve({
      ...graphResponse,
      content_hash: 'sha256:stale-saved',
      outline: {
        ...graphResponse.outline,
        labels: [{ ...graphResponse.outline.labels[0], summary: 'Stale saved label' }],
      },
    });

    await waitFor(() => expect(screen.queryByText('sha256:stale-saved')).not.toBeInTheDocument());
    expect(screen.getByText('Newer preview label')).toBeInTheDocument();
    expect(screen.queryByText('Stale saved label')).not.toBeInTheDocument();
  });

  it('previews the current unsaved draft graph without saving the draft', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    const editor = await screen.findByLabelText('Draft JSON');
    fireEvent.change(editor, {
      target: { value: '{"labels":{"start":[{"op":"narrate","text":"Changed."}]}}' },
    });
    await user.click(screen.getByRole('button', { name: 'Preview current JSON graph' }));

    await waitFor(() => expect(mocks.previewVNScriptDraftGraph).toHaveBeenCalledWith(1, {
      draft_revision: 3,
      draft: { labels: { start: [{ op: 'narrate', text: 'Changed.' }] } },
    }));
    expect(mocks.putVNScriptDraft).not.toHaveBeenCalled();
    expect(screen.getByText('supplied_draft')).toBeInTheDocument();
    expect(screen.getByText('sha256:preview')).toBeInTheDocument();
  });

  it('keeps the graph inspector hidden when backend capabilities disable the graph feature', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      ...vnCapabilitiesResponse,
      features: { ...vnCapabilitiesResponse.features, script_authoring_graph: false },
    });

    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    expect(screen.queryByRole('button', { name: 'Load saved graph' })).not.toBeInTheDocument();
    expect(mocks.getVNScriptDraftGraph).not.toHaveBeenCalled();
  });

  it('loads a published version graph from the version card', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByText('Version 2');
    await user.click(screen.getByRole('button', { name: 'Graph for version 2' }));

    await waitFor(() => expect(mocks.getVNScriptVersionGraph).toHaveBeenCalledWith(1, 12));
    const versionCard = screen.getByTestId('version-12');
    expect(within(versionCard).getByText('published_version')).toBeInTheDocument();
    expect(within(versionCard).getByText('sha256:version')).toBeInTheDocument();
    expect(within(versionCard).getByText('published_version_snapshot')).toBeInTheDocument();
  });

  it('creates a script shell, prepends it, selects it, and loads its details', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Created Route');
    await user.clear(screen.getByLabelText('Primary asset pack ID'));
    await user.type(screen.getByLabelText('Primary asset pack ID'), '44');
    await user.clear(screen.getByLabelText('Policy profile ID'));
    await user.type(screen.getByLabelText('Policy profile ID'), 'policy-explicit');
    await user.clear(screen.getByLabelText('Generation profile ID'));
    await user.type(screen.getByLabelText('Generation profile ID'), 'gen-explicit');
    await user.selectOptions(screen.getByLabelText('Content rating'), 'mature');
    await user.click(screen.getByRole('button', { name: 'Create script' }));

    await waitFor(() => {
      expect(mocks.createVNScript).toHaveBeenCalledWith({
        title: 'Created Route',
        primary_asset_pack_id: 44,
        policy_profile_id: 'policy-explicit',
        generation_profile_id: 'gen-explicit',
        content_rating: 'mature',
      });
    });
    expect(await screen.findByText('Script #9')).toBeInTheDocument();
    expect(mocks.getVNScriptDraft).toHaveBeenCalledWith(9);
    expect(mocks.listVNScriptVersions).toHaveBeenCalledWith(9);
  });

  it('loads templates on mount and renders starter options', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    expect(await screen.findByLabelText('Starter template')).toBeInTheDocument();
    await waitFor(() => expect(mocks.listVNScriptTemplates).toHaveBeenCalledTimes(1));
    expect(screen.getByRole('option', { name: 'Blank/custom JSON' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Linear scene' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Authored choices' })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText('Starter template'), 'linear_scene');
    expect(screen.getByText('Start with one authored scene.')).toBeInTheDocument();
  });

  it('loads the authoring catalog alongside script data when VN capabilities enables it', async () => {
    render(<VNScriptsWorkbench />);

    expect(await screen.findByRole('button', { name: /Opening Route/ })).toBeInTheDocument();
    await waitFor(() => expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-capabilities'));
    await waitFor(() => expect(mocks.getVNScriptAuthoringCatalog).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole('heading', { name: 'Guided insert' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generated choice set/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Live asset generation/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Draft JSON')).toBeInTheDocument();
  });

  it('keeps guided insert hidden when capabilities disable the authoring catalog while raw JSON remains available', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      ...vnCapabilitiesResponse,
      features: { ...vnCapabilitiesResponse.features, script_authoring_catalog: false },
    });
    render(<VNScriptsWorkbench />);

    expect(await screen.findByLabelText('Draft JSON')).toBeInTheDocument();
    await waitFor(() => expect(mocks.apiClient.get).toHaveBeenCalledWith('/vn/vn-capabilities'));
    expect(mocks.getVNScriptAuthoringCatalog).not.toHaveBeenCalled();
    expect(screen.queryByRole('heading', { name: 'Guided insert' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save draft' })).toBeEnabled();
  });

  it('keeps raw JSON usable and shows non-blocking status when catalog loading fails', async () => {
    mocks.getVNScriptAuthoringCatalog.mockRejectedValueOnce(new Error('catalog offline'));
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    const editor = await screen.findByLabelText('Draft JSON');
    expect(await screen.findByText('Guided insert catalog unavailable. Raw JSON editing remains available.')).toBeInTheDocument();
    fireEvent.change(editor, { target: { value: JSON.stringify({ scenes: [{ id: 'manual' }] }) } });
    await user.click(screen.getByRole('button', { name: 'Save draft' }));

    await waitFor(() => {
      expect(mocks.putVNScriptDraft).toHaveBeenCalledWith(1, {
        if_revision: 3,
        draft: { scenes: [{ id: 'manual' }] },
      });
    });
  });

  it('renders simple snippet parameter inputs from schema and defaults after selecting a snippet', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));

    expect(screen.getByLabelText('Prompt')).toHaveValue('Offer three routes.');
    expect(screen.getByLabelText('Choice count')).toHaveValue(3);
    expect(screen.getByLabelText('Shuffle choices')).not.toBeChecked();
    expect((screen.getByRole('option', { name: 'dramatic' }) as HTMLOptionElement).selected).toBe(true);

    await user.clear(screen.getByLabelText('Prompt'));
    await user.type(screen.getByLabelText('Prompt'), 'Offer two clues.');
    await user.clear(screen.getByLabelText('Choice count'));
    await user.type(screen.getByLabelText('Choice count'), '2');
    await user.click(screen.getByLabelText('Shuffle choices'));
    await user.selectOptions(screen.getByLabelText('Tone'), screen.getByRole('option', { name: 'quiet' }));
    expect(screen.getByLabelText('Prompt')).toHaveValue('Offer two clues.');
    expect(screen.getByLabelText('Choice count')).toHaveValue(2);
    expect(screen.getByLabelText('Shuffle choices')).toBeChecked();
    expect((screen.getByRole('option', { name: 'quiet' }) as HTMLOptionElement).selected).toBe(true);
  });

  it('preserves numeric and boolean enum parameter values when previewing', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Typed enum snippet/ }));
    expect((screen.getByRole('option', { name: '2' }) as HTMLOptionElement).selected).toBe(true);
    expect((screen.getByRole('option', { name: 'true' }) as HTMLOptionElement).selected).toBe(true);

    await user.selectOptions(screen.getByLabelText('Difficulty'), screen.getByRole('option', { name: '3' }));
    await user.selectOptions(screen.getByLabelText('Include recap'), screen.getByRole('option', { name: 'false' }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));

    await waitFor(() => {
      expect(mocks.previewVNScriptSnippet).toHaveBeenCalledWith(1, expect.objectContaining({
        snippet_id: 'typed_enum_snippet',
        parameters: {
          difficulty: 3,
          includeRecap: false,
        },
      }));
    });
  });

  it('previews a selected snippet without updating the stored draft revision', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.selectOptions(screen.getByLabelText('Anchor mode'), 'before');
    await user.clear(screen.getByLabelText('Op index'));
    await user.type(screen.getByLabelText('Op index'), '2');
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));

    await waitFor(() => {
      expect(mocks.previewVNScriptSnippet).toHaveBeenCalledWith(1, {
        snippet_id: 'generated_choice_set',
        anchor: { label: 'start', mode: 'before', op_index: 2 },
        parameters: {
          prompt: 'Offer three routes.',
          count: 3,
          shuffle: false,
          tone: 'dramatic',
        },
        draft: { scenes: [{ id: 'start', text: 'Wake up.' }] },
        draft_revision: 3,
      });
    });
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    expect(screen.getByText(/labels.start\[1\]/)).toBeInTheDocument();
    expect(screen.getByText(/preview_nodes/)).toBeInTheDocument();
    expect(screen.getByText(/Revision 3/)).toBeInTheDocument();
    expect(screen.queryByDisplayValue(/choices/)).not.toBeInTheDocument();
  });

  it('applies a selected snippet with current draft revision and updates the draft from the backend response', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));

    await waitFor(() => {
      expect(mocks.applyVNScriptSnippet).toHaveBeenCalledWith(1, {
        if_revision: 3,
        snippet_id: 'generated_choice_set',
        anchor: { label: 'start', mode: 'append', op_index: null },
        parameters: {
          prompt: 'Offer three routes.',
          count: 3,
          shuffle: false,
          tone: 'dramatic',
        },
      });
    });
    expect(await screen.findByText('Applied snippet at revision 4.')).toBeInTheDocument();
    expect(screen.getByText(/labels.choice_intro/)).toBeInTheDocument();
    expect(screen.getByDisplayValue(/choices/)).toBeInTheDocument();
    expect(screen.getByText(/Revision 4/)).toBeInTheDocument();
    expect(screen.getAllByText(/applied_nodes/).length).toBeGreaterThan(0);
  });

  it('invalidates snippet preview before apply when parameters change after preview', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeEnabled();

    await user.clear(screen.getByLabelText('Prompt'));
    await user.type(screen.getByLabelText('Prompt'), 'Offer a revised route.');

    expect(screen.queryByText(/Preview inserted 2 operations/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    expect(mocks.applyVNScriptSnippet).not.toHaveBeenCalled();
  });

  it('keeps apply disabled after previewing unsaved raw JSON edits', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    const editor = screen.getByLabelText('Draft JSON');
    fireEvent.change(editor, {
      target: { value: JSON.stringify({ scenes: [{ id: 'unsaved', text: 'Unsaved buffer.' }] }) },
    });
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));

    await waitFor(() => {
      expect(mocks.previewVNScriptSnippet).toHaveBeenCalledWith(1, expect.objectContaining({
        draft: { scenes: [{ id: 'unsaved', text: 'Unsaved buffer.' }] },
      }));
    });
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    expect(mocks.applyVNScriptSnippet).not.toHaveBeenCalled();
  });

  it('ignores stale snippet preview responses after raw JSON changes while preview is in flight', async () => {
    const user = userEvent.setup();
    const preview = createDeferred<{
      script_id: number;
      base_revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
      warnings: Array<Record<string, unknown>>;
    }>();
    mocks.previewVNScriptSnippet.mockReturnValueOnce(preview.promise);
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    await waitFor(() => expect(mocks.previewVNScriptSnippet).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText('Draft JSON'), {
      target: { value: JSON.stringify({ scenes: [{ id: 'changed', text: 'Changed while previewing.' }] }) },
    });
    preview.resolve({
      script_id: 1,
      base_revision: 3,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'start' }], choices: [{ prompt: 'stale preview' }] },
      diagnostics: { valid: true, preview_nodes: 2 },
      patch_summary: {
        inserted_ops: 2,
        created_labels: ['stale_preview'],
        changed_paths: ['labels.start[1]'],
      },
      warnings: [],
    });

    await waitFor(() => expect(screen.queryByText(/stale_preview/)).not.toBeInTheDocument());
    expect(screen.queryByText(/Preview inserted 2 operations/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeDisabled();
  });

  it('ignores older snippet preview responses when raw JSON changed and a newer preview is current', async () => {
    const user = userEvent.setup();
    const firstPreview = createDeferred<{
      script_id: number;
      base_revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
      warnings: Array<Record<string, unknown>>;
    }>();
    const secondPreview = createDeferred<{
      script_id: number;
      base_revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
      warnings: Array<Record<string, unknown>>;
    }>();
    mocks.previewVNScriptSnippet
      .mockReturnValueOnce(firstPreview.promise)
      .mockReturnValueOnce(secondPreview.promise);
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    await waitFor(() => expect(mocks.previewVNScriptSnippet).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText('Draft JSON'), {
      target: { value: JSON.stringify({ scenes: [{ id: 'changed', text: 'Changed before preview B.' }] }) },
    });
    await waitFor(() => expect(screen.getByDisplayValue(/Changed before preview B/)).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    await waitFor(() => expect(mocks.previewVNScriptSnippet).toHaveBeenCalledTimes(2));

    secondPreview.resolve({
      script_id: 1,
      base_revision: 3,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'changed' }], choices: [{ prompt: 'current preview' }] },
      diagnostics: { valid: true, preview_nodes: 2 },
      patch_summary: {
        inserted_ops: 2,
        created_labels: ['current_preview'],
        changed_paths: ['labels.current_preview[1]'],
      },
      warnings: [],
    });
    expect(await screen.findByText(/current_preview/)).toBeInTheDocument();

    firstPreview.resolve({
      script_id: 1,
      base_revision: 3,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'start' }], choices: [{ prompt: 'stale preview' }] },
      diagnostics: { valid: true, preview_nodes: 2 },
      patch_summary: {
        inserted_ops: 1,
        created_labels: ['stale_preview_a'],
        changed_paths: ['labels.stale_preview_a[1]'],
      },
      warnings: [],
    });

    await waitFor(() => expect(screen.queryByText(/stale_preview_a/)).not.toBeInTheDocument());
    expect(screen.getByText(/current_preview/)).toBeInTheDocument();
  });

  it('keeps preview loading active when an older preview resolves before the current one', async () => {
    const user = userEvent.setup();
    const firstPreview = createDeferred<{
      script_id: number;
      base_revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
      warnings: Array<Record<string, unknown>>;
    }>();
    const secondPreview = createDeferred<{
      script_id: number;
      base_revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
      warnings: Array<Record<string, unknown>>;
    }>();
    mocks.previewVNScriptSnippet
      .mockReturnValueOnce(firstPreview.promise)
      .mockReturnValueOnce(secondPreview.promise);
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    await waitFor(() => expect(mocks.previewVNScriptSnippet).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText('Draft JSON'), {
      target: { value: JSON.stringify({ scenes: [{ id: 'changed', text: 'Changed before preview B.' }] }) },
    });
    await waitFor(() => expect(screen.getByDisplayValue(/Changed before preview B/)).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    await waitFor(() => expect(mocks.previewVNScriptSnippet).toHaveBeenCalledTimes(2));

    firstPreview.resolve({
      script_id: 1,
      base_revision: 3,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'start' }], choices: [{ prompt: 'stale preview' }] },
      diagnostics: { valid: true },
      patch_summary: { inserted_ops: 1, created_labels: ['stale_preview'], changed_paths: [] },
      warnings: [],
    });

    await waitFor(() => expect(screen.getByRole('button', { name: 'Preview snippet' })).toHaveAttribute('aria-busy', 'true'));
    secondPreview.resolve({
      script_id: 1,
      base_revision: 3,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'changed' }], choices: [{ prompt: 'current preview' }] },
      diagnostics: { valid: true },
      patch_summary: { inserted_ops: 1, created_labels: ['current_preview'], changed_paths: ['labels.current_preview[1]'] },
      warnings: [],
    });
    expect(await screen.findByText(/current_preview/)).toBeInTheDocument();
  });

  it('ignores stale snippet apply responses after switching scripts', async () => {
    const user = userEvent.setup();
    const apply = createDeferred<{
      script_id: number;
      revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
    }>();
    mocks.applyVNScriptSnippet.mockReturnValueOnce(apply.promise);
    mocks.getVNScriptDraft.mockImplementation(async (scriptId: number) =>
      scriptId === 2
        ? { ...draftResponse, script_id: 2, draft: { scenes: [{ id: 'second', text: 'Look around.' }] } }
        : draftResponse
    );
    mocks.listVNScriptVersions.mockImplementation((scriptId: number) =>
      Promise.resolve({
        items: scriptId === 2 ? [secondVersionResponse] : [versionResponse],
        limit: 25,
        offset: 0,
        total: 1,
        has_more: false,
        pagination: { limit: 25, offset: 0, total: 1, has_more: false },
      })
    );
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    await waitFor(() => expect(mocks.applyVNScriptSnippet).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole('button', { name: /Second Route/ }));
    await screen.findByText('Script #2');
    apply.resolve({
      script_id: 1,
      revision: 4,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'start' }], choices: [{ prompt: 'stale' }] },
      diagnostics: { valid: true, stale_nodes: 2 },
      patch_summary: {
        inserted_ops: 2,
        created_labels: ['stale_choice'],
        changed_paths: ['labels.start[1]'],
      },
    });

    await waitFor(() => expect(screen.getByDisplayValue(/Look around/)).toBeInTheDocument());
    expect(screen.queryByDisplayValue(/stale/)).not.toBeInTheDocument();
    expect(screen.queryByText('Applied snippet at revision 4.')).not.toBeInTheDocument();
  });

  it('ignores apply responses after same-script raw JSON edits invalidate the preview context', async () => {
    const user = userEvent.setup();
    const apply = createDeferred<{
      script_id: number;
      revision: number;
      snippet_id: string;
      draft: Record<string, unknown>;
      diagnostics: Record<string, unknown>;
      patch_summary: { inserted_ops: number; created_labels: string[]; changed_paths: string[] };
    }>();
    mocks.applyVNScriptSnippet.mockReturnValueOnce(apply.promise);
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    await waitFor(() => expect(mocks.applyVNScriptSnippet).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText('Draft JSON'), {
      target: { value: JSON.stringify({ scenes: [{ id: 'local', text: 'Keep local edit.' }] }) },
    });
    apply.resolve({
      script_id: 1,
      revision: 4,
      snippet_id: 'generated_choice_set',
      draft: { scenes: [{ id: 'server' }], choices: [{ prompt: 'server response' }] },
      diagnostics: { valid: true },
      patch_summary: { inserted_ops: 1, created_labels: [], changed_paths: [] },
    });

    await waitFor(() => expect(screen.getByDisplayValue(/Keep local edit/)).toBeInTheDocument());
    expect(screen.queryByDisplayValue(/server response/)).not.toBeInTheDocument();
    expect(screen.queryByText('Applied snippet at revision 4.')).not.toBeInTheDocument();
  });

  it('ignores stale conflict reload responses after switching scripts', async () => {
    const user = userEvent.setup();
    const conflict = new Error('[object Object]') as Error & {
      detail?: { code?: string; message?: string; details?: { reason?: string } };
      status?: number;
    };
    const conflictReload = createDeferred<typeof draftResponse>();
    let openingDraftCalls = 0;
    conflict.status = 409;
    conflict.detail = {
      code: 'invalid_request',
      message: 'draft_revision_conflict',
      details: { reason: 'draft_revision_conflict' },
    };
    mocks.applyVNScriptSnippet.mockRejectedValueOnce(conflict);
    mocks.getVNScriptDraft.mockImplementation(async (scriptId: number) => {
      if (scriptId === 2) {
        return { ...draftResponse, script_id: 2, draft: { scenes: [{ id: 'second', text: 'Look around.' }] } };
      }
      openingDraftCalls += 1;
      return openingDraftCalls === 1
        ? draftResponse
        : conflictReload.promise;
    });
    mocks.listVNScriptVersions.mockImplementation((scriptId: number) =>
      Promise.resolve({
        items: scriptId === 2 ? [secondVersionResponse] : [versionResponse],
        limit: 25,
        offset: 0,
        total: 1,
        has_more: false,
        pagination: { limit: 25, offset: 0, total: 1, has_more: false },
      })
    );
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    await waitFor(() => expect(mocks.getVNScriptDraft).toHaveBeenCalledTimes(2));

    await user.click(screen.getByRole('button', { name: /Second Route/ }));
    await screen.findByText('Script #2');
    conflictReload.resolve({
      ...draftResponse,
      revision: 5,
      draft: { scenes: [{ id: 'conflict_reload', text: 'Old script reload.' }] },
    });

    await waitFor(() => expect(screen.getByDisplayValue(/Look around/)).toBeInTheDocument());
    expect(screen.queryByDisplayValue(/conflict_reload/)).not.toBeInTheDocument();
    expect(screen.queryByText('Draft changed on the server. Reloaded the latest draft; review before applying again.')).not.toBeInTheDocument();
  });

  it('ignores stale conflict reload errors after switching scripts', async () => {
    const user = userEvent.setup();
    const conflict = new Error('[object Object]') as Error & {
      detail?: { code?: string; message?: string; details?: { reason?: string } };
      status?: number;
    };
    const conflictReload = createDeferred<typeof draftResponse>();
    let openingDraftCalls = 0;
    conflict.status = 409;
    conflict.detail = {
      code: 'invalid_request',
      message: 'draft_revision_conflict',
      details: { reason: 'draft_revision_conflict' },
    };
    mocks.applyVNScriptSnippet.mockRejectedValueOnce(conflict);
    mocks.getVNScriptDraft.mockImplementation(async (scriptId: number) => {
      if (scriptId === 2) {
        return { ...draftResponse, script_id: 2, draft: { scenes: [{ id: 'second', text: 'Look around.' }] } };
      }
      openingDraftCalls += 1;
      return openingDraftCalls === 1
        ? draftResponse
        : conflictReload.promise;
    });
    mocks.listVNScriptVersions.mockImplementation((scriptId: number) =>
      Promise.resolve({
        items: scriptId === 2 ? [secondVersionResponse] : [versionResponse],
        limit: 25,
        offset: 0,
        total: 1,
        has_more: false,
        pagination: { limit: 25, offset: 0, total: 1, has_more: false },
      })
    );
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    await waitFor(() => expect(mocks.getVNScriptDraft).toHaveBeenCalledTimes(2));

    await user.click(screen.getByRole('button', { name: /Second Route/ }));
    await screen.findByText('Script #2');
    conflictReload.reject(new Error('reload_failed'));

    await waitFor(() => expect(screen.getByDisplayValue(/Look around/)).toBeInTheDocument());
    expect(screen.queryByText('Draft changed on the server. Refresh the draft before applying again.')).not.toBeInTheDocument();
  });

  it('invalidates snippet preview before apply when switching scripts with the same draft revision', async () => {
    const user = userEvent.setup();
    mocks.getVNScriptDraft.mockImplementation(async (scriptId: number) =>
      scriptId === 2
        ? { ...draftResponse, script_id: 2, draft: { scenes: [{ id: 'second', text: 'Look around.' }] } }
        : draftResponse
    );
    mocks.listVNScriptVersions.mockImplementation((scriptId: number) =>
      Promise.resolve({
        items: scriptId === 2 ? [secondVersionResponse] : [versionResponse],
        limit: 25,
        offset: 0,
        total: 1,
        has_more: false,
        pagination: { limit: 25, offset: 0, total: 1, has_more: false },
      })
    );
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeEnabled();

    await user.click(screen.getByRole('button', { name: /Second Route/ }));
    await screen.findByText('Script #2');

    expect(screen.queryByText(/Preview inserted 2 operations/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));
    expect(mocks.applyVNScriptSnippet).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    await waitFor(() => {
      expect(mocks.previewVNScriptSnippet).toHaveBeenLastCalledWith(2, expect.objectContaining({
        draft_revision: 3,
      }));
    });
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Apply snippet' })).toBeEnabled();
  });

  it('handles draft revision conflicts without duplicating snippet content', async () => {
    const user = userEvent.setup();
    const conflict = new Error('[object Object]') as Error & {
      detail?: { code?: string; message?: string; details?: { reason?: string } };
      status?: number;
    };
    conflict.status = 409;
    conflict.detail = {
      code: 'invalid_request',
      message: 'draft_revision_conflict',
      details: { reason: 'draft_revision_conflict' },
    };
    mocks.applyVNScriptSnippet.mockRejectedValueOnce(conflict);
    render(<VNScriptsWorkbench />);

    await user.click(await screen.findByRole('button', { name: /Generated choice set/ }));
    await user.click(screen.getByRole('button', { name: 'Preview snippet' }));
    expect(await screen.findByText(/Preview inserted 2 operations/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply snippet' }));

    expect(await screen.findByText('Draft changed on the server. Reloaded the latest draft; review before applying again.')).toBeInTheDocument();
    await waitFor(() => expect(mocks.getVNScriptDraft).toHaveBeenCalledTimes(2));
    expect(screen.getByDisplayValue(/Wake up/)).toBeInTheDocument();
    expect(screen.queryByDisplayValue(/choices/)).not.toBeInTheDocument();
  });

  it('creates from a selected template and shows the returned draft immediately', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.selectOptions(screen.getByLabelText('Starter template'), 'linear_scene');
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Template Route');
    await user.clear(screen.getByLabelText('Primary asset pack ID'));
    await user.type(screen.getByLabelText('Primary asset pack ID'), '55');
    await user.clear(screen.getByLabelText('Policy profile ID'));
    await user.type(screen.getByLabelText('Policy profile ID'), 'teen-policy');
    await user.clear(screen.getByLabelText('Generation profile ID'));
    await user.type(screen.getByLabelText('Generation profile ID'), 'story-default');
    await user.selectOptions(screen.getByLabelText('Content rating'), 'general');
    await user.click(screen.getByRole('button', { name: 'Create script' }));

    await waitFor(() => {
      expect(mocks.createVNScriptFromTemplate).toHaveBeenCalledWith('linear_scene', {
        title: 'Template Route',
        description: 'A simple authored opener.',
        primary_asset_pack_id: 55,
        policy_profile_id: 'teen-policy',
        generation_profile_id: 'story-default',
        content_rating: 'general',
      });
    });
    expect(mocks.createVNScript).not.toHaveBeenCalled();
    expect(await screen.findByText('Script #21')).toBeInTheDocument();
    expect(screen.getByDisplayValue(/template_start/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Template Route/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save draft' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Validate' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Diagnostics' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Publish' })).toBeEnabled();
  });

  it('keeps the blank custom JSON path on the normal create endpoint', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.selectOptions(screen.getByLabelText('Starter template'), 'blank');
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Blank Route');
    await user.clear(screen.getByLabelText('Primary asset pack ID'));
    await user.type(screen.getByLabelText('Primary asset pack ID'), '46');
    await user.click(screen.getByRole('button', { name: 'Create script' }));

    await waitFor(() => {
      expect(mocks.createVNScript).toHaveBeenCalledWith({
        title: 'Blank Route',
        primary_asset_pack_id: 46,
        content_rating: 'teen',
      });
    });
    expect(mocks.createVNScriptFromTemplate).not.toHaveBeenCalled();
  });

  it('omits empty optional profile IDs when creating a script shell', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Default Profiles Route');
    await user.clear(screen.getByLabelText('Primary asset pack ID'));
    await user.type(screen.getByLabelText('Primary asset pack ID'), '45');
    await user.click(screen.getByRole('button', { name: 'Create script' }));

    await waitFor(() => {
      expect(mocks.createVNScript).toHaveBeenCalledWith({
        title: 'Default Profiles Route',
        primary_asset_pack_id: 45,
        content_rating: 'teen',
      });
    });
  });

  it('saves parsed draft JSON with the current revision', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    const editor = await screen.findByLabelText('Draft JSON');
    fireEvent.change(editor, { target: { value: JSON.stringify({ scenes: [{ id: 'edited' }] }) } });
    await user.click(screen.getByRole('button', { name: 'Save draft' }));

    await waitFor(() => {
      expect(mocks.putVNScriptDraft).toHaveBeenCalledWith(1, {
        if_revision: 3,
        draft: { scenes: [{ id: 'edited' }] },
      });
    });
    expect(await screen.findByText('Draft saved at revision 4.')).toBeInTheDocument();
  });

  it('blocks draft save on invalid local JSON without calling the backend', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    const editor = await screen.findByLabelText('Draft JSON');
    fireEvent.change(editor, { target: { value: '{"scenes":' } });
    await user.click(screen.getByRole('button', { name: 'Save draft' }));

    expect(await screen.findByText(/Draft JSON is invalid/)).toBeInTheDocument();
    expect(mocks.putVNScriptDraft).not.toHaveBeenCalled();
  });

  it('blocks validation on invalid local JSON without validating stale saved content', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    const editor = await screen.findByLabelText('Draft JSON');
    fireEvent.change(editor, { target: { value: '{"scenes":' } });
    await user.click(screen.getByRole('button', { name: 'Validate' }));

    expect(await screen.findByText(/Draft JSON is invalid/)).toBeInTheDocument();
    expect(mocks.validateVNScriptDraft).not.toHaveBeenCalled();
  });

  it('validates draft JSON and renders backend errors and warnings', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByLabelText('Draft JSON');
    await user.click(screen.getByRole('button', { name: 'Validate' }));

    await waitFor(() => {
      expect(mocks.validateVNScriptDraft).toHaveBeenCalledWith(1, {
        draft: { scenes: [{ id: 'start', text: 'Wake up.' }] },
      });
    });
    expect(await screen.findByText('Invalid')).toBeInTheDocument();
    expect(screen.getByText(/missing_start/)).toBeInTheDocument();
    expect(screen.getByText(/unused_scene/)).toBeInTheDocument();
  });

  it('loads diagnostics and renders a sanitized backend summary', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Diagnostics' }));

    await waitFor(() => expect(mocks.getVNScriptDiagnostics).toHaveBeenCalledWith(1));
    expect(await screen.findByText(/unreachable/)).toBeInTheDocument();
    expect(screen.getByText(/unused/)).toBeInTheDocument();
    expect(screen.getByText(/\[redacted\]/)).toBeInTheDocument();
    expect(screen.queryByText(/secret raw payload/)).not.toBeInTheDocument();
  });

  it('publishes with an idempotency key and current draft revision without inferred acknowledgements', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.type(screen.getByLabelText('Publish label'), 'Playable v1');
    await user.click(screen.getByRole('button', { name: 'Publish' }));

    await waitFor(() => {
      expect(mocks.publishVNScript).toHaveBeenCalledWith(1, {
        draft_revision: 3,
        label: 'Playable v1',
        idempotency_key: expect.stringMatching(/^vn-script-publish-1-3-/),
        acknowledgements: [],
      });
    });
    expect(mocks.listVNScriptVersions).toHaveBeenCalledWith(1);
  });

  it('reuses the draft-scoped publish idempotency key when a publish is retried', async () => {
    const user = userEvent.setup();
    mocks.publishVNScript
      .mockRejectedValueOnce(new Error('script_publish_acknowledgement_required'))
      .mockResolvedValueOnce({
        script_id: 1,
        version_id: 13,
        version_number: 3,
        status: 'published',
        asset_pack_id: 7,
        manifest_snapshot_id: 111,
        policy_snapshot_id: 222,
        generation_profile_snapshot_id: 333,
        generation_profile_snapshots: {},
        validation: { valid: true },
        created_at: '2026-05-12T10:20:00Z',
      });
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Publish' }));
    expect(await screen.findByText('script_publish_acknowledgement_required')).toBeInTheDocument();
    const firstKey = mocks.publishVNScript.mock.calls[0][1].idempotency_key;

    await user.click(screen.getByRole('button', { name: 'Publish' }));
    await waitFor(() => expect(mocks.publishVNScript).toHaveBeenCalledTimes(2));
    expect(mocks.publishVNScript.mock.calls[1][1].idempotency_key).toBe(firstKey);
  });

  it('keeps the optimistic publish result when version refresh fails after publish succeeds', async () => {
    const user = userEvent.setup();
    let versionListCalls = 0;
    mocks.listVNScriptVersions.mockImplementation(async () => {
      versionListCalls += 1;
      if (versionListCalls === 1) {
        return {
          items: [versionResponse],
          limit: 25,
          offset: 0,
          total: 1,
          has_more: false,
          pagination: { limit: 25, offset: 0, total: 1, has_more: false },
        };
      }
      throw new Error('Failed to fetch versions from backend with internal stack payload');
    });
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Publish' }));

    expect(await screen.findByText('Version 3')).toBeInTheDocument();
    expect(screen.getByText('Published, but failed to refresh versions')).toBeInTheDocument();
  });

  it('surfaces publish acknowledgement-required errors without inferring codes', async () => {
    const user = userEvent.setup();
    mocks.publishVNScript.mockRejectedValueOnce(new Error('script_publish_acknowledgement_required'));
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Publish' }));

    expect(await screen.findByText('script_publish_acknowledgement_required')).toBeInTheDocument();
    expect(mocks.publishVNScript).toHaveBeenCalledWith(1, expect.objectContaining({
      acknowledgements: [],
    }));
  });

  it('loads manifest and policy summaries for a published version', async () => {
    const user = userEvent.setup();
    render(<VNScriptsWorkbench />);

    const version = await screen.findByTestId('version-12');
    await user.click(within(version).getByRole('button', { name: /Load manifest for version 2/i }));
    await user.click(within(version).getByRole('button', { name: /Evaluate policy for version 2/i }));

    await waitFor(() => expect(mocks.getVNScriptManifestSnapshot).toHaveBeenCalledWith(1, 12));
    await waitFor(() => expect(mocks.evaluateVNScriptVersionPolicy).toHaveBeenCalledWith(1, 12));
    expect(await screen.findByText(/background.interior/)).toBeInTheDocument();
    expect(screen.getByText(/allow/)).toBeInTheDocument();
    expect(screen.queryByText(/hidden prompt text/)).not.toBeInTheDocument();
    expect(screen.queryByText(/hidden policy details/)).not.toBeInTheDocument();
  });

  it('does not let a stale publish version refresh overwrite a newly selected script', async () => {
    const user = userEvent.setup();
    const staleRefresh = createDeferred<{
      items: typeof versionResponse[];
      limit: number;
      offset: number;
      total: number;
      has_more: boolean;
      pagination: { limit: number; offset: number; total: number; has_more: boolean };
    }>();
    let openingVersionCalls = 0;
    mocks.getVNScriptDraft.mockImplementation(async (scriptId: number) =>
      scriptId === 2
        ? { ...draftResponse, script_id: 2, draft: { scenes: [{ id: 'second' }] } }
        : draftResponse
    );
    mocks.listVNScriptVersions.mockImplementation((scriptId: number) => {
      if (scriptId === 2) {
        return Promise.resolve({
          items: [secondVersionResponse],
          limit: 25,
          offset: 0,
          total: 1,
          has_more: false,
          pagination: { limit: 25, offset: 0, total: 1, has_more: false },
        });
      }
      openingVersionCalls += 1;
      if (openingVersionCalls === 1) {
        return Promise.resolve({
          items: [versionResponse],
          limit: 25,
          offset: 0,
          total: 1,
          has_more: false,
          pagination: { limit: 25, offset: 0, total: 1, has_more: false },
        });
      }
      return staleRefresh.promise;
    });
    render(<VNScriptsWorkbench />);

    await screen.findByRole('button', { name: /Opening Route/ });
    await user.click(screen.getByRole('button', { name: 'Publish' }));
    await waitFor(() => expect(mocks.publishVNScript).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: /Second Route/ }));
    await screen.findByText('Script #2');

    staleRefresh.resolve({
      items: [versionResponse],
      limit: 25,
      offset: 0,
      total: 1,
      has_more: false,
      pagination: { limit: 25, offset: 0, total: 1, has_more: false },
    });

    expect(await screen.findByTestId('version-22')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByTestId('version-12')).not.toBeInTheDocument());
  });
});
