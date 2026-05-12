import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mocks = vi.hoisted(() => ({
  createVNScript: vi.fn(),
  evaluateVNScriptVersionPolicy: vi.fn(),
  getVNScriptDiagnostics: vi.fn(),
  getVNScriptDraft: vi.fn(),
  getVNScriptManifestSnapshot: vi.fn(),
  listVNScripts: vi.fn(),
  listVNScriptVersions: vi.fn(),
  publishVNScript: vi.fn(),
  putVNScriptDraft: vi.fn(),
  validateVNScriptDraft: vi.fn(),
}));

vi.mock('@web/lib/api/vnScripts', () => ({
  createVNScript: (...args: unknown[]) => mocks.createVNScript(...args),
  evaluateVNScriptVersionPolicy: (...args: unknown[]) => mocks.evaluateVNScriptVersionPolicy(...args),
  getVNScriptDiagnostics: (...args: unknown[]) => mocks.getVNScriptDiagnostics(...args),
  getVNScriptDraft: (...args: unknown[]) => mocks.getVNScriptDraft(...args),
  getVNScriptManifestSnapshot: (...args: unknown[]) => mocks.getVNScriptManifestSnapshot(...args),
  listVNScripts: (...args: unknown[]) => mocks.listVNScripts(...args),
  listVNScriptVersions: (...args: unknown[]) => mocks.listVNScriptVersions(...args),
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

describe('VNScriptsWorkbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockList();
    mocks.createVNScript.mockResolvedValue({
      ...secondScript,
      id: 9,
      title: 'Created Route',
      primary_asset_pack_id: 44,
      policy_profile_id: 'policy-explicit',
      generation_profile_id: 'gen-explicit',
      content_rating: 'mature',
    });
    mocks.getVNScriptDraft.mockResolvedValue(draftResponse);
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
