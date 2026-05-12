import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type {
  VNPlayBranch,
  VNPlayCheckpoint,
  VNPlaySession,
  VNPlaySetupAssetPackOption,
  VNPlaySetupOptionsResponse,
} from '@web/types/vn-play';

const mocks = vi.hoisted(() => ({
  createVNPlayCheckpoint: vi.fn(),
  createVNPlaySession: vi.fn(),
  getVNPlaySession: vi.fn(),
  getVNAssetReadiness: vi.fn(),
  listVNPlayBranches: vi.fn(),
  listVNPlayCheckpoints: vi.fn(),
  listVNPlayEvents: vi.fn(),
  listVNPlaySessions: vi.fn(),
  listVNPlaySetupOptions: vi.fn(),
  listCharacters: vi.fn(),
  listVNAssetPacks: vi.fn(),
  restoreVNPlaySession: vi.fn(),
  retryLastVNPlayTurn: vi.fn(),
  submitVNPlayTurn: vi.fn(),
}));

vi.mock('@web/lib/api/vnPlay', () => ({
  createVNPlayCheckpoint: (...args: unknown[]) => mocks.createVNPlayCheckpoint(...args),
  createVNPlaySession: (...args: unknown[]) => mocks.createVNPlaySession(...args),
  getVNPlaySession: (...args: unknown[]) => mocks.getVNPlaySession(...args),
  listVNPlayBranches: (...args: unknown[]) => mocks.listVNPlayBranches(...args),
  listVNPlayCheckpoints: (...args: unknown[]) => mocks.listVNPlayCheckpoints(...args),
  listVNPlayEvents: (...args: unknown[]) => mocks.listVNPlayEvents(...args),
  listVNPlaySessions: (...args: unknown[]) => mocks.listVNPlaySessions(...args),
  listVNPlaySetupOptions: (...args: unknown[]) => mocks.listVNPlaySetupOptions(...args),
  restoreVNPlaySession: (...args: unknown[]) => mocks.restoreVNPlaySession(...args),
  retryLastVNPlayTurn: (...args: unknown[]) => mocks.retryLastVNPlayTurn(...args),
  submitVNPlayTurn: (...args: unknown[]) => mocks.submitVNPlayTurn(...args),
}));

vi.mock('@web/lib/api/characters', () => ({
  listCharacters: (...args: unknown[]) => mocks.listCharacters(...args),
}));

vi.mock('@web/lib/api/vnAssets', () => ({
  getVNAssetReadiness: (...args: unknown[]) => mocks.getVNAssetReadiness(...args),
  listVNAssetPacks: (...args: unknown[]) => mocks.listVNAssetPacks(...args),
}));

import VNPlayWorkspace from '@web/components/vn-play/VNPlayWorkspace';

const defaultCharacters = [
  {
    id: 7,
    version: 1,
    name: 'Mira Vale',
    description: 'Archive guide',
    tags: ['archive', 'story'],
    image_present: true,
  },
];

const defaultPacks = [
  {
    id: 12,
    title: 'Moonlit Archive Pack',
    primary_character_id: 7,
    description: 'Runtime-ready VN poses and backdrops',
    status: 'approved',
    content_rating: 'general',
    planned_output_count: 8,
  },
];

const defaultSetupOptions: VNPlaySetupOptionsResponse = {
  characters: [
    {
      id: 7,
      name: 'Mira Vale',
      description_preview: 'Archive guide',
      tags: ['archive', 'story'],
      favorite: false,
      deleted: false,
      has_image: true,
    },
  ],
  selected_character: null,
  asset_packs: [
    {
      id: 12,
      title: 'Moonlit Archive Pack',
      primary_character_id: 7,
      content_rating: 'general',
      status: 'approved',
      trust_level: 'local',
      trust_source: 'local_pack',
      ready: true,
      readiness_status: 'ready',
      readiness_warnings: [],
      readiness_errors: [],
      compatibility: { status: 'compatible', reason_codes: [] },
      warning_summary: {
        highest_severity: 'info',
        requires_acknowledgement: false,
        warnings: [],
      },
      recommended: true,
    },
  ],
  defaults: {
    mode: 'freeform',
    character_id: 7,
    asset_pack_id: 12,
    content_rating: 'general',
  },
  pagination: {
    characters: { limit: 25, offset: 0, has_more: false, total: 1 },
    asset_packs: { limit: 25, offset: 0, has_more: false, total: 1 },
  },
  empty_states: [],
  generated_at: '2026-05-09T15:00:00Z',
};

function setupPack(overrides: Partial<VNPlaySetupAssetPackOption> = {}): VNPlaySetupAssetPackOption {
  return {
    ...defaultSetupOptions.asset_packs[0],
    compatibility: {
      ...defaultSetupOptions.asset_packs[0].compatibility,
      ...overrides.compatibility,
    },
    warning_summary: {
      ...defaultSetupOptions.asset_packs[0].warning_summary,
      ...overrides.warning_summary,
    },
    ...overrides,
  };
}

function setupOptions(
  overrides: Partial<VNPlaySetupOptionsResponse> = {}
): VNPlaySetupOptionsResponse {
  const characters = overrides.characters ?? defaultSetupOptions.characters;
  const assetPacks = overrides.asset_packs ?? defaultSetupOptions.asset_packs;
  return {
    ...defaultSetupOptions,
    ...overrides,
    characters,
    asset_packs: assetPacks,
    defaults: {
      ...defaultSetupOptions.defaults,
      ...overrides.defaults,
    },
    pagination: {
      characters: {
        ...defaultSetupOptions.pagination.characters,
        total: characters.length,
        has_more: false,
      },
      asset_packs: {
        ...defaultSetupOptions.pagination.asset_packs,
        total: assetPacks.length,
        has_more: false,
      },
      ...overrides.pagination,
    },
  };
}

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, reject, resolve };
}

function mockVNPlayApi({
  branches = [],
  checkpoints = [],
  sessions = [],
}: {
  branches?: VNPlayBranch[];
  checkpoints?: VNPlayCheckpoint[];
  sessions?: VNPlaySession[];
} = {}) {
  mocks.listVNPlaySessions.mockResolvedValue(sessions);
  mocks.listVNPlayEvents.mockResolvedValue([]);
  mocks.listVNPlayCheckpoints.mockResolvedValue(checkpoints);
  mocks.listVNPlayBranches.mockResolvedValue(branches);
  mocks.getVNPlaySession.mockImplementation(async (sessionId: number) =>
    sessions.find((session) => session.id === sessionId) ?? sessions[0]
  );
  mocks.createVNPlaySession.mockImplementation(async (request) => ({
    id: 9,
    owner_user_id: 42,
    scene_version: 0,
    status: 'active',
    trust_level: 'local',
    linked_chat_mode: 'read_only_context',
    scene_state: { scene_version: 0 },
    ...request,
  }));
  mocks.submitVNPlayTurn.mockResolvedValue({
    turn_request_id: 10,
    status: 'completed',
    scene_version: 1,
    scene_state: { scene_version: 1 },
    events: [
      {
        id: 2,
        session_id: 1,
        owner_user_id: 42,
        sequence_number: 2,
        event_type: 'model_turn',
        event_payload: {
          dialogue: [{ speaker: 'Narrator', text: 'A quiet reply.' }],
        },
        source: 'model',
      },
    ],
  });
  mocks.createVNPlayCheckpoint.mockImplementation(async (_sessionId: number, request) => ({
    id: 7,
    session_id: _sessionId,
    owner_user_id: 42,
    label: request.label,
    scene_version: request.scene_version ?? 0,
  }));
  mocks.restoreVNPlaySession.mockImplementation(async (sessionId: number) =>
    sessions.find((session) => session.id === sessionId) ?? sessions[0]
  );
  mocks.retryLastVNPlayTurn.mockResolvedValue({
    turn_request_id: 11,
    status: 'completed',
    scene_version: 2,
    scene_state: { scene_version: 2 },
    events: [],
  });
}

describe('VNPlayWorkspace', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockVNPlayApi();
    mocks.listCharacters.mockResolvedValue(defaultCharacters);
    mocks.listVNAssetPacks.mockResolvedValue(defaultPacks);
    mocks.listVNPlaySetupOptions.mockResolvedValue(setupOptions());
    mocks.getVNAssetReadiness.mockResolvedValue({
      ready: true,
      status: 'ready',
      warnings: [],
      errors: [],
    });
  });

  it('renders freeform and story session actions', async () => {
    mockVNPlayApi({ sessions: [] });

    render(<VNPlayWorkspace />);

    expect(await screen.findByRole('button', { name: /new freeform/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /new story/i })).toBeInTheDocument();
  });

  it('loads and selects the first session', async () => {
    mockVNPlayApi({
      sessions: [
        {
          id: 1,
          mode: 'freeform',
          title: 'Library',
          primary_character_id: 1,
          vn_asset_pack_id: 2,
          scene_version: 0,
          scene_state: { scene_version: 0 },
        },
      ],
    });

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('Library')).toBeInTheDocument();
    expect(screen.getByText('Selected session: Library')).toBeInTheDocument();
  });

  it('creates a freeform session from named character and asset pack selectors', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    expect(await screen.findByLabelText('Character')).toBeInTheDocument();
    expect(screen.getAllByText('Mira Vale').length).toBeGreaterThan(0);
    expect(screen.getByText(/Archive guide/)).toBeInTheDocument();
    expect(screen.getAllByText(/archive, story/).length).toBeGreaterThan(0);
    expect(screen.getByLabelText('VN asset pack')).toBeInTheDocument();
    expect(screen.getAllByText('Moonlit Archive Pack').length).toBeGreaterThan(0);
    expect(screen.getByText(/Trust level: local/)).toBeInTheDocument();
    expect(screen.getByText(/Readiness: ready/)).toBeInTheDocument();

    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Moonlit Archive');
    await user.selectOptions(screen.getByLabelText('Character'), '7');
    await user.selectOptions(screen.getByLabelText('VN asset pack'), '12');
    await user.click(screen.getByRole('button', { name: 'Create session' }));

    await waitFor(() => {
      expect(mocks.createVNPlaySession).toHaveBeenCalledWith({
        mode: 'freeform',
        title: 'Moonlit Archive',
        primary_character_id: 7,
        vn_asset_pack_id: 12,
        linked_chat_id: null,
        content_rating: 'general',
      });
    });
    expect(await screen.findByText('Moonlit Archive')).toBeInTheDocument();
  });

  it('uses backend setup options instead of client-side setup fan-out', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByLabelText('Character')).toBeInTheDocument();
    expect(mocks.listVNPlaySetupOptions).toHaveBeenCalledWith({
      content_rating: 'general',
      mode: 'freeform',
    });
    expect(mocks.listCharacters).not.toHaveBeenCalled();
    expect(mocks.listVNAssetPacks).not.toHaveBeenCalled();
    expect(mocks.getVNAssetReadiness).not.toHaveBeenCalled();
  });

  it('does not refetch setup options when applying backend defaults', async () => {
    const user = userEvent.setup();
    const firstSetup = createDeferred<VNPlaySetupOptionsResponse>();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockReturnValueOnce(firstSetup.promise);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    expect(mocks.listVNPlaySetupOptions).toHaveBeenCalledTimes(1);

    firstSetup.resolve(setupOptions());
    expect(await screen.findByLabelText('Character')).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(mocks.listVNPlaySetupOptions).toHaveBeenCalledTimes(1);
  });

  it('renders setup loading state before backend options resolve', async () => {
    const user = userEvent.setup();
    const setup = createDeferred<VNPlaySetupOptionsResponse>();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockReturnValueOnce(setup.promise);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByText(/Loading setup options/i)).toBeInTheDocument();
    setup.resolve(setupOptions());
    expect(await screen.findByLabelText('Character')).toBeInTheDocument();
    expect(screen.getAllByText('Mira Vale').length).toBeGreaterThan(0);
    expect(screen.getByRole('option', { name: /Moonlit Archive Pack/i })).toBeInTheDocument();
  });

  it('does not issue client-side readiness checks for setup packs', async () => {
    const user = userEvent.setup();
    const packs = Array.from({ length: 6 }, (_, index) => ({
      ...setupPack(),
      id: index + 1,
      title: `Pack ${index + 1}`,
    }));

    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockResolvedValue(setupOptions({ asset_packs: packs }));

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    expect(await screen.findByRole('option', { name: /Pack 1/i })).toBeInTheDocument();
    expect(mocks.getVNAssetReadiness).not.toHaveBeenCalled();
  });

  it('marks incompatible asset packs as unavailable for the selected character', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockResolvedValue(
      setupOptions({
        asset_packs: [
          setupPack(),
          setupPack({
            id: 13,
            title: 'Rival Pack',
            primary_character_id: 99,
            compatibility: { status: 'different_character', reason_codes: ['character_mismatch'] },
            recommended: false,
            warning_summary: {
              highest_severity: 'high_risk',
              requires_acknowledgement: true,
              warnings: [
                {
                  code: 'pack_character_mismatch',
                  severity: 'high_risk',
                  message: 'Rival Pack is attached to character 99, not Mira Vale.',
                  requires_acknowledgement: true,
                },
              ],
            },
          }),
        ],
      })
    );

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new story/i }));

    expect(await screen.findByRole('option', {
      name: /Rival Pack.*different character/i,
    })).toBeDisabled();
    expect(screen.getByText(/Rival Pack is attached to character 99/i)).toBeInTheDocument();
  });

  it('shows readiness, draft, missing-byte, and content-rating warnings before submit', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockResolvedValue(
      setupOptions({
        asset_packs: [
          setupPack({
            status: 'draft',
            content_rating: 'mature',
            ready: false,
            readiness_status: 'missing_assets',
            readiness_warnings: ['2 required runtime assets are missing bytes'],
            readiness_errors: ['sprite.primary has no approved file bytes'],
            recommended: false,
            warning_summary: {
              highest_severity: 'high_risk',
              requires_acknowledgement: true,
              warnings: [
                {
                  code: 'pack_status_draft',
                  severity: 'high_risk',
                  message: 'Pack status is draft; review or approve it before starting VN Play.',
                  requires_acknowledgement: true,
                },
                {
                  code: 'content_rating_mismatch',
                  severity: 'warning',
                  message: 'Pack content rating mature differs from session rating general.',
                  requires_acknowledgement: false,
                },
              ],
            },
          }),
        ],
        defaults: { asset_pack_id: 12 },
      })
    );

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByText(/Pack status is draft/i)).toBeInTheDocument();
    expect(screen.getByText(/2 required runtime assets are missing bytes/i)).toBeInTheDocument();
    expect(screen.getByText(/sprite.primary has no approved file bytes/i)).toBeInTheDocument();
    expect(screen.getByText(/Pack content rating mature differs from session rating general/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create session' })).toBeDisabled();
  });

  it('allows acknowledged high-risk setup warnings to be submitted in session settings', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockResolvedValue(
      setupOptions({
        asset_packs: [
          setupPack({
            title: 'Imported Archive Pack',
            trust_level: 'untrusted_import',
            trust_source: 'latest_import_journal',
            ready: true,
            readiness_status: 'ready',
            recommended: false,
            warning_summary: {
              highest_severity: 'high_risk',
              requires_acknowledgement: true,
              warnings: [
                {
                  code: 'untrusted_import',
                  severity: 'high_risk',
                  message: 'Imported packs should be reviewed before VN Play.',
                  requires_acknowledgement: true,
                },
              ],
            },
          }),
        ],
        defaults: { asset_pack_id: 12 },
      })
    );

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByRole('option', {
      name: /Imported Archive Pack.*review required/i,
    })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Create session' })).toBeDisabled();

    await user.click(screen.getByRole('checkbox', { name: /I understand and want to proceed/i }));
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Acknowledged Import');
    await user.click(screen.getByRole('button', { name: 'Create session' }));

    await waitFor(() => {
      expect(mocks.createVNPlaySession).toHaveBeenCalledWith({
        mode: 'freeform',
        title: 'Acknowledged Import',
        primary_character_id: 7,
        vn_asset_pack_id: 12,
        linked_chat_id: null,
        content_rating: 'general',
        settings: {
          setup_acknowledgements: [
            {
              asset_pack_id: 12,
              warning_codes: ['untrusted_import'],
              highest_severity: 'high_risk',
            },
          ],
        },
      });
    });
  });

  it('renders duplicate readiness warnings without duplicate React keys', async () => {
    const user = userEvent.setup();
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockResolvedValue(
      setupOptions({
        asset_packs: [
          setupPack({
            readiness_warnings: ['Review duplicate slot metadata', 'Review duplicate slot metadata'],
          }),
        ],
      })
    );

    try {
      render(<VNPlayWorkspace />);

      await user.click(await screen.findByRole('button', { name: /new freeform/i }));

      expect(await screen.findAllByText('Review duplicate slot metadata')).toHaveLength(2);
      expect(
        consoleErrorSpy.mock.calls.some((call) =>
          call.some((value) => String(value).includes('Encountered two children with the same key'))
        )
      ).toBe(false);
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });

  it('resets all new-session form fields when reopened after manual fallback', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockRejectedValue(new Error('setup options offline'));

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    expect(await screen.findByText(/Could not load setup options/i)).toBeInTheDocument();
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Dirty draft');
    await user.clear(screen.getByLabelText('Primary character ID'));
    await user.type(screen.getByLabelText('Primary character ID'), '17');
    await user.clear(screen.getByLabelText('VN asset pack ID'));
    await user.type(screen.getByLabelText('VN asset pack ID'), '21');
    await user.type(screen.getByLabelText('Linked chat ID'), 'chat-to-clear');
    await user.click(screen.getByRole('button', { name: 'Close' }));

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByText(/Could not load setup options/i)).toBeInTheDocument();
    expect(screen.getByLabelText('Title')).toHaveValue('Untitled VN play session');
    expect(screen.getByLabelText('Primary character ID')).toHaveValue('1');
    expect(screen.getByLabelText('VN asset pack ID')).toHaveValue('1');
    expect(screen.getByLabelText('Linked chat ID')).toHaveValue('');
  });

  it('keeps manual ID entry available when setup selectors fail to load', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockRejectedValueOnce(new Error('setup options offline'));

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    expect(await screen.findByText(/Could not load setup options/i)).toBeInTheDocument();

    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Manual Session');
    await user.clear(screen.getByLabelText('Primary character ID'));
    await user.type(screen.getByLabelText('Primary character ID'), '17');
    await user.clear(screen.getByLabelText('VN asset pack ID'));
    await user.type(screen.getByLabelText('VN asset pack ID'), '21');
    await user.click(screen.getByRole('button', { name: 'Create session' }));

    await waitFor(() => {
      expect(mocks.createVNPlaySession).toHaveBeenCalledWith({
        mode: 'freeform',
        title: 'Manual Session',
        primary_character_id: 17,
        vn_asset_pack_id: 21,
        linked_chat_id: null,
        content_rating: 'general',
      });
    });
  });

  it('shows empty-state guidance when no character or runtime-ready pack exists', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNPlaySetupOptions.mockResolvedValue(
      setupOptions({
        characters: [],
        asset_packs: [],
        defaults: { character_id: null, asset_pack_id: null },
        empty_states: [
          { code: 'no_characters', scope: 'global', message: 'No available characters were found.' },
          { code: 'no_asset_packs', scope: 'global', message: 'No VN asset packs were found.' },
        ],
      })
    );

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new story/i }));

    expect(await screen.findByText(/No available characters were found/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open characters/i })).toHaveAttribute('href', '/characters');
    expect(screen.getByText(/No VN asset packs were found/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Open VN asset packs/i })).toHaveAttribute('href', '/vn-assets');
    expect(screen.getByRole('button', { name: 'Create session' })).toBeDisabled();
  });

  it('submits a freeform turn and renders the returned dialogue', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Library',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 0,
      scene_state: { scene_version: 0 },
    };
    mockVNPlayApi({ sessions: [session] });

    render(<VNPlayWorkspace />);

    await screen.findByText('Library');
    await user.type(screen.getByLabelText('Freeform input'), 'Look around');
    await user.click(screen.getByRole('button', { name: 'Send turn' }));

    await waitFor(() => {
      expect(mocks.submitVNPlayTurn).toHaveBeenCalledWith(1, expect.objectContaining({
        input_text: 'Look around',
        client_scene_version: 0,
      }));
    });
    expect(await screen.findByText('A quiet reply.')).toBeInTheDocument();
  });

  it('loads checkpoints and branches for the selected session', async () => {
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 3,
      scene_state: { scene_version: 3 },
    };
    mockVNPlayApi({
      branches: [
        {
          id: 4,
          session_id: 1,
          owner_user_id: 42,
          branch_label: 'Door branch',
          status: 'active',
        },
      ],
      checkpoints: [
        {
          id: 5,
          session_id: 1,
          owner_user_id: 42,
          label: 'Before the door',
          scene_version: 2,
        },
      ],
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('Before the door')).toBeInTheDocument();
    expect(screen.getByText('Door branch')).toBeInTheDocument();
    expect(mocks.listVNPlayCheckpoints).toHaveBeenCalledWith(1);
    expect(mocks.listVNPlayBranches).toHaveBeenCalledWith(1);
  });

  it('creates a checkpoint and refreshes recovery metadata', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Library',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.listVNPlayCheckpoints
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: 9,
          session_id: 1,
          owner_user_id: 42,
          label: 'Before choosing',
          scene_version: 4,
        },
      ]);

    render(<VNPlayWorkspace />);

    await user.type(await screen.findByLabelText('Checkpoint label'), 'Before choosing');
    await user.click(screen.getByRole('button', { name: 'Create checkpoint' }));

    await waitFor(() => {
      expect(mocks.createVNPlayCheckpoint).toHaveBeenCalledWith(1, {
        label: 'Before choosing',
        scene_version: 4,
      });
    });
    expect(await screen.findByText('Before choosing')).toBeInTheDocument();
  });

  it('restores a checkpoint and refreshes the selected session', async () => {
    const user = userEvent.setup();
    const restoredSession: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 2,
      scene_state: { scene_version: 2 },
    };
    mockVNPlayApi({
      checkpoints: [
        {
          id: 5,
          session_id: 1,
          owner_user_id: 42,
          label: 'Before the door',
          scene_version: 2,
        },
      ],
      sessions: [restoredSession],
    });
    mocks.restoreVNPlaySession.mockResolvedValue(restoredSession);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /restore checkpoint: before the door/i }));

    await waitFor(() => {
      expect(mocks.restoreVNPlaySession).toHaveBeenCalledWith(1, expect.objectContaining({
        checkpoint_id: 5,
        client_scene_version: 2,
        idempotency_key: expect.stringMatching(/^restore-/),
      }));
    });
    expect(mocks.getVNPlaySession).not.toHaveBeenCalled();
    expect(await screen.findByText('Scene version')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('retries the last failed turn with a fresh idempotency key', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Library',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 0,
      scene_state: { scene_version: 0 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.submitVNPlayTurn.mockRejectedValueOnce(Object.assign(new Error('parse_failed'), { status: 502 }));

    render(<VNPlayWorkspace />);

    await user.type(await screen.findByLabelText('Freeform input'), 'Look around');
    await user.click(screen.getByRole('button', { name: 'Send turn' }));
    await user.click(await screen.findByRole('button', { name: /retry last turn/i }));

    await waitFor(() => {
      expect(mocks.retryLastVNPlayTurn).toHaveBeenCalledWith(1, expect.objectContaining({
        client_scene_version: 0,
        idempotency_key: expect.stringMatching(/^retry-/),
      }));
    });
  });

  it('shows explicit recovery copy for stale scene conflicts', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Library',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 0,
      scene_state: { scene_version: 0 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.submitVNPlayTurn.mockRejectedValueOnce(Object.assign(new Error('stale_scene_version'), { status: 409 }));

    render(<VNPlayWorkspace />);

    await user.type(await screen.findByLabelText('Freeform input'), 'Look around');
    await user.click(screen.getByRole('button', { name: 'Send turn' }));

    expect(await screen.findByText(/scene changed on another client/i)).toBeInTheDocument();
    expect(screen.queryByText('stale_scene_version')).not.toBeInTheDocument();
  });
});
