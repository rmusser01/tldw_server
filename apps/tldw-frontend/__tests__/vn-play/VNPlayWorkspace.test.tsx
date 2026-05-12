import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type {
  VNPlayBranch,
  VNPlayBranchNavigationResponse,
  VNPlayCheckpoint,
  VNPlayGenerationHistoryResponse,
  VNPlayGenerationRevisionDebugResponse,
  VNPlaySession,
  VNPlaySetupAssetPackOption,
  VNPlaySetupOptionsResponse,
} from '@web/types/vn-play';

const mocks = vi.hoisted(() => ({
  createVNPlayCheckpoint: vi.fn(),
  createVNPlaySession: vi.fn(),
  getVNPlayBranchNavigation: vi.fn(),
  getVNPlayGenerationRevisionDebug: vi.fn(),
  getVNPlaySession: vi.fn(),
  getVNAssetReadiness: vi.fn(),
  listVNPlayGenerationRevisions: vi.fn(),
  listVNPlayGenerations: vi.fn(),
  listVNPlayBranches: vi.fn(),
  listVNPlayCheckpoints: vi.fn(),
  listVNPlayEvents: vi.fn(),
  listVNPlaySessions: vi.fn(),
  listVNPlaySetupOptions: vi.fn(),
  listCharacters: vi.fn(),
  listVNAssetPacks: vi.fn(),
  regenerateVNPlayGeneration: vi.fn(),
  restoreVNPlayBranch: vi.fn(),
  restoreVNPlaySession: vi.fn(),
  retryLastVNPlayTurn: vi.fn(),
  submitVNPlayTurn: vi.fn(),
  activateVNPlayGenerationRevision: vi.fn(),
  cancelVNPlayGenerationRequest: vi.fn(),
  confirmVNPlayGenerationRequest: vi.fn(),
}));

vi.mock('@web/lib/api/vnPlay', () => ({
  activateVNPlayGenerationRevision: (...args: unknown[]) =>
    mocks.activateVNPlayGenerationRevision(...args),
  cancelVNPlayGenerationRequest: (...args: unknown[]) =>
    mocks.cancelVNPlayGenerationRequest(...args),
  confirmVNPlayGenerationRequest: (...args: unknown[]) =>
    mocks.confirmVNPlayGenerationRequest(...args),
  createVNPlayCheckpoint: (...args: unknown[]) => mocks.createVNPlayCheckpoint(...args),
  createVNPlaySession: (...args: unknown[]) => mocks.createVNPlaySession(...args),
  getVNPlayBranchNavigation: (...args: unknown[]) => mocks.getVNPlayBranchNavigation(...args),
  getVNPlayGenerationRevisionDebug: (...args: unknown[]) =>
    mocks.getVNPlayGenerationRevisionDebug(...args),
  getVNPlaySession: (...args: unknown[]) => mocks.getVNPlaySession(...args),
  listVNPlayBranches: (...args: unknown[]) => mocks.listVNPlayBranches(...args),
  listVNPlayCheckpoints: (...args: unknown[]) => mocks.listVNPlayCheckpoints(...args),
  listVNPlayEvents: (...args: unknown[]) => mocks.listVNPlayEvents(...args),
  listVNPlayGenerationRevisions: (...args: unknown[]) =>
    mocks.listVNPlayGenerationRevisions(...args),
  listVNPlayGenerations: (...args: unknown[]) => mocks.listVNPlayGenerations(...args),
  listVNPlaySessions: (...args: unknown[]) => mocks.listVNPlaySessions(...args),
  listVNPlaySetupOptions: (...args: unknown[]) => mocks.listVNPlaySetupOptions(...args),
  regenerateVNPlayGeneration: (...args: unknown[]) => mocks.regenerateVNPlayGeneration(...args),
  restoreVNPlayBranch: (...args: unknown[]) => mocks.restoreVNPlayBranch(...args),
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

const defaultGenerationHistory: VNPlayGenerationHistoryResponse = {
  items: [],
  total: 0,
  limit: 25,
  offset: 0,
  has_more: false,
  next_offset: null,
  pagination: {
    mode: 'offset',
    total: 0,
    limit: 25,
    offset: 0,
    has_more: false,
    next_offset: null,
  },
};

const emptyBranchNavigation: VNPlayBranchNavigationResponse = {
  session_id: 1,
  mode: 'story',
  scene_version: 0,
  last_event_id: null,
  active_branch_node_id: null,
  active_path: [],
  branches: [],
  warnings: [],
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
  branchNavigation = emptyBranchNavigation,
  branches = [],
  checkpoints = [],
  generations = defaultGenerationHistory,
  sessions = [],
}: {
  branchNavigation?: VNPlayBranchNavigationResponse | null;
  branches?: VNPlayBranch[];
  checkpoints?: VNPlayCheckpoint[];
  generations?: VNPlayGenerationHistoryResponse;
  sessions?: VNPlaySession[];
} = {}) {
  mocks.listVNPlaySessions.mockResolvedValue(sessions);
  mocks.listVNPlayEvents.mockResolvedValue([]);
  mocks.listVNPlayCheckpoints.mockResolvedValue(checkpoints);
  mocks.listVNPlayBranches.mockResolvedValue(branches);
  mocks.getVNPlayBranchNavigation.mockResolvedValue(branchNavigation);
  mocks.listVNPlayGenerations.mockResolvedValue(generations);
  mocks.listVNPlayGenerationRevisions.mockResolvedValue(defaultGenerationHistory);
  mocks.getVNPlayGenerationRevisionDebug.mockResolvedValue({
    id: 31,
    generation_id: 12,
    generation_request_id: 91,
    generation_point_key: 'intro:2:choice',
    revision_number: 1,
    status: 'succeeded',
    output_schema: 'choice_set',
    public_output: {},
    raw_output_debug_state: 'absent',
    raw_output_debug: null,
    parser_diagnostics: {},
    moderation_diagnostics: {},
    model_metadata: {},
    usage_metadata: {},
    request: {},
    profile: { profile_key: 'default', snapshot_id: 44 },
    created_at: '2026-05-12T01:00:00Z',
  } satisfies VNPlayGenerationRevisionDebugResponse);
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
  mocks.restoreVNPlayBranch.mockImplementation(async (sessionId: number, branchId: number, request) => {
    const session = sessions.find((candidate) => candidate.id === sessionId) ?? sessions[0];
    return {
      status: 'completed',
      replayed: false,
      restore_event_id: 51,
      target_event_id: request.target === 'choice_point' ? 41 : 50,
      scene_version: session?.scene_version ?? 0,
      session,
      current_scene: session?.scene_state ?? session?.current_scene ?? { scene_version: 0 },
      branch_navigation: branchNavigation ?? emptyBranchNavigation,
      branch_id: branchId,
      target: request.target,
    };
  });
  mocks.retryLastVNPlayTurn.mockResolvedValue({
    turn_request_id: 11,
    status: 'completed',
    scene_version: 2,
    scene_state: { scene_version: 2 },
    events: [],
  });
  mocks.regenerateVNPlayGeneration.mockResolvedValue({
    turn_request_id: 21,
    status: 'completed',
    scene_version: 5,
    scene_state: { scene_version: 5 },
    events: [],
  });
  mocks.activateVNPlayGenerationRevision.mockResolvedValue({
    turn_request_id: 22,
    status: 'completed',
    scene_version: 5,
    scene_state: { scene_version: 5 },
    events: [],
  });
  mocks.confirmVNPlayGenerationRequest.mockResolvedValue({
    turn_request_id: 23,
    status: 'completed',
    scene_version: 5,
    scene_state: { scene_version: 5 },
    events: [],
  });
  mocks.cancelVNPlayGenerationRequest.mockResolvedValue({
    turn_request_id: 24,
    status: 'completed',
    scene_version: 5,
    scene_state: { scene_version: 5 },
    events: [],
  });
}

describe('VNPlayWorkspace', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
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

  it('renders player-facing branch navigation for story sessions', async () => {
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 6,
      scene_state: { scene_version: 6 },
    };
    const branchNavigation: VNPlayBranchNavigationResponse = {
      session_id: 1,
      mode: 'story',
      scene_version: 6,
      last_event_id: 26,
      active_branch_node_id: 12,
      active_path: [
        {
          branch_id: 12,
          branch_label: 'Step inside',
          choice_id: 'open-door',
          choice_text: 'Open the archive door',
          depth: 1,
        },
      ],
      branches: [
        {
          branch_id: 12,
          parent_branch_id: null,
          parent_event_id: 17,
          choice_selected_event_id: 18,
          branch_label: 'Step inside',
          choice_id: 'open-door',
          choice_text: 'Open the archive door',
          branch_path: [],
          depth: 1,
          status: 'active',
          is_active: true,
          is_on_active_path: true,
          event_range: {
            start_event_id: 18,
            start_sequence_number: 18,
            latest_event_id: 26,
            latest_sequence_number: 26,
          },
          subtree_event_range: {
            start_event_id: 18,
            start_sequence_number: 18,
            latest_event_id: 26,
            latest_sequence_number: 26,
          },
          restore: {
            supported: true,
            default_target: 'branch_latest',
            target_names: ['branch_latest', 'choice_point'],
            targets: {
              branch_latest: { event_id: 26, scene_version: 6 },
              choice_point: { event_id: 17, scene_version: 5 },
            },
          },
          warnings: [],
        },
      ],
      warnings: [],
    };
    mockVNPlayApi({ branchNavigation, sessions: [session] });

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('Branch timeline')).toBeInTheDocument();
    expect(screen.getAllByText('Open the archive door').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /resume branch: step inside/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /return to choice: step inside/i })).toBeInTheDocument();
    expect(mocks.getVNPlayBranchNavigation).toHaveBeenCalledWith(1);
  });

  it('restores a story branch using the selected backend target', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 6,
      scene_state: { scene_version: 6 },
    };
    const restoredSession: VNPlaySession = {
      ...session,
      scene_version: 7,
      scene_state: { scene_version: 7 },
    };
    const branchNavigation: VNPlayBranchNavigationResponse = {
      session_id: 1,
      mode: 'story',
      scene_version: 6,
      last_event_id: 26,
      active_branch_node_id: 12,
      active_path: [],
      branches: [
        {
          branch_id: 12,
          parent_branch_id: null,
          parent_event_id: 17,
          choice_selected_event_id: 18,
          branch_label: 'Step inside',
          choice_id: 'open-door',
          choice_text: 'Open the archive door',
          branch_path: [],
          depth: 1,
          status: 'active',
          is_active: true,
          is_on_active_path: true,
          event_range: {
            start_event_id: 18,
            start_sequence_number: 18,
            latest_event_id: 26,
            latest_sequence_number: 26,
          },
          subtree_event_range: {
            start_event_id: 18,
            start_sequence_number: 18,
            latest_event_id: 26,
            latest_sequence_number: 26,
          },
          restore: {
            supported: true,
            default_target: 'branch_latest',
            target_names: ['branch_latest', 'choice_point'],
            targets: {
              branch_latest: { event_id: 26, scene_version: 6 },
              choice_point: { event_id: 17, scene_version: 5 },
            },
          },
          warnings: [],
        },
      ],
      warnings: [],
    };
    const restoredNavigation: VNPlayBranchNavigationResponse = {
      ...branchNavigation,
      scene_version: 7,
    };
    mockVNPlayApi({ branchNavigation, sessions: [session] });
    mocks.restoreVNPlayBranch.mockResolvedValue({
      status: 'completed',
      replayed: false,
      restore_event_id: 51,
      target_event_id: 17,
      scene_version: 7,
      session: restoredSession,
      current_scene: restoredSession.scene_state,
      branch_navigation: restoredNavigation,
      branch_id: 12,
      target: 'choice_point',
    });
    mocks.getVNPlaySession.mockResolvedValue(restoredSession);
    mocks.getVNPlayBranchNavigation.mockResolvedValueOnce(branchNavigation).mockResolvedValueOnce(restoredNavigation);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /return to choice: step inside/i }));

    await waitFor(() => {
      expect(mocks.restoreVNPlayBranch).toHaveBeenCalledWith(1, 12, {
        client_scene_version: 6,
        idempotency_key: expect.stringMatching(/^restore-branch-/),
        target: 'choice_point',
      });
    });
    await waitFor(() => {
      expect(screen.getAllByText('Scene 7').length).toBeGreaterThan(0);
    });
  });

  it('surfaces branch restore in-progress conflicts as recoverable play state', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 6,
      scene_state: { scene_version: 6 },
    };
    const branchNavigation: VNPlayBranchNavigationResponse = {
      session_id: 1,
      mode: 'story',
      scene_version: 6,
      last_event_id: 26,
      active_branch_node_id: 12,
      active_path: [],
      branches: [
        {
          branch_id: 12,
          parent_branch_id: null,
          parent_event_id: 17,
          choice_selected_event_id: 18,
          branch_label: 'Step inside',
          choice_id: 'open-door',
          choice_text: 'Open the archive door',
          branch_path: [],
          depth: 1,
          status: 'active',
          is_active: true,
          is_on_active_path: true,
          event_range: {
            start_event_id: 18,
            start_sequence_number: 18,
            latest_event_id: 26,
            latest_sequence_number: 26,
          },
          subtree_event_range: {
            start_event_id: 18,
            start_sequence_number: 18,
            latest_event_id: 26,
            latest_sequence_number: 26,
          },
          restore: {
            supported: true,
            default_target: 'branch_latest',
            target_names: ['branch_latest'],
            targets: {
              branch_latest: { event_id: 26, scene_version: 6 },
            },
          },
          warnings: [],
        },
      ],
      warnings: [],
    };
    mockVNPlayApi({ branchNavigation, sessions: [session] });
    mocks.restoreVNPlayBranch.mockRejectedValueOnce(
      Object.assign(new Error('restore_action_in_progress'), {
        code: 'restore_action_in_progress',
        status: 409,
      })
    );

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /resume branch: step inside/i }));

    expect(await screen.findByText(/branch restore is already in progress/i)).toBeInTheDocument();
    expect(screen.queryByText('restore_action_in_progress')).not.toBeInTheDocument();
    expect(mocks.getVNPlaySession).toHaveBeenCalledWith(1);
  });

  it('renders scripted generation history without raw debug payloads', async () => {
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
      generations: {
        ...defaultGenerationHistory,
        total: 1,
        items: [
          {
            id: 31,
            generation_id: 12,
            generation_point_key: 'intro:2:choice',
            revision_number: 1,
            status: 'succeeded',
            active: true,
            output_schema: 'choice_set',
            public_output: {
              lead_in: "The map trembles in Mira's hand.",
              choices: [{ id: 'ask-map', text: 'Ask about the map', source: 'generated' }],
              raw_output_debug: 'must not render',
            },
            applied_visuals: [],
            rejected_visuals: [],
            source: 'model',
            profile: {
              profile_key: 'choice_writer',
              snapshot_id: 44,
              provider_class: 'hosted',
              moderation_required: true,
              estimated_cost_class: 'low',
            },
            created_at: 'generated-at-time',
          },
        ],
      },
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('Scripted generations')).toBeInTheDocument();
    expect(screen.getByText('intro:2:choice')).toBeInTheDocument();
    expect(screen.getByText("The map trembles in Mira's hand.")).toBeInTheDocument();
    expect(screen.getByText('Ask about the map')).toBeInTheDocument();
    expect(screen.getByText(/choice_writer/)).toBeInTheDocument();
    expect(screen.getByText(/generated-at-time/)).toBeInTheDocument();
    expect(screen.queryByText('must not render')).not.toBeInTheDocument();
    expect(mocks.listVNPlayGenerations).toHaveBeenCalledWith(1, { limit: 25, offset: 0 });
    expect(screen.getByRole('link', { name: /open generation inspector/i })).toHaveAttribute(
      'href',
      '/vn-play/sessions/1/generations'
    );
  });

  it('keeps scene playback usable while the generation inspector stays separate', async () => {
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 3,
      scene_state: {
        scene_version: 3,
        background: { content_url: '/scene-bg.png' },
        active_sprites: [{ item_id: 2, content_url: '/mira.png' }],
        location_key: 'archive',
        visible_choices: [
          {
            id: 'ask-map',
            text: 'Ask about the map',
            metadata: {
              source: 'generated',
              generation_point_key: 'intro:choices',
            },
          },
        ],
      },
    };
    mockVNPlayApi({ sessions: [session] });

    render(<VNPlayWorkspace />);

    expect(await screen.findByAltText(/scene background/i)).toHaveAttribute('src', '/scene-bg.png');
    expect(screen.getByRole('button', { name: /ask about the map/i })).toBeInTheDocument();
    expect(screen.getByText('Generated')).toBeInTheDocument();
    expect(screen.getByText('intro:choices')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /open generation inspector/i })).toHaveAttribute(
      'href',
      '/vn-play/sessions/1/generations'
    );
  });

  it('selects the requested session for the dedicated generation inspector route', async () => {
    mockVNPlayApi({
      sessions: [
        {
          id: 1,
          mode: 'story',
          title: 'First Door',
          primary_character_id: 1,
          vn_asset_pack_id: 2,
          scene_version: 1,
          scene_state: { scene_version: 1 },
        },
        {
          id: 2,
          mode: 'story',
          title: 'Second Door',
          primary_character_id: 1,
          vn_asset_pack_id: 2,
          scene_version: 2,
          scene_state: { scene_version: 2 },
        },
      ],
    });

    render(<VNPlayWorkspace generationInspectorRoute initialSessionId={2} />);

    expect(await screen.findByText('Selected session: Second Door')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /open generation inspector/i })).not.toBeInTheDocument();
  });

  it('loads additional generation history pages', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 3,
      scene_state: { scene_version: 3 },
    };
    const firstPage: VNPlayGenerationHistoryResponse = {
      ...defaultGenerationHistory,
      items: [
        {
          id: 31,
          generation_id: 12,
          generation_point_key: 'intro:first',
          revision_number: 1,
          status: 'succeeded',
          active: true,
          output_schema: 'choice_set',
          public_output: { lead_in: 'First generated line' },
          profile: { profile_key: 'choice_writer', snapshot_id: 44 },
          created_at: 'first-page-time',
        },
      ],
      pagination: {
        mode: 'offset',
        total: 2,
        limit: 25,
        offset: 0,
        has_more: true,
        next_offset: 25,
      },
    };
    const secondPage: VNPlayGenerationHistoryResponse = {
      ...defaultGenerationHistory,
      items: [
        {
          id: 32,
          generation_id: 13,
          generation_point_key: 'intro:second',
          revision_number: 1,
          status: 'succeeded',
          active: true,
          output_schema: 'choice_set',
          public_output: { lead_in: 'Second generated line' },
          profile: { profile_key: 'choice_writer', snapshot_id: 45 },
          created_at: 'second-page-time',
        },
      ],
      pagination: {
        mode: 'offset',
        total: 2,
        limit: 25,
        offset: 25,
        has_more: false,
        next_offset: null,
      },
    };
    mockVNPlayApi({ generations: firstPage, sessions: [session] });
    mocks.listVNPlayGenerations.mockResolvedValueOnce(firstPage).mockResolvedValueOnce(secondPage);

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('First generated line')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /load more generations/i }));

    expect(await screen.findByText('Second generated line')).toBeInTheDocument();
    expect(mocks.listVNPlayGenerations).toHaveBeenLastCalledWith(1, { limit: 25, offset: 25 });
  });

  it('surfaces major generation error states with readable guidance', async () => {
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
      generations: {
        ...defaultGenerationHistory,
        items: [
          {
            id: 41,
            generation_id: 20,
            generation_point_key: 'provider',
            revision_number: 1,
            status: 'provider_unavailable',
            active: false,
            output_schema: 'choice_set',
            public_output: {},
            public_error_code: 'provider_unavailable',
            profile: { profile_key: 'choice_writer', snapshot_id: 44 },
          },
          {
            id: 42,
            generation_id: 21,
            generation_point_key: 'parser',
            revision_number: 1,
            status: 'parser_failed',
            active: false,
            output_schema: 'choice_set',
            public_output: {},
            public_error_code: 'parser_failed',
            profile: { profile_key: 'choice_writer', snapshot_id: 45 },
          },
          {
            id: 43,
            generation_id: 22,
            generation_point_key: 'activation',
            revision_number: 1,
            status: 'activation_blocked',
            active: false,
            output_schema: 'choice_set',
            public_output: {},
            public_error_code: 'activation_blocked',
            profile: { profile_key: 'choice_writer', snapshot_id: 46 },
          },
          {
            id: 44,
            generation_id: 23,
            generation_point_key: 'abandoned',
            revision_number: 1,
            status: 'abandoned',
            active: false,
            output_schema: 'choice_set',
            public_output: {},
            public_error_code: 'abandoned',
            profile: { profile_key: 'choice_writer', snapshot_id: 47 },
          },
        ],
      },
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    expect(await screen.findByText(/provider was unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/could not be parsed/i)).toBeInTheDocument();
    expect(screen.getByText(/activation was blocked/i)).toBeInTheDocument();
    expect(screen.getByText(/abandoned or timed out/i)).toBeInTheDocument();
  });

  it('runs generation regenerate and activate commands through backend actions', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    mockVNPlayApi({
      generations: {
        ...defaultGenerationHistory,
        items: [
          {
            id: 31,
            generation_id: 12,
            generation_point_key: 'intro:2:choice',
            revision_number: 1,
            status: 'succeeded',
            active: false,
            output_schema: 'choice_set',
            public_output: { choices: [{ id: 'ask-map', text: 'Ask about the map' }] },
            applied_visuals: [],
            rejected_visuals: [],
            source: 'model',
            profile: { profile_key: 'choice_writer', snapshot_id: 44 },
            created_at: '2026-05-12T01:00:00Z',
          },
        ],
        total: 1,
      },
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /regenerate intro:2:choice/i }));
    await waitFor(() => {
      expect(mocks.regenerateVNPlayGeneration).toHaveBeenCalledWith(1, 12, {
        client_scene_version: 4,
        idempotency_key: expect.stringMatching(/^generation-regenerate-/),
      });
    });

    await user.click(screen.getByRole('button', { name: /activate revision 1 for intro:2:choice/i }));
    await waitFor(() => {
      expect(mocks.activateVNPlayGenerationRevision).toHaveBeenCalledWith(1, 12, 31, {
        client_scene_version: 4,
        idempotency_key: expect.stringMatching(/^generation-activate-/),
      });
    });
  });

  it('refreshes generation history when a generation action returns a session payload', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    const refreshedSession: VNPlaySession = {
      ...session,
      scene_version: 5,
      scene_state: { scene_version: 5 },
    };
    const initialGenerations: VNPlayGenerationHistoryResponse = {
      ...defaultGenerationHistory,
      items: [
        {
          id: 31,
          generation_id: 12,
          generation_point_key: 'intro:2:choice',
          revision_number: 1,
          status: 'succeeded',
          active: false,
          output_schema: 'choice_set',
          public_output: { choices: [{ id: 'ask-map', text: 'Ask about the map' }] },
          profile: { profile_key: 'choice_writer', snapshot_id: 44 },
        },
      ],
      total: 1,
    };
    const refreshedGenerations: VNPlayGenerationHistoryResponse = {
      ...defaultGenerationHistory,
      items: [
        {
          id: 32,
          generation_id: 12,
          generation_point_key: 'intro:2:choice',
          revision_number: 2,
          status: 'succeeded',
          active: true,
          output_schema: 'choice_set',
          public_output: { choices: [{ id: 'follow-map', text: 'Follow the refreshed map' }] },
          profile: { profile_key: 'choice_writer', snapshot_id: 45 },
        },
      ],
      total: 1,
    };
    mockVNPlayApi({ generations: initialGenerations, sessions: [session] });
    mocks.regenerateVNPlayGeneration.mockResolvedValue({
      turn_request_id: 21,
      status: 'completed',
      scene_version: 5,
      session: refreshedSession,
      events: [],
    });
    mocks.getVNPlaySession.mockResolvedValue(refreshedSession);
    mocks.listVNPlayGenerations
      .mockResolvedValueOnce(initialGenerations)
      .mockResolvedValueOnce(refreshedGenerations);

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('Ask about the map')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /regenerate intro:2:choice/i }));

    expect(await screen.findByText('Follow the refreshed map')).toBeInTheDocument();
    expect(mocks.listVNPlayGenerations).toHaveBeenCalledTimes(2);
  });

  it('runs pending generation confirm and cancel commands through backend actions', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 6,
      scene_state: {
        scene_version: 6,
        waiting_generation_request_id: 91,
      },
    };
    mockVNPlayApi({
      generations: {
        ...defaultGenerationHistory,
        items: [
          {
            id: 31,
            generation_id: 12,
            generation_point_key: 'intro:2:choice',
            revision_number: 1,
            status: 'pending_confirmation',
            active: false,
            output_schema: 'choice_set',
            public_output: { lead_in: 'Review generated choice.' },
            profile: { profile_key: 'choice_writer', snapshot_id: 44 },
          },
        ],
      },
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /confirm generation/i }));
    await waitFor(() => {
      expect(mocks.confirmVNPlayGenerationRequest).toHaveBeenCalledWith(1, 91, {
        client_scene_version: 6,
        idempotency_key: expect.stringMatching(/^generation-confirm-/),
      });
    });

    await user.click(screen.getByRole('button', { name: /^cancel$/i }));
    await waitFor(() => {
      expect(mocks.cancelVNPlayGenerationRequest).toHaveBeenCalledWith(1, 91, {
        client_scene_version: 6,
        idempotency_key: expect.stringMatching(/^generation-cancel-/),
      });
    });
  });

  it('blocks repeated generation actions while a request is in flight', async () => {
    const user = userEvent.setup();
    const deferred = createDeferred<{
      turn_request_id: number;
      status: 'completed';
      scene_version: number;
      scene_state: { scene_version: number };
      events: [];
    }>();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    mockVNPlayApi({
      generations: {
        ...defaultGenerationHistory,
        items: [
          {
            id: 31,
            generation_id: 12,
            generation_point_key: 'intro:2:choice',
            revision_number: 1,
            status: 'succeeded',
            active: false,
            output_schema: 'choice_set',
            public_output: { choices: [{ id: 'ask-map', text: 'Ask about the map' }] },
            profile: { profile_key: 'choice_writer', snapshot_id: 44 },
          },
        ],
      },
      sessions: [session],
    });
    mocks.regenerateVNPlayGeneration.mockReturnValue(deferred.promise);

    render(<VNPlayWorkspace />);

    const regenerate = await screen.findByRole('button', { name: /regenerate intro:2:choice/i });
    fireEvent.click(regenerate);
    fireEvent.click(regenerate);

    expect(mocks.regenerateVNPlayGeneration).toHaveBeenCalledTimes(1);
    deferred.resolve({ turn_request_id: 30, status: 'completed', scene_version: 5, scene_state: { scene_version: 5 }, events: [] });
    await waitFor(() => expect(regenerate).not.toBeDisabled());
    await user.click(regenerate);
    expect(mocks.regenerateVNPlayGeneration).toHaveBeenCalledTimes(2);
  });

  it('restricts debug controls for non-admin JWT users', async () => {
    window.localStorage.setItem('access_token', 'jwt-token');
    window.localStorage.setItem('user', JSON.stringify({ username: 'reader', role: 'user' }));
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    mockVNPlayApi({
      generations: {
        ...defaultGenerationHistory,
        items: [
          {
            id: 31,
            generation_id: 12,
            generation_point_key: 'intro:2:choice',
            revision_number: 1,
            status: 'succeeded',
            active: true,
            output_schema: 'choice_set',
            public_output: { lead_in: 'Public line' },
            profile: { profile_key: 'choice_writer', snapshot_id: 44 },
          },
        ],
      },
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    expect(await screen.findByText('Public line')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /debug intro:2:choice/i })).not.toBeInTheDocument();
    expect(screen.getByText('Debug restricted')).toBeInTheDocument();
  });

  it('gates moderation-blocked raw debug reveal behind explicit confirmation', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Archive Door',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    mockVNPlayApi({
      generations: {
        ...defaultGenerationHistory,
        items: [
          {
            id: 31,
            generation_id: 12,
            generation_point_key: 'intro:2:choice',
            revision_number: 1,
            status: 'moderation_blocked',
            active: false,
            output_schema: 'choice_set',
            public_output: {},
            applied_visuals: [],
            rejected_visuals: [],
            public_error_code: 'moderation_blocked',
            source: 'model',
            profile: { profile_key: 'choice_writer', snapshot_id: 44 },
            created_at: '2026-05-12T01:00:00Z',
          },
        ],
        total: 1,
      },
      sessions: [session],
    });
    mocks.getVNPlayGenerationRevisionDebug
      .mockResolvedValueOnce({
        id: 31,
        generation_id: 12,
        generation_request_id: 91,
        generation_point_key: 'intro:2:choice',
        revision_number: 1,
        status: 'moderation_blocked',
        output_schema: 'choice_set',
        public_output: {},
        raw_output_debug_state: 'redacted',
        raw_output_debug: null,
        parser_diagnostics: { error_code: 'ok' },
        moderation_diagnostics: { reason: 'policy_block' },
        model_metadata: { provider: 'hosted' },
        usage_metadata: {},
        request: {},
        profile: { profile_key: 'choice_writer', snapshot_id: 44 },
        created_at: '2026-05-12T01:00:00Z',
      } satisfies VNPlayGenerationRevisionDebugResponse)
      .mockResolvedValueOnce({
        id: 31,
        generation_id: 12,
        generation_request_id: 91,
        generation_point_key: 'intro:2:choice',
        revision_number: 1,
        status: 'moderation_blocked',
        output_schema: 'choice_set',
        public_output: {},
        raw_output_debug_state: 'revealed',
        raw_output_debug: { raw_text: 'blocked model text' },
        parser_diagnostics: {},
        moderation_diagnostics: { reason: 'policy_block' },
        model_metadata: { provider: 'hosted' },
        usage_metadata: {},
        request: {},
        profile: { profile_key: 'choice_writer', snapshot_id: 44 },
        created_at: '2026-05-12T01:00:00Z',
      } satisfies VNPlayGenerationRevisionDebugResponse);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /debug intro:2:choice revision 1/i }));
    expect(await screen.findByText(/Raw output: redacted/i)).toBeInTheDocument();
    expect(screen.queryByText('blocked model text')).not.toBeInTheDocument();
    expect(mocks.getVNPlayGenerationRevisionDebug).toHaveBeenCalledWith(1, 12, 31);

    await user.click(screen.getByRole('button', { name: /reveal moderation-blocked raw output/i }));
    expect(await screen.findByRole('dialog', { name: /Reveal moderation-blocked output/i })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Reveal raw output' }));

    await waitFor(() => {
      expect(mocks.getVNPlayGenerationRevisionDebug).toHaveBeenLastCalledWith(1, 12, 31, {
        include_blocked_raw: true,
        confirm: 'REVEAL_MODERATION_BLOCKED',
      });
    });
    expect(await screen.findByText(/blocked model text/)).toBeInTheDocument();
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
