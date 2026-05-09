import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VNAssetReadiness } from '@web/types/vn-assets';
import type { VNPlayBranch, VNPlayCheckpoint, VNPlaySession } from '@web/types/vn-play';

const mocks = vi.hoisted(() => ({
  createVNPlayCheckpoint: vi.fn(),
  createVNPlaySession: vi.fn(),
  getVNPlaySession: vi.fn(),
  getVNAssetReadiness: vi.fn(),
  listVNPlayBranches: vi.fn(),
  listVNPlayCheckpoints: vi.fn(),
  listVNPlayEvents: vi.fn(),
  listVNPlaySessions: vi.fn(),
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

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, reject, resolve };
}

function readyReadiness(): VNAssetReadiness {
  return {
    ready: true,
    status: 'ready',
    warnings: [],
    errors: [],
  };
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
    expect(screen.getByText(/archive, story/)).toBeInTheDocument();
    expect(screen.getByLabelText('VN asset pack')).toBeInTheDocument();
    expect(screen.getAllByText('Moonlit Archive Pack').length).toBeGreaterThan(0);
    expect(screen.getByText(/Runtime-ready VN poses and backdrops/)).toBeInTheDocument();

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

  it('renders character and pack selectors before readiness checks finish', async () => {
    const user = userEvent.setup();
    const readiness = createDeferred<VNAssetReadiness>();
    mockVNPlayApi({ sessions: [] });
    mocks.getVNAssetReadiness.mockReturnValue(readiness.promise);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByLabelText('Character')).toBeInTheDocument();
    expect(screen.getAllByText('Mira Vale').length).toBeGreaterThan(0);
    expect(screen.getByRole('option', { name: /Moonlit Archive Pack/i })).toBeInTheDocument();

    readiness.resolve(readyReadiness());
  });

  it('limits concurrent readiness checks while loading selector data', async () => {
    const user = userEvent.setup();
    const packs = Array.from({ length: 6 }, (_, index) => ({
      ...defaultPacks[0],
      id: index + 1,
      title: `Pack ${index + 1}`,
    }));
    const pendingReadiness = new Map<number, ReturnType<typeof createDeferred<VNAssetReadiness>>>();
    let activeRequests = 0;
    let maxActiveRequests = 0;

    mockVNPlayApi({ sessions: [] });
    mocks.listVNAssetPacks.mockResolvedValue(packs);
    mocks.getVNAssetReadiness.mockImplementation((packId: number) => {
      activeRequests += 1;
      maxActiveRequests = Math.max(maxActiveRequests, activeRequests);
      const readiness = createDeferred<VNAssetReadiness>();
      pendingReadiness.set(packId, readiness);
      return readiness.promise.finally(() => {
        activeRequests -= 1;
      });
    });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    await waitFor(() => expect(mocks.getVNAssetReadiness).toHaveBeenCalledTimes(4));
    expect(maxActiveRequests).toBeLessThanOrEqual(4);

    for (const readiness of pendingReadiness.values()) {
      readiness.resolve(readyReadiness());
    }
    await waitFor(() => expect(mocks.getVNAssetReadiness).toHaveBeenCalledTimes(6));
    for (const readiness of pendingReadiness.values()) {
      readiness.resolve(readyReadiness());
    }
    await waitFor(() => expect(activeRequests).toBe(0));
    expect(maxActiveRequests).toBeLessThanOrEqual(4);
  });

  it('marks incompatible asset packs as unavailable for the selected character', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNAssetPacks.mockResolvedValue([
      ...defaultPacks,
      {
        id: 13,
        title: 'Rival Pack',
        primary_character_id: 99,
        status: 'approved',
        content_rating: 'general',
        planned_output_count: 6,
      },
    ]);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new story/i }));

    expect(await screen.findByRole('option', {
      name: /Rival Pack.*incompatible with Mira Vale/i,
    })).toBeDisabled();
    expect(screen.getByText(/Rival Pack is attached to character 99/i)).toBeInTheDocument();
  });

  it('shows readiness, draft, missing-byte, and content-rating warnings before submit', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listVNAssetPacks.mockResolvedValue([
      {
        ...defaultPacks[0],
        status: 'draft',
        content_rating: 'mature',
      },
    ]);
    mocks.getVNAssetReadiness.mockResolvedValue({
      ready: false,
      status: 'missing_assets',
      warnings: ['2 required runtime assets are missing bytes'],
      errors: ['sprite.primary has no approved file bytes'],
    });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));

    expect(await screen.findByText(/Pack status is draft/i)).toBeInTheDocument();
    expect(screen.getByText(/2 required runtime assets are missing bytes/i)).toBeInTheDocument();
    expect(screen.getByText(/sprite.primary has no approved file bytes/i)).toBeInTheDocument();
    expect(screen.getByText(/Pack content rating mature differs from session rating general/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create session' })).toBeDisabled();
  });

  it('renders duplicate readiness warnings without duplicate React keys', async () => {
    const user = userEvent.setup();
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    mockVNPlayApi({ sessions: [] });
    mocks.getVNAssetReadiness.mockResolvedValue({
      ready: true,
      status: 'ready',
      warnings: ['Review duplicate slot metadata', 'Review duplicate slot metadata'],
      errors: [],
    });

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

  it('keeps manual ID entry available when setup selectors fail to load', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });
    mocks.listCharacters.mockRejectedValueOnce(new Error('characters offline'));

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    expect(await screen.findByText(/Could not load setup selectors/i)).toBeInTheDocument();

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
    mocks.listCharacters.mockResolvedValue([]);
    mocks.listVNAssetPacks.mockResolvedValue([]);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new story/i }));

    expect(await screen.findByText(/No characters available/i)).toBeInTheDocument();
    expect(screen.getByText(/Create or import a character before starting VN Play/i)).toBeInTheDocument();
    expect(screen.getByText(/No VN asset packs available/i)).toBeInTheDocument();
    expect(screen.getByText(/Prepare or review a VN asset pack before starting VN Play/i)).toBeInTheDocument();
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
