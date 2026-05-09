import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VNPlayBranch, VNPlayCheckpoint, VNPlaySession } from '@web/types/vn-play';

const mocks = vi.hoisted(() => ({
  createVNPlayCheckpoint: vi.fn(),
  createVNPlaySession: vi.fn(),
  getVNPlaySession: vi.fn(),
  listVNPlayBranches: vi.fn(),
  listVNPlayCheckpoints: vi.fn(),
  listVNPlayEvents: vi.fn(),
  listVNPlaySessions: vi.fn(),
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

import VNPlayWorkspace from '@web/components/vn-play/VNPlayWorkspace';

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

  it('creates a freeform session from the dialog', async () => {
    const user = userEvent.setup();
    mockVNPlayApi({ sessions: [] });

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /new freeform/i }));
    await user.clear(screen.getByLabelText('Title'));
    await user.type(screen.getByLabelText('Title'), 'Moonlit Archive');
    await user.clear(screen.getByLabelText('Primary character ID'));
    await user.type(screen.getByLabelText('Primary character ID'), '7');
    await user.clear(screen.getByLabelText('VN asset pack ID'));
    await user.type(screen.getByLabelText('VN asset pack ID'), '12');
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
