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
  mocks.createVNPlayCheckpoint.mockResolvedValue(
    checkpoints[0] ?? {
      id: 100,
      session_id: 1,
      owner_user_id: 42,
      label: 'Checkpoint',
      scene_version: 0,
      scene_state_snapshot: { scene_version: 0 },
    }
  );
  mocks.listVNPlaySessions.mockResolvedValue(sessions);
  mocks.listVNPlayEvents.mockResolvedValue([]);
  mocks.listVNPlayBranches.mockResolvedValue(branches);
  mocks.listVNPlayCheckpoints.mockResolvedValue(checkpoints);
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
  mocks.restoreVNPlaySession.mockImplementation(async (sessionId: number) =>
    sessions.find((session) => session.id === sessionId) ?? sessions[0]
  );
  mocks.retryLastVNPlayTurn.mockResolvedValue({
    turn_request_id: 11,
    status: 'completed',
    scene_version: 1,
    scene_state: { scene_version: 1 },
    events: [],
  });
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

  it('loads branch and checkpoint metadata for the selected session', async () => {
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Branching story',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 3,
      scene_state: { scene_version: 3 },
    };
    mockVNPlayApi({
      branches: [
        {
          id: 7,
          session_id: 1,
          owner_user_id: 42,
          branch_label: 'Archive path',
          status: 'active',
        },
      ],
      checkpoints: [
        {
          id: 5,
          session_id: 1,
          owner_user_id: 42,
          label: 'Before archive',
          scene_version: 2,
          scene_state_snapshot: { scene_version: 2 },
        },
      ],
      sessions: [session],
    });

    render(<VNPlayWorkspace />);

    expect(await screen.findByTestId('vn-play-branch-count')).toHaveTextContent('1');
    expect(screen.getByTestId('vn-play-checkpoint-count')).toHaveTextContent('1');
    expect(screen.getByText('Archive path')).toBeInTheDocument();
    expect(screen.getByText('Before archive')).toBeInTheDocument();
  });

  it('creates a checkpoint from the current scene and refreshes metadata', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Checkpoint story',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 4,
      scene_state: { scene_version: 4 },
    };
    const checkpoint: VNPlayCheckpoint = {
      id: 9,
      session_id: 1,
      owner_user_id: 42,
      label: 'Before the tower',
      scene_version: 4,
      scene_state_snapshot: { scene_version: 4 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.listVNPlayCheckpoints
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([checkpoint]);
    mocks.createVNPlayCheckpoint.mockResolvedValueOnce(checkpoint);

    render(<VNPlayWorkspace />);

    await screen.findByText('Checkpoint story');
    await user.clear(screen.getByLabelText('Checkpoint label'));
    await user.type(screen.getByLabelText('Checkpoint label'), 'Before the tower');
    await user.click(screen.getByRole('button', { name: /create checkpoint/i }));

    await waitFor(() => {
      expect(mocks.createVNPlayCheckpoint).toHaveBeenCalledWith(1, {
        label: 'Before the tower',
        scene_version: 4,
      });
    });
    expect(await screen.findByText('Before the tower')).toBeInTheDocument();
  });

  it('restores a checkpoint and refreshes session state', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'story',
      title: 'Restore story',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 5,
      scene_state: { scene_version: 5 },
    };
    const restoredSession: VNPlaySession = {
      ...session,
      scene_version: 2,
      scene_state: { scene_version: 2 },
    };
    mockVNPlayApi({
      checkpoints: [
        {
          id: 5,
          session_id: 1,
          owner_user_id: 42,
          label: 'Before branch',
          scene_version: 2,
          scene_state_snapshot: { scene_version: 2 },
        },
      ],
      sessions: [session],
    });
    mocks.restoreVNPlaySession.mockResolvedValueOnce(restoredSession);
    mocks.getVNPlaySession.mockResolvedValue(restoredSession);

    render(<VNPlayWorkspace />);

    await user.click(await screen.findByRole('button', { name: /restore checkpoint before branch/i }));

    await waitFor(() => {
      expect(mocks.restoreVNPlaySession).toHaveBeenCalledWith(1, expect.objectContaining({
        checkpoint_id: 5,
      }));
    });
    expect(await screen.findByTestId('vn-play-scene-version')).toHaveTextContent('2');
  });

  it('shows stale scene recovery controls instead of a generic error', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Stale story',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 1,
      scene_state: { scene_version: 1 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.submitVNPlayTurn.mockRejectedValueOnce(Object.assign(new Error('stale_scene_version'), { status: 409 }));

    render(<VNPlayWorkspace />);

    await screen.findByText('Stale story');
    await user.type(screen.getByLabelText('Freeform input'), 'Open the archive');
    await user.click(screen.getByRole('button', { name: 'Send turn' }));

    expect(await screen.findByText('Scene changed on the server')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /reload session/i }));

    await waitFor(() => {
      expect(mocks.getVNPlaySession).toHaveBeenCalledWith(1);
    });
  });

  it('shows in-progress turn recovery controls with a poll action', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Busy story',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 1,
      scene_state: { scene_version: 1 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.submitVNPlayTurn.mockRejectedValueOnce(Object.assign(new Error('turn_in_progress'), { status: 409 }));

    render(<VNPlayWorkspace />);

    await screen.findByText('Busy story');
    await user.type(screen.getByLabelText('Freeform input'), 'Ask again');
    await user.click(screen.getByRole('button', { name: 'Send turn' }));

    expect(await screen.findByText('Turn is still in progress')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /poll session/i }));

    await waitFor(() => {
      expect(mocks.getVNPlaySession).toHaveBeenCalledWith(1);
    });
  });

  it('retries the last recoverable turn with a fresh idempotency key', async () => {
    const user = userEvent.setup();
    const session: VNPlaySession = {
      id: 1,
      mode: 'freeform',
      title: 'Retry story',
      primary_character_id: 1,
      vn_asset_pack_id: 2,
      scene_version: 6,
      scene_state: { scene_version: 6 },
    };
    mockVNPlayApi({ sessions: [session] });
    mocks.submitVNPlayTurn.mockResolvedValueOnce({
      turn_request_id: 22,
      status: 'model_failed',
      scene_version: 6,
      scene_state: { scene_version: 6 },
      error_code: 'model_failed',
      error_message: 'The model timed out.',
      events: [],
    });

    render(<VNPlayWorkspace />);

    await screen.findByText('Retry story');
    await user.type(screen.getByLabelText('Freeform input'), 'Try the door');
    await user.click(screen.getByRole('button', { name: 'Send turn' }));
    await user.click(await screen.findByRole('button', { name: /retry last turn/i }));

    await waitFor(() => {
      expect(mocks.retryLastVNPlayTurn).toHaveBeenCalledWith(1, expect.objectContaining({
        client_scene_version: 6,
        idempotency_key: expect.stringMatching(/^retry-/),
      }));
    });
  });
});
