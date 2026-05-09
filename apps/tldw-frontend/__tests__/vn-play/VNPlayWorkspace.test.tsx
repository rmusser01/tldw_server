import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VNPlaySession } from '@web/types/vn-play';

const mocks = vi.hoisted(() => ({
  createVNPlaySession: vi.fn(),
  getVNPlaySession: vi.fn(),
  listVNPlayEvents: vi.fn(),
  listVNPlaySessions: vi.fn(),
  submitVNPlayTurn: vi.fn(),
}));

vi.mock('@web/lib/api/vnPlay', () => ({
  createVNPlaySession: (...args: unknown[]) => mocks.createVNPlaySession(...args),
  getVNPlaySession: (...args: unknown[]) => mocks.getVNPlaySession(...args),
  listVNPlayEvents: (...args: unknown[]) => mocks.listVNPlayEvents(...args),
  listVNPlaySessions: (...args: unknown[]) => mocks.listVNPlaySessions(...args),
  submitVNPlayTurn: (...args: unknown[]) => mocks.submitVNPlayTurn(...args),
}));

import VNPlayWorkspace from '@web/components/vn-play/VNPlayWorkspace';

function mockVNPlayApi({ sessions = [] }: { sessions?: VNPlaySession[] } = {}) {
  mocks.listVNPlaySessions.mockResolvedValue(sessions);
  mocks.listVNPlayEvents.mockResolvedValue([]);
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
});
