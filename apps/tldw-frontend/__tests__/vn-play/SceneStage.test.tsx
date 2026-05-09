import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VNPlayChoice, VNPlayEvent, VNPlaySceneState } from '@web/types/vn-play';

const mocks = vi.hoisted(() => ({
  submitVNPlayTurn: vi.fn(),
}));

vi.mock('@web/lib/api/vnPlay', () => ({
  submitVNPlayTurn: (...args: unknown[]) => mocks.submitVNPlayTurn(...args),
}));

import ChoicePanel from '@web/components/vn-play/ChoicePanel';
import SceneStage from '@web/components/vn-play/SceneStage';

function modelTurnEvent(payload: Record<string, unknown>): VNPlayEvent {
  return {
    id: 1,
    session_id: 1,
    owner_user_id: 42,
    sequence_number: 1,
    event_type: 'model_turn',
    event_payload: payload,
    source: 'model',
  };
}

describe('VN play scene components', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.submitVNPlayTurn.mockResolvedValue({
      turn_request_id: 3,
      status: 'completed',
      scene_version: 2,
      scene_state: { scene_version: 2 },
      events: [],
    });
  });

  it('renders background, sprite, dialogue, and warnings', () => {
    const sceneState = {
      scene_version: 1,
      background: { content_url: '/bg.png', labels: { location: 'library' } },
      active_sprites: [
        { item_id: 2, content_url: '/sprite.png', labels: { emotion: 'happy' } },
      ],
      warnings: [{ reason: 'asset_not_found', slot_key: 'sprite.angry' }],
    } satisfies VNPlaySceneState;

    render(
      <SceneStage
        events={[modelTurnEvent({ dialogue: [{ speaker: 'Mira', text: 'Hello.' }] })]}
        sceneState={sceneState}
      />
    );

    expect(screen.getByAltText(/background/i)).toHaveAttribute('src', '/bg.png');
    expect(screen.getByAltText(/sprite/i)).toHaveAttribute('src', '/sprite.png');
    expect(screen.getByText('Hello.')).toBeInTheDocument();
    expect(screen.getByText(/asset_not_found/i)).toBeInTheDocument();
  });

  it('submits a story choice with current scene version', async () => {
    const user = userEvent.setup();
    const onTurn = vi.fn();
    const choices: VNPlayChoice[] = [{ id: 'c1', text: 'Open the door' }];

    render(
      <ChoicePanel
        choices={choices}
        sceneVersion={1}
        sessionId={1}
        onTurn={onTurn}
      />
    );

    await user.click(screen.getByRole('button', { name: /open the door/i }));

    await waitFor(() => {
      expect(mocks.submitVNPlayTurn).toHaveBeenCalledWith(1, expect.objectContaining({
        choice_id: 'c1',
        client_scene_version: 1,
      }));
    });
    expect(onTurn).toHaveBeenCalledWith(expect.objectContaining({ scene_version: 2 }));
  });
});
