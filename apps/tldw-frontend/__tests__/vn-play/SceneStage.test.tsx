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

  it('renders backend scene metadata, depth layer, and visual fallback copy', () => {
    const sceneState = {
      scene_version: 1,
      background: { content_url: '/bg.png', labels: { location: 'archive' } },
      depth: { content_url: '/depth.png' },
      active_sprite_items: [
        { item_id: 3, content_url: '/sprite.png', metadata: { pose: 'thinking' } },
      ],
      location_key: 'library',
      mood: 'tense',
      time_of_day: 'night',
      weather: 'rain',
      warnings: [
        {
          code: 'visual_directive_rejected',
          message: 'Sprite expression was unavailable.',
          slot_key: 'sprite.mira.angry',
          asset_type: 'sprite',
        },
      ],
    } satisfies VNPlaySceneState;

    render(<SceneStage events={[]} sceneState={sceneState} />);

    expect(screen.getByAltText(/scene depth layer/i)).toHaveAttribute('src', '/depth.png');
    expect(screen.getByText('Location: library')).toBeInTheDocument();
    expect(screen.getByText('Mood: tense')).toBeInTheDocument();
    expect(screen.getByText('Time: night')).toBeInTheDocument();
    expect(screen.getByText('Weather: rain')).toBeInTheDocument();
    expect(screen.getByText(/Sprite expression was unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/sprite.mira.angry/i)).toBeInTheDocument();
  });

  it('renders a user-safe fallback when no scene visuals are available', () => {
    render(<SceneStage events={[]} sceneState={{ scene_version: 1 }} />);

    expect(screen.getByText('No scene visuals available')).toBeInTheDocument();
    expect(screen.getByText(/The backend did not provide a background or active sprite/i)).toBeInTheDocument();
  });

  it('does not show the no-visuals fallback when sprites render without a background', () => {
    render(
      <SceneStage
        events={[]}
        sceneState={{
          scene_version: 1,
          active_sprites: [
            { item_id: 1, content_url: '' },
            { item_id: 2, content_url: '/sprite.png' },
          ],
        }}
      />
    );

    expect(screen.getByAltText('Character sprite 1')).toHaveAttribute('src', '/sprite.png');
    expect(screen.queryByText('No scene visuals available')).not.toBeInTheDocument();
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

  it('labels generated choices from backend metadata', () => {
    const choices: VNPlayChoice[] = [
      {
        id: 'c1',
        text: 'Ask about the generated map',
        metadata: {
          source: 'generated',
          generation_point_key: 'intro:choices',
          status: 'succeeded',
        },
      },
    ];

    render(
      <ChoicePanel
        choices={choices}
        sceneVersion={1}
        sessionId={1}
        onTurn={vi.fn()}
      />
    );

    expect(screen.getByRole('button', { name: /ask about the generated map/i })).toBeInTheDocument();
    expect(screen.getByText('Generated')).toBeInTheDocument();
    expect(screen.getByText('intro:choices')).toBeInTheDocument();
  });

  it('labels generated choices from the top-level source field', () => {
    const choices: VNPlayChoice[] = [
      {
        id: 'c1',
        text: 'Follow the generated clue',
        source: 'generated',
      },
    ];

    render(
      <ChoicePanel
        choices={choices}
        sceneVersion={1}
        sessionId={1}
        onTurn={vi.fn()}
      />
    );

    expect(screen.getByRole('button', { name: /follow the generated clue/i })).toBeInTheDocument();
    expect(screen.getByText('Generated')).toBeInTheDocument();
  });

  it('uses play-focused copy when no choices are available', () => {
    render(
      <ChoicePanel
        choices={[]}
        sceneVersion={1}
        sessionId={1}
        onTurn={vi.fn()}
      />
    );

    expect(screen.getByText('No choices are available yet.')).toBeInTheDocument();
  });
});
