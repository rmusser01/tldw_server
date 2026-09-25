import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import GenerationMonitor from '@web/components/vn-assets/GenerationMonitor';

const slots = [
  {
    id: 1,
    pack_id: 1,
    asset_type: 'sprite',
    slot_key: 'sprite.primary',
    variant_count: 1,
    status: 'planned',
  },
];

describe('GenerationMonitor', () => {
  it('offers a new start instead of retry for a legacy batch without a recipe', () => {
    render(
      <GenerationMonitor
        generation={{ batch_id: 4, status: 'failed', recipe_available: false, selected_slot_ids: [1] }}
        slots={[{ ...slots[0], status: 'failed', last_error: 'worker interrupted' }]}
        onStartGeneration={vi.fn()}
        onRetrySlot={vi.fn()}
      />
    );

    expect(screen.getByRole('button', { name: 'Retry sprite.primary' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start generation' })).toBeEnabled();
    expect(screen.getByText('Original settings unavailable. Start generation to use current settings.')).toBeVisible();
  });

  it('keeps Retry available when an older failed slot has its own recipe', () => {
    render(
      <GenerationMonitor
        generation={{
          batch_id: 4, status: 'failed', recipe_available: false,
          selected_slot_ids: [1], failed_slot_batch_ids: { 1: 3 },
        }}
        slots={[{ ...slots[0], status: 'failed', last_error: 'worker interrupted' }]}
        onStartGeneration={vi.fn()}
        onRetrySlot={vi.fn()}
      />
    );

    expect(screen.getByRole('button', { name: 'Retry sprite.primary' })).toBeEnabled();
  });

  it('only allows generation start outside active lifecycle states', async () => {
    const user = userEvent.setup();
    const onStartGeneration = vi.fn();

    const { rerender } = render(
      <GenerationMonitor
        generation={{ status: 'queued' }}
        slots={slots}
        onStartGeneration={onStartGeneration}
      />
    );

    expect(screen.getByRole('button', { name: 'Start generation' })).toBeDisabled();

    rerender(
      <GenerationMonitor
        generation={{ status: 'failed' }}
        slots={slots}
        onStartGeneration={onStartGeneration}
      />
    );

    await user.click(screen.getByRole('button', { name: 'Start generation' }));

    expect(onStartGeneration).toHaveBeenCalledTimes(1);
  });

  it('only allows cancellation for active generation states', async () => {
    const user = userEvent.setup();
    const onCancelGeneration = vi.fn();

    const { rerender } = render(
      <GenerationMonitor
        generation={{ status: 'cancelled' }}
        slots={slots}
        onCancelGeneration={onCancelGeneration}
      />
    );

    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();

    rerender(
      <GenerationMonitor
        generation={{ status: 'processing' }}
        slots={slots}
        onCancelGeneration={onCancelGeneration}
      />
    );

    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(onCancelGeneration).toHaveBeenCalledTimes(1);
  });
});
