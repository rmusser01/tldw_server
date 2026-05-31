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
