import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import MatrixEditor from '@web/components/vn-assets/MatrixEditor';

describe('MatrixEditor', () => {
  it('updates variant count and total planned count before applying the starter matrix', async () => {
    const user = userEvent.setup();
    const onApplyMatrix = vi.fn();

    render(
      <MatrixEditor
        matrix={{
          key: 'starter',
          title: 'Starter',
          slot_count: 8,
          planned_output_count: 24,
          asset_types: ['background', 'sprite', 'cg'],
        }}
        onApplyMatrix={onApplyMatrix}
        selectedPackId={1}
      />
    );

    expect(screen.getByText('24 planned assets')).toBeInTheDocument();
    await user.clear(screen.getByLabelText('Variants per slot'));
    await user.type(screen.getByLabelText('Variants per slot'), '3');

    expect(screen.getByText('72 planned assets')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Apply starter matrix' }));

    expect(onApplyMatrix).toHaveBeenCalledWith('starter', { variant_count: 3 });
  });

  it('shows prompt preview warnings and omitted source counts', () => {
    render(
      <MatrixEditor
        matrix={{
          key: 'starter',
          title: 'Starter',
          slot_count: 1,
          planned_output_count: 1,
          asset_types: ['sprite'],
        }}
        promptPreview={{
          prompt: 'portrait sprite, neutral expression',
          negative_prompt: 'low quality',
          omitted_source_counts: { world_book: 2 },
          token_estimates: { total: 42 },
          warnings: ['world_book_truncated'],
        }}
        selectedPackId={1}
      />
    );

    expect(screen.getByText('world_book_truncated')).toBeInTheDocument();
    expect(screen.getByText('world_book: 2')).toBeInTheDocument();
    expect(screen.getByText('42 tokens')).toBeInTheDocument();
  });
});
