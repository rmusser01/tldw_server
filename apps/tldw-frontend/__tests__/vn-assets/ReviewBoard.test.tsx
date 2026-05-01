import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import ReadinessPanel from '@web/components/vn-assets/ReadinessPanel';
import ReviewBoard from '@web/components/vn-assets/ReviewBoard';
import type { VNAssetItem } from '@web/types/vn-assets';

const items: VNAssetItem[] = [
  {
    id: 11,
    pack_id: 1,
    slot_id: 4,
    variant_index: 0,
    generated_file_id: 101,
    mime_type: 'image/png',
    width: 512,
    height: 768,
    review_status: 'draft',
    preferred: false,
    source: 'generated',
  },
  {
    id: 12,
    pack_id: 1,
    slot_id: 4,
    variant_index: 1,
    generated_file_id: 102,
    mime_type: 'image/png',
    width: 512,
    height: 768,
    review_status: 'approved',
    preferred: false,
    source: 'generated',
  },
];

describe('ReviewBoard', () => {
  it('bulk approves and rejects selected draft items', async () => {
    const user = userEvent.setup();
    const onBulkReview = vi.fn();

    render(<ReviewBoard items={items} onBulkReview={onBulkReview} />);

    await user.click(screen.getByLabelText('Select item 11'));
    await user.click(screen.getByRole('button', { name: 'Approve selected' }));
    expect(onBulkReview).toHaveBeenCalledWith({ item_ids: [11], review_status: 'approved' });

    await user.click(screen.getByLabelText('Select item 11'));
    await user.click(screen.getByLabelText('Select item 12'));
    await user.click(screen.getByRole('button', { name: 'Reject selected' }));
    expect(onBulkReview).toHaveBeenLastCalledWith({ item_ids: [12], review_status: 'rejected' });
  });

  it('supports setting one preferred item per slot', async () => {
    const user = userEvent.setup();
    const onSetPreferred = vi.fn();

    render(<ReviewBoard items={items} onSetPreferred={onSetPreferred} />);

    await user.click(screen.getByLabelText('Set preferred item 12'));

    expect(onSetPreferred).toHaveBeenCalledWith(12);
  });
});

describe('ReadinessPanel', () => {
  it('shows optional failed slots as warnings without blocking readiness', () => {
    render(
      <ReadinessPanel
        readiness={{
          ready: true,
          status: 'ready_with_warnings',
          warnings: ['optional_slot_failed:3', 'depth_unavailable'],
          errors: [],
        }}
      />
    );

    expect(screen.getByText('Ready with warnings')).toBeInTheDocument();
    expect(screen.getByText('optional_slot_failed:3')).toBeInTheDocument();
    expect(screen.queryByText('Blocked')).not.toBeInTheDocument();
  });
});
