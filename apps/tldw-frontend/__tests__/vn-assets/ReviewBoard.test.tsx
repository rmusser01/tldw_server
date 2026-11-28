import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
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
  afterEach(() => vi.unstubAllGlobals());

  it('bulk approves and rejects selected draft items', async () => {
    const user = userEvent.setup();
    const onBulkReview = vi.fn();

    render(<ReviewBoard items={items} onBulkReview={onBulkReview} />);

    await user.click(screen.getByLabelText('Select item 11'));
    await user.click(screen.getByRole('button', { name: 'Approve selected' }));
    expect(onBulkReview).toHaveBeenCalledWith({ item_ids: [11], review_status: 'approved' });

    expect(screen.getByLabelText('Select item 11')).not.toBeChecked();
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

  it('keeps visible selection and keyboard focus when item data refreshes', async () => {
    const user = userEvent.setup();
    const onBulkReview = vi.fn();
    const { rerender } = render(<ReviewBoard items={items} onBulkReview={onBulkReview} />);
    const filter = screen.getByRole('combobox', { name: 'Review status' });
    await user.selectOptions(filter, 'draft');
    await user.click(screen.getByRole('button', { name: 'Select visible' }));
    const checkbox = screen.getByRole('checkbox', { name: 'Select item 11' });
    checkbox.focus();

    rerender(<ReviewBoard items={items.map((item) => ({ ...item }))} onBulkReview={onBulkReview} />);

    expect(checkbox).toBeChecked();
    expect(checkbox).toHaveFocus();
    expect(screen.queryByRole('checkbox', { name: 'Select item 12' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Approve selected' }));
    expect(onBulkReview).toHaveBeenCalledWith({ item_ids: [11], review_status: 'approved' });
    expect(filter).toHaveFocus();
  });

  it('keeps a failed selection available for retry and prevents concurrent review writes', async () => {
    const user = userEvent.setup();
    let fail!: (error: Error) => void;
    const onBulkReview = vi.fn().mockImplementationOnce(() => new Promise((_resolve, reject) => { fail = reject; }));
    const onSetPreferred = vi.fn();
    render(<ReviewBoard items={items} onBulkReview={onBulkReview} onSetPreferred={onSetPreferred} />);
    const checkbox = screen.getByRole('checkbox', { name: 'Select item 11' });
    const approve = screen.getByRole('button', { name: 'Approve selected' });
    await user.click(checkbox);
    await user.click(approve);
    await user.click(screen.getByRole('button', { name: 'Reject selected' }));
    await user.click(screen.getByRole('button', { name: 'Set preferred item 12' }));
    expect(onBulkReview).toHaveBeenCalledTimes(1);
    expect(onSetPreferred).not.toHaveBeenCalled();

    await act(async () => fail(new Error('Review service unavailable')));

    expect(screen.getByRole('alert')).toHaveTextContent('Review service unavailable');
    expect(checkbox).toBeChecked();
    await user.click(approve);
    expect(onBulkReview).toHaveBeenCalledTimes(2);
    expect(checkbox).not.toBeChecked();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('loads previews on demand and releases the image when the item leaves the view', async () => {
    const user = userEvent.setup();
    const createObjectURL = vi.fn(() => 'blob:test-preview');
    const revokeObjectURL = vi.fn();
    vi.stubGlobal('URL', Object.assign(class extends URL {}, { createObjectURL, revokeObjectURL }));
    const onLoadPreview = vi.fn().mockResolvedValue(new Blob(['preview'], { type: 'image/png' }));
    const onBulkReview = vi.fn();
    const { rerender } = render(<ReviewBoard items={items} onBulkReview={onBulkReview} onLoadPreview={onLoadPreview} />);
    expect(onLoadPreview).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Preview item 11' }));

    expect(await screen.findByRole('img', { name: 'Slot 4 · Variant 1 preview' })).toHaveAttribute('src', 'blob:test-preview');
    const signal = onLoadPreview.mock.calls[0][1] as AbortSignal;
    expect(onLoadPreview).toHaveBeenCalledWith(11, signal);
    rerender(<ReviewBoard items={[items[1]]} onBulkReview={onBulkReview} onLoadPreview={onLoadPreview} />);
    await waitFor(() => expect(revokeObjectURL).toHaveBeenCalledWith('blob:test-preview'));
    expect(signal.aborted).toBe(true);
  });

  it('drops a selection and preview when a refreshed card leaves the active filter', async () => {
    const user = userEvent.setup();
    const onBulkReview = vi.fn();
    const onLoadPreview = vi.fn().mockRejectedValue(new Error('Preview unavailable'));
    const board = (currentItems: VNAssetItem[]) => (
      <ReviewBoard items={currentItems} onBulkReview={onBulkReview} onLoadPreview={onLoadPreview} />
    );
    const { rerender } = render(board(items));
    await user.selectOptions(screen.getByRole('combobox', { name: 'Review status' }), 'draft');
    await user.click(screen.getByRole('checkbox', { name: 'Select item 11' }));
    await user.click(screen.getByRole('button', { name: 'Preview item 11' }));
    await screen.findByRole('button', { name: 'Retry preview' });

    rerender(board([{ ...items[0], review_status: 'approved' }, items[1]]));
    rerender(board(items));

    expect(screen.getByRole('checkbox', { name: 'Select item 11' })).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Preview item 11' })).toHaveAttribute('aria-expanded', 'false');
    expect(onLoadPreview).toHaveBeenCalledTimes(1);
  });

  it('restores focus to the preview toggle when retry starts', async () => {
    const user = userEvent.setup();
    const onLoadPreview = vi.fn()
      .mockRejectedValueOnce(new Error('Preview unavailable'))
      .mockImplementationOnce(() => new Promise(() => {}));
    render(<ReviewBoard items={items} onLoadPreview={onLoadPreview} />);
    await user.click(screen.getByRole('button', { name: 'Preview item 11' }));
    const retry = await screen.findByRole('button', { name: 'Retry preview' });
    retry.focus();
    await user.keyboard('{Enter}');

    expect(screen.getByRole('button', { name: 'Hide preview for item 11' })).toHaveFocus();
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
