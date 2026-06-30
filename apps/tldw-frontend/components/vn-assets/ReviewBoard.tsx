import React, { useMemo, useState } from 'react';
import { Check, Star, X } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import type {
  VNAssetBulkReviewRequest,
  VNAssetItem,
  VNAssetReviewStatus,
} from '@web/types/vn-assets';

export interface ReviewBoardProps {
  items: VNAssetItem[];
  onBulkReview?: (request: VNAssetBulkReviewRequest) => void;
  onSetPreferred?: (itemId: number) => void;
}

function reviewBadgeVariant(status: string): 'danger' | 'neutral' | 'success' | 'warning' {
  if (status === 'approved') return 'success';
  if (status === 'rejected' || status === 'hidden') return 'danger';
  if (status === 'draft') return 'warning';
  return 'neutral';
}

function itemLabel(item: VNAssetItem): string {
  return `Slot ${item.slot_id} · Variant ${item.variant_index + 1}`;
}

export default function ReviewBoard({ items, onBulkReview, onSetPreferred }: ReviewBoardProps) {
  const [selectedItemIds, setSelectedItemIds] = useState<Set<number>>(() => new Set());

  const counts = useMemo(() => {
    return items.reduce(
      (accumulator, item) => {
        if (item.review_status === 'approved') accumulator.approved += 1;
        if (item.review_status === 'draft') accumulator.draft += 1;
        return accumulator;
      },
      { approved: 0, draft: 0 }
    );
  }, [items]);

  const selectedIds = useMemo(() => Array.from(selectedItemIds).sort((left, right) => left - right), [selectedItemIds]);

  const toggleSelection = (itemId: number) => {
    setSelectedItemIds((previous) => {
      const next = new Set(previous);
      if (next.has(itemId)) {
        next.delete(itemId);
      } else {
        next.add(itemId);
      }
      return next;
    });
  };

  const handleBulkReview = (reviewStatus: VNAssetReviewStatus) => {
    if (selectedIds.length === 0) return;
    onBulkReview?.({ item_ids: selectedIds, review_status: reviewStatus });
  };

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Review board</h2>
        <Badge variant="neutral">{items.length} items</Badge>
      </div>
      <div className="mb-4 grid grid-cols-3 gap-3 text-sm">
        <div>
          <p className="text-text-muted">Draft</p>
          <p className="font-medium">{counts.draft}</p>
        </div>
        <div>
          <p className="text-text-muted">Approved</p>
          <p className="font-medium">{counts.approved}</p>
        </div>
        <div>
          <p className="text-text-muted">Selected</p>
          <p className="font-medium">{selectedIds.length}</p>
        </div>
      </div>
      <div className="mb-4 flex flex-wrap gap-2">
        <Button
          className="gap-2"
          disabled={selectedIds.length === 0}
          onClick={() => handleBulkReview('approved')}
          size="sm"
          type="button"
        >
          <Check aria-hidden className="h-4 w-4" />
          Approve selected
        </Button>
        <Button
          className="gap-2"
          disabled={selectedIds.length === 0}
          onClick={() => handleBulkReview('rejected')}
          size="sm"
          type="button"
          variant="danger"
        >
          <X aria-hidden className="h-4 w-4" />
          Reject selected
        </Button>
      </div>

      {items.length === 0 ? (
        <p className="text-sm text-text-muted">Generated and uploaded variants will appear here for review.</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {items.map((item) => {
            const selected = selectedItemIds.has(item.id);

            return (
              <article
                key={item.id}
                className={`rounded-md border bg-bg p-3 transition-colors ${
                  selected ? 'border-primary' : 'border-border'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <label className="flex items-center gap-2 text-sm font-medium">
                    <input
                      aria-label={`Select item ${item.id}`}
                      checked={selected}
                      className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                      type="checkbox"
                      onChange={() => toggleSelection(item.id)}
                    />
                    {itemLabel(item)}
                  </label>
                  <Button
                    aria-label={`Set preferred item ${item.id}`}
                    className={item.preferred ? 'text-primary' : 'text-text-muted'}
                    onClick={() => onSetPreferred?.(item.id)}
                    size="xs"
                    type="button"
                    variant="ghost"
                  >
                    <Star aria-hidden className="h-4 w-4" />
                  </Button>
                </div>
                <div
                  className="mt-3 flex items-center justify-center rounded-md bg-surface2 text-xs text-text-muted"
                  style={{ aspectRatio: '2 / 3' }}
                >
                  {item.width && item.height ? `${item.width} x ${item.height}` : 'Preview pending'}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Badge variant={reviewBadgeVariant(String(item.review_status))}>
                    {item.review_status}
                  </Badge>
                  <Badge variant="neutral">{item.source}</Badge>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
