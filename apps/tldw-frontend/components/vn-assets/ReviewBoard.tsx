import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
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
  reviewBusy?: boolean;
  suspended?: boolean;
  onBulkReview?: (request: VNAssetBulkReviewRequest) => boolean | void | Promise<boolean | void>;
  onSetPreferred?: (itemId: number) => boolean | void | Promise<boolean | void>;
  onLoadPreview?: (itemId: number, signal: AbortSignal) => Promise<Blob>;
}

const statusOptions = [
  ['all', 'All statuses'], ['draft', 'Draft'], ['approved', 'Approved'],
  ['rejected', 'Rejected'], ['hidden', 'Hidden'],
] as const;

type ReviewFilter = (typeof statusOptions)[number][0];

function reviewBadgeVariant(status: string): 'danger' | 'neutral' | 'success' | 'warning' {
  if (status === 'approved') return 'success';
  if (status === 'rejected' || status === 'hidden') return 'danger';
  if (status === 'draft') return 'warning';
  return 'neutral';
}

function itemLabel(item: VNAssetItem): string {
  return `Slot ${item.slot_id} · Variant ${item.variant_index + 1}`;
}

function AssetPreview({ item, load, onRetry }: {
  item: VNAssetItem;
  load: NonNullable<ReviewBoardProps['onLoadPreview']>;
  onRetry: () => void;
}) {
  const [preview, setPreview] = useState<{ url: string } | 'failed' | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let url: string | undefined;
    void load(item.id, controller.signal).then((blob) => {
      if (controller.signal.aborted) return;
      if (!['image/png', 'image/jpeg', 'image/webp'].includes(blob.type.split(';')[0].trim().toLowerCase())) {
        setPreview('failed');
        return;
      }
      url = URL.createObjectURL(blob);
      setPreview({ url });
    }).catch(() => {
      if (!controller.signal.aborted) setPreview('failed');
    });
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [attempt, item.id, load]);

  if (preview === 'failed') {
    return (
      <div className="space-y-2 text-center">
        <p role="status">Preview unavailable.</p>
        <Button type="button" variant="secondary" size="sm" onClick={() => {
          onRetry();
          setPreview(null);
          setAttempt((previous) => previous + 1);
        }}>Retry preview</Button>
      </div>
    );
  }
  if (!preview) return <p role="status">Loading preview…</p>;
  return (
    // eslint-disable-next-line @next/next/no-img-element -- The authenticated preview is an in-memory blob.
    <img src={preview.url} alt={`${itemLabel(item)} preview`} className="h-full w-full object-contain"
      onError={() => setPreview('failed')} />
  );
}

export default function ReviewBoard({
  items, reviewBusy = false, suspended = false, onBulkReview, onSetPreferred, onLoadPreview,
}: ReviewBoardProps) {
  const filterId = useId();
  const filterRef = useRef<HTMLSelectElement>(null);
  const previewToggleRef = useRef<HTMLButtonElement>(null);
  const lastFocusedControl = useRef<HTMLElement | null>(null);
  const saving = useRef(false);
  const [pending, setPending] = useState(false);
  const [feedback, setFeedback] = useState<{ error: boolean; text: string } | null>(null);
  const [filter, setFilter] = useState<ReviewFilter>('all');
  const [previewItemId, setPreviewItemId] = useState<number | null>(null);
  const [selectedItemIds, setSelectedItemIds] = useState<Set<number>>(() => new Set());
  const busy = pending || reviewBusy;

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

  const visibleItems = useMemo(
    () => filter === 'all' ? items : items.filter((item) => item.review_status === filter),
    [filter, items],
  );
  const selectedIds = useMemo(() => visibleItems.filter((item) => selectedItemIds.has(item.id))
    .map((item) => item.id).sort((left, right) => left - right), [selectedItemIds, visibleItems]);

  useEffect(() => {
    const idsInView = new Set(visibleItems.map((item) => item.id));
    setSelectedItemIds((previous) => {
      const remaining = new Set([...previous].filter((id) => idsInView.has(id)));
      return remaining.size === previous.size ? previous : remaining;
    });
    setPreviewItemId((previous) => previous !== null && idsInView.has(previous) ? previous : null);
    const control = lastFocusedControl.current;
    if (!suspended && control && !control.isConnected && document.activeElement === document.body) {
      filterRef.current?.focus();
    }
  }, [suspended, visibleItems]);

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

  const saveReview = async (
    persist: () => boolean | void | Promise<boolean | void>, onSaved?: () => void,
  ) => {
    if (busy || saving.current) return;
    saving.current = true;
    setPending(true);
    setFeedback(null);
    try {
      if (await persist() === false) return;
      onSaved?.();
      setFeedback({ error: false, text: 'Review saved.' });
    } catch (error) {
      setFeedback({ error: true, text: error instanceof Error ? error.message : 'Could not save review. Try again.' });
    } finally {
      saving.current = false;
      setPending(false);
    }
  };

  const handleBulkReview = (reviewStatus: VNAssetReviewStatus, origin: HTMLButtonElement) => {
    if (!onBulkReview || selectedIds.length === 0) return;
    void saveReview(() => onBulkReview({ item_ids: selectedIds, review_status: reviewStatus }), () => {
      const submitted = new Set(selectedIds);
      setSelectedItemIds((previous) => new Set([...previous].filter((id) => !submitted.has(id))));
      if (document.activeElement === origin) filterRef.current?.focus();
    });
  };

  if (suspended) {
    return (
      <section className="rounded-md border border-border bg-surface p-4">
        <h2 className="text-lg font-semibold">Review board</h2>
        <p className="mt-2 text-sm text-text-muted">Verify pack access to continue reviewing.</p>
      </section>
    );
  }

  return (
    <section className="rounded-md border border-border bg-surface p-4" onFocusCapture={(event) => {
      if (event.target instanceof HTMLElement) lastFocusedControl.current = event.target;
    }}>
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
      <div className="mb-4 flex flex-wrap items-end gap-2">
        <div className="flex max-w-full flex-col gap-1">
          <label htmlFor={filterId} className="text-sm font-medium">Review status</label>
          <select id={filterId} ref={filterRef} value={filter}
            className="min-h-11 max-w-full rounded-md border border-border bg-bg px-3 text-base text-text focus:ring-2 focus:ring-primary"
            onChange={(event) => {
              setFilter(event.target.value as ReviewFilter);
              setSelectedItemIds(new Set());
              setPreviewItemId(null);
            }}>
            {statusOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
        <Button size="sm" type="button" variant="secondary" className="min-h-11"
          disabled={busy || !onBulkReview || visibleItems.length === 0}
          onClick={() => setSelectedItemIds(new Set(visibleItems.map((item) => item.id)))}>
          Select visible
        </Button>
        <Button size="sm" type="button" variant="ghost" className="min-h-11"
          disabled={busy || selectedIds.length === 0} onClick={() => setSelectedItemIds(new Set())}>
          Clear selection
        </Button>
        <Button
          className="min-h-11 gap-2 aria-disabled:opacity-50"
          disabled={!onBulkReview || (!busy && selectedIds.length === 0)}
          aria-disabled={busy || undefined}
          onClick={(event) => handleBulkReview('approved', event.currentTarget)}
          size="sm"
          type="button"
        >
          <Check aria-hidden className="h-4 w-4" />
          Approve selected
        </Button>
        <Button
          className="min-h-11 gap-2 aria-disabled:opacity-50"
          disabled={!onBulkReview || (!busy && selectedIds.length === 0)}
          aria-disabled={busy || undefined}
          onClick={(event) => handleBulkReview('rejected', event.currentTarget)}
          size="sm"
          type="button"
          variant="danger"
        >
          <X aria-hidden className="h-4 w-4" />
          Reject selected
        </Button>
      </div>

      <p aria-live="polite" className="mb-3 text-sm text-text-muted empty:hidden">
        {busy ? 'Saving review…' : feedback && !feedback.error ? feedback.text : ''}
      </p>
      {feedback?.error && <p role="alert" className="mb-3 break-words text-sm text-danger">{feedback.text}</p>}

      {visibleItems.length === 0 ? (
        <p className="text-sm text-text-muted">{items.length > 0
          ? 'No variants match this review status.' : 'Generated and uploaded variants will appear here for review.'}</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {visibleItems.map((item) => {
            const selected = selectedItemIds.has(item.id);
            const canPreview = !!onLoadPreview && !!item.generated_file_id;
            const previewOpen = canPreview && previewItemId === item.id;

            return (
              <article
                key={item.id}
                className={`rounded-md border bg-bg p-3 transition-colors ${
                  selected ? 'border-primary' : 'border-border'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <label className="flex min-h-11 items-center gap-2 text-sm font-medium">
                    <input
                      aria-label={`Select item ${item.id}`}
                      checked={selected}
                      disabled={busy || !onBulkReview}
                      className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                      type="checkbox"
                      onChange={() => toggleSelection(item.id)}
                    />
                    {itemLabel(item)}
                  </label>
                  <Button
                    aria-label={`Set preferred item ${item.id}`}
                    aria-pressed={item.preferred}
                    aria-disabled={busy || undefined}
                    className={`min-h-11 min-w-11 aria-disabled:opacity-50 ${item.preferred ? 'text-primary' : 'text-text-muted'}`}
                    disabled={!onSetPreferred || item.review_status !== 'approved'}
                    onClick={() => { if (onSetPreferred) void saveReview(() => onSetPreferred(item.id)); }}
                    size="xs"
                    type="button"
                    variant="ghost"
                  >
                    <Star aria-hidden className="h-4 w-4" />
                  </Button>
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Badge variant={reviewBadgeVariant(String(item.review_status))}>
                    {item.review_status}
                  </Badge>
                  <Badge variant="neutral">{item.source}</Badge>
                  {item.width && item.height && <span className="text-xs text-text-muted">{item.width} × {item.height}</span>}
                  {canPreview ? (
                    <Button variant="ghost" size="sm" type="button" className="min-h-11"
                      ref={previewOpen ? previewToggleRef : undefined}
                      aria-expanded={previewOpen} aria-label={`${previewOpen ? 'Hide preview for' : 'Preview'} item ${item.id}`}
                      onClick={() => setPreviewItemId(previewOpen ? null : item.id)}>
                      {previewOpen ? 'Hide preview' : 'Preview'}
                    </Button>
                  ) : !item.generated_file_id && <span className="text-xs text-text-muted">Preview unavailable</span>}
                </div>
                {previewOpen && onLoadPreview && (
                  <div className="mt-2 flex aspect-[4/3] max-h-80 items-center justify-center overflow-hidden rounded-md bg-surface2 p-2 text-sm text-text-muted">
                    <AssetPreview key={item.generated_file_id} item={item} load={onLoadPreview}
                      onRetry={() => previewToggleRef.current?.focus()} />
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
