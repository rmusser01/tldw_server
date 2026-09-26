import React from 'react';
import { Ban, Play, RefreshCw, RotateCcw } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import type { VNAssetGenerationPreflight, VNAssetGenerationStatus, VNAssetSlot } from '@web/types/vn-assets';

export interface GenerationMonitorProps {
  generation?: VNAssetGenerationStatus | null;
  slots: VNAssetSlot[];
  isCancelling?: boolean;
  isStarting?: boolean;
  onCancelGeneration?: () => void;
  onStartGeneration?: () => void;
  onRetrySlot?: (slotId: number) => void;
  retryingSlotId?: number | null;
  onRefresh?: () => void;
  disabled?: boolean;
  preflight?: VNAssetGenerationPreflight | null;
  preflightError?: string | null;
}

function generationBadgeVariant(status?: string): 'danger' | 'info' | 'neutral' | 'success' | 'warning' {
  if (status === 'failed') return 'danger';
  if (status === 'completed') return 'success';
  if (status === 'queued' || status === 'processing') return 'info';
  if (status === 'cancelled') return 'warning';
  return 'neutral';
}

export default function GenerationMonitor({
  generation,
  slots,
  isCancelling = false,
  isStarting = false,
  onCancelGeneration,
  onStartGeneration,
  onRetrySlot,
  retryingSlotId = null,
  onRefresh,
  disabled = false,
  preflight,
  preflightError,
}: GenerationMonitorProps) {
  const status = generation?.status ?? 'idle';
  const generationActive = status === 'queued' || status === 'enqueued' || status === 'processing';
  const commandBusy = isStarting || isCancelling || retryingSlotId !== null;
  const busy = disabled || commandBusy;
  const canStartGeneration = slots.length > 0 && !generationActive && !busy;
  const canCancelGeneration = generationActive && !busy;
  const failedSlots = slots.filter((slot) => slot.status === 'failed' || slot.last_error);
  const legacyRetryUnavailable = (slotId: number): boolean => {
    const sourceAvailable = generation?.failed_slot_recipe_available?.[slotId];
    if (sourceAvailable !== undefined) return !sourceAvailable;
    if (generation?.recipe_available !== false) return false;
    const sourceBatchId = generation.failed_slot_batch_ids?.[slotId];
    return sourceBatchId === generation.batch_id ||
      (sourceBatchId == null && generation.selected_slot_ids?.includes(slotId) === true);
  };

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Generation monitor</h2>
        <div className="flex items-center gap-2">
          <span role="status" aria-label="Generation status"><Badge variant={generationBadgeVariant(status)}>{status}</Badge></span>
          {onRefresh && (
            <Button aria-label="Refresh generation status" title="Refresh generation status"
              size="sm" variant="secondary" disabled={commandBusy} onClick={onRefresh}>
              <RefreshCw aria-hidden className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-text-muted">Slots</dt>
          <dd className="font-medium">{slots.length}</dd>
        </div>
        <div>
          <dt className="text-text-muted">Queued variants</dt>
          <dd className="font-medium">{generation?.planned_count ?? 0}</dd>
        </div>
        <div>
          <dt className="text-text-muted">Completed</dt>
          <dd className="font-medium">{generation?.completed_count ?? 0}</dd>
        </div>
        <div>
          <dt className="text-text-muted">Failed</dt>
          <dd className="font-medium">{generation?.failed_count ?? 0}</dd>
        </div>
      </dl>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          className="gap-2"
          disabled={!canStartGeneration}
          loading={isStarting}
          onClick={onStartGeneration}
          size="sm"
          type="button"
        >
          <Play aria-hidden className="h-4 w-4" />
          Start generation
        </Button>
        <Button
          className="gap-2"
          disabled={!canCancelGeneration}
          loading={isCancelling}
          onClick={onCancelGeneration}
          size="sm"
          type="button"
          variant="secondary"
        >
          <Ban aria-hidden className="h-4 w-4" />
          Cancel
        </Button>
      </div>
      {generation?.enqueue_error && (
        <p role="alert" className="mt-3 text-sm text-danger">
          Generation could not be queued. Check worker availability and job limits, then retry.
        </p>
      )}
      {preflightError && <p role="alert" className="mt-3 text-sm text-danger">{preflightError}</p>}
      {preflight && (
        <details className="mt-3 text-sm" open={!preflight.local_workers_enabled || preflight.slots.some((slot) => slot.status !== 'configured')}>
          <summary className="cursor-pointer font-medium">Generation configuration</summary>
          <ul className="mt-2 space-y-2 text-text-muted">
            {preflight.warnings.map((warning) => <li key={warning}>{warning}</li>)}
            {preflight.slots.filter((check) => check.message).map((check) => (
              <li key={check.slot_id} className="break-words">
                {slots.find((slot) => slot.id === check.slot_id)?.slot_key ?? 'Asset slot'}: {check.message}
              </li>
            ))}
          </ul>
        </details>
      )}
      {failedSlots.length > 0 && (
        <ul className="mt-4 divide-y divide-border text-sm">
          {failedSlots.map((slot) => (
            <li key={slot.id} className="flex items-start justify-between gap-3 py-3">
              <div className="min-w-0">
                <p className="break-words font-medium">{slot.slot_key}</p>
                <p className="mt-1 text-text-muted">
                  {legacyRetryUnavailable(slot.id)
                    ? 'Original settings unavailable. Start generation to use current settings.'
                    : slot.last_error === 'vn_asset_local_model_changed'
                      ? 'Local model configuration changed. Restore it to retry, or start generation with current settings.'
                    : slot.last_error === 'image_backend_unavailable' || slot.last_error === 'image_adapter_unavailable'
                    ? 'Image backend unavailable. Check its configuration before retrying.'
                    : slot.last_error === 'vn_asset_backend_busy'
                      ? 'Image backend busy. Retry when capacity is available.'
                      : 'Generation failed. Check the image provider, then retry this slot.'}
                </p>
              </div>
              {onRetrySlot && (
                <Button aria-label={`Retry ${slot.slot_key}`} className="shrink-0 gap-2"
                  disabled={generationActive || busy || legacyRetryUnavailable(slot.id)}
                  loading={retryingSlotId === slot.id}
                  onClick={() => onRetrySlot(slot.id)} size="sm" variant="secondary">
                  <RotateCcw aria-hidden className="h-4 w-4" /> Retry
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
