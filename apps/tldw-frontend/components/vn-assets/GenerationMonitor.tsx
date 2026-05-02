import React from 'react';
import { Ban, Play } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import type { VNAssetGenerationStatus, VNAssetSlot } from '@web/types/vn-assets';

export interface GenerationMonitorProps {
  generation?: VNAssetGenerationStatus | null;
  slots: VNAssetSlot[];
  isCancelling?: boolean;
  isStarting?: boolean;
  onCancelGeneration?: () => void;
  onStartGeneration?: () => void;
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
}: GenerationMonitorProps) {
  const status = generation?.status ?? 'idle';
  const generationActive = status === 'queued' || status === 'enqueued' || status === 'processing';
  const canStartGeneration = slots.length > 0 && !generationActive && !isStarting;
  const canCancelGeneration = generationActive && !isCancelling;

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Generation monitor</h2>
        <Badge variant={generationBadgeVariant(status)}>{status}</Badge>
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
        <p className="mt-3 text-sm text-danger">{generation.enqueue_error}</p>
      )}
    </section>
  );
}
