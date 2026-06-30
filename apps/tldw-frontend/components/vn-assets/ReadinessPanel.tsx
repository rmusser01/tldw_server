import React from 'react';
import { Badge } from '@web/components/ui/Badge';
import type { VNAssetReadiness } from '@web/types/vn-assets';

export interface ReadinessPanelProps {
  readiness?: VNAssetReadiness | null;
}

function formatStatus(status: string): string {
  const label = status.replace(/_/g, ' ').toLowerCase();
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export default function ReadinessPanel({ readiness }: ReadinessPanelProps) {
  const warnings = readiness?.warnings ?? [];
  const errors = readiness?.errors ?? [];
  const statusLabel = readiness ? formatStatus(readiness.status) : 'Setup';
  const badgeVariant = errors.length > 0 ? 'danger' : readiness?.ready ? 'success' : 'warning';

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Readiness</h2>
        <Badge variant={badgeVariant}>{statusLabel}</Badge>
      </div>
      {errors.length > 0 && (
        <div className="mb-3 rounded-md bg-danger/10 p-3 text-sm text-danger">
          <p className="font-medium">Blocked</p>
          <ul className="mt-2 grid gap-1">
            {errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      {warnings.length > 0 ? (
        <ul className="grid gap-2 text-sm text-text-muted">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-text-muted">
          {readiness?.ready ? 'Approved assets are ready for manifest export.' : 'Apply a matrix and approve variants to make the pack ready.'}
        </p>
      )}
    </section>
  );
}
