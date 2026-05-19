'use client';

import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatDateTime } from '@/lib/format';
import {
  formatCostUsd,
  formatErrorRate24h,
  formatRequestCount24h,
  formatTokenCount,
  getKeyAgeIndicator,
  getKeyExpiryIndicator,
  isInactiveKey,
  type UnifiedApiKeyRow,
  type UnifiedApiKeyStatus,
} from '@/lib/api-keys-hub';

type UnifiedApiKeysTableProps = {
  rows: UnifiedApiKeyRow[];
  selectedRowIds?: Set<string>;
  onToggleRowSelection?: (rowId: string, checked: boolean) => void;
  onToggleAllSelection?: (rowIds: string[], checked: boolean) => void;
};

const statusBadgeVariant = (status: UnifiedApiKeyStatus): 'default' | 'secondary' | 'destructive' => {
  switch (status) {
    case 'revoked':
      return 'destructive';
    case 'expired':
      return 'secondary';
    default:
      return 'default';
  }
};

const statusLabel = (status: UnifiedApiKeyStatus): string => {
  switch (status) {
    case 'revoked':
      return 'Revoked';
    case 'expired':
      return 'Expired';
    default:
      return 'Active';
  }
};

const ageBadgeClass = (color: 'green' | 'yellow' | 'red'): string => {
  switch (color) {
    case 'green':
      return 'bg-green-600 text-white';
    case 'yellow':
      return 'bg-yellow-500 text-black';
    default:
      return 'bg-red-600 text-white';
  }
};

const expiryBadgeClass = (color: 'yellow' | 'red'): string => {
  if (color === 'red') {
    return 'bg-red-600 text-white';
  }
  return 'bg-yellow-500 text-black';
};

export const UnifiedApiKeysTable = ({
  rows,
  selectedRowIds,
  onToggleRowSelection,
  onToggleAllSelection,
}: UnifiedApiKeysTableProps) => {
  if (rows.length === 0) {
    return (
      <div className="py-8 text-center text-muted-foreground">No API keys match the current filters.</div>
    );
  }

  const rowIds = rows.map((row) => `${row.ownerUserId}:${row.keyId}`);
  const selectedCount = rowIds.filter((rowId) => selectedRowIds?.has(rowId)).length;
  const allSelected = selectedCount > 0 && selectedCount === rowIds.length;
  const hasTelemetry = rows.some(
    (row) => row.requestCount24h !== null || row.errorRate24h !== null
  );

  // Show usage attribution columns when any row has token or cost data
  const hasUsage = rows.some(
    (row) => row.totalTokens !== null || row.estimatedCostUsd !== null
  );

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-10">
            <Checkbox
              aria-label="Select all keys"
              checked={allSelected}
              onCheckedChange={(checked) => {
                onToggleAllSelection?.(rowIds, checked);
              }}
            />
          </TableHead>
          <TableHead>Key ID</TableHead>
          <TableHead>Owner</TableHead>
          <TableHead>Created</TableHead>
          <TableHead>Last Used</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>
            <button type="button" className="group relative cursor-help border-0 bg-transparent p-0 font-medium text-inherit focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 rounded" title="Green: <90d, Yellow: 90-180d, Red: >180d" aria-label="Age — threshold guide">
              Age
              <span className="ml-1 inline-block text-muted-foreground text-xs align-super">?</span>
              <span className="pointer-events-none absolute left-0 top-full z-10 mt-1 hidden w-48 rounded-md border bg-popover p-2 text-xs font-normal text-popover-foreground shadow-md group-hover:block group-focus:block">
                <span className="block"><span className="inline-block h-2 w-2 rounded-full bg-green-600 mr-1" />Green: &lt;90 days</span>
                <span className="block"><span className="inline-block h-2 w-2 rounded-full bg-yellow-500 mr-1" />Yellow: 90-180 days</span>
                <span className="block"><span className="inline-block h-2 w-2 rounded-full bg-red-600 mr-1" />Red: &gt;180 days</span>
              </span>
            </button>
            <span className="sr-only">
              Green under 90 days, yellow 90 to 180 days, red over 180 days.
            </span>
          </TableHead>
          <TableHead>Expiry</TableHead>
          <TableHead>Activity</TableHead>
          {hasUsage && <TableHead>Tokens</TableHead>}
          {hasUsage && <TableHead>Est. Cost</TableHead>}
          {hasTelemetry && <TableHead>Requests (24h)</TableHead>}
          {hasTelemetry && <TableHead>Error Rate (24h)</TableHead>}
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row) => {
          const ageIndicator = getKeyAgeIndicator(row.createdAt);
          const expiryIndicator = getKeyExpiryIndicator(row.expiresAt);
          const inactive = isInactiveKey(row.lastUsedAt);
          const rowId = `${row.ownerUserId}:${row.keyId}`;
          const isSelected = selectedRowIds?.has(rowId) ?? false;

          return (
            <TableRow key={`${row.ownerUserId}-${row.keyId}`}>
              <TableCell>
                <Checkbox
                  aria-label={`Select key ${row.keyPrefix || row.keyId}`}
                  checked={isSelected}
                  onCheckedChange={(checked) => onToggleRowSelection?.(rowId, checked)}
                />
              </TableCell>
              <TableCell>
                <code className="rounded bg-muted px-2 py-1 text-xs">
                  {row.keyPrefix || `key-${row.keyId}`}
                </code>
              </TableCell>
              <TableCell>
                <Link href={`/users/${row.ownerUserId}/api-keys`} className="font-medium hover:underline">
                  {row.ownerUsername}
                </Link>
                <p className="text-xs text-muted-foreground">{row.ownerEmail}</p>
              </TableCell>
              <TableCell>{formatDateTime(row.createdAt, { fallback: '—' })}</TableCell>
              <TableCell>{formatDateTime(row.lastUsedAt, { fallback: '—' })}</TableCell>
              <TableCell>
                <Badge variant={statusBadgeVariant(row.status)}>{statusLabel(row.status)}</Badge>
              </TableCell>
              <TableCell>
                {ageIndicator ? (
                  <Badge className={ageBadgeClass(ageIndicator.color)}>{ageIndicator.label}</Badge>
                ) : '—'}
              </TableCell>
              <TableCell>
                {row.status === 'expired' ? (
                  <Badge variant="destructive">Expired</Badge>
                ) : expiryIndicator ? (
                  <Badge className={expiryBadgeClass(expiryIndicator.color)}>{expiryIndicator.label}</Badge>
                ) : (
                  '—'
                )}
              </TableCell>
              <TableCell>
                {inactive ? (
                  <Badge className="bg-yellow-500 text-black">Inactive &gt;30d</Badge>
                ) : (
                  <Badge variant="secondary">Normal</Badge>
                )}
              </TableCell>
              {hasUsage && <TableCell>{formatTokenCount(row.totalTokens)}</TableCell>}
              {hasUsage && <TableCell>{formatCostUsd(row.estimatedCostUsd)}</TableCell>}
              {hasTelemetry && <TableCell>{formatRequestCount24h(row.requestCount24h)}</TableCell>}
              {hasTelemetry && <TableCell>{formatErrorRate24h(row.errorRate24h)}</TableCell>}
              <TableCell className="text-right">
                <Link href={`/users/${row.ownerUserId}/api-keys`}>
                  <Button variant="outline" size="sm">Manage</Button>
                </Link>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
};
