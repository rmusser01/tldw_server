import { useCallback, useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { api } from '@/lib/api-client';
import Link from 'next/link';

type ErrorBreakdownItem = {
  endpoint: string;
  status_code: number;
  count: number;
  last_occurred: string | null;
};

type ErrorBreakdownData = {
  items: ErrorBreakdownItem[];
  total_errors: number;
  period: string;
};

const formatTimestamp = (value: string | null): string => {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

const getStatusBadgeVariant = (statusCode: number): 'destructive' | 'secondary' | 'default' => {
  if (statusCode >= 500) return 'destructive';
  if (statusCode === 429) return 'secondary';
  return 'default';
};

const DEFAULT_HOURS = '24';

export default function ErrorBreakdownPanel({ hours = DEFAULT_HOURS }: { hours?: string }) {
  const [data, setData] = useState<ErrorBreakdownData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const result = await api.getErrorBreakdown({ hours });
      const record = result && typeof result === 'object' ? (result as Record<string, unknown>) : {};
      setData({
        items: Array.isArray(record.items) ? (record.items as ErrorBreakdownItem[]) : [],
        total_errors: typeof record.total_errors === 'number' ? record.total_errors : 0,
        period: typeof record.period === 'string' ? record.period : '24h',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load error breakdown');
    } finally {
      setLoading(false);
    }
  }, [hours]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  return (
    <Card data-testid="error-breakdown-panel">
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Error Breakdown
            </CardTitle>
            <CardDescription>
              Top error actions from the audit log over the last {data?.period ?? '24h'}.
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {data && data.total_errors > 0 ? (
              <Badge variant="destructive" data-testid="error-breakdown-total">
                {data.total_errors} total
              </Badge>
            ) : null}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => void loadData()}
              disabled={loading}
              title="Refresh error breakdown"
              aria-label="Refresh error breakdown"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="text-sm text-destructive">{error}</div>
        ) : loading ? (
          <div className="text-sm text-muted-foreground">Loading error breakdown...</div>
        ) : !data || data.items.length === 0 ? (
          <div className="text-sm text-muted-foreground">
            No errors detected in the last {data?.period ?? '24h'}.
          </div>
        ) : (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Action / Endpoint</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Count</TableHead>
                  <TableHead>Last Occurred</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.items.slice(0, 10).map((item, index) => (
                  <TableRow key={`${item.endpoint}-${item.status_code}-${index}`}>
                    <TableCell className="font-mono text-sm">{item.endpoint}</TableCell>
                    <TableCell>
                      <Badge variant={getStatusBadgeVariant(item.status_code)}>
                        {item.status_code}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{item.count}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatTimestamp(item.last_occurred)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {data.items.length > 10 ? (
              <div className="mt-2 text-xs text-muted-foreground">
                Showing top 10 of {data.items.length} error actions.
              </div>
            ) : null}
            <Link
              href="/audit?action=error"
              className="mt-3 inline-block text-xs text-primary hover:underline"
              data-testid="error-breakdown-audit-link"
            >
              View all errors in audit log
            </Link>
          </>
        )}
      </CardContent>
    </Card>
  );
}
