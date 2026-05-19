'use client';

import { useEffect, useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Button } from '@/components/ui/button';
import { Pause, Play, RefreshCw } from 'lucide-react';

type DashboardHeaderProps = {
  serverStatusLabel: string;
  serverStatusDotClass: string;
  checkedAtLabel?: string | null;
  uptimePercent: number | null;
  lastIncidentAt: string | null;
  uptimeWindowDays?: number;
  loading: boolean;
  onRefresh: () => Promise<void> | void;
  lastRefreshed?: Date | null;
  autoRefreshEnabled?: boolean;
  onAutoRefreshToggle?: () => void;
};

const formatUptimeValue = (value: number | null) => {
  if (value === null || !Number.isFinite(value)) {
    return 'N/A';
  }
  return `${value.toFixed(2)}%`;
};

const formatIncidentTimestamp = (timestamp: string | null) => {
  if (!timestamp) return 'No incidents recorded';
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) return 'Unavailable';
  return parsed.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const DashboardHeader = ({
  serverStatusLabel,
  serverStatusDotClass,
  checkedAtLabel,
  uptimePercent,
  lastIncidentAt,
  uptimeWindowDays = 30,
  loading,
  onRefresh,
  lastRefreshed,
  autoRefreshEnabled,
  onAutoRefreshToggle,
}: DashboardHeaderProps) => {
  // Re-render the "Last updated X ago" text every 15 seconds so it stays fresh
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!lastRefreshed) return;
    const tickId = setInterval(() => setTick((t) => t + 1), 15_000);
    return () => clearInterval(tickId);
  }, [lastRefreshed]);

  const lastUpdatedLabel = lastRefreshed
    ? `Updated ${formatDistanceToNow(lastRefreshed, { addSuffix: true })}`
    : null;

  return (
    <div className="mb-8 flex items-center justify-between">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">Overview of your tldw_server instance</p>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex flex-wrap items-center gap-2 rounded-full border px-3 py-1 text-sm">
          <span className={`h-2 w-2 rounded-full ${serverStatusDotClass}`} />
          <span className="font-medium">{serverStatusLabel}</span>
          {checkedAtLabel && (
            <span className="text-xs text-muted-foreground">
              Checked {checkedAtLabel}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 rounded-full border px-3 py-1 text-sm">
          <span className="font-medium">Uptime {loading ? '...' : formatUptimeValue(uptimePercent)}</span>
          <span className="text-xs text-muted-foreground">{uptimeWindowDays}d window</span>
          <span className="text-xs text-muted-foreground">
            Last incident {loading ? 'loading...' : formatIncidentTimestamp(lastIncidentAt)}
          </span>
        </div>
        {lastUpdatedLabel && (
          <span className="text-xs text-muted-foreground" data-testid="last-updated-label">
            {lastUpdatedLabel}
          </span>
        )}
        {onAutoRefreshToggle && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onAutoRefreshToggle}
            aria-label={autoRefreshEnabled ? 'Pause auto-refresh' : 'Resume auto-refresh'}
            data-testid="auto-refresh-toggle"
            title={autoRefreshEnabled ? 'Pause auto-refresh' : 'Resume auto-refresh'}
          >
            {autoRefreshEnabled ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
        )}
        <Button
          variant="outline"
          onClick={() => {
            void onRefresh();
          }}
          disabled={loading}
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>
    </div>
  );
};
