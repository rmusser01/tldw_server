import { useEffect, useState } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Button } from '@/components/ui/button';
import { Pause, Play, RefreshCw } from 'lucide-react';

type MonitoringPageHeaderProps = {
  lastUpdated: Date | null;
  loading: boolean;
  onRefresh: () => Promise<void> | void;
  lastRefreshed?: Date | null;
  autoRefreshEnabled?: boolean;
  onAutoRefreshToggle?: () => void;
};

export default function MonitoringPageHeader({
  lastUpdated,
  loading,
  onRefresh,
  lastRefreshed,
  autoRefreshEnabled,
  onAutoRefreshToggle,
}: MonitoringPageHeaderProps) {
  // Re-render the "Updated X ago" text every 15 seconds so it stays fresh
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
        <h1 className="text-3xl font-bold">Monitoring</h1>
        <p className="text-muted-foreground">System health, metrics, and alerts</p>
      </div>
      <div className="flex items-center gap-4">
        {lastUpdatedLabel && (
          <span
            className="text-sm text-muted-foreground"
            aria-live="polite"
            data-testid="last-updated-label"
          >
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
        <Button variant="outline" onClick={() => { void onRefresh(); }} disabled={loading}>
          <RefreshCw
            className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`}
            aria-hidden="true"
          />
          Refresh
        </Button>
      </div>
    </div>
  );
}
