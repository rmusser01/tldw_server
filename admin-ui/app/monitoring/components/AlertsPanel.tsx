import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Bell, Check, CheckCircle, ChevronDown, X } from 'lucide-react';
import {
  ALERT_SNOOZE_OPTIONS,
  formatAlertHistoryActionLabel,
  formatSnoozeCountdown,
  isAlertSnoozed,
  sortAlertHistoryEntries,
} from '@/lib/monitoring-alerts';
import { CardSkeleton } from '@/components/ui/skeleton';
import type {
  AlertAssignableUser,
  AlertHistoryEntry,
  SnoozeDurationOption,
  SystemAlert,
} from '../types';
import { useCallback, useMemo, useState } from 'react';

type AlertsPanelProps = {
  alerts: SystemAlert[];
  history: AlertHistoryEntry[];
  showSnoozed: boolean;
  assignableUsers: AlertAssignableUser[];
  loading: boolean;
  localActionsEnabled: boolean;
  onToggleShowSnoozed: () => void;
  onAcknowledge: (alert: SystemAlert) => void;
  onDismiss: (alert: SystemAlert) => void;
  onAssign: (alert: SystemAlert, userId: string) => void;
  onSnooze: (alert: SystemAlert, duration: SnoozeDurationOption) => void;
  onEscalate: (alert: SystemAlert) => void;
};

const getSeverityBadge = (severity: string) => {
  switch (severity) {
    case 'critical':
      return <Badge variant="destructive">Critical</Badge>;
    case 'error':
      return <Badge variant="destructive">Error</Badge>;
    case 'warning':
      return <Badge className="bg-yellow-500">Warning</Badge>;
    default:
      return <Badge variant="secondary">Info</Badge>;
  }
};

const formatTimestamp = (timestamp?: string) => {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleString();
};

const ALERTS_PAGE_SIZE = 10;

export default function AlertsPanel({
  alerts,
  history,
  showSnoozed,
  assignableUsers,
  loading,
  localActionsEnabled,
  onToggleShowSnoozed,
  onAcknowledge,
  onDismiss,
  onAssign,
  onSnooze,
  onEscalate,
}: AlertsPanelProps) {
  const [snoozeSelections, setSnoozeSelections] = useState<Record<string, SnoozeDurationOption>>(
    {}
  );
  const [alertsDisplayLimit, setAlertsDisplayLimit] = useState(ALERTS_PAGE_SIZE);
  const now = new Date();
  const snoozedCount = alerts.filter((alert) => isAlertSnoozed(alert, now)).length;
  const visibleAlerts = showSnoozed
    ? alerts
    : alerts.filter((alert) => !isAlertSnoozed(alert, now));
  const paginatedAlerts = visibleAlerts.slice(0, alertsDisplayLimit);
  const hasMoreAlerts = visibleAlerts.length > alertsDisplayLimit;
  const handleShowMoreAlerts = useCallback(() => {
    setAlertsDisplayLimit((prev) => prev + ALERTS_PAGE_SIZE);
  }, []);
  const activeCount = alerts.filter(
    (alert) => !alert.acknowledged && !isAlertSnoozed(alert, now)
  ).length;
  const acknowledgedCount = alerts.filter((alert) => alert.acknowledged).length;
  const orderedHistory = useMemo(
    () => sortAlertHistoryEntries(history).slice(0, 50),
    [history]
  );

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Alerts
          </CardTitle>
          <CardDescription>
            {activeCount} active, {acknowledgedCount} acknowledged
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              document.querySelector('[data-testid="alert-rule-create"]')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }}
            data-testid="alerts-create-rule-shortcut"
          >
            Create Rule
          </Button>
          <Button
            type="button"
            variant={showSnoozed ? 'secondary' : 'outline'}
            size="sm"
            onClick={onToggleShowSnoozed}
            data-testid="alerts-show-snoozed-toggle"
            aria-label={showSnoozed ? 'Hide snoozed alerts' : `Show snoozed alerts (${snoozedCount})`}
          >
            <ChevronDown className="mr-2 h-4 w-4" />
            Show snoozed ({snoozedCount})
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-3"><CardSkeleton /><CardSkeleton /><CardSkeleton /></div>
        ) : visibleAlerts.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">
            <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-500" />
            <p>No alerts - system is healthy</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground mb-2">
              Showing {paginatedAlerts.length} of {visibleAlerts.length} alerts
            </div>
            {paginatedAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`flex items-start justify-between p-3 rounded-lg border ${
                  alert.acknowledged ? 'bg-muted/30 opacity-60' : 'bg-background'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {getSeverityBadge(alert.severity)}
                    {isAlertSnoozed(alert, now) && (
                      <Badge variant="secondary" className="text-xs">
                        Snoozed {formatSnoozeCountdown(alert.snoozed_until as string, now)}
                      </Badge>
                    )}
                    {alert.acknowledged && (
                      <Badge variant="outline" className="text-xs">
                        <Check className="mr-1 h-3 w-3" />
                        Acknowledged
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm font-medium truncate">{alert.message}</p>
                  <p className="text-xs text-muted-foreground">
                    {alert.source && `${alert.source} • `}
                    {formatTimestamp(alert.timestamp)}
                  </p>
                  <div className="mt-2 max-w-xs space-y-1">
                    <Label htmlFor={`alert-assignee-${alert.id}`} className="text-xs">
                      Assign to
                    </Label>
                    <Select
                      id={`alert-assignee-${alert.id}`}
                      value={alert.assigned_to ?? ''}
                      disabled={!localActionsEnabled}
                      onChange={(event) => onAssign(alert, event.target.value)}
                      data-testid={`alert-assignee-select-${alert.id}`}
                    >
                      <option value="">Unassigned</option>
                      {assignableUsers.map((user) => (
                        <option key={user.id} value={user.id}>
                          {user.label}
                        </option>
                      ))}
                    </Select>
                  </div>
                </div>
                <div className="ml-2 flex flex-col gap-2">
                  <div className="flex items-center gap-1">
                    <Select
                      value={snoozeSelections[alert.id] ?? '15m'}
                      disabled={!localActionsEnabled}
                      onChange={(event) => {
                        setSnoozeSelections((prev) => ({
                          ...prev,
                          [alert.id]: event.target.value as SnoozeDurationOption,
                        }));
                      }}
                      data-testid={`alert-snooze-duration-${alert.id}`}
                    >
                      {ALERT_SNOOZE_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </Select>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!localActionsEnabled}
                      onClick={() => onSnooze(alert, snoozeSelections[alert.id] ?? '15m')}
                      data-testid={`alert-snooze-button-${alert.id}`}
                    >
                      Snooze
                    </Button>
                  </div>
                  <div className="flex gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!localActionsEnabled}
                      onClick={() => onEscalate(alert)}
                      title="Escalate"
                      aria-label="Escalate"
                    >
                      Escalate
                    </Button>
                  </div>
                  {!alert.acknowledged && (
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onAcknowledge(alert)}
                        title="Acknowledge"
                        aria-label="Acknowledge alert"
                      >
                        <Check className="h-4 w-4 text-green-500" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onDismiss(alert)}
                        title="Dismiss"
                        aria-label="Dismiss alert"
                      >
                        <X className="h-4 w-4 text-red-500" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {hasMoreAlerts && (
              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-muted-foreground">
                  Showing {paginatedAlerts.length} of {visibleAlerts.length} alerts
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleShowMoreAlerts}
                  data-testid="alerts-load-more"
                >
                  Load more
                </Button>
              </div>
            )}
          </div>
        )}

        <details className="mt-4" data-testid="alert-history-panel">
          <summary className="cursor-pointer text-sm font-medium">
            Alert History ({orderedHistory.length})
          </summary>
          <div className="mt-3 space-y-2" data-testid="alert-history-timeline">
            {orderedHistory.length === 0 ? (
              <div className="text-sm text-muted-foreground">No alert history yet.</div>
            ) : (
              orderedHistory.map((entry) => (
                <div key={entry.id} className="rounded-md border p-2">
                  <div className="text-sm font-medium">{formatAlertHistoryActionLabel(entry.action)}</div>
                  {entry.details ? (
                    <div className="text-sm text-muted-foreground">{entry.details}</div>
                  ) : null}
                  <div className="text-xs text-muted-foreground">
                    {formatTimestamp(entry.timestamp)}
                    {entry.actor ? ` • ${entry.actor}` : ''}
                  </div>
                </div>
              ))
            )}
          </div>
        </details>
        <div className="mt-4 text-center">
          <Button
            type="button"
            variant="link"
            size="sm"
            className="text-xs text-muted-foreground"
            disabled
            title="Preset alert rules are not available yet"
          >
            Create Alert Rule for common patterns
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
