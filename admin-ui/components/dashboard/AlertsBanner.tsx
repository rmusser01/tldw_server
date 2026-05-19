'use client';

import Link from 'next/link';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { AlertTriangle, CheckCheck } from 'lucide-react';

type AlertSeverity = 'info' | 'warning' | 'error' | 'critical';

export type DashboardAlertSummaryItem = {
  id?: string | number;
  severity?: AlertSeverity;
  message?: string;
  created_at?: string;
};

type DashboardAlertFull = DashboardAlertSummaryItem & {
  id?: string | number;
  message?: string;
  acknowledged?: boolean;
};

type AlertsBannerProps = {
  alerts: DashboardAlertFull[];
  onAcknowledgeAll?: () => void;
};

export const summarizeAlertSeverities = (alerts: DashboardAlertSummaryItem[]) => {
  const summary = {
    critical: 0,
    warning: 0,
    info: 0,
  };

  alerts.forEach((alert) => {
    const severity = (alert.severity ?? 'info').toLowerCase();
    if (severity === 'critical' || severity === 'error') {
      summary.critical += 1;
      return;
    }
    if (severity === 'warning') {
      summary.warning += 1;
      return;
    }
    summary.info += 1;
  });

  return summary;
};

const SEVERITY_PRIORITY: Record<string, number> = {
  critical: 0,
  error: 1,
  warning: 2,
  info: 3,
};

export const findMostRelevantAlert = (
  alerts: DashboardAlertSummaryItem[]
): DashboardAlertSummaryItem | null => {
  if (alerts.length === 0) return null;

  const withMessage = alerts.filter((a) => a.message);
  if (withMessage.length === 0) return null;

  return withMessage.sort((a, b) => {
    const severityA = SEVERITY_PRIORITY[(a.severity ?? 'info').toLowerCase()] ?? 3;
    const severityB = SEVERITY_PRIORITY[(b.severity ?? 'info').toLowerCase()] ?? 3;
    if (severityA !== severityB) return severityA - severityB;

    const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
    return dateB - dateA;
  })[0];
};

const getAlertToneClasses = (critical: number, warning: number) => {
  if (critical > 0) {
    return {
      container: 'border-red-200 bg-red-50',
      icon: 'text-red-700',
      description: 'text-red-900',
    };
  }
  if (warning > 0) {
    return {
      container: 'border-yellow-200 bg-yellow-50',
      icon: 'text-yellow-700',
      description: 'text-yellow-900',
    };
  }
  return {
    container: 'border-blue-200 bg-blue-50',
    icon: 'text-blue-700',
    description: 'text-blue-900',
  };
};

export const AlertsBanner = ({ alerts, onAcknowledgeAll }: AlertsBannerProps) => {
  if (alerts.length <= 0) return null;

  const summary = summarizeAlertSeverities(alerts);
  const tone = getAlertToneClasses(summary.critical, summary.warning);
  const total = summary.critical + summary.warning + summary.info;
  const topAlert = findMostRelevantAlert(alerts);

  return (
    <Alert className={cn('mb-6', tone.container)}>
      <AlertTriangle className={cn('h-4 w-4', tone.icon)} />
      <AlertDescription className={cn('space-y-2', tone.description)} aria-live="polite" aria-atomic="true">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={summary.critical > 0 ? 'destructive' : 'secondary'}>
            {summary.critical} critical
          </Badge>
          <Badge className="bg-yellow-500 text-white">
            {summary.warning} warning
          </Badge>
          <Badge className="bg-blue-500 text-white">
            {summary.info} info
          </Badge>
          {topAlert?.message && (
            <span className="text-sm truncate max-w-md">
              — {topAlert.message}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span>
            {total} active alert{total !== 1 ? 's' : ''} require attention.{' '}
            <Link href="/monitoring" className="underline font-medium">View all</Link>
          </span>
          {onAcknowledgeAll && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-xs"
              onClick={onAcknowledgeAll}
              title="Acknowledge all alerts"
              data-testid="acknowledge-alerts-btn"
            >
              <CheckCheck className="h-3.5 w-3.5 mr-1" />
              Acknowledge All
            </Button>
          )}
        </div>
      </AlertDescription>
    </Alert>
  );
};
