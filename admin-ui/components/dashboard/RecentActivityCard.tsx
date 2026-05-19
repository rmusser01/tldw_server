'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import type { AuditLog } from '@/types';
import { Activity, AlertTriangle, ArrowRight, FileText, ShieldAlert } from 'lucide-react';

type RecentActivityCardProps = {
  loading: boolean;
  recentActivity: AuditLog[];
  formatTimeAgo: (dateStr: string) => string;
};

type ActivitySeverity = 'info' | 'warning' | 'critical';

const getActivitySeverity = (log: AuditLog): ActivitySeverity => {
  const value = `${log.action} ${log.resource ?? ''}`.toLowerCase();
  if (value.includes('error') || value.includes('failed') || value.includes('breach')) {
    return 'critical';
  }
  if (value.includes('delete') || value.includes('revoke') || value.includes('disable')) {
    return 'warning';
  }
  return 'info';
};

const getSeverityIcon = (severity: ActivitySeverity) => {
  if (severity === 'critical') {
    return <ShieldAlert className="h-3 w-3 text-red-600" />;
  }
  if (severity === 'warning') {
    return <AlertTriangle className="h-3 w-3 text-yellow-600" />;
  }
  return <Activity className="h-3 w-3 text-blue-600" />;
};

const getSeverityContainerClass = (severity: ActivitySeverity) => {
  if (severity === 'critical') {
    return 'bg-red-100';
  }
  if (severity === 'warning') {
    return 'bg-yellow-100';
  }
  return 'bg-blue-100';
};

export const getResourceTypeLabel = (resource?: string) => {
  if (!resource) return 'system';
  const [type] = resource.split(':');
  return type.replace(/[_-]+/g, ' ').trim() || 'system';
};

const formatAuditDetails = (log: AuditLog): string => {
  const detailSource = log.details ?? {
    resource: log.resource,
    ip_address: log.ip_address,
    action: log.action,
    raw: log.raw,
  };
  return JSON.stringify(detailSource, null, 2);
};

const SEVERITY_LEVELS: ActivitySeverity[] = ['critical', 'warning', 'info'];

const SEVERITY_LABELS: Record<ActivitySeverity, string> = {
  critical: 'Critical',
  warning: 'Warning',
  info: 'Info',
};

const SEVERITY_BADGE_CLASS: Record<ActivitySeverity, string> = {
  critical: 'bg-red-100 text-red-700 border-red-200',
  warning: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  info: 'bg-blue-100 text-blue-700 border-blue-200',
};

export const RecentActivityCard = ({
  loading,
  recentActivity,
  formatTimeAgo,
}: RecentActivityCardProps) => {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});
  const [activeSeverities, setActiveSeverities] = useState<Set<ActivitySeverity>>(
    new Set(SEVERITY_LEVELS)
  );

  const activityWithMetadata = useMemo(() => recentActivity.map((log) => ({
    ...log,
    severity: getActivitySeverity(log),
    resourceLabel: getResourceTypeLabel(log.resource),
    actorLabel: log.username?.trim() ? log.username : `User ${log.user_id}`,
    detailText: formatAuditDetails(log),
  })), [recentActivity]);

  const severityCounts = useMemo(() => {
    const counts: Record<ActivitySeverity, number> = { critical: 0, warning: 0, info: 0 };
    activityWithMetadata.forEach((log) => { counts[log.severity]++; });
    return counts;
  }, [activityWithMetadata]);

  const filteredActivity = useMemo(
    () => activityWithMetadata.filter((log) => activeSeverities.has(log.severity)),
    [activityWithMetadata, activeSeverities]
  );

  const toggleSeverity = (severity: ActivitySeverity) => {
    setActiveSeverities((prev) => {
      const next = new Set(prev);
      if (next.has(severity)) {
        if (next.size > 1) {
          next.delete(severity);
        }
      } else {
        next.add(severity);
      }
      return next;
    });
  };

  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Recent Activity
            </CardTitle>
            <CardDescription>Latest system events</CardDescription>
          </div>
          <Link href="/audit">
            <Button variant="ghost" size="sm">
              View All
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
        <div className="flex items-center gap-2 flex-wrap" role="group" aria-label="Severity filter">
          {SEVERITY_LEVELS.map((severity) => (
            <Button
              key={severity}
              type="button"
              size="sm"
              variant={activeSeverities.has(severity) ? 'default' : 'outline'}
              onClick={() => toggleSeverity(severity)}
              aria-pressed={activeSeverities.has(severity)}
              className="text-xs gap-1.5"
            >
              {SEVERITY_LABELS[severity]}
              <Badge
                variant="outline"
                className={`ml-0.5 text-xs px-1.5 py-0 h-4 ${SEVERITY_BADGE_CLASS[severity]}`}
              >
                {severityCounts[severity]}
              </Badge>
            </Button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex items-start gap-3">
                <Skeleton className="h-8 w-8 rounded-full" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
            ))}
          </div>
        ) : filteredActivity.length === 0 ? (
          <p className="text-center text-muted-foreground py-8">
            {recentActivity.length === 0 ? 'No recent activity' : 'No events match selected filters'}
          </p>
        ) : (
          <div className="space-y-3">
            {filteredActivity.map((log) => {
              const logId = String(log.id);
              const expanded = expandedIds[logId] === true;
              return (
                <div key={log.id} className="rounded-lg border p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex min-w-0 items-start gap-3">
                      <div className={`rounded-full p-2 ${getSeverityContainerClass(log.severity)}`}>
                        {getSeverityIcon(log.severity)}
                      </div>
                      <div className="min-w-0 space-y-1">
                        <p className="text-sm font-medium break-words">
                          {log.action}
                        </p>
                        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                          <Badge variant="outline">{log.resourceLabel}</Badge>
                          <span>{log.actorLabel}</span>
                          <span>{formatTimeAgo(log.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setExpandedIds((prev) => ({ ...prev, [logId]: !expanded }));
                      }}
                      aria-expanded={expanded}
                    >
                      {expanded ? 'Hide' : 'Details'}
                    </Button>
                  </div>
                  {expanded && (
                    <pre className="mt-3 max-h-48 overflow-auto rounded-md bg-muted/40 p-2 text-xs">
                      {log.detailText}
                    </pre>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
