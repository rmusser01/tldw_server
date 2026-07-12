import { useCallback, useEffect, useRef } from 'react';

import { useOptionalNotificationLifecycle } from '@web/components/notifications/NotificationLifecycleProvider';
import { useToast } from '@web/components/ui/ToastProvider';
import type { NotificationItem } from '@web/lib/api/notifications';

const TOAST_COALESCE_MS = 800;

function severityToVariant(severity?: string): 'info' | 'success' | 'warning' | 'danger' {
  if (severity === 'error') return 'danger';
  if (severity === 'warning') return 'warning';
  return 'info';
}

function toNotificationFromStream(payload: unknown): NotificationItem | null {
  if (!payload || typeof payload !== 'object') return null;
  const data = payload as Record<string, unknown>;
  const rawId = Number(data.notification_id ?? data.event_id);
  if (!Number.isFinite(rawId) || rawId <= 0) return null;
  return {
    id: rawId,
    kind: String(data.kind ?? 'notification'),
    title: String(data.title ?? 'Notification'),
    message: String(data.message ?? ''),
    severity: String(data.severity ?? 'info'),
    created_at: String(data.created_at ?? new Date().toISOString()),
    read_at: null,
    dismissed_at: null,
  };
}

export function NotificationToastBridge() {
  const { show } = useToast();
  const lifecycle = useOptionalNotificationLifecycle();
  const pendingToastCountRef = useRef(0);
  const latestToastItemRef = useRef<NotificationItem | null>(null);
  const toastTimerRef = useRef<number | null>(null);
  const handledEventSequenceRef = useRef(0);

  const flushQueuedToast = useCallback(() => {
    const burstCount = pendingToastCountRef.current;
    const latestItem = latestToastItemRef.current;
    pendingToastCountRef.current = 0;
    latestToastItemRef.current = null;
    toastTimerRef.current = null;
    if (burstCount <= 0) return;

    if (burstCount === 1 && latestItem) {
      show({
        title: latestItem.title || 'New notification',
        description: latestItem.message || 'A new notification is available.',
        variant: severityToVariant(String(latestItem.severity)),
      });
      return;
    }

    show({
      title: `${burstCount} new notifications`,
      description: 'Your inbox has been updated.',
      variant: 'info',
    });
  }, [show]);

  const queueToast = useCallback(
    (item: NotificationItem | null, incrementBy: number = 1) => {
      if (item) {
        latestToastItemRef.current = item;
      }
      pendingToastCountRef.current += Math.max(1, incrementBy);
      if (toastTimerRef.current !== null) return;
      toastTimerRef.current = window.setTimeout(() => flushQueuedToast(), TOAST_COALESCE_MS);
    },
    [flushQueuedToast]
  );

  const clearQueuedToast = useCallback(() => {
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = null;
    pendingToastCountRef.current = 0;
    latestToastItemRef.current = null;
  }, []);

  useEffect(() => {
    clearQueuedToast();
    handledEventSequenceRef.current = 0;
  }, [clearQueuedToast, lifecycle?.scopeKey]);

  useEffect(() => {
    const pendingEvents = (lifecycle?.events ?? []).filter(
      ({ sequence }) => sequence > handledEventSequenceRef.current
    );
    for (const { sequence, event } of pendingEvents) {
      handledEventSequenceRef.current = sequence;
      if (event.event === 'notification') {
        const nextItem = toNotificationFromStream(event.payload);
        if (nextItem) queueToast(nextItem, 1);
        continue;
      }
      if (event.event === 'notifications_coalesced') {
        const payload = event.payload as Record<string, unknown> | undefined;
        const count = Number(payload?.count ?? 0);
        if (Number.isFinite(count) && count > 0) queueToast(null, count);
      }
    }
  }, [lifecycle?.events, queueToast]);

  useEffect(() => {
    return () => {
      clearQueuedToast();
    };
  }, [clearQueuedToast]);

  return null;
}
