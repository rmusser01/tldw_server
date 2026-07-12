import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/router';
import { useNotificationLifecycle } from '@web/components/notifications/NotificationLifecycleProvider';
import { useToast } from '@web/components/ui/ToastProvider';
import {
  cancelNotificationSnooze,
  dismissNotification,
  getNotificationPreferences,
  updateNotificationPreferences,
  listNotifications,
  markNotificationsRead,
  NotificationItem,
  NotificationPreferences,
  snoozeNotification,
} from '@web/lib/api/notifications';
import { formatRelativeTime } from '@web/lib/utils';
import { classifyNotificationError } from '@/services/notification-lifecycle';

const DEFAULT_SNOOZE_MINUTES = 15;
const NOTIFICATIONS_FETCH_LIMIT = 100;
type PreferenceKey = 'reminder_enabled' | 'job_completed_enabled' | 'job_failed_enabled';

function resolveRouteForLinkType(linkType: string | null | undefined): string | undefined {
  if (!linkType) return undefined
  const lt = linkType.toLowerCase()
  if (lt.includes("reading")) return "/collections"
  if (lt.includes("note") || lt.includes("document")) return "/notes"
  if (lt.includes("watchlist") || lt.includes("job")) return "/watchlists"
  return undefined
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

export default function NotificationsPage() {
  const { show } = useToast();
  const router = useRouter();
  const {
    scopeKey,
    state: lifecycleState,
    unreadCount: lifecycleUnreadCount,
    updatedAt: lifecycleUpdatedAt,
    events,
    reportRequestError,
    reportMutationError,
  } = useNotificationLifecycle();
  const isTerminal = lifecycleState === 'auth-required' || lifecycleState === 'unavailable';
  const [items, setItems] = useState<NotificationItem[]>([]);
  const [snoozedItems, setSnoozedItems] = useState<NotificationItem[]>([]);
  const [loadedScopeKey, setLoadedScopeKey] = useState(scopeKey);
  const [showSnoozed, setShowSnoozed] = useState(false);
  const [unreadCount, setUnreadCount] = useState(lifecycleUnreadCount);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const handledEventSequenceRef = useRef(0);
  const inboxRetryAttemptRef = useRef(0);
  const inboxRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const refreshInboxRef = useRef<() => Promise<void>>(async () => undefined);
  const pageGenerationRef = useRef(0);
  const inboxAbortRef = useRef<AbortController | null>(null);
  const previousScopeRef = useRef(scopeKey);
  const [showPrefs, setShowPrefs] = useState(false);
  const [prefs, setPrefs] = useState<NotificationPreferences | null>(null);
  const [prefsLoading, setPrefsLoading] = useState(false);
  const [prefsError, setPrefsError] = useState<string | null>(null);
  const [prefsSavingKey, setPrefsSavingKey] = useState<PreferenceKey | null>(null);

  const clearInboxRetry = useCallback(() => {
    if (inboxRetryTimerRef.current === null) return;
    clearTimeout(inboxRetryTimerRef.current);
    inboxRetryTimerRef.current = null;
  }, []);

  const refreshInbox = useCallback(async () => {
    clearInboxRetry();
    if (isTerminal) return;
    inboxAbortRef.current?.abort();
    const requestAbort = new AbortController();
    inboxAbortRef.current = requestAbort;
    const requestGeneration = pageGenerationRef.current;
    try {
      const [list, snoozed] = await Promise.all([
        listNotifications({
          limit: NOTIFICATIONS_FETCH_LIMIT,
          offset: 0,
          include_archived: false,
          signal: requestAbort.signal,
        }),
        listNotifications({
          limit: NOTIFICATIONS_FETCH_LIMIT,
          offset: 0,
          include_archived: true,
          only_snoozed: true,
          signal: requestAbort.signal,
        }),
      ]);
      if (requestAbort.signal.aborted || requestGeneration !== pageGenerationRef.current) return;
      setItems(list.items);
      setSnoozedItems(snoozed.items);
      setLoadedScopeKey(scopeKey);
      setError(null);
      inboxRetryAttemptRef.current = 0;
    } catch (refreshError) {
      if (requestAbort.signal.aborted || requestGeneration !== pageGenerationRef.current) return;
      reportRequestError(refreshError);
      const message = refreshError instanceof Error ? refreshError.message : 'Failed to load notifications';
      setError(message);
      const classification = classifyNotificationError(refreshError, {
        attempt: inboxRetryAttemptRef.current,
      });
      if (classification.kind === 'retry') {
        inboxRetryAttemptRef.current += 1;
        inboxRetryTimerRef.current = setTimeout(() => {
          inboxRetryTimerRef.current = null;
          void refreshInboxRef.current();
        }, classification.delayMs);
      } else {
        inboxRetryAttemptRef.current = 0;
      }
    } finally {
      if (requestGeneration === pageGenerationRef.current) setIsLoading(false);
    }
  }, [clearInboxRetry, isTerminal, reportRequestError, scopeKey]);

  const handleSnooze = useCallback(
    async (notificationId: number, minutes: number = DEFAULT_SNOOZE_MINUTES) => {
      const requestGeneration = pageGenerationRef.current;
      try {
        const result = await snoozeNotification(notificationId, minutes);
        if (requestGeneration !== pageGenerationRef.current) return;
        const target = items.find((item) => item.id === notificationId);
        if (target) {
          const dismissedAt = new Date().toISOString();
          setItems((previous) => previous.filter((item) => item.id !== notificationId));
          setSnoozedItems((previous) => [
            {
              ...target,
              dismissed_at: dismissedAt,
              snooze_until: result.run_at,
            },
            ...previous.filter((item) => item.id !== notificationId),
          ]);
          if (!target.read_at && !target.dismissed_at) {
            setUnreadCount((count) => Math.max(0, count - 1));
          }
        }
        show({
          title: 'Snoozed',
          description: `We will remind you again in ${minutes} minutes.`,
          variant: 'success',
        });
      } catch (snoozeError) {
        if (requestGeneration !== pageGenerationRef.current) return;
        reportMutationError(snoozeError);
        const message = snoozeError instanceof Error ? snoozeError.message : 'Failed to snooze notification';
        show({
          title: 'Snooze failed',
          description: message,
          variant: 'danger',
        });
      }
    },
    [items, reportMutationError, show]
  );

  const handleCancelSnooze = useCallback(async (notificationId: number) => {
    const requestGeneration = pageGenerationRef.current;
    try {
      await cancelNotificationSnooze(notificationId);
      if (requestGeneration !== pageGenerationRef.current) return;
      setSnoozedItems((previous) => previous.filter((item) => item.id !== notificationId));
      show({
        title: 'Snooze cancelled',
        description: 'This reminder will not return.',
        variant: 'success',
      });
    } catch (cancelError) {
      if (requestGeneration !== pageGenerationRef.current) return;
      reportMutationError(cancelError);
      const message = cancelError instanceof Error ? cancelError.message : 'Failed to cancel snooze';
      show({ title: 'Cancel snooze failed', description: message, variant: 'danger' });
    }
  }, [reportMutationError, show]);

  const loadPrefs = useCallback(async () => {
    if (prefsLoading) return;
    setPrefsLoading(true);
    setPrefsError(null);
    const requestGeneration = pageGenerationRef.current;
    try {
      const nextPrefs = await getNotificationPreferences();
      if (requestGeneration !== pageGenerationRef.current) return;
      setPrefs(nextPrefs);
    } catch (preferenceError) {
      if (requestGeneration !== pageGenerationRef.current) return;
      reportRequestError(preferenceError);
      setPrefs(null);
      setPrefsError('Notification preferences are currently unavailable.');
    } finally {
      if (requestGeneration === pageGenerationRef.current) setPrefsLoading(false);
    }
  }, [prefsLoading, reportRequestError]);

  const togglePref = useCallback(
    async (key: PreferenceKey) => {
      if (!prefs || prefsSavingKey) return;
      const nextValue = !prefs[key];
      const updated =
        key === 'reminder_enabled'
          ? { reminder_enabled: nextValue }
          : key === 'job_completed_enabled'
            ? { job_completed_enabled: nextValue }
            : { job_failed_enabled: nextValue };
      setPrefsSavingKey(key);
      const requestGeneration = pageGenerationRef.current;
      try {
        const result = await updateNotificationPreferences(updated);
        if (requestGeneration !== pageGenerationRef.current) return;
        setPrefs(result);
      } catch (preferenceError) {
        if (requestGeneration !== pageGenerationRef.current) return;
        reportMutationError(preferenceError);
        show({ title: 'Failed to update preference', variant: 'danger' });
      } finally {
        if (requestGeneration === pageGenerationRef.current) setPrefsSavingKey(null);
      }
    },
    [prefs, prefsSavingKey, reportMutationError, show]
  );

  const applyIncomingNotification = useCallback(
    (incoming: NotificationItem) => {
      setItems((previous) => {
        if (previous.some((item) => item.id === incoming.id)) {
          return previous;
        }
        return [incoming, ...previous].slice(0, 200);
      });
    },
    []
  );

  useEffect(() => {
    if (previousScopeRef.current === scopeKey) return;
    previousScopeRef.current = scopeKey;
    pageGenerationRef.current += 1;
    inboxAbortRef.current?.abort();
    inboxAbortRef.current = null;
    clearInboxRetry();
    handledEventSequenceRef.current = 0;
    inboxRetryAttemptRef.current = 0;
    setItems([]);
    setSnoozedItems([]);
    setLoadedScopeKey(scopeKey);
    setPrefs(null);
    setPrefsError(null);
    setPrefsLoading(false);
    setPrefsSavingKey(null);
    setShowPrefs(false);
    setError(null);
    setIsLoading(true);
  }, [clearInboxRetry, scopeKey]);

  useEffect(() => {
    refreshInboxRef.current = refreshInbox;
  }, [refreshInbox]);

  useEffect(() => {
    void refreshInbox();
  }, [refreshInbox]);

  useEffect(() => {
    if (isTerminal) clearInboxRetry();
    return () => {
      clearInboxRetry();
      inboxAbortRef.current?.abort();
    };
  }, [clearInboxRetry, isTerminal]);

  useEffect(() => {
    setUnreadCount(lifecycleUnreadCount);
  }, [lifecycleUnreadCount, lifecycleUpdatedAt]);

  useEffect(() => {
    const pendingEvents = events.filter(
      ({ sequence }) => sequence > handledEventSequenceRef.current
    );
    let needsRefresh = false;
    for (const { sequence, event } of pendingEvents) {
      handledEventSequenceRef.current = sequence;
      if (event.event === 'notification') {
        const nextItem = toNotificationFromStream(event.payload);
        if (nextItem) applyIncomingNotification(nextItem);
      } else if (event.event === 'notifications_coalesced' || event.event === 'reset_required') {
        needsRefresh = true;
      }
    }
    if (needsRefresh) void refreshInbox();
  }, [applyIncomingNotification, events, refreshInbox]);

  const handleMarkRead = useCallback(async (notificationId: number) => {
    const requestGeneration = pageGenerationRef.current;
    try {
      await markNotificationsRead([notificationId]);
      if (requestGeneration !== pageGenerationRef.current) return;
      setItems((previous) =>
        previous.map((item) =>
          item.id === notificationId
            ? { ...item, read_at: item.read_at || new Date().toISOString() }
            : item
        )
      );
      setUnreadCount((count) => Math.max(0, count - 1));
    } catch (markError) {
      if (requestGeneration !== pageGenerationRef.current) return;
      reportMutationError(markError);
      const message = markError instanceof Error ? markError.message : 'Failed to mark notification as read';
      show({ title: 'Mark read failed', description: message, variant: 'danger' });
    }
  }, [reportMutationError, show]);

  const handleDismiss = useCallback(async (notificationId: number) => {
    const requestGeneration = pageGenerationRef.current;
    try {
      await dismissNotification(notificationId);
      if (requestGeneration !== pageGenerationRef.current) return;
      setItems((previous) => {
        const target = previous.find((item) => item.id === notificationId);
        if (target && !target.read_at && !target.dismissed_at) {
          setUnreadCount((count) => Math.max(0, count - 1));
        }
        return previous.filter((item) => item.id !== notificationId);
      });
    } catch (dismissError) {
      if (requestGeneration !== pageGenerationRef.current) return;
      reportMutationError(dismissError);
      const message = dismissError instanceof Error ? dismissError.message : 'Failed to dismiss notification';
      show({ title: 'Dismiss failed', description: message, variant: 'danger' });
    }
  }, [reportMutationError, show]);

  const scopedItems = loadedScopeKey === scopeKey ? items : [];
  const scopedSnoozedItems = loadedScopeKey === scopeKey ? snoozedItems : [];
  const hasNotifications = scopedItems.length > 0;
  const hasAnyNotifications = hasNotifications || scopedSnoozedItems.length > 0;
  const unreadLabel = useMemo(() => `Unread: ${unreadCount}`, [unreadCount]);

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6 lg:px-8">
      <section className="rounded-lg border border-border bg-card p-4 shadow-sm">
        <header className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-foreground">Notifications</h1>
            <p className="mt-1 text-sm text-muted-foreground">{unreadLabel}</p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              className="rounded border border-border px-3 py-2 text-sm font-medium hover:bg-muted"
              disabled={isTerminal}
              onClick={() => void refreshInbox()}
            >
              Refresh
            </button>
            <button
              type="button"
              className="rounded border border-border px-3 py-2 text-sm font-medium hover:bg-muted"
              disabled={isTerminal}
              onClick={() => {
                const nextShowPrefs = !showPrefs;
                setShowPrefs(nextShowPrefs);
                if (nextShowPrefs && !prefs && !prefsLoading) {
                  void loadPrefs();
                }
              }}
              aria-expanded={showPrefs}
            >
              {showPrefs ? 'Hide Preferences' : 'Preferences'}
            </button>
          </div>
        </header>

        {showPrefs && (
          <div className="mb-4 rounded-lg border border-border bg-muted/30 p-4">
            <h3 className="mb-3 text-sm font-semibold">Notification Preferences</h3>
            {prefsLoading ? (
              <p className="text-sm text-muted-foreground">Loading preferences...</p>
            ) : prefsError ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">{prefsError}</p>
                <button
                  type="button"
                  className="rounded border border-border px-3 py-2 text-sm font-medium hover:bg-muted"
                  onClick={() => void loadPrefs()}
                >
                  Retry
                </button>
              </div>
            ) : !prefs ? (
              <p className="text-sm text-muted-foreground">
                Notification preferences are currently unavailable.
              </p>
            ) : (
              <div className="space-y-3">
                {([
                  { key: 'job_completed_enabled' as const, label: 'Job completed notifications', desc: 'Notify when watchlist jobs finish successfully' },
                  { key: 'job_failed_enabled' as const, label: 'Job failed notifications', desc: 'Notify when watchlist jobs encounter errors' },
                  { key: 'reminder_enabled' as const, label: 'Reminder notifications', desc: 'Notify when snoozed items resurface' },
                ]).map(({ key, label, desc }) => (
                  <label key={key} className="flex cursor-pointer items-start gap-3">
                    <input
                      type="checkbox"
                      checked={prefs[key]}
                      disabled={isTerminal || prefsSavingKey !== null}
                      onChange={() => void togglePref(key)}
                      className="mt-1 h-4 w-4 rounded border-border disabled:cursor-not-allowed disabled:opacity-50"
                    />
                    <div>
                      <span className="text-sm font-medium">{label}</span>
                      <p className="text-xs text-muted-foreground">{desc}</p>
                    </div>
                  </label>
                ))}
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mb-4 rounded border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {error}
          </div>
        )}

        {isLoading && !hasAnyNotifications ? (
          <div className="py-8 text-center text-sm text-muted-foreground">Loading notifications...</div>
        ) : !hasAnyNotifications ? (
          <div className="py-8 text-center text-sm text-muted-foreground">No notifications yet.</div>
        ) : hasNotifications ? (
          <ul className="space-y-3">
            {scopedItems.map((item) => (
              <li key={item.id} className="rounded border border-border/70 bg-card/80 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-semibold text-foreground">{item.title}</h2>
                    <p className="mt-1 text-sm text-muted-foreground">{item.message}</p>
                  </div>
                  <span className="whitespace-nowrap text-xs text-muted-foreground">
                    {formatRelativeTime(item.created_at)}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {(item.link_url || resolveRouteForLinkType(item.link_type)) && (
                    <button
                      type="button"
                      className="rounded border border-primary/30 bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/20"
                      disabled={isTerminal}
                      onClick={async () => {
                        if (!item.read_at) {
                          try { await handleMarkRead(item.id) } catch {}
                        }
                        if (item.link_url) {
                          try {
                            const url = new URL(item.link_url, window.location.origin)
                            if (url.origin === window.location.origin) {
                              void router.push(url.pathname + url.search + url.hash)
                            }
                          } catch {}
                        } else {
                          const route = resolveRouteForLinkType(item.link_type)
                          if (route) void router.push(route)
                        }
                      }}
                    >
                      View
                    </button>
                  )}
                  {!item.read_at && (
                    <button
                      type="button"
                      className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted"
                      disabled={isTerminal}
                      onClick={() => void handleMarkRead(item.id)}
                    >
                      Mark read
                    </button>
                  )}
                  <button
                    type="button"
                    className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted"
                    disabled={isTerminal}
                    onClick={() => void handleSnooze(item.id, DEFAULT_SNOOZE_MINUTES)}
                  >
                    Snooze {DEFAULT_SNOOZE_MINUTES}m
                  </button>
                  <button
                    type="button"
                    className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted"
                    disabled={isTerminal}
                    onClick={() => void handleDismiss(item.id)}
                  >
                    Dismiss
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ) : null}

        {scopedSnoozedItems.length > 0 && (
          <div className="mt-6 border-t border-border pt-4">
            <button
              type="button"
              className="mb-3 text-sm font-medium text-muted-foreground hover:text-foreground"
              onClick={() => setShowSnoozed(!showSnoozed)}
            >
              {showSnoozed ? 'Hide' : 'Show'} snoozed ({scopedSnoozedItems.length})
            </button>
            {showSnoozed && (
              <ul className="space-y-2">
                {scopedSnoozedItems.map((item) => (
                  <li key={item.id} className="rounded border border-border/50 bg-muted/30 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <h2 className="text-sm font-medium text-foreground">{item.title}</h2>
                        <p className="mt-1 text-xs text-muted-foreground">{item.message}</p>
                      </div>
                      <div className="text-right text-xs text-muted-foreground">
                        <div>Snoozed {item.dismissed_at ? formatRelativeTime(item.dismissed_at) : 'recently'}</div>
                        {item.snooze_until ? <div>Returns {formatRelativeTime(item.snooze_until)}</div> : null}
                      </div>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        type="button"
                        className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted"
                        disabled={isTerminal}
                        onClick={() => void handleCancelSnooze(item.id)}
                      >
                        Cancel snooze
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
