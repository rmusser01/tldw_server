import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/router';
import { useToast } from '@web/components/ui/ToastProvider';
import {
  cancelNotificationSnooze,
  dismissNotification,
  getUnreadCount,
  getNotificationPreferences,
  updateNotificationPreferences,
  listNotifications,
  markNotificationsRead,
  NotificationItem,
  NotificationPreferences,
  NotificationStreamEvent,
  snoozeNotification,
  subscribeNotificationsStream,
} from '@web/lib/api/notifications';
import { formatRelativeTime } from '@web/lib/utils';

const POLL_INTERVAL_MS = 30_000;
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
  const [items, setItems] = useState<NotificationItem[]>([]);
  const [snoozedItems, setSnoozedItems] = useState<NotificationItem[]>([]);
  const [showSnoozed, setShowSnoozed] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const cursorRef = useRef(0);
  const [showPrefs, setShowPrefs] = useState(false);
  const [prefs, setPrefs] = useState<NotificationPreferences | null>(null);
  const [prefsLoading, setPrefsLoading] = useState(false);
  const [prefsError, setPrefsError] = useState<string | null>(null);
  const [prefsSavingKey, setPrefsSavingKey] = useState<PreferenceKey | null>(null);

  const refreshInbox = useCallback(async () => {
    try {
      const [list, snoozed, unread] = await Promise.all([
        listNotifications({ limit: NOTIFICATIONS_FETCH_LIMIT, offset: 0, include_archived: false }),
        listNotifications({
          limit: NOTIFICATIONS_FETCH_LIMIT,
          offset: 0,
          include_archived: true,
          only_snoozed: true,
        }),
        getUnreadCount(),
      ]);
      setItems(list.items);
      setSnoozedItems(snoozed.items);
      setUnreadCount(unread.unread_count);
      const maxSeen = list.items.reduce((maxId, item) => Math.max(maxId, item.id), cursorRef.current);
      cursorRef.current = maxSeen;
      setError(null);
    } catch (refreshError) {
      const message = refreshError instanceof Error ? refreshError.message : 'Failed to load notifications';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleSnooze = useCallback(
    async (notificationId: number, minutes: number = DEFAULT_SNOOZE_MINUTES) => {
      try {
        const result = await snoozeNotification(notificationId, minutes);
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
        const message = snoozeError instanceof Error ? snoozeError.message : 'Failed to snooze notification';
        show({
          title: 'Snooze failed',
          description: message,
          variant: 'danger',
        });
      }
    },
    [items, show]
  );

  const handleCancelSnooze = useCallback(async (notificationId: number) => {
    try {
      await cancelNotificationSnooze(notificationId);
      setSnoozedItems((previous) => previous.filter((item) => item.id !== notificationId));
      show({
        title: 'Snooze cancelled',
        description: 'This reminder will not return.',
        variant: 'success',
      });
    } catch (cancelError) {
      const message = cancelError instanceof Error ? cancelError.message : 'Failed to cancel snooze';
      show({ title: 'Cancel snooze failed', description: message, variant: 'danger' });
    }
  }, [show]);

  const loadPrefs = useCallback(async () => {
    if (prefsLoading) return;
    setPrefsLoading(true);
    setPrefsError(null);
    try {
      const nextPrefs = await getNotificationPreferences();
      setPrefs(nextPrefs);
    } catch {
      setPrefs(null);
      setPrefsError('Notification preferences are currently unavailable.');
    } finally {
      setPrefsLoading(false);
    }
  }, [prefsLoading]);

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
      try {
        const result = await updateNotificationPreferences(updated);
        setPrefs(result);
      } catch {
        show({ title: 'Failed to update preference', variant: 'danger' });
      } finally {
        setPrefsSavingKey(null);
      }
    },
    [prefs, prefsSavingKey, show]
  );

  const applyIncomingNotification = useCallback(
    (incoming: NotificationItem) => {
      setItems((previous) => {
        if (previous.some((item) => item.id === incoming.id)) {
          return previous;
        }
        return [incoming, ...previous].slice(0, 200);
      });
      setUnreadCount((count) => count + 1);
      cursorRef.current = Math.max(cursorRef.current, incoming.id);
    },
    []
  );

  useEffect(() => {
    void refreshInbox();
  }, [refreshInbox]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshInbox();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(intervalId);
  }, [refreshInbox]);

  useEffect(() => {
    const unsubscribe = subscribeNotificationsStream({
      after: cursorRef.current,
      onEvent: (event: NotificationStreamEvent) => {
        if (typeof event.id === 'number' && Number.isFinite(event.id)) {
          cursorRef.current = Math.max(cursorRef.current, event.id);
        }
        if (event.event === 'notification') {
          const nextItem = toNotificationFromStream(event.payload);
          if (nextItem) {
            applyIncomingNotification(nextItem);
          }
          return;
        }
        if (event.event === 'notifications_coalesced') {
          void refreshInbox();
          return;
        }
        if (event.event === 'reset_required') {
          void refreshInbox();
        }
      },
      onError: () => {
        // Polling remains active as a fallback path.
      },
    });
    return () => {
      unsubscribe();
    };
  }, [applyIncomingNotification, refreshInbox]);

  const handleMarkRead = useCallback(async (notificationId: number) => {
    try {
      await markNotificationsRead([notificationId]);
      setItems((previous) =>
        previous.map((item) =>
          item.id === notificationId
            ? { ...item, read_at: item.read_at || new Date().toISOString() }
            : item
        )
      );
      setUnreadCount((count) => Math.max(0, count - 1));
    } catch (markError) {
      const message = markError instanceof Error ? markError.message : 'Failed to mark notification as read';
      show({ title: 'Mark read failed', description: message, variant: 'danger' });
    }
  }, [show]);

  const handleDismiss = useCallback(async (notificationId: number) => {
    try {
      await dismissNotification(notificationId);
      setItems((previous) => {
        const target = previous.find((item) => item.id === notificationId);
        if (target && !target.read_at && !target.dismissed_at) {
          setUnreadCount((count) => Math.max(0, count - 1));
        }
        return previous.filter((item) => item.id !== notificationId);
      });
    } catch (dismissError) {
      const message = dismissError instanceof Error ? dismissError.message : 'Failed to dismiss notification';
      show({ title: 'Dismiss failed', description: message, variant: 'danger' });
    }
  }, [show]);

  const hasNotifications = items.length > 0;
  const hasAnyNotifications = hasNotifications || snoozedItems.length > 0;
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
              onClick={() => void refreshInbox()}
            >
              Refresh
            </button>
            <button
              type="button"
              className="rounded border border-border px-3 py-2 text-sm font-medium hover:bg-muted"
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
                      disabled={prefsSavingKey !== null}
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
            {items.map((item) => (
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
                      onClick={() => void handleMarkRead(item.id)}
                    >
                      Mark read
                    </button>
                  )}
                  <button
                    type="button"
                    className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted"
                    onClick={() => void handleSnooze(item.id, DEFAULT_SNOOZE_MINUTES)}
                  >
                    Snooze {DEFAULT_SNOOZE_MINUTES}m
                  </button>
                  <button
                    type="button"
                    className="rounded border border-border px-2 py-1 text-xs font-medium hover:bg-muted"
                    onClick={() => void handleDismiss(item.id)}
                  >
                    Dismiss
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ) : null}

        {snoozedItems.length > 0 && (
          <div className="mt-6 border-t border-border pt-4">
            <button
              type="button"
              className="mb-3 text-sm font-medium text-muted-foreground hover:text-foreground"
              onClick={() => setShowSnoozed(!showSnoozed)}
            >
              {showSnoozed ? 'Hide' : 'Show'} snoozed ({snoozedItems.length})
            </button>
            {showSnoozed && (
              <ul className="space-y-2">
                {snoozedItems.map((item) => (
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
