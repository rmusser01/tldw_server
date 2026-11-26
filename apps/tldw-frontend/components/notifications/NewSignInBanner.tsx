import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/router';
import { useOptionalNotificationLifecycle } from '@web/components/notifications/NotificationLifecycleProvider';
import { dismissNotification, listNotifications, type NotificationItem } from '@web/lib/api/notifications';
import { Alert } from '@/components/ui/primitives/Alert';
import { listAuthSessions, type AuthSession } from '@/services/auth-sessions';
import { isProtectedAlertError, NEW_SIGN_IN_KIND, visibleSignInAlerts } from '@/services/sign-in-alert';

const SESSIONS_PATH = '/settings/sessions';
const HIDDEN_KEY_PREFIX = 'tldw:new-sign-in-hidden:';
const PROTECTED_TEXT =
  'This alert can be dismissed from a browser that was signed in before it, or after 24 hours.';
const RETRY_TEXT = 'Could not dismiss the alert. Try again.';

const isHiddenHere = (id: number): boolean => {
  try {
    return window.localStorage.getItem(HIDDEN_KEY_PREFIX + id) === '1';
  } catch {
    return false;
  }
};

const rememberHidden = (id: number): void => {
  try {
    window.localStorage.setItem(HIDDEN_KEY_PREFIX + id, '1');
  } catch {
    // Storage can be absent or full; the in-memory set still hides it for this page.
  }
};

/** Reload triggers: a new sign-in, a coalesced burst (which hides the individual rows) or a stream reset. */
const needsReload = ({ event, payload }: { event: string; payload?: unknown }): boolean =>
  event === 'notifications_coalesced' ||
  event === 'reset_required' ||
  (event === 'notification' &&
    (payload as { kind?: unknown } | null | undefined)?.kind === NEW_SIGN_IN_KIND);

export function NewSignInBanner() {
  const router = useRouter();
  const lifecycle = useOptionalNotificationLifecycle();
  const active = lifecycle?.connectionVerified === true && lifecycle.state === 'active';
  const epoch = lifecycle?.lifecycleEpoch;
  const events = lifecycle?.events;
  const [items, setItems] = useState<NotificationItem[]>([]);
  const [hiddenIds, setHiddenIds] = useState<ReadonlySet<number>>(new Set());
  const [notice, setNotice] = useState<{ id: number; text: string; refused: boolean } | null>(null);
  const loadRef = useRef(0);
  const handledSequenceRef = useRef(0);

  const load = useCallback(async () => {
    const token = ++loadRef.current;
    try {
      const [list, sessions] = await Promise.all([
        listNotifications({ kind: NEW_SIGN_IN_KIND, limit: 100, offset: 0, include_archived: false }),
        // A failed session lookup must never hide a security alert.
        listAuthSessions().catch((): AuthSession[] => []),
      ]);
      if (token === loadRef.current) setItems(visibleSignInAlerts(list.items ?? [], sessions));
    } catch {
      // Keep what is shown; the next trigger retries.
    }
  }, []);

  useEffect(() => {
    handledSequenceRef.current = 0;
    if (!active) {
      loadRef.current += 1;
      setItems([]);
      return;
    }
    void load();
  }, [active, epoch, load]);

  useEffect(() => {
    if (!active) return;
    let reload = false;
    for (const { sequence, event } of events ?? []) {
      if (sequence <= handledSequenceRef.current) continue;
      handledSequenceRef.current = sequence;
      if (needsReload(event)) reload = true;
    }
    if (reload) void load();
  }, [active, events, load]);

  const shown = items.filter((item) => !hiddenIds.has(item.id) && !isHiddenHere(item.id));
  const newest = shown[0];
  if (!active || router.pathname === SESSIONS_PATH || !newest) return null;

  const refused = notice?.id === newest.id && notice.refused;
  const dismiss = async () => {
    try {
      await dismissNotification(newest.id);
      setItems((prev) => prev.filter((item) => item.id !== newest.id));
      setNotice(null);
    } catch (error) {
      // Deliberately not reported to the lifecycle: a refusal would stop the runtime and hide this alert.
      setNotice(
        isProtectedAlertError(error)
          ? { id: newest.id, text: PROTECTED_TEXT, refused: true }
          : { id: newest.id, text: RETRY_TEXT, refused: false }
      );
    }
  };
  const hideHere = () => {
    rememberHidden(newest.id);
    setHiddenIds((prev) => new Set(prev).add(newest.id));
    setNotice(null);
  };
  const more = shown.length - 1;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-50 mx-auto max-w-3xl px-4 pt-2">
      <Alert
        variant="warning"
        className="pointer-events-auto bg-surface shadow-lg"
        title={newest.title}
        action={{ label: 'Review sessions', onClick: () => void router.push(SESSIONS_PATH) }}
        secondaryAction={
          refused
            ? { label: 'Hide on this browser', onClick: hideHere }
            : { label: 'Dismiss', onClick: () => void dismiss() }
        }
      >
        <p>{notice?.id === newest.id ? notice.text : newest.message}</p>
        {more > 0 && <p className="mt-1">{more === 1 ? '1 more new sign-in.' : `${more} more new sign-ins.`}</p>}
      </Alert>
    </div>
  );
}

export default NewSignInBanner;
