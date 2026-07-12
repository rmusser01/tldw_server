import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  listNotifications: vi.fn(),
  getUnreadCount: vi.fn(),
  getNotificationPreferences: vi.fn(),
  updateNotificationPreferences: vi.fn(),
  markNotificationsRead: vi.fn(),
  dismissNotification: vi.fn(),
  cancelNotificationSnooze: vi.fn(),
  snoozeNotification: vi.fn(),
  subscribeNotificationsStream: vi.fn(),
  showToast: vi.fn(),
  reportMutationError: vi.fn(),
  reportRequestError: vi.fn(),
  tryAgain: vi.fn(),
  refreshPermissions: vi.fn(),
  lifecycle: {
    scopeKey: 'notifications:server-a:user-a',
    lifecycleEpoch: 1,
    state: 'active',
    unreadCount: 2,
    updatedAt: 1,
    latestEvent: null as { event: string; id?: number; payload?: unknown } | null,
    eventSequence: 0,
    events: [] as Array<{ sequence: number; event: { event: string; id?: number; payload?: unknown } }>,
  },
}));

vi.mock('next/router', () => ({
  useRouter: () => ({
    push: mocks.push,
  }),
}));

vi.mock('@web/lib/api/notifications', () => ({
  listNotifications: (...args: unknown[]) => mocks.listNotifications(...args),
  getUnreadCount: (...args: unknown[]) => mocks.getUnreadCount(...args),
  getNotificationPreferences: (...args: unknown[]) => mocks.getNotificationPreferences(...args),
  updateNotificationPreferences: (...args: unknown[]) => mocks.updateNotificationPreferences(...args),
  markNotificationsRead: (...args: unknown[]) => mocks.markNotificationsRead(...args),
  dismissNotification: (...args: unknown[]) => mocks.dismissNotification(...args),
  cancelNotificationSnooze: (...args: unknown[]) => mocks.cancelNotificationSnooze(...args),
  snoozeNotification: (...args: unknown[]) => mocks.snoozeNotification(...args),
  subscribeNotificationsStream: (...args: unknown[]) => mocks.subscribeNotificationsStream(...args),
}));

vi.mock('@web/components/ui/ToastProvider', () => ({
  useToast: () => ({ show: mocks.showToast }),
}));

vi.mock('@web/components/notifications/NotificationLifecycleProvider', () => ({
  useNotificationLifecycle: () => ({
    ...mocks.lifecycle,
    reportMutationError: mocks.reportMutationError,
    reportRequestError: mocks.reportRequestError,
    tryAgain: mocks.tryAgain,
    refreshPermissions: mocks.refreshPermissions,
  }),
}));

import NotificationsPage from '@web/pages/notifications';

describe('NotificationsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.lifecycle.state = 'active';
    mocks.lifecycle.scopeKey = 'notifications:server-a:user-a';
    mocks.lifecycle.lifecycleEpoch = 1;
    mocks.lifecycle.unreadCount = 2;
    mocks.lifecycle.updatedAt = 1;
    mocks.lifecycle.latestEvent = null;
    mocks.lifecycle.eventSequence = 0;
    mocks.lifecycle.events = [];
    mocks.getNotificationPreferences.mockResolvedValue({
      user_id: 'user-1',
      reminder_enabled: true,
      job_completed_enabled: true,
      job_failed_enabled: true,
      updated_at: '2026-04-02T00:00:00Z',
    });
    mocks.updateNotificationPreferences.mockImplementation(async (payload: Record<string, boolean>) => ({
      user_id: 'user-1',
      reminder_enabled: payload.reminder_enabled ?? true,
      job_completed_enabled: payload.job_completed_enabled ?? true,
      job_failed_enabled: payload.job_failed_enabled ?? true,
      updated_at: '2026-04-02T00:01:00Z',
    }));
    mocks.listNotifications.mockImplementation(({ only_snoozed }: { only_snoozed?: boolean } = {}) =>
      Promise.resolve({
        items: only_snoozed
          ? []
          : [
              {
                id: 101,
                kind: 'job_failed',
                title: 'Job failed',
                message: 'chatbooks/export failed.',
                severity: 'error',
                created_at: '2026-02-26T00:00:00+00:00',
                read_at: null,
                dismissed_at: null,
              },
            ],
        total: 1,
      })
    );
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 2 });
    mocks.markNotificationsRead.mockResolvedValue({ updated: 1 });
    mocks.dismissNotification.mockResolvedValue({ dismissed: true });
    mocks.cancelNotificationSnooze.mockResolvedValue({ cancelled: true, deleted_tasks: 1 });
    mocks.snoozeNotification.mockResolvedValue({
      task_id: 'task-123',
      run_at: '2026-02-26T00:15:00+00:00',
    });
    mocks.subscribeNotificationsStream.mockImplementation(() => () => {});
  });

  it('renders unread count and marks notification read', async () => {
    const user = userEvent.setup();

    render(<NotificationsPage />);

    expect(await screen.findByText('Unread: 2')).toBeInTheDocument();
    const markReadButton = await screen.findByRole('button', { name: 'Mark read' });
    await user.click(markReadButton);

    await waitFor(() => {
      expect(mocks.markNotificationsRead).toHaveBeenCalledWith([101]);
    });
    expect(screen.getByText('Unread: 1')).toBeInTheDocument();
  });

  it('updates the inbox from provider-owned stream events without showing a duplicate toast', async () => {
    const view = render(<NotificationsPage />);

    expect(await screen.findByText('Unread: 2')).toBeInTheDocument();

    mocks.lifecycle.latestEvent = {
      event: 'notification',
      id: 102,
      payload: {
        notification_id: 102,
        kind: 'deep_research_completed',
        title: 'Deep research completed',
        message: 'Open the report in Deep Research.',
        severity: 'info',
        created_at: '2026-03-08T01:00:00Z',
      },
    };
    mocks.lifecycle.unreadCount = 3;
    mocks.lifecycle.updatedAt = 2;
    mocks.lifecycle.eventSequence = 1;
    mocks.lifecycle.events = [{ sequence: 1, event: mocks.lifecycle.latestEvent }];
    view.rerender(<NotificationsPage />);

    expect(await screen.findByText('Open the report in Deep Research.')).toBeInTheDocument();
    expect(screen.getByText('Unread: 3')).toBeInTheDocument();
    expect(mocks.showToast).not.toHaveBeenCalled();
  });

  it('does not create an inbox poll or a second notification stream', async () => {
    vi.useFakeTimers();
    render(<NotificationsPage />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(mocks.listNotifications).toHaveBeenCalledTimes(2);
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(mocks.listNotifications).toHaveBeenCalledTimes(2);
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it('refreshes page-specific lists after a provider coalescing event', async () => {
    const view = render(<NotificationsPage />);
    await waitFor(() => expect(mocks.listNotifications).toHaveBeenCalledTimes(2));

    mocks.lifecycle.latestEvent = { event: 'notifications_coalesced', id: 102 };
    mocks.lifecycle.eventSequence = 1;
    mocks.lifecycle.events = [{ sequence: 1, event: mocks.lifecycle.latestEvent }];
    view.rerender(<NotificationsPage />);

    await waitFor(() => expect(mocks.listNotifications).toHaveBeenCalledTimes(4));
  });

  it('renders every provider event delivered in one React batch', async () => {
    const view = render(<NotificationsPage />);
    await screen.findByText('Job failed');
    const first = {
      event: 'notification', id: 102,
      payload: { notification_id: 102, title: 'First event', message: 'First body' },
    };
    const second = {
      event: 'notification', id: 103,
      payload: { notification_id: 103, title: 'Second event', message: 'Second body' },
    };
    mocks.lifecycle.eventSequence = 2;
    mocks.lifecycle.events = [
      { sequence: 1, event: first },
      { sequence: 2, event: second },
    ];
    mocks.lifecycle.latestEvent = second;
    view.rerender(<NotificationsPage />);

    expect(await screen.findByText('First body')).toBeInTheDocument();
    expect(screen.getByText('Second body')).toBeInTheDocument();
  });

  it('accepts sequence one again after a same-scope lifecycle restart', async () => {
    const view = render(<NotificationsPage />);
    await screen.findByText('Job failed');
    mocks.lifecycle.eventSequence = 1;
    mocks.lifecycle.events = [{
      sequence: 1,
      event: { event: 'notification', id: 102, payload: { notification_id: 102, message: 'Before restart' } },
    }];
    view.rerender(<NotificationsPage />);
    expect(await screen.findByText('Before restart')).toBeInTheDocument();

    mocks.lifecycle.lifecycleEpoch = 2;
    mocks.lifecycle.eventSequence = 1;
    mocks.lifecycle.events = [{
      sequence: 1,
      event: { event: 'notification', id: 103, payload: { notification_id: 103, message: 'After restart' } },
    }];
    view.rerender(<NotificationsPage />);

    expect(await screen.findByText('After restart')).toBeInTheDocument();
  });

  it('clears old-account items and ignores delayed list responses after scope change', async () => {
    let resolveOldInbox: ((value: { items: unknown[]; total: number }) => void) | undefined;
    let resolveOldSnoozed: ((value: { items: unknown[]; total: number }) => void) | undefined;
    mocks.listNotifications
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOldInbox = resolve; }))
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOldSnoozed = resolve; }))
      .mockResolvedValue({
        items: [{
          id: 202, kind: 'notification', title: 'New account', message: 'New body',
          severity: 'info', created_at: '2026-07-11T00:00:00Z', read_at: null, dismissed_at: null,
        }],
        total: 1,
      });
    const view = render(<NotificationsPage />);
    await waitFor(() => expect(mocks.listNotifications).toHaveBeenCalledTimes(2));

    mocks.lifecycle.scopeKey = 'notifications:server-a:user-b';
    mocks.lifecycle.unreadCount = 0;
    mocks.lifecycle.updatedAt = 2;
    view.rerender(<NotificationsPage />);
    expect(screen.queryByText('Job failed')).not.toBeInTheDocument();
    expect(await screen.findAllByText('New body')).not.toHaveLength(0);

    resolveOldInbox?.({
      items: [{
        id: 101, kind: 'notification', title: 'Old account', message: 'Old body',
        severity: 'info', created_at: '2026-07-10T00:00:00Z', read_at: null, dismissed_at: null,
      }],
      total: 1,
    });
    resolveOldSnoozed?.({ items: [], total: 0 });
    await act(async () => Promise.resolve());

    expect(screen.queryByText('Old body')).not.toBeInTheDocument();
  });

  it('reports list failures to the shared lifecycle', async () => {
    const failure = Object.assign(new Error('forbidden'), { status: 403 });
    mocks.listNotifications.mockRejectedValue(failure);

    render(<NotificationsPage />);

    await waitFor(() => expect(mocks.reportRequestError).toHaveBeenCalledWith(failure));
    expect(mocks.reportRequestError).toHaveBeenCalledTimes(1);
  });

  it('retries a transient page-list bootstrap with bounded backoff', async () => {
    vi.useFakeTimers();
    const transient = Object.assign(new Error('offline'), { status: 503 });
    mocks.listNotifications
      .mockRejectedValueOnce(transient)
      .mockResolvedValue({ items: [], total: 0 });
    render(<NotificationsPage />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(mocks.reportRequestError).toHaveBeenCalledWith(transient);
    expect(mocks.listNotifications).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_199);
    });
    expect(mocks.listNotifications).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(mocks.listNotifications).toHaveBeenCalledTimes(4);
    vi.useRealTimers();
  });

  it('reports a failed mutation once without replaying it', async () => {
    const user = userEvent.setup();
    const failure = Object.assign(new Error('offline'), { status: 503 });
    mocks.markNotificationsRead.mockRejectedValue(failure);
    render(<NotificationsPage />);

    await user.click(await screen.findByRole('button', { name: 'Mark read' }));

    await waitFor(() => expect(mocks.reportMutationError).toHaveBeenCalledWith(failure));
    expect(mocks.markNotificationsRead).toHaveBeenCalledTimes(1);
    expect(mocks.reportMutationError).toHaveBeenCalledTimes(1);
  });

  it('retries a transient mutation only after one explicit user action', async () => {
    const user = userEvent.setup();
    const failure = Object.assign(new Error('offline'), { status: 503 });
    mocks.markNotificationsRead
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce({ updated: 1 });
    render(<NotificationsPage />);

    await user.click(await screen.findByRole('button', { name: 'Mark read' }));
    const retryButton = await screen.findByRole('button', { name: 'Retry action' });

    expect(mocks.markNotificationsRead).toHaveBeenCalledTimes(1);
    await user.click(retryButton);

    await waitFor(() => expect(mocks.markNotificationsRead).toHaveBeenCalledTimes(2));
    expect(screen.queryByRole('button', { name: 'Retry action' })).not.toBeInTheDocument();
    expect(screen.getByText('Unread: 1')).toBeInTheDocument();
  });

  it('suppresses page requests and actions while lifecycle state is terminal', async () => {
    mocks.lifecycle.state = 'unavailable';
    render(<NotificationsPage />);
    await act(async () => Promise.resolve());

    expect(mocks.listNotifications).not.toHaveBeenCalled();
    expect(mocks.getUnreadCount).not.toHaveBeenCalled();
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
  });

  it.each([
    ['connecting', 'Connecting to notifications'],
    ['degraded', 'Notifications are reconnecting'],
    ['auth-required', 'Sign in to view notifications'],
    ['unavailable', 'Notifications unavailable for this account'],
  ] as const)('renders the %s recovery state on direct navigation', async (state, copy) => {
    mocks.lifecycle.state = state;
    render(<NotificationsPage />);
    expect(await screen.findByText(copy)).toBeInTheDocument();
  });

  it('makes exactly one explicit retry from degraded inbox state', async () => {
    const user = userEvent.setup();
    mocks.lifecycle.state = 'degraded';
    render(<NotificationsPage />);

    await user.click(await screen.findByRole('button', { name: 'Try again' }));

    expect(mocks.tryAgain).toHaveBeenCalledTimes(1);
  });

  it('shows snoozed notifications instead of the empty state when only snoozed items remain', async () => {
    const calls: Array<{ include_archived?: boolean; only_snoozed?: boolean }> = [];
    mocks.listNotifications.mockImplementation((
      {
        include_archived,
        only_snoozed,
      }: { include_archived?: boolean; only_snoozed?: boolean } = {}
    ) => {
      calls.push({ include_archived, only_snoozed });
      return Promise.resolve({
        items: only_snoozed
          ? [
              {
                id: 201,
                kind: 'reminder_due',
                title: 'Snoozed item',
                message: 'Will return later.',
                severity: 'info',
                created_at: '2026-02-26T00:00:00+00:00',
                read_at: null,
                dismissed_at: '2026-02-26T00:05:00+00:00',
                snooze_until: '2026-02-26T00:20:00+00:00',
              },
            ]
          : [],
        total: 1,
      });
    });
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 0 });
    const user = userEvent.setup();

    render(<NotificationsPage />);

    expect(await screen.findByRole('button', { name: 'Show snoozed (1)' })).toBeInTheDocument();
    expect(screen.queryByText('No notifications yet.')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Show snoozed (1)' }));

    expect(await screen.findByText('Snoozed item')).toBeInTheDocument();
    expect(calls).toContainEqual(expect.objectContaining({ include_archived: true, only_snoozed: true }));
  });

  it('does not treat dismissed-only archived notifications as snoozed without an active reminder', async () => {
    mocks.listNotifications.mockImplementation(({ only_snoozed }: { only_snoozed?: boolean } = {}) =>
      Promise.resolve({
        items: only_snoozed ? [] : [],
        total: 1,
      })
    );
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 0 });

    render(<NotificationsPage />);

    expect(await screen.findByText('No notifications yet.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Show snoozed (1)' })).not.toBeInTheDocument();
  });

  it('moves a notification into the snoozed section immediately after snoozing', async () => {
    const user = userEvent.setup();

    render(<NotificationsPage />);

    expect(await screen.findByText('Job failed')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Snooze 15m' }));

    await waitFor(() => {
      expect(mocks.snoozeNotification).toHaveBeenCalledWith(101, 15);
    });
    expect(screen.queryByText('Job failed')).not.toBeInTheDocument();
    expect(screen.getByText('Unread: 1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Show snoozed (1)' })).toBeInTheDocument();
  });

  it('cancels a snoozed reminder from the snoozed view', async () => {
    mocks.listNotifications.mockImplementation(({ only_snoozed }: { only_snoozed?: boolean } = {}) =>
      Promise.resolve({
        items: only_snoozed
          ? [
              {
                id: 401,
                kind: 'reminder_due',
                title: 'Cancel me',
                message: 'This should go away.',
                severity: 'info',
                created_at: '2026-02-26T00:00:00+00:00',
                read_at: null,
                dismissed_at: '2026-02-26T00:05:00+00:00',
                snooze_until: '2026-02-26T00:20:00+00:00',
              },
            ]
          : [],
        total: 1,
      })
    );
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 0 });
    const user = userEvent.setup();

    render(<NotificationsPage />);

    await user.click(await screen.findByRole('button', { name: 'Show snoozed (1)' }));
    await user.click(await screen.findByRole('button', { name: 'Cancel snooze' }));

    await waitFor(() => {
      expect(mocks.cancelNotificationSnooze).toHaveBeenCalledWith(401);
    });
    expect(screen.queryByText('Cancel me')).not.toBeInTheDocument();
  });

  it('shows an unavailable state when notification preferences fail to load', async () => {
    const user = userEvent.setup();
    mocks.getNotificationPreferences.mockRejectedValueOnce(new Error('preferences unavailable'));

    render(<NotificationsPage />);

    expect(await screen.findByText('Unread: 2')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Preferences' }));

    expect(
      await screen.findByText('Notification preferences are currently unavailable.')
    ).toBeInTheDocument();
    expect(screen.queryByText('Loading preferences...')).not.toBeInTheDocument();
  });

  it('disables preference toggles while a save is in flight and ignores duplicate clicks', async () => {
    const user = userEvent.setup();
    let resolveUpdate:
      | ((value: {
          user_id: string;
          reminder_enabled: boolean;
          job_completed_enabled: boolean;
          job_failed_enabled: boolean;
          updated_at: string;
        }) => void)
      | null = null;

    mocks.updateNotificationPreferences.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveUpdate = resolve;
        })
    );

    render(<NotificationsPage />);

    expect(await screen.findByText('Unread: 2')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Preferences' }));

    const [jobCompletedToggle] = await screen.findAllByRole('checkbox');

    await user.click(jobCompletedToggle);

    await waitFor(() => {
      expect(mocks.updateNotificationPreferences).toHaveBeenCalledTimes(1);
      expect(jobCompletedToggle).toBeDisabled();
    });

    await user.click(jobCompletedToggle);

    expect(mocks.updateNotificationPreferences).toHaveBeenCalledTimes(1);

    resolveUpdate?.({
      user_id: 'user-1',
      reminder_enabled: true,
      job_completed_enabled: false,
      job_failed_enabled: true,
      updated_at: '2026-04-02T00:01:00Z',
    });

    await waitFor(() => {
      expect(jobCompletedToggle).not.toBeDisabled();
      expect(jobCompletedToggle).not.toBeChecked();
    });
  });
});
