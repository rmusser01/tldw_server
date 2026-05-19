import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
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

import NotificationsPage from '@web/pages/notifications';

describe('NotificationsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  it('updates the inbox from stream events without showing a duplicate toast', async () => {
    let onEvent: ((event: { event: string; id?: number; payload?: unknown }) => void) | null = null;
    mocks.subscribeNotificationsStream.mockImplementation((options: { onEvent: typeof onEvent }) => {
      onEvent = options.onEvent;
      return () => {};
    });

    render(<NotificationsPage />);

    expect(await screen.findByText('Unread: 2')).toBeInTheDocument();

    onEvent?.({
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
    });

    expect(await screen.findByText('Open the report in Deep Research.')).toBeInTheDocument();
    expect(mocks.showToast).not.toHaveBeenCalled();
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
