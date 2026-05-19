import { useCallback, useState } from 'react';
import {
  buildNotificationSettingsUpdate,
  normalizeNotificationSettings,
  normalizeRecentNotifications,
} from './notification-utils';
import type {
  NotificationSettings,
  RecentNotification,
} from './types';
import { logger } from '@/lib/logger';

type SetState<T> = (next: T | ((prev: T) => T)) => void;

export type NotificationActionsApiClient = {
  updateNotificationSettings: (payload: Record<string, unknown>) => Promise<unknown>;
  testNotification: (payload?: { message?: string; severity?: string }) => Promise<unknown>;
  getRecentNotifications: () => Promise<unknown>;
};

type UseNotificationActionsArgs = {
  apiClient: NotificationActionsApiClient;
  canSave: boolean;
  setNotificationSettings: SetState<NotificationSettings | null>;
  setRecentNotifications: SetState<RecentNotification[]>;
  setError: (message: string) => void;
  setSuccess: (message: string) => void;
};

export const useNotificationActions = ({
  apiClient,
  canSave,
  setNotificationSettings,
  setRecentNotifications,
  setError,
  setSuccess,
}: UseNotificationActionsArgs) => {
  const [notificationsSaving, setNotificationsSaving] = useState(false);

  const handleSaveNotificationSettings = useCallback(async (settings: NotificationSettings) => {
    if (!canSave) {
      return false;
    }
    try {
      setNotificationsSaving(true);
      setError('');
      const payload = buildNotificationSettingsUpdate(settings);
      const updated = await apiClient.updateNotificationSettings(payload as unknown as Record<string, unknown>);
      setNotificationSettings(normalizeNotificationSettings(updated));
      setSuccess('Notification settings saved');
      return true;
    } catch (err: unknown) {
      logger.error('Failed to save notification settings', { component: 'useNotificationActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to save notification settings');
      return false;
    } finally {
      setNotificationsSaving(false);
    }
  }, [apiClient, canSave, setError, setNotificationSettings, setSuccess]);

  const handleTestNotification = useCallback(async (
    payload?: {
      message?: string;
      severity?: string;
    },
  ) => {
    try {
      setError('');
      await apiClient.testNotification(payload ?? {});
      setSuccess('Test notification sent');
      try {
        const data = await apiClient.getRecentNotifications();
        setRecentNotifications(normalizeRecentNotifications(data));
      } catch {
        // Ignore reload errors
      }
    } catch (err: unknown) {
      logger.error('Failed to send test notification', { component: 'useNotificationActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to send test notification');
    }
  }, [apiClient, setError, setRecentNotifications, setSuccess]);

  return {
    notificationsSaving,
    handleSaveNotificationSettings,
    handleTestNotification,
  };
};
