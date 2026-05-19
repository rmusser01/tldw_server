'use client';

import { useCallback, useEffect, useState } from 'react';
import type { PrivilegedActionOptions, PrivilegedActionResult } from '@/components/ui/privileged-action-dialog';
import { api } from '@/lib/api-client';
import type { AuditLog } from '@/types';
import { logger } from '@/lib/logger';

export type MfaStatus = {
  enabled: boolean;
  has_secret: boolean;
  has_backup_codes: boolean;
  method?: string | null;
};

export type UserSession = {
  id: number;
  ip_address?: string | null;
  user_agent?: string | null;
  created_at: string;
  last_activity?: string | null;
  expires_at?: string | null;
};

type PasswordResetResponse = {
  force_password_change?: boolean;
  message?: string;
};

type LoginHistoryStatus = 'success' | 'failure';

export type LoginHistoryEntry = {
  id: string;
  timestamp: string;
  ipAddress?: string;
  userAgent?: string;
  status: LoginHistoryStatus;
};

interface UseUserSecurityOptions {
  userId: string;
  user: {
    username?: string | null;
    email?: string | null;
    metadata?: Record<string, unknown>;
  } | null;
  requirePasswordReauth: boolean;
  promptPrivilegedAction: (options: PrivilegedActionOptions) => Promise<PrivilegedActionResult | null>;
  onUserRefresh: () => void;
  showError: (title: string, message: string) => void;
  toastSuccess: (title: string, message: string) => void;
}

const getLoginAttemptStatus = (entry: AuditLog): LoginHistoryStatus => {
  const details = entry.details ?? {};
  const raw = entry.raw ?? {};
  const successValue = details.success ?? raw.success;
  if (typeof successValue === 'boolean') {
    return successValue ? 'success' : 'failure';
  }

  const statusText = String(details.status ?? details.result ?? raw.status ?? '').toLowerCase();
  if (statusText.includes('fail') || statusText.includes('error') || statusText.includes('denied')) {
    return 'failure';
  }
  if (statusText.includes('success') || statusText.includes('ok')) {
    return 'success';
  }

  const actionText = String(entry.action ?? raw.action ?? '').toLowerCase();
  if (actionText.includes('fail') || actionText.includes('denied')) {
    return 'failure';
  }
  return 'success';
};

const toLoginHistoryEntry = (entry: AuditLog, index: number): LoginHistoryEntry => {
  const details = entry.details ?? {};
  const raw = entry.raw ?? {};
  const ipValue = entry.ip_address
    ?? (typeof details.ip_address === 'string' ? details.ip_address : undefined)
    ?? (typeof details.ip === 'string' ? details.ip : undefined)
    ?? (typeof raw.ip_address === 'string' ? raw.ip_address : undefined)
    ?? (typeof raw.ip === 'string' ? raw.ip : undefined);
  const detailsUserAgent = details.user_agent;
  const detailsUserAgentAlt = details.userAgent;
  const rawUserAgent = raw.user_agent;
  const rawUserAgentAlt = raw.userAgent;
  const userAgentValue = (typeof detailsUserAgent === 'string' ? detailsUserAgent : undefined)
    ?? (typeof detailsUserAgentAlt === 'string' ? detailsUserAgentAlt : undefined)
    ?? (typeof rawUserAgent === 'string' ? rawUserAgent : undefined)
    ?? (typeof rawUserAgentAlt === 'string' ? rawUserAgentAlt : undefined);

  return {
    id: entry.id || `${entry.timestamp}-${index}`,
    timestamp: entry.timestamp,
    ipAddress: ipValue,
    userAgent: userAgentValue,
    status: getLoginAttemptStatus(entry),
  };
};

const getUserLabel = (user: UseUserSecurityOptions['user']): string =>
  user?.username || user?.email || 'this user';

export function useUserSecurity({
  userId,
  user,
  requirePasswordReauth,
  promptPrivilegedAction,
  onUserRefresh,
  showError,
  toastSuccess,
}: UseUserSecurityOptions) {
  const [securityLoading, setSecurityLoading] = useState(false);
  const [securityError, setSecurityError] = useState('');
  const [mfaStatus, setMfaStatus] = useState<MfaStatus | null>(null);
  const [sessions, setSessions] = useState<UserSession[]>([]);
  const [loginHistory, setLoginHistory] = useState<LoginHistoryEntry[]>([]);
  const [loginHistoryLoading, setLoginHistoryLoading] = useState(false);
  const [loginHistoryError, setLoginHistoryError] = useState('');
  const [forcePasswordChangeOnNextLogin, setForcePasswordChangeOnNextLogin] = useState(true);
  const [passwordResetLoading, setPasswordResetLoading] = useState(false);
  const [resetPasswordValue, setResetPasswordValue] = useState('');

  useEffect(() => {
    const forceChange =
      typeof user?.metadata?.force_password_change === 'boolean'
        ? user.metadata.force_password_change
        : true;
    setForcePasswordChangeOnNextLogin(forceChange);
  }, [user?.metadata?.force_password_change]);

  const loadSecurity = useCallback(async () => {
    if (!userId) return;
    try {
      setSecurityLoading(true);
      setSecurityError('');
      const [mfaResult, sessionsResult] = await Promise.allSettled([
        api.getUserMfaStatus(userId),
        api.getUserSessions(userId),
      ]);

      if (mfaResult.status === 'fulfilled') {
        setMfaStatus(mfaResult.value as MfaStatus);
      } else {
        setMfaStatus(null);
      }

      if (sessionsResult.status === 'fulfilled') {
        setSessions(Array.isArray(sessionsResult.value) ? (sessionsResult.value as UserSession[]) : []);
      } else {
        setSessions([]);
      }

      if (mfaResult.status === 'rejected' || sessionsResult.status === 'rejected') {
        setSecurityError('Failed to load security controls.');
      }
    } catch (error) {
      logger.error('Failed to load security controls', { component: 'useUserSecurity', error: error instanceof Error ? error.message : String(error) });
      setSecurityError('Failed to load security controls.');
      setMfaStatus(null);
      setSessions([]);
    } finally {
      setSecurityLoading(false);
    }
  }, [userId]);

  const loadLoginHistory = useCallback(async () => {
    if (!userId) return;
    try {
      setLoginHistoryLoading(true);
      setLoginHistoryError('');
      const response = await api.getAuditLogs({
        user_id: String(userId),
        action: 'login',
        limit: '20',
        offset: '0',
      });
      const entries = Array.isArray(response?.entries)
        ? (response.entries as AuditLog[])
        : [];
      const mapped = entries.map((entry, index) => toLoginHistoryEntry(entry, index));
      mapped.sort((a, b) => {
        const aTime = Date.parse(a.timestamp || '');
        const bTime = Date.parse(b.timestamp || '');
        return (Number.isFinite(bTime) ? bTime : 0) - (Number.isFinite(aTime) ? aTime : 0);
      });
      setLoginHistory(mapped.slice(0, 20));
    } catch (error) {
      logger.error('Failed to load login history', { component: 'useUserSecurity', error: error instanceof Error ? error.message : String(error) });
      setLoginHistory([]);
      setLoginHistoryError('Failed to load login history.');
    } finally {
      setLoginHistoryLoading(false);
    }
  }, [userId]);

  const handleDisableMfa = useCallback(async () => {
    if (!mfaStatus?.enabled) return;
    const approval = await promptPrivilegedAction({
      title: 'Disable MFA',
      message: `Disable MFA for ${getUserLabel(user)}?`,
      confirmText: 'Disable MFA',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      await api.disableUserMfa(userId, {
        reason: approval.reason,
        admin_password: approval.adminPassword,
      });
      toastSuccess('MFA disabled', 'Multi-factor authentication has been turned off.');
      void loadSecurity();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to disable MFA';
      showError('Disable MFA failed', message);
    }
  }, [loadSecurity, mfaStatus?.enabled, promptPrivilegedAction, requirePasswordReauth, showError, toastSuccess, user, userId]);

  const handleRevokeSession = useCallback(async (session: UserSession) => {
    const approval = await promptPrivilegedAction({
      title: 'Revoke Session',
      message: 'Revoke this session? The user will be signed out on that device.',
      confirmText: 'Revoke',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      await api.revokeUserSession(userId, session.id.toString(), {
        reason: approval.reason,
        admin_password: approval.adminPassword,
      });
      toastSuccess('Session revoked', 'The session has been revoked.');
      void loadSecurity();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to revoke session';
      showError('Revoke failed', message);
    }
  }, [loadSecurity, promptPrivilegedAction, requirePasswordReauth, showError, toastSuccess, userId]);

  const handleRevokeAllSessions = useCallback(async () => {
    const approval = await promptPrivilegedAction({
      title: 'Revoke All Sessions',
      message: 'Revoke all active sessions for this user? They will be signed out everywhere.',
      confirmText: 'Revoke all',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      await api.revokeAllUserSessions(userId, {
        reason: approval.reason,
        admin_password: approval.adminPassword,
      });
      toastSuccess('Sessions revoked', 'All sessions have been revoked.');
      void loadSecurity();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to revoke sessions';
      showError('Revoke failed', message);
    }
  }, [loadSecurity, promptPrivilegedAction, requirePasswordReauth, showError, toastSuccess, userId]);

  const handleResetPassword = useCallback(async () => {
    const normalizedTemporaryPassword = resetPasswordValue.trim();
    if (normalizedTemporaryPassword.length < 10) {
      showError('Password reset failed', 'Enter a temporary password with at least 10 characters.');
      return;
    }

    const approval = await promptPrivilegedAction({
      title: 'Reset Password',
      message: `Reset password for ${getUserLabel(user)}?`,
      confirmText: 'Reset password',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      setPasswordResetLoading(true);
      await api.resetUserPassword(userId, {
        temporary_password: normalizedTemporaryPassword,
        force_password_change: forcePasswordChangeOnNextLogin,
        reason: approval.reason,
        admin_password: approval.adminPassword,
      }) as PasswordResetResponse;
      setResetPasswordValue('');
      toastSuccess(
        'Password reset',
        'Temporary password updated. Share it with the user through an approved secure channel.'
      );
      onUserRefresh();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to reset password';
      showError('Password reset failed', message);
    } finally {
      setPasswordResetLoading(false);
    }
  }, [
    forcePasswordChangeOnNextLogin,
    onUserRefresh,
    promptPrivilegedAction,
    requirePasswordReauth,
    resetPasswordValue,
    showError,
    toastSuccess,
    user,
    userId,
  ]);

  return {
    securityLoading,
    securityError,
    mfaStatus,
    sessions,
    loginHistory,
    loginHistoryLoading,
    loginHistoryError,
    forcePasswordChangeOnNextLogin,
    passwordResetLoading,
    resetPasswordValue,
    setForcePasswordChangeOnNextLogin,
    setResetPasswordValue,
    loadSecurity,
    loadLoginHistory,
    handleDisableMfa,
    handleRevokeSession,
    handleRevokeAllSessions,
    handleResetPassword,
  };
}
