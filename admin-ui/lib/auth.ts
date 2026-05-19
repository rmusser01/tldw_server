'use client';

import { clearScopedStorage } from '@/lib/scoped-storage';
import { logger } from '@/lib/logger';

const AUTH_CHANGE_EVENT = 'tldw-admin-auth-change';
const SESSION_MARKER_COOKIE = 'admin_session';
const AUTH_MODE_COOKIE = 'admin_auth_mode';

// In-memory storage for legacy single-user API key mode.
let inMemoryApiKey: string | null = null;

const clearApiKeyStorage = (): void => {
  inMemoryApiKey = null;
};

const clearJwtStorage = (): void => {
  if (typeof localStorage !== 'undefined') {
    localStorage.removeItem('access_token');
  }
};

const hasSessionMarker = (): boolean => {
  if (typeof document === 'undefined') return false;
  return document.cookie
    .split(';')
    .some((cookie) => cookie.trim().startsWith(`${SESSION_MARKER_COOKIE}=`));
};

const getCookieValue = (name: string): string | null => {
  if (typeof document === 'undefined') return null;
  const cookie = document.cookie
    .split(';')
    .map((value) => value.trim())
    .find((value) => value.startsWith(`${name}=`));
  if (!cookie) return null;
  const [, rawValue] = cookie.split('=');
  return rawValue ? decodeURIComponent(rawValue) : null;
};

const emitAuthChange = (): void => {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
};

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

const waitForSessionMarker = async (timeoutMs = 2_000): Promise<boolean> => {
  if (typeof window === 'undefined') return false;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (hasSessionMarker()) {
      return true;
    }
    await sleep(50);
  }
  return hasSessionMarker();
};

const storeUser = (user: AdminUser): void => {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem('user', JSON.stringify(user));
};

const clearStoredUser = (): void => {
  if (typeof localStorage === 'undefined') return;
  localStorage.removeItem('user');
};

const waitForAuthenticatedUser = async (timeoutMs = 2_000): Promise<AdminUser | null> => {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const user = await fetchAndStoreUser();
    if (user) {
      return user;
    }
    await sleep(50);
  }
  return fetchAndStoreUser();
};

const finalizeAuthenticatedLogin = async (): Promise<AuthenticatedLoginResult | null> => {
  if (!await waitForSessionMarker()) {
    return null;
  }
  const user = await waitForAuthenticatedUser();
  if (!user) {
    return null;
  }
  clearApiKeyStorage();
  clearJwtStorage();
  emitAuthChange();
  return {
    status: 'authenticated',
  };
};

export const subscribeAuthChange = (handler: () => void): (() => void) => {
  if (typeof window === 'undefined') return () => {};
  window.addEventListener(AUTH_CHANGE_EVENT, handler);
  return () => window.removeEventListener(AUTH_CHANGE_EVENT, handler);
};

export interface AdminUser {
  id: number;
  uuid: string;
  username: string;
  email: string;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  storage_quota_mb: number;
  storage_used_mb: number;
  created_at: string;
  updated_at: string;
  last_login?: string;
}

export interface AuthenticatedLoginResult {
  status: 'authenticated';
}

export interface MfaRequiredLoginResult {
  status: 'mfa_required';
  sessionToken: string;
  expiresIn: number;
  message: string;
}

export type PasswordLoginResult = AuthenticatedLoginResult | MfaRequiredLoginResult;

type LoginSuccessPayload = {
  token_type?: string;
  session_token?: string;
  mfa_required?: boolean;
  expires_in?: number;
  message?: string;
};

type ApiKeyLoginPayload = {
  user?: AdminUser;
};

const isMfaChallengePayload = (
  value: LoginSuccessPayload
): value is Required<Pick<LoginSuccessPayload, 'session_token' | 'expires_in' | 'message'>> & {
  mfa_required: true;
} =>
  value.mfa_required === true
  && typeof value.session_token === 'string'
  && typeof value.expires_in === 'number'
  && typeof value.message === 'string';

const isAuthenticatedLoginPayload = (value: LoginSuccessPayload): boolean =>
  value.mfa_required !== true && typeof value.token_type === 'string';

/**
 * Check if we're in single-user mode (X-API-KEY auth)
 */
export function isSingleUserMode(): boolean {
  if (typeof window === 'undefined') return false;
  return getCookieValue(AUTH_MODE_COOKIE) === 'single_user' || !!getApiKey();
}

/**
 * Get X-API-KEY for single-user mode
 */
export function getApiKey(): string | null {
  if (typeof window === 'undefined') return null;
  return inMemoryApiKey;
}

/**
 * Set X-API-KEY for single-user mode
 */
export function setApiKey(key: string): void {
  if (typeof window === 'undefined') return;
  inMemoryApiKey = key;
  emitAuthChange();
}

/**
 * Admin bearer tokens are stored in httpOnly cookies and are never exposed to browser JS.
 */
export function getJWTToken(): string | null {
  return null;
}

export function hasStoredAuth(): boolean {
  if (typeof window === 'undefined') return false;
  return hasSessionMarker() || !!getApiKey();
}

/**
 * Login with username/email and password (multi-user JWT mode)
 */
export async function loginWithPassword(
  username: string,
  password: string
): Promise<PasswordLoginResult | null> {
  try {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData.toString(),
      credentials: 'include',
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json() as LoginSuccessPayload;
    if (isMfaChallengePayload(data)) {
      return {
        status: 'mfa_required',
        sessionToken: data.session_token,
        expiresIn: data.expires_in,
        message: data.message,
      };
    }

    if (isAuthenticatedLoginPayload(data)) {
      return await finalizeAuthenticatedLogin();
    }

    return null;
  } catch (error) {
    logger.error('Login failed', { component: 'auth', error: error instanceof Error ? error.message : String(error) });
    return null;
  }
}

export async function completeMfaLogin(
  sessionToken: string,
  mfaToken: string
): Promise<AuthenticatedLoginResult | null> {
  try {
    const response = await fetch('/api/auth/mfa/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_token: sessionToken,
        mfa_token: mfaToken,
      }),
      credentials: 'include',
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json() as LoginSuccessPayload;
    if (!isAuthenticatedLoginPayload(data)) {
      return null;
    }

    return await finalizeAuthenticatedLogin();
  } catch (error) {
    logger.error('MFA login failed', { component: 'auth', error: error instanceof Error ? error.message : String(error) });
    return null;
  }
}

/**
 * Login with API key (single-user mode)
 */
export async function loginWithApiKey(apiKey: string): Promise<boolean> {
  try {
    const response = await fetch('/api/auth/apikey', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ apiKey }),
      credentials: 'include',
    });

    if (!response.ok) {
      return false;
    }

    const payload = await response.json() as ApiKeyLoginPayload;
    if (!payload.user) {
      return false;
    }

    storeUser(payload.user);

    if (!await waitForSessionMarker()) {
      clearStoredUser();
      return false;
    }

    clearJwtStorage();
    clearApiKeyStorage();
    emitAuthChange();

    void waitForAuthenticatedUser().catch(() => null);
    return true;
  } catch (error) {
    logger.error('API key validation failed', { component: 'auth', error: error instanceof Error ? error.message : String(error) });
    return false;
  }
}

/**
 * Fetch and store current user info using the server-managed session.
 */
async function fetchAndStoreUser(): Promise<AdminUser | null> {
  try {
    const response = await fetch('/api/proxy/users/me', {
      credentials: 'include',
    });

    if (!response.ok) {
      return null;
    }

    const user: AdminUser = await response.json();
    storeUser(user);
    return user;
  } catch (error) {
    logger.error('Failed to fetch user', { component: 'auth', error: error instanceof Error ? error.message : String(error) });
    return null;
  }
}

/**
 * Logout - clear all stored credentials
 */
export async function logout(): Promise<void> {
  try {
    await fetch('/api/auth/logout', {
      method: 'POST',
      credentials: 'include',
    }).catch(() => {
      // Ignore errors - local auth state must still be cleared.
    });
  } finally {
    clearScopedStorage();
    clearStoredUser();
    clearJwtStorage();
    clearApiKeyStorage();
    emitAuthChange();
  }
}
