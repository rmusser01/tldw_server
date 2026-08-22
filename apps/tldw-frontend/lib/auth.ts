import { apiClient } from './api';
import { dispatchAuthCredentialsChanged } from './auth-events';
import { getRuntimeApiBearer, getRuntimeApiKey } from './authStorage';
import { clearRequestHistory } from './history';

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}

// Mirrors the backend /users/me shape closely but remains tolerant to changes.
export interface User {
  id?: number | string;
  username: string;
  email?: string;
  role?: string;
  is_active?: boolean;
  is_verified?: boolean;
  created_at?: string;
  last_login?: string;
  storage_quota_mb?: number;
  storage_used_mb?: number;
  media_count?: number;
  notes_count?: number;
  prompts_count?: number;
  last_activity?: string;
  // Optional richer RBAC fields
  is_admin?: boolean;
  roles?: string[] | string;
  permissions?: string[] | string;
  scopes?: string[] | string;
}

export type AuthMode = 'env_single_user' | 'env_bearer' | 'jwt' | 'none';

type ApiErrorLike = {
  status?: number;
  statusCode?: number;
  detail?: string;
  retryAfter?: number;
};

const getStoredApiKey = (): string | null => {
  if (typeof window === 'undefined') return null;
  return getRuntimeApiKey();
};

const getStoredApiBearer = (): string | null => {
  if (typeof window === 'undefined') return null;
  return getRuntimeApiBearer();
};

const hasStoredApiKey = (): boolean => !!getStoredApiKey();
const hasStoredApiBearer = (): boolean => !!getStoredApiBearer();

const hasJwtToken = (): boolean => {
  if (typeof window === 'undefined') return false;
  return !!localStorage.getItem('access_token');
};

const getApiErrorInfo = (error: unknown): ApiErrorLike => {
  if (!error || typeof error !== 'object') {
    return {};
  }
  const data = error as Record<string, unknown>;
  const status = typeof data.status === 'number' ? data.status : undefined;
  const statusCode = typeof data.statusCode === 'number' ? data.statusCode : undefined;
  const detail = typeof data.detail === 'string' ? data.detail : undefined;
  const retryAfter = typeof data.retryAfter === 'number' ? data.retryAfter : undefined;
  return {
    status,
    statusCode,
    detail,
    retryAfter,
  };
};

export function getAuthMode(): AuthMode {
  const hasApiKey = !!process.env.NEXT_PUBLIC_X_API_KEY;
  const hasBearer = !!process.env.NEXT_PUBLIC_API_BEARER;

  if (hasApiKey) return 'env_single_user';
  if (hasBearer) return 'env_bearer';

  if (hasJwtToken()) return 'jwt';
  if (hasStoredApiKey()) return 'env_single_user';
  if (hasStoredApiBearer()) return 'env_bearer';

  return 'none';
}

class AuthService {
  private static instance: AuthService;

  private constructor() {}

  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  private hasEnvAuth(): boolean {
    // Either X-API-KEY or API_BEARER provided via env or local storage config
    return !!(process.env.NEXT_PUBLIC_X_API_KEY || process.env.NEXT_PUBLIC_API_BEARER || hasStoredApiKey() || hasStoredApiBearer());
  }

  private hasApiBearer(): boolean {
    return !!(process.env.NEXT_PUBLIC_API_BEARER || hasStoredApiBearer());
  }

  private getEnvUser(): User {
    // Synthetic user when using env-provided API credentials
    return { username: this.hasApiBearer() ? 'api-bearer-auth' : 'api-key-auth' };
  }

  /**
   * Fetch the current user profile from the backend and cache it locally.
   * Only used when operating in JWT mode.
   */
  async fetchCurrentUser(): Promise<User | null> {
    try {
      const user = await apiClient.get<User>('/users/me');
      this.setUser(user);
      return user;
    } catch {
      return null;
    }
  }

  async login(credentials: LoginCredentials): Promise<AuthToken> {
    // OAuth2 compatible login using form data
    const formData = new URLSearchParams();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);

    try {
      const response = await apiClient.post<AuthToken>('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      // Store token and user info
      if (response.access_token) {
        this.setToken(response.access_token);
        // Attempt to hydrate full user profile; fall back to a minimal shape on failure
        const user = await this.fetchCurrentUser();
        if (!user) {
          this.setUser({ username: credentials.username });
        }
      }

      return response;
    } catch (error: unknown) {
      // Map known AuthNZ error patterns (rate limiting, lockout, server errors) into clearer messages
      if (error instanceof Error) {
        const info = getApiErrorInfo(error);
        const status: number | undefined = info.status ?? info.statusCode;
        const detail: string | undefined = info.detail || error.message;
        const retryAfter: number | undefined = info.retryAfter;
        const lowerDetail = (detail || '').toLowerCase();

        // 423: account locked
        if (status === 423) {
          const msg = detail || 'Your account is locked. Please contact an administrator.';
          throw new Error(msg);
        }

        // 429: rate limited
        if (status === 429) {
          if (retryAfter && Number.isFinite(retryAfter) && retryAfter > 0) {
            throw new Error(`Too many login attempts. Please wait ${retryAfter} seconds and try again.`);
          }
          throw new Error('Too many login attempts. Please wait and try again.');
        }

        // 401: invalid credentials or MFA required
        if (status === 401) {
          if (lowerDetail.includes('mfa')) {
            throw new Error(detail || 'Multi-factor authentication is required to complete login.');
          }
          throw new Error(detail || 'Invalid username or password.');
        }

        // 5xx: server error
        if (status && status >= 500) {
          throw new Error(detail || 'Authentication service is temporarily unavailable. Please try again later.');
        }

        // Fallback: surface the original message
        throw new Error(detail || error.message || 'Login failed. Please try again.');
      }

      throw new Error('Login failed. Please try again.');
    }
  }

  logout(): void {
    const mode = getAuthMode();
    const token = this.getToken();
    if (mode === 'jwt' && token) {
      void apiClient.post('/auth/logout').catch(() => undefined);
    }
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed', {
        detail: { kind: 'logout' },
      }));
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      dispatchAuthCredentialsChanged(false);
      // Purge any credentials/tokens that may have been captured in the
      // request-history ring so they cannot survive logout.
      clearRequestHistory();
    }
  }

  getToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('access_token');
    }
    return null;
  }

  setToken(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('access_token', token);
      dispatchAuthCredentialsChanged(true);
    }
  }

  getUser(): User | null {
    if (typeof window !== 'undefined') {
      // Env or config-based auth returns a synthetic user (no local storage)
      if (!hasJwtToken() && this.hasEnvAuth()) {
        return this.getEnvUser();
      }
      const userStr = localStorage.getItem('user');
      if (userStr) {
        try {
          return JSON.parse(userStr) as User;
        } catch {
          return null;
        }
      }
    }
    return null;
  }

  setUser(user: User): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('user', JSON.stringify(user));
    }
  }

  isAuthenticated(): boolean {
    // Auth when: JWT token present, or env credentials present
    return getAuthMode() !== 'none';
  }

  async validateToken(): Promise<boolean> {
    try {
      const mode = getAuthMode();

      // Env-provided credentials: assume valid here (connectivity is checked via feature-specific flows)
      if (mode === 'env_single_user' || mode === 'env_bearer') {
        return true;
      }

      if (mode !== 'jwt') {
        return false;
      }

      const user = await this.fetchCurrentUser();
      return !!user;
    } catch {
      return false;
    }
  }
}

export const authService = AuthService.getInstance();
