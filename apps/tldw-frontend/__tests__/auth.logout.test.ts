import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { authService } from '@web/lib/auth';
import { apiClient } from '@web/lib/api';
import { clearRuntimeAuth, setRuntimeApiKey } from '@web/lib/authStorage';
import {
  AUTH_CREDENTIALS_CHANGED_EVENT,
  type AuthCredentialsChangedDetail,
} from '@web/lib/auth-events';

vi.mock('@web/lib/api', () => {
  return {
    apiClient: {
      post: vi.fn(),
    },
  };
});

const mockedApiClient = apiClient as unknown as {
  post: ReturnType<typeof vi.fn>;
};

const originalApiKey = process.env.NEXT_PUBLIC_X_API_KEY;
const originalBearer = process.env.NEXT_PUBLIC_API_BEARER;

const clearEnv = () => {
  delete process.env.NEXT_PUBLIC_X_API_KEY;
  delete process.env.NEXT_PUBLIC_API_BEARER;
};

const restoreEnv = () => {
  if (originalApiKey === undefined) {
    delete process.env.NEXT_PUBLIC_X_API_KEY;
  } else {
    process.env.NEXT_PUBLIC_X_API_KEY = originalApiKey;
  }
  if (originalBearer === undefined) {
    delete process.env.NEXT_PUBLIC_API_BEARER;
  } else {
    process.env.NEXT_PUBLIC_API_BEARER = originalBearer;
  }
};

describe('authService.logout', () => {
  beforeEach(() => {
    clearEnv();
    localStorage.clear();
    clearRuntimeAuth();
    mockedApiClient.post.mockResolvedValue({});
  });

  afterEach(() => {
    localStorage.clear();
    clearRuntimeAuth();
    restoreEnv();
    vi.resetAllMocks();
  });

  it('calls /auth/logout when a JWT token is present', () => {
    const listener = vi.fn<(event: Event) => void>();
    window.addEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, listener);
    localStorage.setItem('access_token', 'token');
    localStorage.setItem('user', JSON.stringify({ username: 'user' }));

    authService.logout();

    expect(mockedApiClient.post).toHaveBeenCalledWith('/auth/logout');
    expect(localStorage.getItem('access_token')).toBeNull();
    expect(localStorage.getItem('user')).toBeNull();
    expect((listener.mock.calls[0]?.[0] as CustomEvent<AuthCredentialsChangedDetail>).detail).toEqual({
      authenticated: false,
    });
    window.removeEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, listener);
  });

  it('announces token rotation without exposing the token', () => {
    const listener = vi.fn<(event: Event) => void>();
    window.addEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, listener);

    authService.setToken('secret-token');

    const event = listener.mock.calls[0]?.[0] as CustomEvent<AuthCredentialsChangedDetail>;
    expect(event.detail).toEqual({ authenticated: true });
    expect(JSON.stringify(event.detail)).not.toContain('secret-token');
    window.removeEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, listener);
  });

  it('does not call /auth/logout when env auth is present', () => {
    process.env.NEXT_PUBLIC_X_API_KEY = 'key';

    authService.logout();

    expect(mockedApiClient.post).not.toHaveBeenCalled();
  });

  it('does not call /auth/logout when no token exists', () => {
    setRuntimeApiKey('key');

    authService.logout();

    expect(mockedApiClient.post).not.toHaveBeenCalled();
  });
});
