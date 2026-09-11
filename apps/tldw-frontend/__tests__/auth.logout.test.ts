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
    sessionStorage.clear();
    clearRuntimeAuth();
    mockedApiClient.post.mockResolvedValue({});
  });

  afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
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

  it('clears local auth before synchronously announcing the trusted logout boundary', () => {
    localStorage.setItem('access_token', 'token');
    localStorage.setItem('user', JSON.stringify({ id: 42, username: 'user' }));
    const observed: Array<{ token: string | null; user: string | null; kind?: string }> = [];
    const listener = (event: Event) => {
      observed.push({
        token: localStorage.getItem('access_token'),
        user: localStorage.getItem('user'),
        kind: (event as CustomEvent<{ kind?: string }>).detail?.kind,
      });
    };
    window.addEventListener('tldw:auth-principal-changed', listener);

    authService.logout();

    expect(observed).toEqual([{ token: null, user: null, kind: 'logout' }]);
    window.removeEventListener('tldw:auth-principal-changed', listener);
  });

  it('removes only Task 14 recovery keys without reading source-bearing values when no route hook is mounted', () => {
    const draftKey = 'tldw:presentation-studio:html:draft:v1:https%3A%2F%2Ftldw.example:42';
    const resumeKey = 'tldw:presentation-studio:html:resume:v1:https%3A%2F%2Ftldw.example:42';
    sessionStorage.setItem(draftKey, 'PRIVATE DIRECT MATERIAL');
    sessionStorage.setItem(resumeKey, '{"idempotencyKey":"PRIVATE-KEY"}');
    sessionStorage.setItem('unrelated:session:key', 'keep');
    const getSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, 'getItem');

    authService.logout();

    const remainingKeys = Array.from(
      { length: sessionStorage.length },
      (_, index) => sessionStorage.key(index),
    );
    expect(remainingKeys).toEqual(['unrelated:session:key']);
    expect(sessionStorage.getItem('unrelated:session:key')).toBe('keep');
    expect(getSpy).not.toHaveBeenCalledWith(draftKey);
    expect(getSpy).not.toHaveBeenCalledWith(resumeKey);
    getSpy.mockRestore();
  });
});
