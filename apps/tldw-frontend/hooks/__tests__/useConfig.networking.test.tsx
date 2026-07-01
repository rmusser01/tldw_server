import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const authStorageMocks = vi.hoisted(() => ({
  setRuntimeApiBearer: vi.fn(),
  setRuntimeApiKey: vi.fn(),
}));

vi.mock('@web/lib/authStorage', () => authStorageMocks);

describe('useConfig networking', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    localStorage.clear();
    vi.unstubAllGlobals();
    delete process.env.NEXT_PUBLIC_API_URL;
    delete process.env.NEXT_PUBLIC_API_BASE_URL;
    delete process.env.NEXT_PUBLIC_API_VERSION;
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
    delete process.env.NEXT_PUBLIC_X_API_KEY;
    delete process.env.NEXT_PUBLIC_API_BEARER;
  });

  it('keeps a relative /api/v1 base in quickstart mode', async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart';

    const apiModule = await import('@web/lib/api');
    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    await waitFor(() => {
      expect(apiModule.getApiBaseUrl()).toBe('/api/v1');
    });

    expect(result.current.config.apiVersion).toBe('v1');
  });

  it('does not let a stored absolute host override quickstart mode', async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart';
    localStorage.setItem('tldw-api-host', 'http://127.0.0.1:8000');

    const apiModule = await import('@web/lib/api');
    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    await waitFor(() => {
      expect(apiModule.getApiBaseUrl()).toBe('/api/v1');
    });

    expect(localStorage.getItem('tldw-api-host')).not.toBe('http://127.0.0.1:8000');
  });

  it('fetches docs-info from the quickstart same-origin api root', async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart';

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    await result.current.reloadBootstrapConfig();

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/config/docs-info', {
      credentials: 'omit',
    });
  });

  it('pins quickstart docs-info fetches to api v1 even when a different version is stored', async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart';
    localStorage.setItem('tldw-api-version', 'v9');

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    await result.current.reloadBootstrapConfig();

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/config/docs-info', {
      credentials: 'omit',
    });
  });

  it('fetches docs-info from the advanced api origin', async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'advanced';
    process.env.NEXT_PUBLIC_API_URL = 'https://api.example.test';

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    await result.current.reloadBootstrapConfig();

    expect(fetchMock).toHaveBeenCalledWith('https://api.example.test/api/v1/config/docs-info', {
      credentials: 'omit',
    });
  });

  it('hydrates single-user api keys from the canonical browser config', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8000';
    localStorage.setItem(
      'tldwConfig',
      JSON.stringify({
        authMode: 'single-user',
        apiKey: 'stored-api-key',
        serverUrl: 'http://127.0.0.1:8123',
      })
    );

    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    expect(result.current.config.xApiKey).toBe('stored-api-key');
    expect(result.current.config.apiBaseHost).toBe('http://127.0.0.1:8123');

    await waitFor(() => {
      expect(authStorageMocks.setRuntimeApiKey).toHaveBeenLastCalledWith('stored-api-key');
    });
  });

  it('persists manually entered single-user api keys for reloads', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8000';
    localStorage.setItem('apiBearer', 'legacy-bearer');
    localStorage.setItem('refreshToken', 'legacy-refresh');
    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    act(() => {
      result.current.setXApiKey('saved-api-key');
    });

    await waitFor(() => {
      expect(authStorageMocks.setRuntimeApiKey).toHaveBeenLastCalledWith('saved-api-key');
    });
    expect(localStorage.getItem('apiKey')).toBe('saved-api-key');
    expect(localStorage.getItem('apiBearer')).toBeNull();
    expect(localStorage.getItem('tldwConfig')).toContain('saved-api-key');
    expect(localStorage.getItem('refreshToken')).toBeNull();
  });

  it('persists manually entered multi-user bearer tokens for reloads', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8000';
    localStorage.setItem('apiKey', 'legacy-api-key');
    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    act(() => {
      result.current.setApiBearer('Bearer saved-bearer-token');
    });

    await waitFor(() => {
      expect(authStorageMocks.setRuntimeApiBearer).toHaveBeenLastCalledWith('Bearer saved-bearer-token');
    });
    expect(localStorage.getItem('accessToken')).toBe('saved-bearer-token');
    expect(localStorage.getItem('apiKey')).toBeNull();
    expect(localStorage.getItem('tldwConfig')).toContain('saved-bearer-token');
  });

  it('refreshes live config after settings writes canonical tldw config', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8000';
    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    act(() => {
      localStorage.setItem(
        'tldwConfig',
        JSON.stringify({
          authMode: 'single-user',
          apiKey: 'event-api-key',
          serverUrl: 'http://127.0.0.1:8222',
        })
      );
      window.dispatchEvent(new CustomEvent('tldw:config-updated'));
    });

    await waitFor(() => {
      expect(result.current.config.xApiKey).toBe('event-api-key');
      expect(result.current.config.apiBaseHost).toBe('http://127.0.0.1:8222');
      expect(authStorageMocks.setRuntimeApiKey).toHaveBeenLastCalledWith('event-api-key');
    });
    expect(localStorage.getItem('apiKey')).toBe('event-api-key');
    expect(localStorage.getItem('tldwConfig')).toContain('event-api-key');
  });

  it('keeps environment api keys ahead of stale browser config', async () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8000';
    process.env.NEXT_PUBLIC_X_API_KEY = 'env-api-key';
    localStorage.setItem(
      'tldwConfig',
      JSON.stringify({
        authMode: 'single-user',
        apiKey: 'stale-browser-key',
      })
    );

    const { ConfigProvider, useConfig } = await import('@web/hooks/useConfig');

    const { result } = renderHook(() => useConfig(), {
      wrapper: ({ children }) => <ConfigProvider>{children}</ConfigProvider>,
    });

    expect(result.current.config.xApiKey).toBe('env-api-key');

    await waitFor(() => {
      expect(authStorageMocks.setRuntimeApiKey).toHaveBeenLastCalledWith('env-api-key');
    });
  });
});
