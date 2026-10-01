import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Page } from '@playwright/test';
import { seedAuth, TEST_CONFIG } from '../helpers';
import { toPersistedTldwConfig } from '../../../../packages/ui/src/services/tldw/single-user-credential';
import type { TldwConfig } from '../../../../packages/ui/src/services/tldw/TldwApiClient';
import { isExtensionRuntime } from '../../../../packages/ui/src/utils/browser-runtime';

type SeededChrome = {
  runtime?: { id?: string };
  storage: {
    local: {
      get: (key: string) => Promise<Record<string, unknown>>;
      set: (value: Record<string, unknown>) => Promise<void>;
    };
  };
};

describe('seedAuth WebUI runtime identity (UAT397)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    localStorage.clear();
  });

  it.each([
    { name: 'strict live default', live: '1', allowOffline: undefined, stale: false, expected: null },
    { name: 'strict live explicit bypass', live: '1', allowOffline: true, stale: false, expected: null },
    { name: 'strict live stale bypass', live: '1', allowOffline: undefined, stale: true, expected: null },
    { name: 'isolated verified connection', live: '0', allowOffline: false, stale: true, expected: null },
    { name: 'deliberate isolated offline mode', live: '0', allowOffline: true, stale: false, expected: 'true' },
  ])('seeds the connection policy for $name', async ({ live, allowOffline, stale, expected }) => {
    vi.stubEnv('TLDW_LIVE_TIER_UAT', live);
    vi.stubGlobal('chrome', undefined);
    vi.stubGlobal('browser', {});
    if (stale) localStorage.setItem('__tldw_allow_offline', 'true');
    const page = {
      addInitScript: async (
        init: (config: typeof TEST_CONFIG) => void,
        config: typeof TEST_CONFIG
      ) => init(config),
      route: vi.fn(),
    } as unknown as Page;

    await seedAuth(page, allowOffline === undefined ? {} : { allowOffline });

    // This flag makes the real connection store bypass its credential/health check,
    // which correctly prevents NotificationLifecycleProvider from requesting inbox data.
    expect(localStorage.getItem('__tldw_allow_offline')).toBe(expected);
  });

  it.each([
    { name: 'no chrome runtime', chrome: undefined, id: undefined },
    { name: 'an empty chrome runtime', chrome: { runtime: {} }, id: undefined },
    {
      name: 'a genuine extension',
      chrome: { runtime: { id: 'genuine-extension-id' } },
      id: 'genuine-extension-id',
    },
  ])('seeds working storage without inventing identity for $name', async ({ chrome, id }) => {
    vi.stubEnv('TLDW_LIVE_TIER_UAT', '1');
    vi.stubGlobal('chrome', chrome);
    vi.stubGlobal('browser', {});
    const page = {
      // Execute the actual production fixture callback, not a copied shim.
      addInitScript: async (
        init: (config: typeof TEST_CONFIG) => void,
        config: typeof TEST_CONFIG
      ) => {
        init(config);
      },
    } as unknown as Page;
    await seedAuth(page, { serverUrl: 'http://127.0.0.1:19897', apiKey: 'uat397-fixture-key' });
    const seeded = (window as Window & { chrome: SeededChrome }).chrome;
    expect(seeded.runtime?.id).toBe(id);
    expect(isExtensionRuntime()).toBe(id !== undefined);
    expect(await seeded.storage.local.get('tldwConfig')).toMatchObject({
      tldwConfig: {
        serverUrl: 'http://127.0.0.1:19897',
        authMode: 'single-user',
        apiKey: 'uat397-fixture-key',
      },
    });
    expect(await seeded.storage.local.get('notes-tutorial-shown')).toEqual({
      'notes-tutorial-shown': '1',
    });
    await seeded.storage.local.set({ uat397_probe: 'persisted' });
    expect(JSON.parse(localStorage.getItem('uat397_probe')!)).toBe('persisted');
  });
});

describe('seedAuth production credential persistence (UAT398)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    localStorage.clear();
  });

  it.each(['http://127.0.0.1:19897', 'http://localhost:19898/'])(
    'retains only the intended server credential through production serialization: %s',
    async (serverUrl) => {
      vi.stubEnv('TLDW_LIVE_TIER_UAT', '1');
      vi.stubGlobal('chrome', undefined);
      vi.stubGlobal('browser', {});
      const page = {
        addInitScript: async (
          init: (config: typeof TEST_CONFIG) => void,
          config: typeof TEST_CONFIG
        ) => init(config),
      } as unknown as Page;
      await seedAuth(page, { serverUrl, apiKey: 'uat398-owned-key' });
      const stored: TldwConfig = JSON.parse(localStorage.getItem('tldwConfig')!);
      // Quickstart initialization uses this real policy to remove incomplete keys.
      expect(toPersistedTldwConfig(stored).apiKey).toBe('uat398-owned-key');
      expect(stored.apiKeyServerOrigin).toBe(new URL(serverUrl).origin);
      expect(
        toPersistedTldwConfig({ ...stored, serverUrl: 'http://localhost:19899' })
      ).not.toHaveProperty('apiKey');
    }
  );
});
