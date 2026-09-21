import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Page } from '@playwright/test';
import { seedAuth, TEST_CONFIG } from '../helpers';
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
    expect(await seeded.storage.local.get('tldwConfig')).toEqual({
      tldwConfig: {
        serverUrl: 'http://127.0.0.1:19897',
        authMode: 'single-user',
        apiKey: 'uat397-fixture-key',
      },
    });
    await seeded.storage.local.set({ uat397_probe: 'persisted' });
    expect(JSON.parse(localStorage.getItem('uat397_probe')!)).toBe('persisted');
  });
});
