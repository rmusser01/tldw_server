import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock('@/services/background-proxy', () => ({
  bgRequest: (...args: unknown[]) => transport.request(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn(),
}));
vi.mock('@/utils/safe-storage', () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(),
    remove: vi.fn(),
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value,
  },
}));

import { TldwApiClient, TldwApiClientBase } from '../TldwApiClient';
import type { ScopedRequestOptions } from '../TldwApiClient';

const scope: NonNullable<ScopedRequestOptions['requestScope']> = {
  config: { serverUrl: 'https://original-image.test', authMode: 'multi-user' },
  userId: 42,
};
const id = '83569797-3613-42e0-b2a2-498794fc1cd0';

describe('captured Data Tables owner: composed client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    transport.request.mockReset();
    transport.request.mockResolvedValue({ job_id: 71 });
  });
  it('keeps scoped table methods canonical in the domain', () => {
    for (const name of ['getDataTable', 'regenerateDataTable']) {
      expect(Object.prototype.hasOwnProperty.call(TldwApiClientBase.prototype, name)).toBe(false);
    }
  });
  it.each(['generate', 'regenerate', 'poll', 'read', 'cancel'] as const)(
    'keeps the original scope for %s',
    async (action) => {
      const client = new TldwApiClient();
      const controller = new AbortController();
      const options: ScopedRequestOptions = { requestScope: scope, signal: controller.signal };
      if (action === 'generate')
        await client.generateDataTable(
          { name: 'Rowan', prompt: 'Extract facts', sources: [] },
          options
        );
      if (action === 'regenerate') await client.regenerateDataTable(id, {}, options);
      if (action === 'poll') await client.getDataTableJob(71, options);
      if (action === 'read') await client.getDataTable(id, {}, options);
      if (action === 'cancel') await client.cancelDataTableJob(71, options);
      const paths = {
        generate: '/api/v1/data-tables/generate',
        regenerate: `/api/v1/data-tables/${id}/regenerate`,
        poll: '/api/v1/data-tables/jobs/71',
        read: `/api/v1/data-tables/${id}`,
        cancel: '/api/v1/data-tables/jobs/71',
      };
      expect(transport.request).toHaveBeenCalledOnce();
      expect(transport.request.mock.calls[0][0]).toMatchObject({
        path: paths[action],
        method: action === 'cancel' ? 'DELETE' : ['poll', 'read'].includes(action) ? 'GET' : 'POST',
        abortSignal: controller.signal,
        servicePromptConfig: { ...scope.config, expectedUserId: 42 },
        headers: { 'X-TLDW-Expected-User-ID': '42' },
      });
    }
  );
});
