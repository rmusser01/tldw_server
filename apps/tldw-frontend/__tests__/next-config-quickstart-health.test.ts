import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';

let loadId = 0;
const loadConfig = async (mode: string, internalOrigin: string) => {
  vi.stubEnv('NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE', mode);
  vi.stubEnv('TLDW_INTERNAL_API_ORIGIN', internalOrigin);
  vi.stubEnv('NEXT_PUBLIC_API_URL', mode === 'quickstart' ? '' : 'https://public.example.test');
  const url = pathToFileURL(path.resolve(__dirname, '../next.config.mjs'));
  url.searchParams.set('health-routing-test', String(++loadId));
  return (await import(/* @vite-ignore */ url.href)).default;
};

afterEach(() => vi.unstubAllEnvs());

describe('quickstart public health routing', () => {
  it.each(['http://app:8000', 'http://app:8000/'])(
    'forwards readiness to the public backend endpoint for %s',
    async (internalOrigin) => {
      const config = await loadConfig('quickstart', internalOrigin);
      const rewrites = await config.rewrites();
      expect(rewrites).toContainEqual({
        source: '/health',
        destination: 'http://app:8000/health',
      });
      expect(
        rewrites.filter(({ source }: { source: string }) => source.startsWith('/api'))
      ).toEqual([
        { source: '/api/v1/media', destination: 'http://app:8000/api/v1/media/' },
        { source: '/api/:path*/', destination: 'http://app:8000/api/:path*/' },
        { source: '/api/:path*', destination: 'http://app:8000/api/:path*' },
      ]);
    }
  );

  it.each(['', 'http://app:8000'])(
    'does not proxy advanced mode (internal origin: %s)',
    async (origin) => {
      const config = await loadConfig('advanced', origin);
      expect(await config.rewrites()).toEqual([]);
    }
  );

  it('still rejects quickstart without an internal backend', async () => {
    await expect(loadConfig('quickstart', '')).rejects.toThrow(
      'quickstart mode requires TLDW_INTERNAL_API_ORIGIN'
    );
  });
});
