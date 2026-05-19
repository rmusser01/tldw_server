import { describe, it, expect } from 'vitest';

describe('auth cache invalidation', () => {
  it('exports invalidateAuthCache function from middleware', async () => {
    const mod = await import('../../middleware');
    expect(typeof mod.invalidateAuthCache).toBe('function');
  });

  it('invalidateAuthCache accepts a token string without throwing', async () => {
    const { invalidateAuthCache } = await import('../../middleware');
    await expect(invalidateAuthCache('some-test-token')).resolves.not.toThrow();
  });
});
