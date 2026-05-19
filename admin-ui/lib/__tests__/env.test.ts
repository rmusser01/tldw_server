import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('env validation', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe('validateEnv (build-time)', () => {
    it('succeeds when NEXT_PUBLIC_API_URL is set', async () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000';
      const { validateEnv } = await import('../env');
      expect(() => validateEnv()).not.toThrow();
    });

    it('throws when NEXT_PUBLIC_API_URL is missing', async () => {
      delete process.env.NEXT_PUBLIC_API_URL;
      const { validateEnv } = await import('../env');
      expect(() => validateEnv()).toThrow(/NEXT_PUBLIC_API_URL/);
    });

    it('throws when NEXT_PUBLIC_API_URL is not a valid URL', async () => {
      process.env.NEXT_PUBLIC_API_URL = 'not-a-url';
      const { validateEnv } = await import('../env');
      expect(() => validateEnv()).toThrow();
    });

    it('defaults NEXT_PUBLIC_API_VERSION to v1', async () => {
      process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000';
      delete process.env.NEXT_PUBLIC_API_VERSION;
      const { validateEnv } = await import('../env');
      const env = validateEnv();
      expect(env.NEXT_PUBLIC_API_VERSION).toBe('v1');
    });
  });

  describe('validateRuntimeEnv (server-side)', () => {
    it('succeeds when JWT_SECRET_KEY is set', async () => {
      process.env.JWT_SECRET_KEY = 'test-secret-key-value';
      const { validateRuntimeEnv } = await import('../env');
      expect(() => validateRuntimeEnv()).not.toThrow();
    });

    it('throws when JWT_SECRET_KEY is missing', async () => {
      delete process.env.JWT_SECRET_KEY;
      const { validateRuntimeEnv } = await import('../env');
      expect(() => validateRuntimeEnv()).toThrow(/JWT_SECRET_KEY/);
    });

    it('throws when JWT_SECRET_KEY is empty', async () => {
      process.env.JWT_SECRET_KEY = '';
      const { validateRuntimeEnv } = await import('../env');
      expect(() => validateRuntimeEnv()).toThrow(/JWT_SECRET_KEY/);
    });

    it('defaults JWT_ALGORITHM to HS256', async () => {
      process.env.JWT_SECRET_KEY = 'test-secret-key-value';
      delete process.env.JWT_ALGORITHM;
      const { validateRuntimeEnv } = await import('../env');
      const env = validateRuntimeEnv();
      expect(env.JWT_ALGORITHM).toBe('HS256');
    });

    it('allows JWT_SECONDARY_SECRET to be absent', async () => {
      process.env.JWT_SECRET_KEY = 'test-secret-key-value';
      delete process.env.JWT_SECONDARY_SECRET;
      const { validateRuntimeEnv } = await import('../env');
      const env = validateRuntimeEnv();
      expect(env.JWT_SECONDARY_SECRET).toBeUndefined();
    });
  });
});
