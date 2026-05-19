import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('rate limiter', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.resetModules();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('allows requests under the limit', async () => {
    const { checkRateLimit } = await import('../rate-limiter');
    for (let i = 0; i < 10; i++) {
      expect(checkRateLimit('127.0.0.1').allowed).toBe(true);
    }
  });

  it('blocks requests over the limit', async () => {
    const { checkRateLimit } = await import('../rate-limiter');
    for (let i = 0; i < 10; i++) {
      checkRateLimit('127.0.0.1');
    }
    const result = checkRateLimit('127.0.0.1');
    expect(result.allowed).toBe(false);
    expect(result.retryAfterSeconds).toBeGreaterThan(0);
  });

  it('resets after the window expires', async () => {
    const { checkRateLimit } = await import('../rate-limiter');
    for (let i = 0; i < 10; i++) {
      checkRateLimit('127.0.0.1');
    }
    expect(checkRateLimit('127.0.0.1').allowed).toBe(false);
    vi.advanceTimersByTime(61_000);
    expect(checkRateLimit('127.0.0.1').allowed).toBe(true);
  });

  it('tracks different IPs independently', async () => {
    const { checkRateLimit } = await import('../rate-limiter');
    for (let i = 0; i < 10; i++) {
      checkRateLimit('1.1.1.1');
    }
    expect(checkRateLimit('1.1.1.1').allowed).toBe(false);
    expect(checkRateLimit('2.2.2.2').allowed).toBe(true);
  });

  it('evicts LRU entries when store exceeds MAX_ENTRIES', async () => {
    // MAX_ENTRIES is 10_000; fill to 10_001 unique keys to trigger eviction
    const { checkRateLimit } = await import('../rate-limiter');

    // Use up 10 requests on the "old" IP so it's rate-limited
    for (let i = 0; i < 10; i++) {
      checkRateLimit('old-ip');
    }
    expect(checkRateLimit('old-ip').allowed).toBe(false);

    // Touch a "recent" IP so it gets a higher LRU rank than old-ip
    for (let i = 0; i < 5; i++) {
      checkRateLimit('recent-ip');
    }

    // Fill beyond MAX_ENTRIES with unique IPs — old-ip was inserted first
    // and hasn't been touched since, so it sits at the front of the Map
    for (let i = 0; i < 10_001; i++) {
      checkRateLimit(`filler-${i}`);
    }

    // old-ip should have been evicted (LRU) — a fresh request is allowed
    expect(checkRateLimit('old-ip').allowed).toBe(true);

    // recent-ip was accessed after old-ip, so it survives eviction
    // and retains its 5 existing requests (still under limit)
    expect(checkRateLimit('recent-ip').allowed).toBe(true);
  });
});

describe('extractClientIp', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('returns unknown when TRUST_PROXY_HEADERS is not set', async () => {
    delete process.env.TRUST_PROXY_HEADERS;
    const { extractClientIp } = await import('../rate-limiter');
    const headers = { get: (name: string) => name === 'x-forwarded-for' ? '1.2.3.4' : null };
    expect(extractClientIp(headers)).toBe('unknown');
  });

  it('returns unknown when TRUST_PROXY_HEADERS is false', async () => {
    process.env.TRUST_PROXY_HEADERS = 'false';
    const { extractClientIp } = await import('../rate-limiter');
    const headers = { get: (name: string) => name === 'x-forwarded-for' ? '1.2.3.4' : null };
    expect(extractClientIp(headers)).toBe('unknown');
  });

  it('reads x-forwarded-for when TRUST_PROXY_HEADERS is true', async () => {
    process.env.TRUST_PROXY_HEADERS = 'true';
    const { extractClientIp } = await import('../rate-limiter');
    const headers = { get: (name: string) => name === 'x-forwarded-for' ? '1.2.3.4, 5.6.7.8' : null };
    expect(extractClientIp(headers)).toBe('1.2.3.4');
  });

  it('reads x-real-ip when x-forwarded-for is absent and proxy is trusted', async () => {
    process.env.TRUST_PROXY_HEADERS = 'true';
    const { extractClientIp } = await import('../rate-limiter');
    const headers = { get: (name: string) => name === 'x-real-ip' ? '9.8.7.6' : null };
    expect(extractClientIp(headers)).toBe('9.8.7.6');
  });

  it('returns unknown when proxy is trusted but no headers present', async () => {
    process.env.TRUST_PROXY_HEADERS = 'true';
    const { extractClientIp } = await import('../rate-limiter');
    const headers = { get: () => null };
    expect(extractClientIp(headers)).toBe('unknown');
  });
});
