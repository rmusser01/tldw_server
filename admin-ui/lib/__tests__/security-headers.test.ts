import { describe, it, expect } from 'vitest';

const nextConfig = require('../../next.config.js');

describe('next.config.js security', () => {
  it('enables standalone output for Docker', () => {
    expect(nextConfig.output).toBe('standalone');
  });

  it('disables X-Powered-By header', () => {
    expect(nextConfig.poweredByHeader).toBe(false);
  });

  it('defines security headers for all routes', async () => {
    expect(typeof nextConfig.headers).toBe('function');
    const headers = await nextConfig.headers();
    expect(headers).toHaveLength(1);
    expect(headers[0].source).toBe('/:path*');

    const headerMap = Object.fromEntries(
      headers[0].headers.map((h: { key: string; value: string }) => [h.key, h.value])
    );

    expect(headerMap['X-Frame-Options']).toBe('DENY');
    expect(headerMap['X-Content-Type-Options']).toBe('nosniff');
    expect(headerMap['Referrer-Policy']).toBe('strict-origin-when-cross-origin');
    expect(headerMap['Content-Security-Policy']).toContain("frame-ancestors 'none'");
    expect(headerMap['Strict-Transport-Security']).toContain('max-age=');
    expect(headerMap['Permissions-Policy']).toContain('camera=()');
  });
});
