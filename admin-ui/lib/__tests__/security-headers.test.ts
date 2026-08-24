import { spawnSync } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { describe, it, expect } from 'vitest';

import nextConfig from '../../next.config.mjs';

describe('next.config.mjs security', () => {
  it('loads the optional bundle analyzer only when analysis is enabled', async () => {
    const tempDirectory = await mkdtemp(join(tmpdir(), 'tldw-admin-next-config-'));
    const analyzerUrl = pathToFileURL(join(tempDirectory, 'bundle-analyzer.mjs')).href;
    const loaderUrl = pathToFileURL(join(tempDirectory, 'loader.mjs')).href;
    const configUrl = pathToFileURL(resolve(process.cwd(), 'next.config.mjs')).href;

    try {
      await writeFile(
        new URL(analyzerUrl),
        'export default ({ enabled }) => (config) => ({ ...config, testAnalyzerEnabled: enabled });\n'
      );
      await writeFile(
        new URL(loaderUrl),
        `const analyzerUrl = ${JSON.stringify(analyzerUrl)};
export async function resolve(specifier, context, nextResolve) {
  if (specifier === '@next/bundle-analyzer') {
    if (process.env.TEST_ANALYZER_MODE === 'block') {
      throw new Error('bundle analyzer resolved while disabled');
    }
    return { url: analyzerUrl, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
`
      );

      const runConfig = (analyze: string, mode: string, script: string) =>
        spawnSync(
          process.execPath,
          ['--experimental-loader', loaderUrl, '--input-type=module', '--eval', script],
          {
            cwd: process.cwd(),
            encoding: 'utf8',
            env: {
              ...process.env,
              ANALYZE: analyze,
              NEXT_PUBLIC_SENTRY_DSN: '',
              TEST_ANALYZER_MODE: mode,
            },
          }
        );

      const disabled = runConfig('false', 'block', `await import(${JSON.stringify(configUrl)});`);
      expect(disabled.status, disabled.stderr).toBe(0);

      const enabled = runConfig(
        'true',
        'stub',
        `const config = (await import(${JSON.stringify(configUrl)})).default;
if (config.testAnalyzerEnabled !== true) throw new Error('analyzer wrapper was not applied');`
      );
      expect(enabled.status, enabled.stderr).toBe(0);
    } finally {
      await rm(tempDirectory, { recursive: true, force: true });
    }
  });

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
