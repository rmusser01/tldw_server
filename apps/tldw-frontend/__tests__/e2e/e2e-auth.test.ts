import { describe, expect, it } from 'vitest';

import {
  DEFAULT_E2E_API_KEY,
  isLocalE2eServerUrl,
  resolveE2eApiKey,
  resolveExplicitE2eApiKey,
} from '../../e2e/utils/e2e-auth';

describe('E2E auth helpers', () => {
  it('prefers explicit API key environment values', () => {
    expect(resolveExplicitE2eApiKey({ TLDW_API_KEY: ' primary ' })).toBe('primary');
    expect(resolveExplicitE2eApiKey({ TLDW_E2E_API_KEY: ' e2e ' })).toBe('e2e');
    expect(resolveExplicitE2eApiKey({ SINGLE_USER_API_KEY: ' single ' })).toBe('single');
  });

  it('allows the placeholder key only for local E2E server URLs', () => {
    expect(resolveE2eApiKey({ serverUrl: 'http://127.0.0.1:8000', env: {} })).toBe(
      DEFAULT_E2E_API_KEY
    );
    expect(resolveE2eApiKey({ serverUrl: 'http://localhost:8000', env: {} })).toBe(
      DEFAULT_E2E_API_KEY
    );
    expect(resolveE2eApiKey({ serverUrl: 'http://[::1]:8000', env: {} })).toBe(
      DEFAULT_E2E_API_KEY
    );
  });

  it('detects localhost variants used by Playwright and dev servers', () => {
    expect(isLocalE2eServerUrl('http://localhost:8000')).toBe(true);
    expect(isLocalE2eServerUrl('http://app.localhost:8000')).toBe(true);
    expect(isLocalE2eServerUrl('http://127.0.0.1:8000')).toBe(true);
    expect(isLocalE2eServerUrl('http://[::1]:8000')).toBe(true);
    expect(isLocalE2eServerUrl('https://example.com')).toBe(false);
  });

  it('fails fast for remote E2E server URLs without an explicit key', () => {
    expect(() =>
      resolveE2eApiKey({ serverUrl: 'https://example.com', env: {} })
    ).toThrow(/Remote E2E server URL/);
  });
});
