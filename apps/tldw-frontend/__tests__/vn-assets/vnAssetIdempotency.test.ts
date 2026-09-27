import { afterEach, describe, expect, it, vi } from 'vitest';
import { createVNAssetIdempotencyKey } from '@web/lib/vnAssetIdempotency';

describe('VN asset idempotency keys', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('uses the browser UUID when available', () => {
    vi.stubGlobal('crypto', { randomUUID: () => 'test-uuid' });
    expect(createVNAssetIdempotencyKey('vn-generation')).toBe('vn-generation-test-uuid');
  });

  it('supports self-hosted HTTP without randomUUID', () => {
    vi.stubGlobal('crypto', {});
    expect(createVNAssetIdempotencyKey('vn-generation')).toMatch(/^vn-generation-\d+-[a-z0-9]+$/);
  });
});
