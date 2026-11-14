import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createVNPlayIdempotencyKey,
  getVNPlayErrorInfo,
  isRecoverableVNPlayConflict,
} from '@web/components/vn-play/runtime';

describe('VN Play runtime helpers', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('creates prefixed idempotency keys with random UUIDs when available', () => {
    vi.stubGlobal('crypto', { randomUUID: () => 'uuid-1' });

    expect(createVNPlayIdempotencyKey('retry')).toBe('retry-uuid-1');
  });

  it('uses cryptographic random bytes when randomUUID is unavailable', () => {
    vi.stubGlobal('crypto', { getRandomValues: (bytes: Uint8Array) => bytes.fill(0x02) });
    expect(createVNPlayIdempotencyKey('retry')).toBe(`retry-${'02'.repeat(16)}`);
  });

  it('extracts structured API error details without [object Object] fallback text', () => {
    const info = getVNPlayErrorInfo({
      status: 409,
      detail: {
        error_code: 'stale_scene_version',
        message: 'Scene version is stale',
      },
    });

    expect(info).toEqual({
      code: 'stale_scene_version',
      message: 'Scene version is stale',
      status: 409,
    });
    expect(info.message).not.toBe('[object Object]');
  });

  it('recognizes recoverable stale and in-progress turn conflicts by code', () => {
    expect(isRecoverableVNPlayConflict({ status: 400, error_code: 'turn_in_progress' })).toBe(true);
    expect(isRecoverableVNPlayConflict({ status: 400, detail: { code: 'stale_scene_version' } })).toBe(true);
    expect(isRecoverableVNPlayConflict(new Error('parse_failed'))).toBe(false);
  });
});
