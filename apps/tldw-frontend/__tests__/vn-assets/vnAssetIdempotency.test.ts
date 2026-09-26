import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearPendingVNAssetGeneration,
  createVNAssetIdempotencyKey,
  readPendingVNAssetGeneration,
  writePendingVNAssetGeneration,
} from '@web/lib/vnAssetIdempotency';

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

describe('VN asset pending generation storage', () => {
  const storageKey = 'vn-assets:pending-generation:v1:1:7';

  beforeEach(() => window.sessionStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  describe.each(['start', 'retry'] as const)('persisted %s key length', (kind) => {
    it.each([
      { label: 'empty', key: '' },
      { label: '161-character ASCII', key: 'k'.repeat(161) },
      { label: '161-code-point non-BMP', key: '\u{1F600}'.repeat(161) },
    ])('rejects and removes a $label key only for its owner and pack', ({ key }) => {
      const pending = { kind, ...(kind === 'retry' ? { slotId: 12 } : {}), key };
      const otherOwnerKey = 'vn-assets:pending-generation:v1:2:7';
      const otherPackKey = 'vn-assets:pending-generation:v1:1:8';
      const otherOwner = JSON.stringify({ kind: 'start', key: 'other-owner-key' });
      const otherPack = JSON.stringify({ kind: 'retry', slotId: 13, key: 'other-pack-key' });
      window.sessionStorage.setItem(storageKey, JSON.stringify(pending));
      window.sessionStorage.setItem(otherOwnerKey, otherOwner);
      window.sessionStorage.setItem(otherPackKey, otherPack);

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(window.sessionStorage.getItem(storageKey)).toBeNull();
      expect(window.sessionStorage.getItem(otherOwnerKey)).toBe(otherOwner);
      expect(window.sessionStorage.getItem(otherPackKey)).toBe(otherPack);
    });

    it.each([
      { label: 'one-character ASCII', key: 'k' },
      { label: '160-character ASCII', key: 'k'.repeat(160) },
      { label: 'one-code-point non-BMP', key: '\u{1F600}' },
      { label: '160-code-point non-BMP', key: '\u{1F600}'.repeat(160) },
      { label: '160-code-point mixed', key: `${'k'.repeat(159)}\u{1F600}` },
    ])('preserves a $label key and its owner/pack-scoped receipt', ({ key }) => {
      const pending = { kind, ...(kind === 'retry' ? { slotId: 12 } : {}), key };
      const raw = JSON.stringify(pending);
      window.sessionStorage.setItem(storageKey, raw);

      expect(readPendingVNAssetGeneration(2, 7)).toBeNull();
      expect(readPendingVNAssetGeneration(1, 8)).toBeNull();
      expect(readPendingVNAssetGeneration(1, 7)).toEqual(pending);
      expect(window.sessionStorage.getItem(storageKey)).toBe(raw);
    });
  });

  it.each([
    { label: 'zero', slotId: 0 },
    { label: 'negative', slotId: -1 },
    { label: 'negative slot', slotId: -12 },
    { label: 'unsafe positive', slotId: Number.MAX_SAFE_INTEGER + 1 },
    { label: 'unsafe negative', slotId: Number.MIN_SAFE_INTEGER - 1 },
    { label: 'fractional', slotId: 1.5 },
    { label: 'numeric string', slotId: '12' },
    { label: 'null', slotId: null },
    { label: 'missing', slotId: undefined },
    { label: 'boolean', slotId: true },
    { label: 'object', slotId: {} },
    { label: 'array', slotId: [] },
  ])('rejects and removes a persisted retry with a $label slot ID', ({ slotId }) => {
    window.sessionStorage.setItem(storageKey, JSON.stringify({ kind: 'retry', slotId, key: 'retry-key' }));

    expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
    expect(window.sessionStorage.getItem(storageKey)).toBeNull();
  });

  it.each([1, 12, Number.MAX_SAFE_INTEGER])('preserves a positive safe retry slot ID %s and its key', (slotId) => {
    writePendingVNAssetGeneration(1, 7, { kind: 'retry', slotId, key: 'retry-key' });

    expect(readPendingVNAssetGeneration(1, 7)).toEqual({ kind: 'retry', slotId, key: 'retry-key' });
  });

  it('preserves a start receipt without a retry slot ID', () => {
    writePendingVNAssetGeneration(1, 7, { kind: 'start', key: 'start-key' });

    expect(readPendingVNAssetGeneration(1, 7)).toEqual({ kind: 'start', key: 'start-key' });
  });

  it('keeps retry receipts scoped to their owner and pack', () => {
    writePendingVNAssetGeneration(1, 7, { kind: 'retry', slotId: 12, key: 'retry-key' });

    expect(readPendingVNAssetGeneration(2, 7)).toBeNull();
    expect(readPendingVNAssetGeneration(1, 8)).toBeNull();
    expect(readPendingVNAssetGeneration(1, 7)).toEqual({ kind: 'retry', slotId: 12, key: 'retry-key' });
  });

  it('clears a retry receipt only for the matching owner, pack, and key', () => {
    writePendingVNAssetGeneration(1, 7, { kind: 'retry', slotId: 12, key: 'retry-key' });

    clearPendingVNAssetGeneration(2, 7, 'retry-key');
    clearPendingVNAssetGeneration(1, 8, 'retry-key');
    clearPendingVNAssetGeneration(1, 7, 'other-key');
    expect(readPendingVNAssetGeneration(1, 7)).toEqual({ kind: 'retry', slotId: 12, key: 'retry-key' });

    clearPendingVNAssetGeneration(1, 7, 'retry-key');
    expect(window.sessionStorage.getItem(storageKey)).toBeNull();
  });

  it('does not throw when session storage is disabled', () => {
    vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('Storage disabled', 'SecurityError');
    });

    expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
    expect(() => writePendingVNAssetGeneration(1, 7, { kind: 'retry', slotId: 12, key: 'retry-key' })).not.toThrow();
    expect(() => clearPendingVNAssetGeneration(1, 7, 'retry-key')).not.toThrow();
  });
});
