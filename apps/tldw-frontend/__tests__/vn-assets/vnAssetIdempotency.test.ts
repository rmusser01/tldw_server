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

  describe('receipt failure diagnosis', () => {
    const malformedWarning = '[vn-assets] Ignoring malformed pending generation receipt JSON.';
    const invalidWarning = '[vn-assets] Ignoring invalid pending generation receipt.';
    const readWarning = '[vn-assets] Could not read pending generation receipt: session storage unavailable.';
    const removeWarning = '[vn-assets] Could not remove invalid pending generation receipt: session storage unavailable.';

    beforeEach(() => vi.spyOn(console, 'warn').mockImplementation(() => {}));

    it.each([
      { label: 'truncated JSON', raw: '{"kind":"retry","key":"private-request-key","slotId":' },
      { label: 'empty stored JSON', raw: '' },
    ])('diagnoses and removes $label only for the selected owner and pack', ({ raw }) => {
      const otherOwnerKey = 'vn-assets:pending-generation:v1:2:7';
      const otherPackKey = 'vn-assets:pending-generation:v1:1:8';
      const otherReceipt = JSON.stringify({ kind: 'start', key: 'other-request-key' });
      window.sessionStorage.setItem(storageKey, raw);
      window.sessionStorage.setItem(otherOwnerKey, otherReceipt);
      window.sessionStorage.setItem(otherPackKey, otherReceipt);

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(window.sessionStorage.getItem(storageKey)).toBeNull();
      expect(window.sessionStorage.getItem(otherOwnerKey)).toBe(otherReceipt);
      expect(window.sessionStorage.getItem(otherPackKey)).toBe(otherReceipt);
      expect(vi.mocked(console.warn).mock.calls).toEqual([[malformedWarning]]);

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(vi.mocked(console.warn).mock.calls).toEqual([[malformedWarning]]);
    });

    it('distinguishes an invalid receipt from malformed JSON without logging the payload', () => {
      window.sessionStorage.setItem(storageKey, JSON.stringify({ kind: 'unknown', key: 'private-request-key' }));

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(window.sessionStorage.getItem(storageKey)).toBeNull();
      expect(vi.mocked(console.warn).mock.calls).toEqual([[invalidWarning]]);
    });

    it.each(['getter', 'getItem'] as const)('diagnoses a storage %s failure without throwing or removing a receipt', (boundary) => {
      const storage = window.sessionStorage;
      const storagePrototype = Object.getPrototypeOf(storage) as Storage;
      const raw = JSON.stringify({ kind: 'retry', slotId: 12, key: 'private-request-key' });
      storage.setItem(storageKey, raw);
      const remove = vi.spyOn(storagePrototype, 'removeItem');
      const failure = new DOMException(`${storageKey}: ${raw}`, 'SecurityError');
      const read = boundary === 'getter'
        ? vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => { throw failure; })
        : vi.spyOn(storagePrototype, 'getItem').mockImplementation(() => { throw failure; });

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(remove).not.toHaveBeenCalled();
      expect(vi.mocked(console.warn).mock.calls).toEqual([[readWarning]]);
      read.mockRestore();
      expect(storage.getItem(storageKey)).toBe(raw);
    });

    it.each([
      { label: 'malformed', raw: '{"key":"private-request-key"', warning: malformedWarning },
      { label: 'invalid', raw: '{"kind":"unknown","key":"private-request-key"}', warning: invalidWarning },
    ])('diagnoses a $label receipt and a removal failure separately without throwing', ({ raw, warning }) => {
      const storage = window.sessionStorage;
      storage.setItem(storageKey, raw);
      vi.spyOn(Object.getPrototypeOf(storage) as Storage, 'removeItem').mockImplementation(() => {
        throw new DOMException(`${storageKey}: ${raw}`, 'SecurityError');
      });

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(window.sessionStorage.getItem(storageKey)).toBe(raw);
      expect(vi.mocked(console.warn).mock.calls).toEqual([[warning], [removeWarning]]);
    });

    it('removes malformed JSON using the acquired storage handle without rereading its getter', () => {
      const storage = window.sessionStorage;
      storage.setItem(storageKey, '{"key":"private-request-key"');
      const getter = vi.spyOn(window, 'sessionStorage', 'get').mockReturnValueOnce(storage).mockImplementation(() => {
        throw new DOMException(storageKey, 'SecurityError');
      });

      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(storage.getItem(storageKey)).toBeNull();
      expect(getter).toHaveBeenCalledTimes(1);
      expect(vi.mocked(console.warn).mock.calls).toEqual([[malformedWarning]]);
    });

    it.each(['start', 'retry'] as const)('does not diagnose or rewrite a valid %s receipt', (kind) => {
      const pending = { kind, ...(kind === 'retry' ? { slotId: 12 } : {}), key: 'private-request-key' };
      const raw = JSON.stringify(pending);
      window.sessionStorage.setItem(storageKey, raw);

      expect(readPendingVNAssetGeneration(1, 7)).toEqual(pending);
      expect(window.sessionStorage.getItem(storageKey)).toBe(raw);
      expect(console.warn).not.toHaveBeenCalled();
    });

    it('does not diagnose an absent receipt', () => {
      expect(readPendingVNAssetGeneration(1, 7)).toBeNull();
      expect(console.warn).not.toHaveBeenCalled();
    });

    it.each([
      { ownerUserId: undefined, packId: 7 },
      { ownerUserId: 1, packId: Number.NaN },
    ])('does not access storage or diagnose an invalid owner/pack scope $ownerUserId/$packId', ({ ownerUserId, packId }) => {
      const getter = vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
        throw new DOMException('Private storage failure', 'SecurityError');
      });

      expect(readPendingVNAssetGeneration(ownerUserId, packId)).toBeNull();
      expect(getter).not.toHaveBeenCalled();
      expect(console.warn).not.toHaveBeenCalled();
    });
  });

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
