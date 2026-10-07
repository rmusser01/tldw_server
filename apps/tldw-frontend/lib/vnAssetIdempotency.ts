import { createVNCommandScope, sameVNCommandScope, type VNCommandScope } from '@web/lib/vnGenerationRecovery';

export function createVNAssetIdempotencyKey(prefix: string): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${prefix}-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

export interface PendingVNAssetGeneration {
  kind: 'start' | 'retry';
  slotId?: number;
  key: string;
}

function generationStorageKey(scope: VNCommandScope | null, packId: number): string | null {
  if (!scope || !Number.isSafeInteger(packId) || packId <= 0 || typeof window === 'undefined') return null;
  try {
    if (!sameVNCommandScope(scope, createVNCommandScope(scope.server, Number(scope.principal)))) return null;
    return `vn-assets:pending-generation:v2:${encodeURIComponent(scope.server)}:${encodeURIComponent(scope.principal)}:${packId}`;
  } catch {
    return null;
  }
}

/** Read only a canonical, verified server/account receipt; unscoped v1 bytes stay inert. */
export function readPendingVNAssetGeneration(
  scope: VNCommandScope | null, packId: number
): PendingVNAssetGeneration | null {
  const storageKey = generationStorageKey(scope, packId);
  if (!storageKey || typeof window === 'undefined') return null;
  let storage: Storage;
  let raw: string | null;
  try {
    storage = window.sessionStorage;
    raw = storage.getItem(storageKey);
  } catch {
    console.warn('[vn-assets] Could not read pending generation receipt: session storage unavailable.');
    return null;
  }
  if (raw === null) return null;
  try {
    const value: unknown = JSON.parse(raw);
    if (typeof value === 'object' && value !== null && 'kind' in value && 'key' in value) {
      const pending = value as Record<string, unknown>;
      // API string limits count Unicode code points, not UTF-16 code units.
      if (typeof pending.key === 'string' && pending.key.length > 0 && Array.from(pending.key).length <= 160 && (
        pending.kind === 'start' || (
          pending.kind === 'retry' && typeof pending.slotId === 'number' &&
          Number.isSafeInteger(pending.slotId) && pending.slotId > 0
        )
      )) return pending as unknown as PendingVNAssetGeneration;
    }
    console.warn('[vn-assets] Ignoring invalid pending generation receipt.');
  } catch {
    console.warn('[vn-assets] Ignoring malformed pending generation receipt JSON.');
  }
  try {
    storage.removeItem(storageKey);
  } catch {
    console.warn('[vn-assets] Could not remove invalid pending generation receipt: session storage unavailable.');
  }
  return null;
}

export function writePendingVNAssetGeneration(
  scope: VNCommandScope | null, packId: number, pending: PendingVNAssetGeneration
): void {
  const storageKey = generationStorageKey(scope, packId);
  if (!storageKey || typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(storageKey, JSON.stringify(pending));
  } catch {
    // Same-mount retries still use the in-memory key when tab storage is unavailable.
  }
}

export function clearPendingVNAssetGeneration(
  scope: VNCommandScope | null, packId: number, key: string
): void {
  const storageKey = generationStorageKey(scope, packId);
  if (!storageKey || typeof window === 'undefined') return;
  try {
    if (readPendingVNAssetGeneration(scope, packId)?.key === key) {
      window.sessionStorage.removeItem(storageKey);
    }
  } catch {
    // Storage can be disabled independently of the API request.
  }
}
