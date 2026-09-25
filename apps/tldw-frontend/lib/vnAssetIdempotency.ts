export function createVNAssetIdempotencyKey(prefix: string): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${prefix}-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

export interface PendingVNAssetGeneration {
  kind: 'start' | 'retry';
  slotId?: number;
  key: string;
}

function generationStorageKey(ownerUserId: number | undefined, packId: number): string | null {
  if (!Number.isSafeInteger(ownerUserId) || !Number.isSafeInteger(packId)) return null;
  return `vn-assets:pending-generation:v1:${ownerUserId}:${packId}`;
}

export function readPendingVNAssetGeneration(
  ownerUserId: number | undefined, packId: number
): PendingVNAssetGeneration | null {
  const storageKey = generationStorageKey(ownerUserId, packId);
  if (!storageKey || typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(storageKey);
    if (!raw) return null;
    const value: unknown = JSON.parse(raw);
    if (typeof value === 'object' && value !== null && 'kind' in value && 'key' in value) {
      const pending = value as Record<string, unknown>;
      if (typeof pending.key === 'string' && pending.key.length > 0 && (
        pending.kind === 'start' || (pending.kind === 'retry' && Number.isSafeInteger(pending.slotId))
      )) return pending as unknown as PendingVNAssetGeneration;
    }
    window.sessionStorage.removeItem(storageKey);
  } catch {
    return null;
  }
  return null;
}

export function writePendingVNAssetGeneration(
  ownerUserId: number | undefined, packId: number, pending: PendingVNAssetGeneration
): void {
  const storageKey = generationStorageKey(ownerUserId, packId);
  if (!storageKey || typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(storageKey, JSON.stringify(pending));
  } catch {
    // Same-mount retries still use the in-memory key when tab storage is unavailable.
  }
}

export function clearPendingVNAssetGeneration(
  ownerUserId: number | undefined, packId: number, key: string
): void {
  const storageKey = generationStorageKey(ownerUserId, packId);
  if (!storageKey || typeof window === 'undefined') return;
  try {
    if (readPendingVNAssetGeneration(ownerUserId, packId)?.key === key) {
      window.sessionStorage.removeItem(storageKey);
    }
  } catch {
    // Storage can be disabled independently of the API request.
  }
}
