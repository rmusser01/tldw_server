export function createVNIdempotencyKey(prefix: string): string {
  const crypto = globalThis.crypto;
  if (typeof crypto?.randomUUID === 'function') {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  if (typeof crypto?.getRandomValues !== 'function') {
    throw new Error('Secure random generation is unavailable');
  }
  // Self-hosted HTTP can use getRandomValues even when randomUUID is unavailable.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${prefix}-${hex}`;
}
