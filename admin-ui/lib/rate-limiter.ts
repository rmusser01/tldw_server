const MAX_REQUESTS = 10;
const WINDOW_MS = 60_000;
const MAX_ENTRIES = 10_000;

const store = new Map<string, number[]>();

function pruneEntry(timestamps: number[], now: number): number[] {
  const cutoff = now - WINDOW_MS;
  return timestamps.filter((t) => t > cutoff);
}

function pruneStore(): void {
  const now = Date.now();

  // Pass 1: remove fully expired entries (cheap wins)
  for (const [key, timestamps] of store) {
    if (pruneEntry(timestamps, now).length === 0) {
      store.delete(key);
    }
  }

  // Pass 2: enforce hard cap via LRU eviction — Map iterates in
  // insertion order and checkRateLimit re-inserts on every access,
  // so the first keys are the least-recently-used.
  let toEvict = store.size - MAX_ENTRIES;
  if (toEvict > 0) {
    for (const key of store.keys()) {
      if (toEvict-- <= 0) break;
      store.delete(key);
    }
  }
}

/**
 * Extract client IP with an explicit proxy trust model.
 * Only reads x-forwarded-for / x-real-ip when TRUST_PROXY_HEADERS=true.
 * Without a trusted proxy, all clients share the 'unknown' bucket —
 * deploy behind a reverse proxy (nginx, traefik, etc.) for per-IP limiting.
 */
export function extractClientIp(headers: {
  get(name: string): string | null;
}): string {
  if (process.env.TRUST_PROXY_HEADERS === 'true') {
    const forwarded = headers.get('x-forwarded-for')?.split(',')[0]?.trim();
    if (forwarded) return forwarded;
    const realIp = headers.get('x-real-ip');
    if (realIp) return realIp;
  }
  return 'unknown';
}

export function checkRateLimit(ip: string): {
  allowed: boolean;
  retryAfterSeconds?: number;
} {
  const now = Date.now();

  if (store.size > MAX_ENTRIES) pruneStore();

  const timestamps = pruneEntry(store.get(ip) ?? [], now);

  if (timestamps.length >= MAX_REQUESTS) {
    const oldestInWindow = timestamps[0];
    const retryAfterMs = oldestInWindow + WINDOW_MS - now;
    return {
      allowed: false,
      retryAfterSeconds: Math.ceil(Math.max(retryAfterMs, 1000) / 1000),
    };
  }

  timestamps.push(now);
  // Re-insert to move key to end of Map (maintains LRU order)
  store.delete(ip);
  store.set(ip, timestamps);
  return { allowed: true };
}
