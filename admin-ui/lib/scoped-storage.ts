/**
 * Scoped localStorage helper that prefixes keys with a session marker
 * to prevent data leakage between admin users sharing a browser.
 *
 * Theme preference is intentionally NOT scoped -- it is a device preference.
 */

const SESSION_COOKIE = 'admin_session';
const SCOPE_PREFIX = 'tldw_admin_';

function getSessionScope(): string {
  // Use a prefix derived from the session marker cookie as scope identifier.
  // Falls back to 'default' if no session (pre-login state).
  if (typeof document === 'undefined') return 'default';
  const match = document.cookie.match(new RegExp(`${SESSION_COOKIE}=([^;]+)`));
  return match?.[1] ? `s${match[1].slice(0, 8)}` : 'default';
}

function scopedKey(key: string): string {
  return `${SCOPE_PREFIX}${getSessionScope()}_${key}`;
}

export function getScopedItem(key: string): string | null {
  if (typeof localStorage === 'undefined') return null;
  // Try scoped key first, then fall back to legacy unscoped key (migration)
  const scoped = localStorage.getItem(scopedKey(key));
  if (scoped !== null) return scoped;

  // Migrate: read legacy key, write to scoped, delete legacy
  const legacy = localStorage.getItem(key);
  if (legacy !== null) {
    localStorage.setItem(scopedKey(key), legacy);
    localStorage.removeItem(key);
    return legacy;
  }
  return null;
}

export function setScopedItem(key: string, value: string): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(scopedKey(key), value);
}

export function removeScopedItem(key: string): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.removeItem(scopedKey(key));
}

/** Clear all scoped storage for the current session. Called on logout. */
export function clearScopedStorage(): void {
  if (typeof localStorage === 'undefined') return;
  const prefix = scopedKey('');
  const keysToRemove: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith(prefix)) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((k) => localStorage.removeItem(k));
}

// Exported for tests only
export { scopedKey as _scopedKeyForTesting };
