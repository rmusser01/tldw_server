import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// We need to control document.cookie to simulate different sessions.
// The module reads document.cookie at call time, so we can change it between calls.

let scopedStorage: typeof import('../scoped-storage');

describe('scoped-storage', () => {
  const originalCookie = Object.getOwnPropertyDescriptor(document, 'cookie');
  let cookieJar: string;

  beforeEach(() => {
    localStorage.clear();
    cookieJar = 'admin_session=abc12345xyz';
    // Mock document.cookie getter to return our controlled value
    Object.defineProperty(document, 'cookie', {
      get: () => cookieJar,
      set: () => {},
      configurable: true,
    });
    vi.resetModules();
  });

  afterEach(() => {
    // Restore original cookie descriptor
    if (originalCookie) {
      Object.defineProperty(document, 'cookie', originalCookie);
    }
    localStorage.clear();
  });

  async function loadModule() {
    scopedStorage = await import('../scoped-storage');
    return scopedStorage;
  }

  it('getScopedItem returns null for missing keys', async () => {
    const { getScopedItem } = await loadModule();
    expect(getScopedItem('nonexistent')).toBeNull();
  });

  it('setScopedItem + getScopedItem round-trips correctly', async () => {
    const { getScopedItem, setScopedItem } = await loadModule();
    setScopedItem('testKey', 'hello world');
    expect(getScopedItem('testKey')).toBe('hello world');
  });

  it('scopes keys per session cookie value', async () => {
    const { getScopedItem, setScopedItem } = await loadModule();

    // Write with session abc12345xyz
    setScopedItem('orgId', '42');
    expect(getScopedItem('orgId')).toBe('42');

    // Switch to a different session
    cookieJar = 'admin_session=def98765uvw';
    expect(getScopedItem('orgId')).toBeNull();

    // Write with the new session
    setScopedItem('orgId', '99');
    expect(getScopedItem('orgId')).toBe('99');

    // Switch back to original session -- old value still there
    cookieJar = 'admin_session=abc12345xyz';
    expect(getScopedItem('orgId')).toBe('42');
  });

  it('clearScopedStorage removes all keys for current session only', async () => {
    const { getScopedItem, setScopedItem, clearScopedStorage } = await loadModule();

    // Write keys under session A
    setScopedItem('key1', 'a');
    setScopedItem('key2', 'b');

    // Write keys under session B
    cookieJar = 'admin_session=othersess1234';
    setScopedItem('key1', 'x');

    // Clear session B
    clearScopedStorage();
    expect(getScopedItem('key1')).toBeNull();

    // Session A keys should be untouched
    cookieJar = 'admin_session=abc12345xyz';
    expect(getScopedItem('key1')).toBe('a');
    expect(getScopedItem('key2')).toBe('b');
  });

  it('migrates legacy unscoped key on first read', async () => {
    const { getScopedItem, _scopedKeyForTesting } = await loadModule();

    // Place a legacy (unscoped) key directly in localStorage
    localStorage.setItem('admin_org_saved_views', '["view1"]');

    // getScopedItem should find it, migrate it, and delete the legacy key
    const result = getScopedItem('admin_org_saved_views');
    expect(result).toBe('["view1"]');

    // Legacy key should be gone
    expect(localStorage.getItem('admin_org_saved_views')).toBeNull();

    // Scoped key should now exist
    const scoped = _scopedKeyForTesting('admin_org_saved_views');
    expect(localStorage.getItem(scoped)).toBe('["view1"]');
  });

  it('does not migrate when scoped key already exists', async () => {
    const { getScopedItem, _scopedKeyForTesting } = await loadModule();
    const scoped = _scopedKeyForTesting('myKey');

    // Pre-populate both scoped and legacy keys
    localStorage.setItem(scoped, 'scoped-value');
    localStorage.setItem('myKey', 'legacy-value');

    // Should return the scoped value, not the legacy one
    expect(getScopedItem('myKey')).toBe('scoped-value');

    // Legacy key should remain (not touched because scoped key was found)
    expect(localStorage.getItem('myKey')).toBe('legacy-value');
  });

  it('removeScopedItem removes only the scoped key', async () => {
    const { getScopedItem, setScopedItem, removeScopedItem } = await loadModule();
    setScopedItem('toRemove', 'value');
    expect(getScopedItem('toRemove')).toBe('value');

    removeScopedItem('toRemove');
    expect(getScopedItem('toRemove')).toBeNull();
  });

  it('falls back to "default" scope when no session cookie is present', async () => {
    cookieJar = '';
    const { getScopedItem, setScopedItem } = await loadModule();

    setScopedItem('noSession', 'val');
    expect(getScopedItem('noSession')).toBe('val');

    // Verify the key uses the "default" scope
    const allKeys: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      allKeys.push(localStorage.key(i)!);
    }
    expect(allKeys.some((k) => k.includes('default'))).toBe(true);
  });
});
