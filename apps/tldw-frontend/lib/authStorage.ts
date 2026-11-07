import { isPlaceholderApiKey } from "@/utils/api-key";
import { createSafeStorage } from "@/utils/safe-storage";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import { isCookieSessionConfigInvalidated } from "@/services/tldw/runtime-auth-override";
import {
  applyRefreshRotation,
  MANUAL_SESSION_KEY,
  resolveManualSessionCredential,
  refreshSessionInvalidationKey,
  REFRESH_ROTATION_KEY
} from "@/services/tldw/single-user-credential";
import {
  COOKIE_SESSION_CONFIG_KEY,
  isCookieSessionBrowserTransport,
  isExactOriginCookieSessionConfig
} from "@/services/tldw/browser-networking";
import { AUTH_CREDENTIALS_CHANGED_EVENT } from "@web/lib/auth-events";

let runtimeApiKey: string | null = null;
let runtimeApiBearer: string | null = null;
let suppressEnvApiKeyForSession = false;

type StoredTldwConfig = Partial<TldwConfig>;

const normalizeValue = (value?: string | null): string | null => {
  const raw = (value ?? '').trim();
  return raw ? raw : null;
};

const normalizeApiKeyValue = (value?: string | null): string | null => {
  const normalized = normalizeValue(value);
  if (!normalized) return null;
  if (/\s/.test(normalized)) {
    console.warn('Runtime API key contains whitespace; ignoring value.');
    return null;
  }
  if (isPlaceholderApiKey(normalized)) return null;
  return normalized;
};

const normalizeBearerValue = (value?: string | null): string | null => {
  const normalized = normalizeValue(value);
  if (!normalized) return null;
  const stripped = normalized.replace(/^Bearer\s+/i, '').trim();
  if (!stripped) return null;
  if (/\s/.test(stripped)) {
    console.warn('Runtime API bearer contains whitespace; ignoring value.');
    return null;
  }
  return stripped;
};

export const setRuntimeApiKey = (value?: string | null): void => {
  runtimeApiKey = normalizeApiKeyValue(value);
};

export const setRuntimeApiBearer = (value?: string | null): void => {
  runtimeApiBearer = normalizeBearerValue(value);
};

export const getRuntimeApiKey = (): string | null => runtimeApiKey;
export const getRuntimeApiBearer = (): string | null => runtimeApiBearer;

export const setEnvApiKeySuppressedForSession = (suppressed: boolean): void => {
  suppressEnvApiKeyForSession = suppressed;
};

const readStoredTldwConfig = (
  key = "tldwConfig"
): StoredTldwConfig | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    return parsed as StoredTldwConfig;
  } catch {
    return null;
  }
};

const readStoredLocalValue = (key: string): string | null => {
  if (typeof window === "undefined") return null;
  try {
    return normalizeValue(window.localStorage.getItem(key));
  } catch {
    return null;
  }
};

/**
 * One generation of every stored value the auth getters consult. A request
 * burst re-reads storage only until the first getter populates this snapshot;
 * afterwards every signal below drops it so the next read is fresh.
 */
type StoredAuthSnapshot = {
  cookieConfig: StoredTldwConfig | null;
  effectiveConfig: StoredTldwConfig | null;
  legacySessionToken: string | null;
  legacyApiKey: string | null;
  legacyBearerToken: string | null;
  manualSessionRecord: unknown;
};

let storedAuthSnapshot: StoredAuthSnapshot | null = null;

const dropStoredAuthSnapshot = (): void => {
  storedAuthSnapshot = null;
};

/** Test seam for suites that re-seed storage out-of-band between assertions. */
export const __clearAuthStorageCacheForTests = (): void => {
  dropStoredAuthSnapshot();
};

const readManualSessionRecord = (): unknown => {
  if (typeof window === "undefined") return null;
  try {
    return JSON.parse(window.sessionStorage.getItem(MANUAL_SESSION_KEY) || "null");
  } catch {
    // Missing or unavailable session storage cannot authorize a legacy key.
    return null;
  }
};

const buildStoredAuthSnapshot = (): StoredAuthSnapshot => {
  const cookieConfig = readStoredTldwConfig(COOKIE_SESSION_CONFIG_KEY);
  let effectiveConfig: StoredTldwConfig | null = null;
  if (hasActiveCookieSessionAuth(cookieConfig) &&
    typeof window !== "undefined" &&
    isExactOriginCookieSessionConfig(cookieConfig, window.location.origin)) {
    effectiveConfig = cookieConfig;
  } else {
    const stored = readStoredTldwConfig();
    if (stored) {
      let rotation: unknown = null;
      try {
        rotation = JSON.parse(window.localStorage.getItem(REFRESH_ROTATION_KEY) || "null");
      } catch {
        // Invalid rotation records are ignored by the canonical projection.
      }
      const effective = { ...applyRefreshRotation(stored as TldwConfig, rotation) };
      const invalidationKey = typeof effective.accessToken === "string" &&
        typeof effective.refreshToken === "string"
        ? refreshSessionInvalidationKey(effective) : null;
      if (invalidationKey && readStoredLocalValue(invalidationKey) === "true") {
        delete effective.accessToken;
        delete effective.refreshToken;
      }
      effectiveConfig = effective;
    }
  }
  return {
    cookieConfig,
    effectiveConfig,
    legacySessionToken: readStoredLocalValue("access_token"),
    legacyApiKey: readStoredLocalValue("apiKey"),
    legacyBearerToken: readStoredLocalValue("accessToken"),
    manualSessionRecord: readManualSessionRecord()
  };
};

let storedAuthInvalidationRegistered = false;

/**
 * Mirror every invalidation signal the existing consumers already listen to:
 * the cross-tab `storage` event, the same-tab storage-instance watch (the
 * web shim notifies watchers on same-tab writes), the `tldw:config-updated`
 * and auth-credentials-changed emits, plus the focus/online catch-alls
 * `_app.tsx` uses for shim bootstrap writes that emit nothing else.
 */
const registerStoredAuthSnapshotInvalidation = (): void => {
  if (storedAuthInvalidationRegistered || typeof window === "undefined") return;
  storedAuthInvalidationRegistered = true;
  try {
    createSafeStorage({ area: "local" }).watch({
      tldwConfig: dropStoredAuthSnapshot,
      [REFRESH_ROTATION_KEY]: dropStoredAuthSnapshot,
      [COOKIE_SESSION_CONFIG_KEY]: dropStoredAuthSnapshot
    });
  } catch {
    // Watch support is best-effort; the window listeners below still cover
    // cross-tab writes and every same-tab emit.
  }
  // Over-invalidate on purpose: any stored-key change drops the snapshot, so
  // no legacy-key write (access_token, apiKey, accessToken, session records)
  // can survive without a fresh read.
  window.addEventListener("storage", dropStoredAuthSnapshot);
  window.addEventListener("tldw:config-updated", dropStoredAuthSnapshot);
  window.addEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, dropStoredAuthSnapshot);
  window.addEventListener("focus", dropStoredAuthSnapshot);
  window.addEventListener("online", dropStoredAuthSnapshot);
};

const currentStoredAuthSnapshot = (): StoredAuthSnapshot => {
  if (!storedAuthSnapshot) {
    storedAuthSnapshot = buildStoredAuthSnapshot();
    registerStoredAuthSnapshotInvalidation();
  }
  return storedAuthSnapshot;
};

/** Project the canonical rotation and rejection marker without rewriting storage. */
export const getEffectiveStoredTldwConfig = (): StoredTldwConfig | null =>
  currentStoredAuthSnapshot().effectiveConfig;

/** Legacy sessions are eligible only when no canonical auth configuration owns them. */
export const getSessionAccessToken = (): string | null => {
  const snapshot = currentStoredAuthSnapshot();
  const config = snapshot.effectiveConfig;
  if (config?.authMode === "multi-user" || config?.authMode === "single-user") return null;
  return normalizeBearerValue(snapshot.legacySessionToken);
};

/**
 * Authoritative (uncached) read for security decisions such as the 401
 * logout check: a token rotation that raced an in-flight request burst must
 * be observed even when its invalidation signal has not been processed yet.
 */
export const getFreshSessionAccessToken = (): string | null => {
  dropStoredAuthSnapshot();
  return getSessionAccessToken();
};

const readRuntimeWindowApiKey = (): string | null => {
  if (typeof window === "undefined") return null;
  const runtimeValue = (window as Window & { __tldwRuntimeApiKey?: unknown })
    .__tldwRuntimeApiKey;
  return typeof runtimeValue === "string" ? normalizeApiKeyValue(runtimeValue) : null;
};

const isQuickstartDeployment = (): boolean =>
  String(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || "").trim() ===
  "quickstart";

export const hasActiveCookieSessionAuth = (
  config: { authMode?: unknown; authSource?: unknown } | null | undefined
): boolean =>
  !isCookieSessionConfigInvalidated() &&
  isCookieSessionBrowserTransport({
    authMode: config?.authMode,
    authSource: config?.authSource,
    transportMode: isQuickstartDeployment() ? 'quickstart' : 'advanced',
    transportKind: isQuickstartDeployment() ? 'same-origin' : 'absolute',
    pageOrigin: typeof window === 'undefined' ? null : String(window.location?.origin || ''),
  });

export const getApiKey = (): string | null => {
  const snapshot = currentStoredAuthSnapshot();
  const storedConfig = snapshot.effectiveConfig;
  if (storedConfig?.authMode === "multi-user") return null;
  if (runtimeApiKey) return runtimeApiKey;
  if (hasActiveCookieSessionAuth(snapshot.cookieConfig)) {
    return null;
  }
  if (hasActiveCookieSessionAuth(storedConfig)) return null;

  const runtimeWindowKey = readRuntimeWindowApiKey();
  if (runtimeWindowKey) return runtimeWindowKey;
  const configuredValue = isQuickstartDeployment()
    ? null
    : normalizeApiKeyValue(process.env.NEXT_PUBLIC_X_API_KEY || null);
  if (configuredValue && !suppressEnvApiKeyForSession) return configuredValue;

  const storedMode = normalizeValue(String(storedConfig?.authMode || ""));
  if (storedMode === "single-user") {
    if (storedConfig?.credentialSource === "manual" && storedConfig.apiKeyPersistence === "session") {
      return normalizeApiKeyValue(resolveManualSessionCredential(storedConfig as TldwConfig, snapshot.manualSessionRecord));
    }
    const storedConfigKey = normalizeApiKeyValue(String(storedConfig?.apiKey || ""));
    if (storedConfigKey) return storedConfigKey;
  }

  return normalizeApiKeyValue(snapshot.legacyApiKey);
};

export const getApiBearer = (): string | null => {
  const snapshot = currentStoredAuthSnapshot();
  const storedConfig = snapshot.effectiveConfig;
  if (storedConfig?.authMode === "multi-user") {
    return normalizeBearerValue(String(storedConfig.accessToken || ""));
  }
  if (runtimeApiBearer) return runtimeApiBearer;
  if (hasActiveCookieSessionAuth(snapshot.cookieConfig)) {
    return null;
  }
  if (hasActiveCookieSessionAuth(storedConfig)) return null;

  const configuredValue = normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER || null);
  if (configuredValue) return configuredValue;

  return normalizeBearerValue(snapshot.legacyBearerToken);
};

export const hasEnvApiAuth = (): boolean =>
  (!isQuickstartDeployment() &&
    !suppressEnvApiKeyForSession &&
    normalizeApiKeyValue(process.env.NEXT_PUBLIC_X_API_KEY || null) !== null) ||
  normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER || null) !== null;

export const clearRuntimeAuth = (): void => {
  runtimeApiKey = null;
  runtimeApiBearer = null;
  suppressEnvApiKeyForSession = false;
  // A runtime-auth reset is also a natural cache-drop point: existing reset
  // callers (logout, bootstrap recovery) expect the next read to hit storage.
  dropStoredAuthSnapshot();
};
