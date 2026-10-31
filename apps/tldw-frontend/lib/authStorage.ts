import { isPlaceholderApiKey } from "@/utils/api-key";
import type { TldwConfig } from "@/services/tldw/TldwApiClient";
import {
  applyRefreshRotation,
  refreshSessionInvalidationKey,
  REFRESH_ROTATION_KEY
} from "@/services/tldw/single-user-credential";
import {
  COOKIE_SESSION_CONFIG_KEY,
  isCookieSessionBrowserTransport,
  isExactOriginCookieSessionConfig
} from "@/services/tldw/browser-networking";

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

/** Project the canonical rotation and rejection marker without rewriting storage. */
export const getEffectiveStoredTldwConfig = (): StoredTldwConfig | null => {
  const cookieConfig = readStoredTldwConfig(COOKIE_SESSION_CONFIG_KEY);
  if (hasActiveCookieSessionAuth(cookieConfig) &&
    isExactOriginCookieSessionConfig(cookieConfig, window.location.origin)) return cookieConfig;
  const stored = readStoredTldwConfig();
  if (!stored) return null;
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
  return effective;
};

/** Legacy sessions are eligible only when no canonical auth configuration owns them. */
export const getSessionAccessToken = (): string | null => {
  const config = getEffectiveStoredTldwConfig();
  if (config?.authMode === "multi-user" || config?.authMode === "single-user") return null;
  return normalizeBearerValue(readStoredLocalValue("access_token"));
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
  isCookieSessionBrowserTransport({
    authMode: config?.authMode,
    authSource: config?.authSource,
    transportMode: isQuickstartDeployment() ? "quickstart" : "advanced",
    transportKind: isQuickstartDeployment() ? "same-origin" : "absolute",
    pageOrigin:
      typeof window === "undefined" ? null : String(window.location?.origin || "")
  });

export const getApiKey = (): string | null => {
  const storedConfig = getEffectiveStoredTldwConfig();
  if (storedConfig?.authMode === "multi-user") return null;
  if (runtimeApiKey) return runtimeApiKey;
  if (hasActiveCookieSessionAuth(readStoredTldwConfig(COOKIE_SESSION_CONFIG_KEY))) {
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
    const storedConfigKey = normalizeApiKeyValue(String(storedConfig?.apiKey || ""));
    if (storedConfigKey) return storedConfigKey;
  }

  return normalizeApiKeyValue(readStoredLocalValue("apiKey"));
};

export const getApiBearer = (): string | null => {
  const storedConfig = getEffectiveStoredTldwConfig();
  if (storedConfig?.authMode === "multi-user") {
    return normalizeBearerValue(String(storedConfig.accessToken || ""));
  }
  if (runtimeApiBearer) return runtimeApiBearer;
  if (hasActiveCookieSessionAuth(readStoredTldwConfig(COOKIE_SESSION_CONFIG_KEY))) {
    return null;
  }
  if (hasActiveCookieSessionAuth(storedConfig)) return null;

  const configuredValue = normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER || null);
  if (configuredValue) return configuredValue;

  return normalizeBearerValue(readStoredLocalValue("accessToken"));
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
};
