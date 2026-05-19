let runtimeApiKey: string | null = null;
let runtimeApiBearer: string | null = null;

type StoredTldwConfig = {
  authMode?: unknown;
  apiKey?: unknown;
  accessToken?: unknown;
};

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

const readStoredTldwConfig = (): StoredTldwConfig | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem("tldwConfig");
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

export const getApiKey = (): string | null => {
  const configuredValue = normalizeApiKeyValue(process.env.NEXT_PUBLIC_X_API_KEY || null);
  if (runtimeApiKey) return runtimeApiKey;
  if (configuredValue) return configuredValue;

  const storedConfig = readStoredTldwConfig();
  const storedMode = normalizeValue(String(storedConfig?.authMode || ""));
  if (storedMode === "single-user") {
    const storedConfigKey = normalizeApiKeyValue(String(storedConfig?.apiKey || ""));
    if (storedConfigKey) return storedConfigKey;
  }

  return normalizeApiKeyValue(readStoredLocalValue("apiKey"));
};

export const getApiBearer = (): string | null => {
  const configuredValue = normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER || null);
  if (runtimeApiBearer) return runtimeApiBearer;
  if (configuredValue) return configuredValue;

  const storedConfig = readStoredTldwConfig();
  const storedMode = normalizeValue(String(storedConfig?.authMode || ""));
  if (storedMode === "multi-user") {
    const storedAccessToken = normalizeBearerValue(String(storedConfig?.accessToken || ""));
    if (storedAccessToken) return storedAccessToken;
  }

  return normalizeBearerValue(readStoredLocalValue("accessToken"));
};

export const hasEnvApiAuth = (): boolean =>
  normalizeApiKeyValue(process.env.NEXT_PUBLIC_X_API_KEY || null) !== null ||
  normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER || null) !== null;

export const clearRuntimeAuth = (): void => {
  runtimeApiKey = null;
  runtimeApiBearer = null;
};
