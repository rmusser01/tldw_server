import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import api, { getApiBaseUrl } from '@web/lib/api';
import { buildApiBaseUrl, resolveDeploymentMode, resolvePublicApiOrigin } from '@web/lib/api-base';
import { setRuntimeApiBearer, setRuntimeApiKey } from '@web/lib/authStorage';

type Theme = 'light' | 'dark' | 'system';

interface AppConfig {
  apiBaseHost: string; // e.g., http://127.0.0.1:8000
  apiVersion: string; // e.g., v1
  xApiKey?: string;
  apiBearer?: string;
  theme: Theme;
  csrfToken?: string | null;
}

interface ConfigContextType {
  config: AppConfig;
  setApiBaseHost: (host: string) => void;
  setApiVersion: (version: string) => void;
  setXApiKey: (key: string) => void;
  setApiBearer: (bearer: string) => void;
  setTheme: (theme: Theme) => void;
  reloadBootstrapConfig: () => Promise<void>;
}

type StoredTldwConfig = {
  authMode?: unknown;
  apiKey?: unknown;
  apiBearer?: unknown;
  accessToken?: unknown;
  refreshToken?: unknown;
  serverUrl?: unknown;
  [key: string]: unknown;
};

const DEFAULT_HOST = (typeof window !== 'undefined' && window.location?.origin) || (process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000');
const DEPLOYMENT_ENV = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
};
const DOCS_INFO_API_VERSION = 'v1';

const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

function getPageOrigin(): string | undefined {
  return typeof window !== 'undefined' ? window.location?.origin : undefined;
}

function getDefaultHost(): string {
  const pageOrigin = getPageOrigin();
  const resolvedOrigin = resolvePublicApiOrigin(DEPLOYMENT_ENV, pageOrigin);
  return resolvedOrigin || pageOrigin || DEFAULT_HOST;
}

function normalizeTextValue(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function normalizeApiKeyValue(value: unknown): string | null {
  const normalized = normalizeTextValue(value);
  if (!normalized || /\s/.test(normalized)) return null;
  return normalized;
}

function normalizeBearerValue(value: unknown): string | null {
  const normalized = normalizeTextValue(value);
  if (!normalized) return null;
  const stripped = normalized.replace(/^Bearer\s+/i, '').trim();
  if (!stripped || /\s/.test(stripped)) return null;
  return stripped;
}

function readStoredTldwConfig(): StoredTldwConfig | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem('tldwConfig');
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return null;
    }
    return parsed as StoredTldwConfig;
  } catch {
    return null;
  }
}

function readStoredValue(key: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return normalizeTextValue(window.localStorage.getItem(key));
  } catch {
    return null;
  }
}

function isTheme(value: string | null): value is Theme {
  return value === 'light' || value === 'dark' || value === 'system';
}

function getStoredApiKey(storedConfig: StoredTldwConfig | null): string | null {
  if (normalizeTextValue(storedConfig?.authMode) === 'single-user') {
    const canonicalKey = normalizeApiKeyValue(storedConfig?.apiKey);
    if (canonicalKey) return canonicalKey;
  }
  return normalizeApiKeyValue(readStoredValue('apiKey'));
}

function getStoredApiBearer(storedConfig: StoredTldwConfig | null): string | null {
  if (normalizeTextValue(storedConfig?.authMode) === 'multi-user') {
    const canonicalBearer =
      normalizeBearerValue(storedConfig?.accessToken) ||
      normalizeBearerValue(storedConfig?.apiBearer);
    if (canonicalBearer) return canonicalBearer;
  }
  return normalizeBearerValue(readStoredValue('accessToken'));
}

function loadBrowserConfig(current?: AppConfig): AppConfig {
  const storedConfig = readStoredTldwConfig();
  const deploymentMode = resolveDeploymentMode(DEPLOYMENT_ENV);
  const canonicalHost = normalizeTextValue(storedConfig?.serverUrl);
  const legacyHost = readStoredValue('tldw-api-host');
  const apiBaseHost =
    deploymentMode === 'quickstart'
      ? getDefaultHost()
      : canonicalHost || legacyHost || current?.apiBaseHost || getDefaultHost();
  const storedVersion = readStoredValue('tldw-api-version');
  const apiVersion = storedVersion || current?.apiVersion || process.env.NEXT_PUBLIC_API_VERSION || 'v1';
  const storedTheme = readStoredValue('theme') || readStoredValue('tldw-theme');
  const theme = isTheme(storedTheme) ? storedTheme : current?.theme || 'dark';
  const envApiKey = normalizeApiKeyValue(process.env.NEXT_PUBLIC_X_API_KEY);
  const envApiBearer = normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER);
  const xApiKey = envApiKey || getStoredApiKey(storedConfig) || undefined;
  const apiBearer = envApiBearer || getStoredApiBearer(storedConfig) || undefined;

  return {
    apiBaseHost,
    apiVersion,
    xApiKey,
    apiBearer,
    theme,
    csrfToken: current?.csrfToken ?? null,
  };
}

function writeBrowserConfig(config: AppConfig): void {
  if (typeof window === 'undefined') return;

  // codeql[js/clear-text-storage-of-sensitive-data]: tldw-api-host stores non-secret server metadata only.
  window.localStorage.setItem('tldw-api-host', config.apiBaseHost);
  window.localStorage.setItem('tldw-api-version', config.apiVersion);
  window.localStorage.setItem('theme', config.theme);
  window.localStorage.removeItem('tldw-theme');

  const existingConfig = readStoredTldwConfig();
  const envApiKey = normalizeApiKeyValue(process.env.NEXT_PUBLIC_X_API_KEY);
  const envApiBearer = normalizeBearerValue(process.env.NEXT_PUBLIC_API_BEARER);
  const apiKey = normalizeApiKeyValue(config.xApiKey);
  const apiBearer = normalizeBearerValue(config.apiBearer);
  const shouldPersistApiKey = !!apiKey && apiKey !== envApiKey;
  const shouldPersistApiBearer = !!apiBearer && apiBearer !== envApiBearer;

  window.localStorage.removeItem('apiKey');
  window.localStorage.removeItem('apiBearer');
  window.localStorage.removeItem('accessToken');
  window.localStorage.removeItem('refreshToken');

  if (!existingConfig && !shouldPersistApiKey && !shouldPersistApiBearer) {
    return;
  }

  const nextConfig: StoredTldwConfig = { ...(existingConfig || {}) };
  nextConfig.serverUrl = config.apiBaseHost;
  delete nextConfig.apiKey;
  delete nextConfig.apiBearer;
  delete nextConfig.accessToken;
  delete nextConfig.refreshToken;

  if (shouldPersistApiKey) {
    nextConfig.authMode = 'single-user';
    nextConfig.apiKey = apiKey;
    // codeql[js/clear-text-storage-of-sensitive-data]: local self-hosted credential persistence is explicit user config.
    window.localStorage.setItem('apiKey', apiKey);
    window.localStorage.removeItem('accessToken');
  } else if (!apiKey && normalizeTextValue(nextConfig.authMode) === 'single-user') {
    delete nextConfig.apiKey;
    window.localStorage.removeItem('apiKey');
  }

  if (shouldPersistApiBearer) {
    nextConfig.authMode = 'multi-user';
    nextConfig.accessToken = apiBearer;
    delete nextConfig.apiKey;
    // codeql[js/clear-text-storage-of-sensitive-data]: local self-hosted credential persistence is explicit user config.
    window.localStorage.setItem('accessToken', apiBearer);
    window.localStorage.removeItem('apiKey');
  } else if (!apiBearer && normalizeTextValue(nextConfig.authMode) === 'multi-user') {
    delete nextConfig.accessToken;
    window.localStorage.removeItem('accessToken');
  }

  // codeql[js/clear-text-storage-of-sensitive-data]: this persists the user's chosen local/self-hosted auth mode.
  window.localStorage.setItem('tldwConfig', JSON.stringify(nextConfig));
}

function computeBaseURL(host: string, version: string) {
  if (resolveDeploymentMode(DEPLOYMENT_ENV) === 'quickstart') {
    return buildApiBaseUrl('', version);
  }
  return buildApiBaseUrl(host || resolvePublicApiOrigin(DEPLOYMENT_ENV, getPageOrigin()), version);
}

function normalizeDocsInfoOrigin(value: string): string {
  return value.replace(/\/api\/[^/]+\/?$/, '').replace(/\/$/, '');
}

function computeDocsInfoUrl(host: string): string {
  if (resolveDeploymentMode(DEPLOYMENT_ENV) === 'quickstart') {
    return `${buildApiBaseUrl('', DOCS_INFO_API_VERSION)}/config/docs-info`;
  }

  const preferredOrigin = (process.env.NEXT_PUBLIC_API_BASE_URL || '').toString().trim();
  const resolvedOrigin = normalizeDocsInfoOrigin(
    preferredOrigin || host || resolvePublicApiOrigin(DEPLOYMENT_ENV, getPageOrigin())
  );
  return `${buildApiBaseUrl(resolvedOrigin, DOCS_INFO_API_VERSION)}/config/docs-info`;
}

function applyTheme(theme: Theme) {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  const isDark =
    theme === 'dark' ||
    (theme === 'system' &&
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
  root.classList.toggle('dark', isDark);
  // Keep legacy aliases to avoid breaking existing selectors/readers.
  root.classList.toggle('theme-dark', isDark);
  root.classList.toggle('theme-light', !isDark);
  root.setAttribute('data-theme', isDark ? 'dark' : 'light');
}

export function ConfigProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<AppConfig>(() => {
    if (typeof window === 'undefined') {
      return {
        apiBaseHost: getDefaultHost(),
        apiVersion: process.env.NEXT_PUBLIC_API_VERSION || 'v1',
        xApiKey: process.env.NEXT_PUBLIC_X_API_KEY,
        apiBearer: process.env.NEXT_PUBLIC_API_BEARER,
        theme: 'dark',
        csrfToken: null,
      };
    }
    return loadBrowserConfig();
  });

  // Initialize API baseURL and theme on mount
  useEffect(() => {
    const current = computeBaseURL(config.apiBaseHost, config.apiVersion);
    if (getApiBaseUrl() !== current) {
      api.defaults.baseURL = current;
    }
    applyTheme(config.theme);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist config changes and update API baseURL
  useEffect(() => {
    if (typeof window === 'undefined') return;
    setRuntimeApiKey(config.xApiKey);
    setRuntimeApiBearer(config.apiBearer);
    // Persist
    try {
      writeBrowserConfig(config);
    } catch {
      // localStorage may be unavailable in some contexts
    }
    // Apply API base URL
    const nextBase = computeBaseURL(config.apiBaseHost, config.apiVersion);
    api.defaults.baseURL = nextBase;
    // Apply theme
    applyTheme(config.theme);
  }, [config]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handleConfigUpdated = () => {
      setConfig((current) => loadBrowserConfig(current));
    };
    window.addEventListener('tldw:config-updated', handleConfigUpdated);
    return () => {
      window.removeEventListener('tldw:config-updated', handleConfigUpdated);
    };
  }, []);

  const setApiBaseHost = (host: string) => setConfig((c) => ({ ...c, apiBaseHost: host }));
  const setApiVersion = (ver: string) => setConfig((c) => ({ ...c, apiVersion: ver || 'v1' }));
  const setXApiKey = (key: string) => setConfig((c) => ({ ...c, xApiKey: key || undefined }));
  const setApiBearer = (bearer: string) => setConfig((c) => ({ ...c, apiBearer: bearer || undefined }));
  const setTheme = (t: Theme) => setConfig((c) => ({ ...c, theme: t }));

  const reloadBootstrapConfig = useCallback(async () => {
    try {
      const docsInfoUrl = computeDocsInfoUrl(config.apiBaseHost);
      // docs-info is intentionally non-sensitive; avoid credentialed CORS requirements.
      const resp = await fetch(docsInfoUrl, { credentials: 'omit' });
      if (!resp.ok) return;
      const json = await resp.json();
      const host =
        resolveDeploymentMode(DEPLOYMENT_ENV) === 'quickstart'
          ? getDefaultHost()
          : json?.base_url || json?.api_base_url || config.apiBaseHost;
      const version = config.apiVersion || 'v1';
      const rawKey = json?.api_key || json?.x_api_key || '';
      const key = rawKey && rawKey !== 'YOUR_API_KEY' ? rawKey : config.xApiKey;
      const bearer = json?.api_bearer || config.apiBearer;
      setConfig((c) => ({ ...c, apiBaseHost: host, apiVersion: version, xApiKey: key, apiBearer: bearer }));
    } catch {
      // ignore bootstrap config fetch failures
    }
  }, [config.apiBaseHost, config.apiVersion, config.xApiKey, config.apiBearer]);

  const value = useMemo(
    () => ({
      config,
      setApiBaseHost,
      setApiVersion,
      setXApiKey,
      setApiBearer,
      setTheme,
      reloadBootstrapConfig,
    }),
    [config, reloadBootstrapConfig]
  );

  return <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>;
}

export function useConfig() {
  const ctx = useContext(ConfigContext);
  if (!ctx) throw new Error('useConfig must be used within ConfigProvider');
  return ctx;
}
