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
    const storedHost = localStorage.getItem('tldw-api-host');
    const apiBaseHost =
      resolveDeploymentMode(DEPLOYMENT_ENV) === 'quickstart'
        ? getDefaultHost()
        : storedHost || getDefaultHost();
    const storedVersion = localStorage.getItem('tldw-api-version') || (process.env.NEXT_PUBLIC_API_VERSION || 'v1');
    const storedKey = process.env.NEXT_PUBLIC_X_API_KEY || '';
    const storedBearer = process.env.NEXT_PUBLIC_API_BEARER || '';
    const storedTheme = (localStorage.getItem('theme') || localStorage.getItem('tldw-theme') || 'dark') as Theme;
    return { apiBaseHost, apiVersion: storedVersion, xApiKey: storedKey || undefined, apiBearer: storedBearer || undefined, theme: storedTheme, csrfToken: null };
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
      localStorage.setItem('tldw-api-host', config.apiBaseHost);
      localStorage.setItem('tldw-api-version', config.apiVersion);
      localStorage.setItem('theme', config.theme);
      localStorage.removeItem('tldw-theme');
    } catch {
      // localStorage may be unavailable in some contexts
    }
    // Apply API base URL
    const nextBase = computeBaseURL(config.apiBaseHost, config.apiVersion);
    api.defaults.baseURL = nextBase;
    // Apply theme
    applyTheme(config.theme);
  }, [config]);

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
