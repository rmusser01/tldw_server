const API_HOST = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = process.env.NEXT_PUBLIC_API_VERSION || 'v1';
const REAL_BACKEND_E2E_DEFAULT_API_HOSTS: Record<string, string> = {
  '3101': 'http://127.0.0.1:8101',
  '3102': 'http://127.0.0.1:8102',
};

export const API_BASE_URL = `${API_HOST.replace(/\/$/, '')}/api/${API_VERSION}`;

export const buildApiUrl = (endpoint?: string): string => {
  if (!endpoint) return API_BASE_URL;
  return `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
};

type RequestLike = {
  url?: string;
};

const isRealBackendE2eMode = (): boolean =>
  process.env.TLDW_ADMIN_E2E_REAL_BACKEND === 'true';

const getConfiguredRealBackendApiHost = (uiPort: string): string | null => {
  let configuredHost: string | undefined;
  if (uiPort === '3101') {
    configuredHost = process.env.TLDW_ADMIN_E2E_JWT_API_URL;
  } else if (uiPort === '3102') {
    configuredHost = process.env.TLDW_ADMIN_E2E_SINGLE_USER_API_URL;
  } else {
    return null;
  }

  const apiUrl = new URL(configuredHost || REAL_BACKEND_E2E_DEFAULT_API_HOSTS[uiPort]);
  if (!['http:', 'https:'].includes(apiUrl.protocol)) {
    throw new Error('Real-backend E2E API URL must use HTTP or HTTPS');
  }
  return apiUrl.origin;
};

const getRealBackendApiHost = (request: RequestLike): string | null => {
  if (!isRealBackendE2eMode() || !request.url) {
    return null;
  }

  let uiPort: string;
  try {
    uiPort = new URL(request.url).port;
  } catch {
    return null;
  }

  // The request selects a known project, but never contributes destination
  // protocol or hostname to a credential-bearing backend request.
  return getConfiguredRealBackendApiHost(uiPort);
};

export const buildApiUrlForRequest = (request: RequestLike, endpoint?: string): string => {
  const apiHost = getRealBackendApiHost(request);
  if (!apiHost) {
    return buildApiUrl(endpoint);
  }

  const apiBaseUrl = `${apiHost.replace(/\/$/, '')}/api/${API_VERSION}`;
  if (!endpoint) {
    return apiBaseUrl;
  }
  return `${apiBaseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
};
