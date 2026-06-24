import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { getAuthMode } from '@web/lib/auth';
import { clearRuntimeAuth, setRuntimeApiBearer, setRuntimeApiKey } from '@web/lib/authStorage';

vi.mock('@web/lib/api', () => {
  return {
    apiClient: {
      get: vi.fn(),
      post: vi.fn(),
    },
  };
});

const originalApiKey = process.env.NEXT_PUBLIC_X_API_KEY;
const originalBearer = process.env.NEXT_PUBLIC_API_BEARER;
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;

const clearEnv = () => {
  delete process.env.NEXT_PUBLIC_X_API_KEY;
  delete process.env.NEXT_PUBLIC_API_BEARER;
  delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
};

const restoreEnv = () => {
  if (originalApiKey === undefined) {
    delete process.env.NEXT_PUBLIC_X_API_KEY;
  } else {
    process.env.NEXT_PUBLIC_X_API_KEY = originalApiKey;
  }
  if (originalBearer === undefined) {
    delete process.env.NEXT_PUBLIC_API_BEARER;
  } else {
    process.env.NEXT_PUBLIC_API_BEARER = originalBearer;
  }
  if (originalDeploymentMode === undefined) {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
  } else {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode;
  }
};

describe('getAuthMode', () => {
  beforeEach(() => {
    clearEnv();
    localStorage.clear();
    clearRuntimeAuth();
  });

  afterEach(() => {
    localStorage.clear();
    clearRuntimeAuth();
    restoreEnv();
  });

  it('returns env_single_user when NEXT_PUBLIC_X_API_KEY is set', () => {
    process.env.NEXT_PUBLIC_X_API_KEY = 'key';
    expect(getAuthMode()).toBe('env_single_user');
  });

  it('returns env_bearer when NEXT_PUBLIC_API_BEARER is set', () => {
    process.env.NEXT_PUBLIC_API_BEARER = 'bearer';
    expect(getAuthMode()).toBe('env_bearer');
  });

  it('returns jwt when access_token is present', () => {
    localStorage.setItem('access_token', 'token');
    expect(getAuthMode()).toBe('jwt');
  });

  it('still returns jwt when hosted mode is set and a local JWT exists', () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'hosted';
    localStorage.setItem('access_token', 'token');
    expect(getAuthMode()).toBe('jwt');
  });

  it('prefers jwt when access_token and stored x_api_key are both present', () => {
    localStorage.setItem('access_token', 'token');
    setRuntimeApiKey('key');
    expect(getAuthMode()).toBe('jwt');
  });

  it('returns env_single_user when stored x_api_key is present', () => {
    setRuntimeApiKey('key');
    expect(getAuthMode()).toBe('env_single_user');
  });

  it('returns env_bearer when stored api bearer is present', () => {
    setRuntimeApiBearer('bearer');
    expect(getAuthMode()).toBe('env_bearer');
  });

  it('returns none when no auth credentials are present', () => {
    expect(getAuthMode()).toBe('none');
  });
});
