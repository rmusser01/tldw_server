import { NextRequest } from 'next/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const PRIMARY_SECRET = 'primary-jwt-test-secret-1234567890';
const SECONDARY_SECRET = 'secondary-jwt-test-secret-1234567890';

const originalJwtSecret = process.env.JWT_SECRET_KEY;
const originalSecondarySecret = process.env.JWT_SECONDARY_SECRET;
const originalJwtAlgorithm = process.env.JWT_ALGORITHM;

const encodeBase64Url = (bytes: Uint8Array): string => {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
};

const encodeJson = (value: object): string =>
  encodeBase64Url(new TextEncoder().encode(JSON.stringify(value)));

const signJwt = async (
  secret: string,
  payload: Record<string, unknown>,
  header: Record<string, unknown> = { alg: 'HS256', typ: 'JWT' }
): Promise<string> => {
  const headerSegment = encodeJson(header);
  const payloadSegment = encodeJson(payload);
  const signingInput = `${headerSegment}.${payloadSegment}`;
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    new TextEncoder().encode(signingInput)
  );
  return `${signingInput}.${encodeBase64Url(new Uint8Array(signature))}`;
};

const requestWithJwt = (token: string): NextRequest =>
  new NextRequest('http://localhost/dashboard', {
    headers: { cookie: `access_token=${token}` },
  });

const requestWithBearerJwt = (token: string): NextRequest =>
  new NextRequest('http://localhost/dashboard', {
    headers: { authorization: `Bearer ${token}` },
  });

const stubStrictCryptoVerifyBoundary = () => {
  const realCrypto = crypto;
  const verify = async (
    ...args: Parameters<SubtleCrypto['verify']>
  ): Promise<boolean> => {
    const signature = args[2];
    if (!ArrayBuffer.isView(signature)) {
      throw new TypeError('HMAC signature must remain an ArrayBuffer view');
    }
    return realCrypto.subtle.verify(...args);
  };
  const strictSubtle = new Proxy(realCrypto.subtle, {
    get(target, property) {
      if (property === 'verify') return verify;
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  const strictCrypto = new Proxy(realCrypto, {
    get(target, property) {
      if (property === 'subtle') return strictSubtle;
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  vi.stubGlobal('crypto', strictCrypto);
};

const restoreEnv = (name: string, value: string | undefined): void => {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
};

describe('middleware local JWT verification', () => {
  beforeEach(() => {
    process.env.JWT_SECRET_KEY = PRIMARY_SECRET;
    process.env.JWT_SECONDARY_SECRET = SECONDARY_SECRET;
    process.env.JWT_ALGORITHM = 'HS256';
    vi.resetModules();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    restoreEnv('JWT_SECRET_KEY', originalJwtSecret);
    restoreEnv('JWT_SECONDARY_SECRET', originalSecondarySecret);
    restoreEnv('JWT_ALGORITHM', originalJwtAlgorithm);
  });

  it('passes the decoded signature view through the WebCrypto boundary', async () => {
    const token = await signJwt(PRIMARY_SECRET, {
      sub: 'jwt-test-user',
      exp: Math.floor(Date.now() / 1000) + 60,
    });
    stubStrictCryptoVerifyBoundary();
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new Error('unexpected fallback')))
    );
    const { middleware } = await import('../middleware');

    const response = await middleware(requestWithJwt(token));

    expect(response.headers.get('location')).toBeNull();
  });

  it('accepts a valid secondary-secret JWT during key rotation', async () => {
    const token = await signJwt(SECONDARY_SECRET, {
      sub: 'jwt-test-user',
      exp: Math.floor(Date.now() / 1000) + 60,
    });
    const { middleware } = await import('../middleware');

    const response = await middleware(requestWithJwt(token));

    expect(response.headers.get('location')).toBeNull();
  });

  it('rejects a JWT with a bad signature', async () => {
    const token = await signJwt('wrong-jwt-test-secret-1234567890', {
      sub: 'jwt-test-user',
      exp: Math.floor(Date.now() / 1000) + 60,
    });
    const { middleware } = await import('../middleware');

    const response = await middleware(requestWithJwt(token));

    expect(response.headers.get('location')).toContain('/login');
  });

  it('rejects an expired JWT', async () => {
    const token = await signJwt(PRIMARY_SECRET, {
      sub: 'jwt-test-user',
      exp: Math.floor(Date.now() / 1000) - 1,
    });
    const { middleware } = await import('../middleware');

    const response = await middleware(requestWithJwt(token));

    expect(response.headers.get('location')).toContain('/login');
  });

  it('fails closed when the algorithm header is not a string', async () => {
    const token = await signJwt(
      PRIMARY_SECRET,
      {
        sub: 'jwt-test-user',
        exp: Math.floor(Date.now() / 1000) + 60,
      },
      { alg: 256, typ: 'JWT' }
    );
    const { middleware } = await import('../middleware');

    const response = await middleware(requestWithJwt(token));

    expect(response.headers.get('location')).toContain('/login');
  });

  it('fails closed when the signature segment is malformed base64url', async () => {
    const headerSegment = encodeJson({ alg: 'HS256', typ: 'JWT' });
    const payloadSegment = encodeJson({
      sub: 'jwt-test-user',
      exp: Math.floor(Date.now() / 1000) + 60,
    });
    const { middleware } = await import('../middleware');

    const response = await middleware(
      requestWithBearerJwt(`${headerSegment}.${payloadSegment}.%%%`)
    );

    expect(response.headers.get('location')).toContain('/login');
  });
});
