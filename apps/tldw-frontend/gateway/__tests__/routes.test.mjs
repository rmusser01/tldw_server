import assert from 'node:assert/strict';
import test from 'node:test';

import { routeForPath } from '../routes.mjs';

test('routes the managed status namespace before application paths', () => {
  for (const path of ['/_tldw', '/_tldw/', '/_tldw/status', '/_tldw/update']) {
    assert.equal(routeForPath(path), 'managed', path);
  }
});

test('keeps existing Next API endpoints and assets on Next', () => {
  for (const path of [
    '/api/_tldw-webui/session',
    '/api/_tldw-webui/runtime-config',
    '/api/documentation',
    '/api/documentation/page',
    '/api/hello',
    '/_next/static/app.js',
    '/settings',
    '/api/v1x',
  ]) {
    assert.equal(routeForPath(path), 'next', path);
  }
});

test('routes only declared backend prefixes and exact paths to FastAPI', () => {
  for (const path of [
    '/api/v1',
    '/api/v1/chat/completions',
    '/health',
    '/internal/ready',
    '/openapi.json',
    '/docs',
    '/docs/oauth2-redirect',
    '/redoc',
    '/setup',
    '/setup/step-1',
  ]) {
    assert.equal(routeForPath(path), 'backend', path);
  }
  for (const path of ['/healthy', '/internal/readiness', '/docs-static', '/setup-extra']) {
    assert.equal(routeForPath(path), 'next', path);
  }
});
