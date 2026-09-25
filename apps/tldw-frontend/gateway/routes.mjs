const prefix = (path, root) => path === root || path.startsWith(`${root}/`);

export function routeForPath(pathname) {
  if (prefix(pathname, '/_tldw')) return 'managed';

  if (
    prefix(pathname, '/api/_tldw-webui') ||
    prefix(pathname, '/api/documentation') ||
    pathname === '/api/hello'
  ) return 'next';

  if (
    prefix(pathname, '/api/v1') ||
    pathname === '/health' ||
    pathname === '/internal/ready' ||
    pathname === '/openapi.json' ||
    prefix(pathname, '/docs') ||
    pathname === '/redoc' ||
    prefix(pathname, '/setup')
  ) return 'backend';

  return 'next';
}
