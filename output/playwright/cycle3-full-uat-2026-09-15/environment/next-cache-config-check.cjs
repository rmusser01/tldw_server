const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const cwd = '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend';
const child = `
  const assert = require('node:assert/strict');
  (async () => {
    const raw = (await import('./next.config.mjs')).default;
    const load = require('next/dist/server/config').default;
    const phase = require('next/constants').PHASE_DEVELOPMENT_SERVER;
    const conf = await load(phase, process.cwd(), {customConfig: raw, silent: true});
    const cacheEnabled = conf.experimental.turbopackFileSystemCacheForDev;
    const expected = !process.env.TLDW_NEXT_DIST_DIR;
    assert.equal(cacheEnabled, expected);
    const headers = await raw.headers();
    const redirects = await raw.redirects();
    const rewrites = await raw.rewrites();
    const comparable = {...conf};
    delete comparable.distDir;
    delete comparable.distDirRoot;
    comparable.experimental = {...comparable.experimental};
    delete comparable.experimental.turbopackFileSystemCacheForDev;
    console.log(JSON.stringify({cacheEnabled, distDir: conf.distDir, comparable,
      headers, redirects, rewrites, webpack: String(raw.webpack)}));
  })().catch(e => {console.error(e.message); process.exit(1)});
`;
const env = {PATH: process.env.PATH, HOME: process.env.HOME, NODE_ENV: 'development',
  NEXT_TELEMETRY_DISABLED: '1', NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: 'advanced',
  NEXT_PUBLIC_API_URL: 'http://127.0.0.1:18201',
  TLDW_INTERNAL_API_ORIGIN: 'http://127.0.0.1:18201'};
function inspect(dist) {
  const runEnv = {...env};
  if (dist) runEnv.TLDW_NEXT_DIST_DIR = dist;
  return JSON.parse(execFileSync(process.execPath, ['-e', child], {cwd, env: runEnv, encoding: 'utf8'}));
}
const normal = inspect();
const isolated = inspect('.next-live-tier-cycle3-multi-20260915');
assert.equal(normal.distDir, '.next/dev');
assert.equal(isolated.distDir, '.next-live-tier-cycle3-multi-20260915/dev');
for (const key of ['comparable','headers','redirects','rewrites','webpack']) assert.deepEqual(normal[key], isolated[key]);
console.log(JSON.stringify({normalDevCache: normal.cacheEnabled, isolatedUatDevCache: isolated.cacheEnabled,
  allOtherNormalizedConfigEqual: true, headersRedirectsRewritesWebpackEqual: true}));
