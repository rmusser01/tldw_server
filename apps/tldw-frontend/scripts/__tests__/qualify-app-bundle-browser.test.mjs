import assert from 'node:assert/strict'
import { test } from 'node:test'
import { existsSync, readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve, join } from 'node:path'
import { spawnSync } from 'node:child_process'
import { pathToFileURL } from 'node:url'

const moduleUrl = new URL('../qualify-app-bundle-browser.mjs', import.meta.url)
const load = async () => {
  assert.ok(existsSync(moduleUrl), 'browser qualification implementation is missing')
  return import(pathToFileURL(moduleUrl.pathname).href)
}
const input = () => ({ instances: [
  { publicUrl: 'http://127.0.0.1:18081', sessionCookieName: 'session_a', csrfCookieName: 'csrf_a' },
  { publicUrl: 'http://127.0.0.1:18082', sessionCookieName: 'session_b', csrfCookieName: 'csrf_b' },
] })

test('accepts exactly two distinct public loopback origins and cookie pairs', async () => {
  const { validateInput } = await load()
  assert.deepEqual(validateInput(input()), input().instances)
})

for (const [name, change] of [
  ['remote URL', x => { x.instances[0].publicUrl = 'http://example.org:18081' }],
  ['credentials in URL', x => { x.instances[0].publicUrl = 'http://secret@127.0.0.1:18081' }],
  ['noncanonical origin', x => { x.instances[0].publicUrl += '/setup' }],
  ['duplicate origin', x => { x.instances[1].publicUrl = x.instances[0].publicUrl }],
  ['missing instance', x => { x.instances.pop() }],
  ['duplicate session cookie', x => { x.instances[1].sessionCookieName = 'session_a' }],
  ['cross-kind cookie collision', x => { x.instances[1].csrfCookieName = 'session_a' }],
  ['cookie injection', x => { x.instances[1].csrfCookieName = 'csrf=x' }],
  ['unsupported secure cookie', x => { x.instances[0].sessionCookieName = '__Host-session' }],
  ['unknown instance secret', x => { x.instances[0].apiKey = 'secret' }],
  ['unknown root secret', x => { x.signingKey = 'secret' }],
]) {
  test(`rejects ${name} before launch with a secret-free failure`, async () => {
    const { validateInput } = await load()
    const value = input(); change(value)
    assert.throws(() => validateInput(value), { message: 'invalid_public_input' })
  })
}

const cookies = () => [
  { name: 'session_a', value: 'secret-session', domain: '127.0.0.1', path: '/api', httpOnly: true, secure: false, sameSite: 'Lax', expires: 2000000000 },
  { name: 'csrf_a', value: 'secret-csrf', domain: '127.0.0.1', path: '/', httpOnly: false, secure: false, sameSite: 'Lax', expires: 2000000000 },
]
test('requires an HttpOnly session and readable CSRF with loopback attributes', async () => {
  const { assertCookiePolicy } = await load()
  assert.doesNotThrow(() => assertCookiePolicy(cookies(), input().instances[0], 1000000000))
})
for (const [name, change] of [
  ['missing session', x => x.shift()],
  ['readable session', x => { x[0].httpOnly = false }],
  ['HttpOnly CSRF', x => { x[1].httpOnly = true }],
  ['empty token', x => { x[1].value = '' }],
  ['broader session path', x => { x[0].path = '/' }],
  ['wrong session path', x => { x[0].path = '/api/v1' }],
  ['wrong CSRF path', x => { x[1].path = '/api' }],
  ['remote domain', x => { x[0].domain = 'example.org' }],
  ['insecure SameSite', x => { x[0].sameSite = 'None' }],
  ['expired session', x => { x[0].expires = 1 }],
]) {
  test(`rejects ${name} without revealing cookie values`, async () => {
    const { assertCookiePolicy } = await load()
    const value = cookies(); change(value)
    assert.throws(() => assertCookiePolicy(value, input().instances[0], 1000000000), { message: 'cookie_attributes_failed' })
  })
}

test('failed check is retained and exception secrets never enter evidence', async () => {
  const { createEvidence, recordCheck } = await load()
  const evidence = createEvidence()
  await assert.rejects(recordCheck(evidence, 'cookie_only_profile', async () => {
    throw new Error('secret-session and secret-api-key')
  }), { message: 'cookie_only_profile' })
  assert.equal(evidence.checks.cookie_only_profile.passed, false)
  assert.ok(evidence.checks.cookie_only_profile.duration_ms >= 0)
  assert.ok(!JSON.stringify(evidence).includes('secret'))
  assert.equal(evidence.planned_setup_complete, false)
})

test('evidence rejects arbitrary check labels instead of copying unsafe input', async () => {
  const { createEvidence, recordCheck } = await load()
  const evidence = createEvidence()
  await assert.rejects(recordCheck(evidence, 'secret-api-key', async () => {}), { message: 'invalid_check' })
  assert.deepEqual(evidence.checks, {})
})


test('per-instance failed checks cannot be overwritten by the other instance', async () => {
  const { createEvidence, recordCheck } = await load()
  const evidence = createEvidence()
  await assert.rejects(recordCheck(evidence, 'manual_master_key_absent_1', async () => { throw new Error('secret') }))
  await recordCheck(evidence, 'manual_master_key_absent_2', async () => {})
  assert.equal(evidence.checks.manual_master_key_absent_1.passed, false)
  assert.equal(evidence.checks.manual_master_key_absent_2.passed, true)
  assert.equal(evidence.failure_code, 'manual_master_key_required')
})

test('visible master-key setup fails without entering credentials', async () => {
  const { chromium } = await import('playwright')
  const { inspectManagedSetup } = await load()
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage()
    page.setDefaultTimeout(5_000)
    await page.setContent('<h1>Setup</h1><label>API Key<input type="password"></label><button>Test connection</button>')
    await assert.rejects(inspectManagedSetup(page), { message: 'manual_master_key_required' })
    assert.equal(await page.getByLabel('API Key').inputValue(), '')
  } finally {
    await browser.close()
  }
})

test('keyless setup uses accessible Docker and privacy controls', async () => {
  const { chromium } = await import('playwright')
  const { inspectManagedSetup } = await load()
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage()
    page.setDefaultTimeout(5_000)
    // A UI-only fixture tests control navigation; live qualification has no API mocks.
    await page.setContent('<button>Set up in WebUI</button>')
    await page.evaluate(() => {
      document.querySelector('button').onclick = () => {
        document.body.innerHTML = '<h1>First-time setup</h1><button>Solo, Docker</button>'
        document.querySelector('button').onclick = () => {
          document.body.innerHTML = '<h2>Privacy and security</h2><label><input type="checkbox">I understand local or remote setup access and provider secret storage.</label><button disabled>Continue</button>'
          document.querySelector('input').onchange = event => {
            document.querySelector('button').disabled = !event.currentTarget.checked
          }
          document.querySelector('button').onclick = () => {
            document.body.innerHTML = '<h2>Chat provider</h2>'
          }
        }
      }
    })
    await inspectManagedSetup(page)
    assert.equal(await page.getByRole('heading', { name: 'Chat provider' }).isVisible(), true)
  } finally {
    await browser.close()
  }
})


function runtimeFixture(root) {
  const fixtures = {}
  for (const index of [1, 2]) {
    const cfg = {
      TLDW_PROJECT_ID: `paired-${index}`, TLDW_PUBLIC_PORT: `${18080 + index}`,
      SINGLE_USER_API_KEY: `fixture-api-${index}`, SINGLE_USER_SESSION_COOKIE_NAME: `session_${index}`,
      CSRF_COOKIE_NAME: `csrf_${index}`, TLDW_GATEWAY_HOP_SECRET: `fixture-hop-${index}`,
      TLDW_BACKEND_IMAGE: 'local/backend@sha256:' + 'a'.repeat(64),
      TLDW_WEBUI_IMAGE: 'local/webui@sha256:' + 'b'.repeat(64),
      TLDW_GATEWAY_IMAGE: 'local/gateway@sha256:' + 'c'.repeat(64),
    }
    mkdirSync(join(root, `instance-${index}`, 'instance'), { recursive: true })
    writeFileSync(join(root, `instance-${index}`, 'instance', 'config.env'), Object.entries(cfg).map(([k, v]) => `${k}=${v}`).join('\n'))
    fixtures[index] = ['app', 'webui', 'gateway'].map(role => {
      const env = role === 'gateway' ? {} : {
        SINGLE_USER_API_KEY: cfg.SINGLE_USER_API_KEY,
        SINGLE_USER_SESSION_COOKIE_NAME: cfg.SINGLE_USER_SESSION_COOKIE_NAME,
        CSRF_COOKIE_NAME: cfg.CSRF_COOKIE_NAME,
      }
      if (role !== 'app') env.TLDW_GATEWAY_HOP_SECRET = cfg.TLDW_GATEWAY_HOP_SECRET
      if (index === 2) {
        if (role === 'app') env.PORT = '18101'
        if (role === 'webui') env.PORT = '18102'
        if (role !== 'app') env.TLDW_INTERNAL_API_ORIGIN = 'http://backend-qualification:18101'
        if (role === 'gateway') env.TLDW_INTERNAL_WEBUI_ORIGIN = 'http://next-qualification:18102'
      }
      return {
        Image: `image-${role}`,
        Config: { Image: cfg[{ app: 'TLDW_BACKEND_IMAGE', webui: 'TLDW_WEBUI_IMAGE', gateway: 'TLDW_GATEWAY_IMAGE' }[role]],
          Labels: { 'com.docker.compose.service': role }, Env: Object.entries(env).map(([k, v]) => `${k}=${v}`) },
        NetworkSettings: { Ports: role === 'gateway' ? { '8080/tcp': [{ HostIp: '127.0.0.1', HostPort: cfg.TLDW_PUBLIC_PORT }] } : { '8000/tcp': null },
          Networks: { private: { Aliases: [role === 'app' ? 'backend-qualification' : 'next-qualification'] } } },
        Mounts: role === 'app' ? [{ Type: 'volume', Name: `data-${index}`, Destination: '/app/Databases' }] : [],
      }
    })
  }
  writeFileSync(join(root, 'mounts.before.json'), JSON.stringify(fixtures[2][0].Mounts))
  return fixtures
}

for (const [name, mutate, success] of [
  ['valid signed runtime identities', () => {}, true],
  ['wrong backend port', f => { f[2][0].Config.Env.push('PORT=8000') }, false],
  ['wrong private alias', f => { f[2][0].NetworkSettings.Networks.private.Aliases = ['app'] }, false],
  ['replaced trust secret', f => { f[2][2].Config.Env.push('TLDW_GATEWAY_HOP_SECRET=foreign') }, false],
  ['different WebUI image', f => { f[2][1].Image = 'different-image' }, false],
  ['changed data volume', f => { f[2][0].Mounts[0].Name = 'fresh-volume' }, false],
]) {
  test(`runtime inspection ${success ? 'accepts' : 'rejects'} ${name}`, () => {
    const root = mkdtempSync(join(tmpdir(), 'bundle-inspection-test-'))
    try {
      const repo = resolve(new URL('../../../..', import.meta.url).pathname)
      const shell = readFileSync(join(repo, 'Helper_Scripts/test_app_bundle_browser.sh'), 'utf8')
      const embedded = shell.match(/python - "\$test_root" "\$bundle_dir" <<'PY'\n([\s\S]*?)\nPY/)[1]
      const fixtures = runtimeFixture(root); mutate(fixtures)
      writeFileSync(join(root, 'fixtures.json'), JSON.stringify(fixtures))
      mkdirSync(join(root, 'bin'))
      writeFileSync(join(root, 'bin/docker'), `#!${process.execPath}\nconst fs=require('node:fs'); const args=process.argv.slice(2); if(args[0]==='compose'){ const i=args[args.indexOf('--project-name')+1].split('-')[1]; process.stdout.write('id'+i+'a id'+i+'w id'+i+'g\\n'); } else { if(args.length!==4 || !args.slice(1).every(x=>/^id[12][awg]$/.test(x))) process.exit(2); const data=JSON.parse(fs.readFileSync(${JSON.stringify(join(root, 'fixtures.json'))},'utf8')); process.stdout.write(JSON.stringify(data[args[1][2]])); }`, { mode: 0o700 })
      writeFileSync(join(root, 'inspect.py'), embedded)
      const result = spawnSync('/bin/bash', ['-c', 'source "$1/.venv/bin/activate" && python "$2" "$3" "$4"', 'fixture', repo, join(root, 'inspect.py'), root, root], {
        env: { ...process.env, PATH: join(root, 'bin') + ':' + process.env.PATH }, encoding: 'utf8', timeout: 10_000,
      })
      assert.equal(result.status, success ? 0 : 1, result.stderr)
      if (success) {
        assert.equal(JSON.parse(readFileSync(join(root, 'public-input.json'), 'utf8')).instances.length, 2)
      } else {
        assert.equal(existsSync(join(root, 'public-input.json')), false)
        assert.equal(result.stderr.trim(), 'Paired runtime identity check failed (private details suppressed).')
      }
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
}

for (const timing of ['early', 'late']) {
  test(`setup tracker rejects ${timing} 403 and keeps success false`, async () => {
    const { createEvidence, createSetupResponseTracker, completeEvidence, REQUIRED_CHECKS } = await load()
    assert.equal(typeof createSetupResponseTracker, 'function', 'setup response tracker is missing')
    const evidence = createEvidence()
    for (const name of REQUIRED_CHECKS) evidence.checks[name] = { passed: true }
    const first = createSetupResponseTracker(evidence, 1, 'http://127.0.0.1:18081')
    const second = createSetupResponseTracker(evidence, 2, 'http://127.0.0.1:18082')
    if (timing === 'early') first.observe('http://127.0.0.1:18081/api/v1/setup/first-run/state', 403)
    first.observe('http://127.0.0.1:18081/api/v1/setup/first-run/state', 200)
    second.observe('http://127.0.0.1:18082/api/v1/setup/first-run/state', 200)
    if (timing === 'late') {
      completeEvidence(evidence)
      first.observe('http://127.0.0.1:18081/api/v1/setup/readiness/status', 403)
    }
    assert.throws(() => completeEvidence(evidence), { message: 'required_checks_failed' })
    assert.equal(evidence.checks.setup_api_access_1.passed, false)
    assert.equal(evidence.passed, false)
  })
}

test('expected negative auth probes do not fail setup access', async () => {
  const { createEvidence, createSetupResponseTracker, completeEvidence, REQUIRED_CHECKS } = await load()
  assert.equal(typeof createSetupResponseTracker, 'function', 'setup response tracker is missing')
  const evidence = createEvidence()
  for (const name of REQUIRED_CHECKS) evidence.checks[name] = { passed: true }
  for (const index of [1, 2]) {
    const origin = `http://127.0.0.1:${18080 + index}`
    const tracker = createSetupResponseTracker(evidence, index, origin)
    tracker.observe(origin + '/api/v1/setup/first-run/state', 200)
    tracker.observe(origin + '/api/v1/users/me/profile', 401)
    tracker.observe(origin + '/api/v1/auth/single-user/session', 403)
    tracker.observe('http://unrelated.invalid/api/v1/setup/first-run/state', 403)
  }
  completeEvidence(evidence)
  assert.equal(evidence.passed, true)
  assert.equal(evidence.G2, false)
  assert.equal(evidence.G4, false)
  assert.equal(evidence.G12, false)
})

test('any required false check prohibits successful evidence', async () => {
  const { createEvidence, completeEvidence } = await load()
  assert.equal(typeof completeEvidence, 'function', 'evidence completion gate is missing')
  const evidence = createEvidence()
  evidence.checks = { setup_api_access_1: { passed: true }, setup_api_access_2: { passed: true }, cookie_only_profile_1: { passed: false } }
  assert.throws(() => completeEvidence(evidence), { message: 'required_checks_failed' })
  assert.equal(evidence.passed, false)
})

test('missing setup responses cannot qualify an instance', async () => {
  const { createEvidence, createSetupResponseTracker, completeEvidence } = await load()
  assert.equal(typeof createSetupResponseTracker, 'function', 'setup response tracker is missing')
  const evidence = createEvidence()
  createSetupResponseTracker(evidence, 1, 'http://127.0.0.1:18081')
  createSetupResponseTracker(evidence, 2, 'http://127.0.0.1:18082')
  assert.throws(() => completeEvidence(evidence), { message: 'required_checks_failed' })
})

for (const [name, downExit, originalExit, expectedExit, browserFailed = false] of [
  ['cleanup succeeds', 0, 0, 0],
  ['cleanup fails after passing probe', 1, 0, 1],
  ['cleanup fails after original failure', 1, 7, 7],
  ['browser shutdown fails with retained recovery', 0, 7, 7, true],
]) {
  test(`helper ${name} with recoverable state and bounded output`, () => {
    const root = mkdtempSync(join(tmpdir(), 'bundle-cleanup-test-'))
    try {
      const owned = join(root, 'owned-state')
      mkdirSync(owned)
      runtimeFixture(owned)
      const evidencePath = join(root, 'browser-evidence.json')
      writeFileSync(evidencePath, JSON.stringify({ schema_version: 1, passed: originalExit === 0,
        failure_code: originalExit === 0 ? undefined : 'manual_master_key_required',
        G2: false, G4: false, G12: false, checks: browserFailed ? { browser_shutdown: { passed: false } } : {} }))
      const repo = resolve(new URL('../../../..', import.meta.url).pathname)
      const shell = readFileSync(join(repo, 'Helper_Scripts/test_app_bundle_browser.sh'), 'utf8')
      const cleanup = shell.slice(shell.indexOf('cleanup() {'), shell.indexOf('trap cleanup EXIT'))
      mkdirSync(join(root, 'bin'))
      writeFileSync(join(root, 'bin/docker'), `#!${process.execPath}\nconst fs=require('node:fs');const args=process.argv.slice(2);fs.appendFileSync(${JSON.stringify(join(root, 'calls.jsonl'))},JSON.stringify(args)+'\\n');console.error('secret-token-must-be-suppressed');process.exit(${downExit});`, { mode: 0o700 })
      writeFileSync(join(root, 'cleanup.sh'), `#!/bin/bash\nset -Eeuo pipefail\numask 077\ntest_root=$1\nbundle_dir=$2\nevidence_path=$3\n${cleanup}\ntrap cleanup EXIT\nexit "$4"\n`)
      const result = spawnSync('/bin/bash', ['-c', 'source "$1/.venv/bin/activate" && bash "$2" "$3" "$4" "$5" "$6"', 'fixture', repo, join(root, 'cleanup.sh'), owned, root, evidencePath, `${originalExit}`], {
        env: { ...process.env, PATH: join(root, 'bin') + ':' + process.env.PATH }, encoding: 'utf8', timeout: 10_000,
      })
      assert.equal(result.status, expectedExit, result.stderr)
      assert.equal(existsSync(join(owned, 'instance-1/instance/config.env')), downExit !== 0 || browserFailed)
      const calls = readFileSync(join(root, 'calls.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line))
      assert.deepEqual(calls.map(args => args[args.indexOf('--project-name') + 1]), ['paired-1', 'paired-2'])
      assert.ok(calls.every(args => args.slice(-2).join() === 'down,--volumes'))
      assert.ok(!result.stderr.includes('secret-token'))
      const evidence = JSON.parse(readFileSync(evidencePath, 'utf8'))
      assert.ok(evidence.checks.owned_resources_removed, 'cleanup outcome is missing from evidence')
      assert.equal(evidence.checks.owned_resources_removed.passed, downExit === 0 && !browserFailed)
      if (downExit !== 0 || browserFailed) {
        assert.equal(evidence.passed, false)
        assert.ok(result.stderr.includes('Cleanup failed; disposable instance state retained for recovery.'))
        const pointer = join(root, '.browser-cleanup-recovery')
        assert.equal(readFileSync(pointer, 'utf8').trim(), owned)
        assert.equal(statSync(pointer).mode & 0o777, 0o600)
        if (originalExit !== 0) assert.equal(evidence.failure_code, 'manual_master_key_required')
      }
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
}

test('missing transport check refuses successful evidence even when all present checks passed', async () => {
  const { createEvidence, completeEvidence } = await load()
  const evidence = createEvidence()
  evidence.checks = { setup_api_access_1: { passed: true }, setup_api_access_2: { passed: true } }
  assert.throws(() => completeEvidence(evidence), { message: 'required_checks_failed' })
  assert.equal(evidence.passed, false)
})

test('browser shutdown rejection fails closed with bounded evidence', async () => {
  const { createEvidence, closeBrowser } = await load()
  const evidence = createEvidence()
  evidence.passed = true
  await assert.rejects(closeBrowser({ close: async () => { throw new Error('secret-cookie') } }, evidence), { message: 'browser_shutdown' })
  assert.equal(evidence.passed, false)
  assert.equal(evidence.checks.browser_shutdown.passed, false)
  assert.ok(!JSON.stringify(evidence).includes('secret-cookie'))
})

test('browser shutdown keeps an earlier failure code', async () => {
  const { createEvidence, closeBrowser } = await load()
  const evidence = createEvidence()
  evidence.failure_code = 'setup_api_access_1'
  await assert.rejects(closeBrowser({ close: async () => { throw new Error('secret') } }, evidence))
  assert.equal(evidence.failure_code, 'setup_api_access_1')
})

// These fixtures use real HTTP, multipart bytes, streaming cancellation and
// Chromium WebSockets. They are deliberately separate from the live candidate.
async function transportFixture(change = '', pairedIndex) {
  const { createServer } = await import('node:http')
  const { createHash } = await import('node:crypto')
  const sockets = new Set()
  let uploadSeen = false
  let streamClosed = false
  let sessionActive = false
  const setupMutations = []
  const foreignSessionRequests = []
  const logoutRequests = []
  const profilesAfterForeignSession = []
  const sessionName = pairedIndex ? `session_${pairedIndex}` : 'fixture_session'
  const csrfName = `csrf_${pairedIndex}`
  const sessionValue = pairedIndex ? `public-fixture-session-${pairedIndex}` : 'public-fixture-session'
  const csrfValue = pairedIndex ? `public-fixture-csrf-${pairedIndex}` : 'public-fixture-csrf'
  const server = createServer((req, res) => {
    const path = new URL(req.url, 'http://fixture').pathname
    if (req.headers.host === 'hostile.invalid' || (req.headers.origin && req.headers.origin !== origin)) {
      res.writeHead(403); res.end(); return
    }
    if (pairedIndex && path === '/setup') {
      res.setHeader('Content-Type', 'text/html')
      res.end(`<h1>Setup</h1><div id="controls"></div><script>
        const controls = document.getElementById('controls');
        window.setupEvents = [];
        async function save(step, data) {
          const csrf = document.cookie.split('; ').find(value => value.startsWith('${csrfName}='))?.split('=')[1];
          const response = await fetch('/api/v1/setup/first-run/state', {
            method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf },
            body: JSON.stringify({ step, data })
          });
          await response.arrayBuffer();
          if (!response.ok) { controls.insertAdjacentHTML('beforeend', '<p role="alert">Setup progress could not be saved.</p>'); return false; }
          return true;
        }
        async function provider() {
          const checkbox = controls.querySelector('input');
          if (!checkbox?.checked) return;
          window.setupEvents.push('continue');
          controls.querySelector('button').disabled = true;
          if (await save('privacy_security', { acknowledged: true, local_only: true, allow_remote_setup_access: false })) {
            controls.innerHTML = '<h1>Chat provider setup</h1>';
          }
        }
        async function docker() {
          if (!await save('setup_path', { acknowledged: true, selected_path: 'docker_single_user', setup_path_key: 'docker_single_user', install_method: 'docker', deployment_mode: 'single_user' })) return;
          controls.innerHTML = '<h2>Privacy and security</h2>${change === 'missing-acknowledgement' ? '' : `<label><input type="checkbox" ${change === 'disabled-acknowledgement' ? 'disabled' : ''}>I understand local or remote setup access and provider secret storage.</label>`}<button disabled onclick="provider()">Continue</button>';
          window.setupEvents.push('privacy-unchecked-continue-disabled');
          const checkbox = controls.querySelector('input');
          if (checkbox) checkbox.onchange = () => {
            window.setupEvents.push(checkbox.checked ? 'acknowledged' : 'unacknowledged');
            controls.querySelector('button').disabled = !checkbox.checked;
          };
        }
        function wizard() { controls.innerHTML = '<h1>First-time setup</h1><button onclick="docker()">Solo, Docker</button>' }
        (async () => {
          const session = await fetch('/api/_tldw-webui/session', { method: 'POST' }); await session.arrayBuffer();
          const state = await fetch('/api/v1/setup/first-run/state'); await state.arrayBuffer();
          controls.innerHTML = '<button onclick="wizard()">Set up in WebUI</button>';
        })()
      </script>`)
      return
    }
    if (pairedIndex && path === '/api/_tldw-webui/session') {
      sessionActive = true
      res.setHeader('Set-Cookie', [
        `${sessionName}=${sessionValue}; Path=/api; HttpOnly; SameSite=Lax`,
        `${csrfName}=${csrfValue}; Path=/; SameSite=Lax`,
      ])
      res.end(); return
    }
    if (pairedIndex && path === '/api/v1/users/me/profile') {
      const requestCookies = Object.fromEntries((req.headers.cookie || '').split('; ').map(cookie => cookie.split('=')))
      const status = sessionActive && requestCookies[sessionName] === sessionValue ? 200 : 401
      // Actual CSRF middleware also issues this cookie on a refused profile GET.
      if (!requestCookies[csrfName]) {
        res.setHeader('Set-Cookie', `${csrfName}=public-fixture-replacement-csrf-${pairedIndex}; Path=/; SameSite=Lax`)
      }
      if (requestCookies[sessionName] === `public-fixture-session-${3 - pairedIndex}`) {
        foreignSessionRequests.push({ cookie: req.headers.cookie, csrfIssued: !requestCookies[csrfName], status })
      } else if (foreignSessionRequests.length) {
        profilesAfterForeignSession.push({
          sessionUnchanged: requestCookies[sessionName] === sessionValue,
          csrfUnchanged: requestCookies[csrfName] === csrfValue, status,
        })
      }
      res.writeHead(status)
      res.end('{}'); return
    }
    if (pairedIndex && path === '/api/v1/auth/single-user/session' && req.method === 'DELETE') {
      const requestCookies = Object.fromEntries((req.headers.cookie || '').split('; ').map(cookie => cookie.split('=')))
      const status = req.headers['x-csrf-token'] && req.headers['x-csrf-token'] === requestCookies[csrfName] ? 200 : 403
      logoutRequests.push({
        sessionUnchanged: requestCookies[sessionName] === sessionValue,
        csrfUnchanged: requestCookies[csrfName] === csrfValue,
        capturedToken: req.headers['x-csrf-token'] === csrfValue, status,
      })
      if (status !== 200) { res.writeHead(status); res.end(); return }
      sessionActive = false
      res.setHeader('Set-Cookie', `${sessionName}=; Path=/api; Max-Age=0; HttpOnly; SameSite=Lax`)
      res.end('{}'); return
    }
    if (pairedIndex && path === '/api/v1/setup/first-run/state' && req.method === 'POST') {
      if (!sessionActive || !req.headers.cookie?.includes(`${sessionName}=${sessionValue}`) ||
          !req.headers.cookie?.includes(`${csrfName}=${csrfValue}`) || req.headers['x-csrf-token'] !== csrfValue) {
        res.writeHead(403); res.end('{}'); return
      }
      let body = ''
      req.on('data', chunk => { body += chunk })
      req.on('end', () => {
        const mutation = JSON.parse(body)
        const expected = setupMutations.length === 0
          ? { step: 'setup_path', data: { acknowledged: true, selected_path: 'docker_single_user', setup_path_key: 'docker_single_user', install_method: 'docker', deployment_mode: 'single_user' } }
          : { step: 'privacy_security', data: { acknowledged: true, local_only: true, allow_remote_setup_access: false } }
        setupMutations.push(mutation)
        const status = JSON.stringify(mutation) !== JSON.stringify(expected) ? 400
          : mutation.step === 'privacy_security' && change === 'privacy-refused' ? 403
            : mutation.step === 'privacy_security' && change === 'privacy-server-error' ? 500 : 200
        res.writeHead(status); res.end('{}')
      }); return
    }
    if (path === '/api/documentation/manifest') { res.end(JSON.stringify({ docsBySource: { server: [{ source: 'server', relativePath: 'API-related/AuthNZ-API-Guide.md' }] } })); return }
    if (path === '/api/documentation/content') {
      if (req.url.includes('..')) { res.writeHead(400); res.end(); return }
      res.end(JSON.stringify({ content: '# AuthNZ API Guide\nPublic fixture.' })); return
    }
    if (path === '/api/v1/health/live/') { res.writeHead(307, { Location: origin + '/api/v1/health/live' }); res.end(); return }
    if (path === '/api/v1/media/process-documents') {
      let body = ''; req.on('data', chunk => { body += chunk }); req.on('end', () => {
        uploadSeen = body.includes('A harmless public upload sentinel.') && body.includes('name="files"') && body.includes('name="perform_analysis"') && body.includes('false') && req.headers['x-csrf-token'] === csrfValue
        res.end(JSON.stringify({ results: [{ status: 'Success', content: change === 'content' ? 'lost' : '# WP1 qualification\nA harmless public upload sentinel.\n' }], errors: [] }))
      }); return
    }
    if (path === '/api/v1/notifications/stream') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' }); res.write('data: public\n\n')
      res.on('close', () => { streamClosed = true }); return
    }
    if (change === 'reflect' && path === '/api/v1/setup/first-run/state') { res.setHeader('X-Tldw-Gateway-Hop', 'public-forged-hop'); res.end('public-forged-hop'); return }
    res.end('{}')
  })
  server.on('upgrade', (req, socket) => {
    sockets.add(socket); socket.on('error', () => {}); socket.on('close', () => sockets.delete(socket))
    if (req.headers.origin !== origin || !req.headers.cookie?.includes(`${sessionName}=${sessionValue}`)) {
      socket.end('HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n'); return
    }
    const key = createHash('sha1').update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64')
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${key}\r\n\r\n`)
    socket.once('data', () => {
      const text = Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: 1, result: { protocolVersion: '2024-11-05', serverInfo: { name: change === 'ws' ? 'wrong' : 'tldw-mcp-unified' } } }))
      const header = Buffer.from([0x81, text.length])
      socket.write(Buffer.concat([header, text])); socket.once('data', () => socket.end(Buffer.from([0x88, 0])))
    })
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const origin = `http://127.0.0.1:${server.address().port}`
  return { origin, setupMutations, foreignSessionRequests, logoutRequests, profilesAfterForeignSession,
    observed: () => ({ uploadSeen, streamClosed }), close: async () => {
    for (const socket of sockets) socket.destroy()
    server.closeAllConnections()
    await new Promise(resolve => server.close(resolve))
  } }
}

test('managed setup acknowledges privacy before successful cookie-CSRF writes', async () => {
  const fixture = await transportFixture('', 1)
  const { chromium } = await import('playwright')
  const { inspectManagedSetup, createEvidence, createSetupResponseTracker, createNetworkTracker } = await load()
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage()
    page.setDefaultTimeout(2_000)
    const evidence = createEvidence()
    const setup = createSetupResponseTracker(evidence, 1, fixture.origin)
    const network = createNetworkTracker(page, fixture.origin)
    page.on('response', response => setup.observe(response.url(), response.status()))
    await page.goto(fixture.origin + '/setup')
    await inspectManagedSetup(page)
    assert.deepEqual(await page.evaluate(() => window.setupEvents), ['privacy-unchecked-continue-disabled', 'acknowledged', 'continue'])
    assert.deepEqual(fixture.setupMutations, [
      { step: 'setup_path', data: { acknowledged: true, selected_path: 'docker_single_user', setup_path_key: 'docker_single_user', install_method: 'docker', deployment_mode: 'single_user' } },
      { step: 'privacy_security', data: { acknowledged: true, local_only: true, allow_remote_setup_access: false } },
    ])
    setup.assertSucceeded()
    assert.equal(network.failed(), false)
  } finally { await browser.close(); await fixture.close() }
})

for (const change of ['missing-acknowledgement', 'disabled-acknowledgement', 'privacy-refused', 'privacy-server-error']) {
  test(`managed setup fails closed for ${change} despite later successful GET`, async () => {
    const fixture = await transportFixture(change, 1)
    const { chromium } = await import('playwright')
    const { inspectManagedSetup, createEvidence, recordCheck, createSetupResponseTracker, createNetworkTracker, completeEvidence, REQUIRED_CHECKS } = await load()
    const browser = await chromium.launch({ headless: true })
    try {
      const page = await browser.newPage()
      page.setDefaultTimeout(1_000)
      const evidence = createEvidence()
      for (const name of REQUIRED_CHECKS) evidence.checks[name] = { passed: true }
      const setup = createSetupResponseTracker(evidence, 1, fixture.origin)
      const network = createNetworkTracker(page, fixture.origin, () => {
        evidence.checks.live_errors_absent = { passed: false }
        evidence.failure_code ||= 'live_errors_absent'
        evidence.passed = false
      })
      page.on('response', response => setup.observe(response.url(), response.status()))
      await page.goto(fixture.origin + '/setup')
      await assert.rejects(recordCheck(evidence, 'setup_interaction_1', () => inspectManagedSetup(page)), { message: 'setup_interaction_1' })
      assert.equal(await page.getByRole('heading', { name: 'Chat provider setup', exact: true }).isVisible(), false)
      const laterSuccess = page.waitForResponse(response => response.url() === fixture.origin + '/api/v1/setup/first-run/state' && response.request().method() === 'GET')
      await page.evaluate(() => fetch('/api/v1/setup/first-run/state').then(response => response.arrayBuffer()))
      assert.equal((await laterSuccess).status(), 200)
      assert.equal(evidence.checks.setup_interaction_1.passed, false)
      if (change.startsWith('privacy-')) {
        assert.deepEqual(fixture.setupMutations.map(mutation => mutation.step), ['setup_path', 'privacy_security'])
        assert.throws(() => setup.assertSucceeded(), { message: 'setup_api_access_1' })
        assert.equal(evidence.checks.setup_api_access_1.passed, false)
      } else {
        assert.deepEqual(fixture.setupMutations.map(mutation => mutation.step), ['setup_path'])
      }
      assert.equal(network.failed(), change === 'privacy-server-error')
      assert.throws(() => completeEvidence(evidence), { message: 'required_checks_failed' })
      assert.equal(evidence.passed, false)
      if (change === 'privacy-server-error') assert.equal(evidence.checks.live_errors_absent.passed, false)
    } finally { await browser.close(); await fixture.close() }
  })
}

test('paired browser preserves session and CSRF cookies despite foreign-session Set-Cookie through logout and rebootstrap', async () => {
  const first = await transportFixture('', 1)
  const second = await transportFixture('', 2)
  const root = mkdtempSync(join(tmpdir(), 'wp1-cookie-path-'))
  try {
    const { qualify } = await load()
    const evidence = await qualify({ instances: [first, second].map((fixture, index) => ({
      publicUrl: fixture.origin, sessionCookieName: `session_${index + 1}`, csrfCookieName: `csrf_${index + 1}`,
    })) }, join(root, 'evidence.json'))
    for (const name of ['cookie_attributes_1', 'cookie_attributes_2', 'hostile_inputs_1', 'hostile_inputs_2', 'foreign_session_refused', 'logout_isolated', 'rebootstrap']) {
      assert.equal(evidence.checks[name].passed, true)
    }
    assert.equal(evidence.passed, true)
    for (const [index, fixture] of [first, second].entries()) {
      assert.deepEqual(fixture.setupMutations.map(mutation => mutation.step), ['setup_path', 'privacy_security'])
      assert.deepEqual(fixture.foreignSessionRequests, [{
        cookie: `session_${index + 1}=public-fixture-session-${2 - index}`, csrfIssued: true, status: 401,
      }])
    }
    assert.deepEqual(first.logoutRequests.at(-1), { sessionUnchanged: true, csrfUnchanged: true, capturedToken: true, status: 200 })
    assert.deepEqual(first.profilesAfterForeignSession, [
      { sessionUnchanged: false, csrfUnchanged: true, status: 401 },
      { sessionUnchanged: true, csrfUnchanged: true, status: 200 },
    ])
    assert.deepEqual(second.profilesAfterForeignSession, [
      { sessionUnchanged: true, csrfUnchanged: true, status: 200 },
      { sessionUnchanged: true, csrfUnchanged: true, status: 200 },
    ])
    assert.ok(!JSON.stringify(evidence).includes('public-fixture-session'))
  } finally { await first.close(); await second.close(); rmSync(root, { recursive: true, force: true }) }
})

for (const change of ['', 'content', 'ws', 'reflect']) {
  test(`real browser transports ${change ? 'reject ' + change + ' contract loss' : 'verify content, cancellation and cookie MCP roundtrip'}`, async () => {
    const fixture = await transportFixture(change)
    const { chromium } = await import('playwright')
    const { qualifyTransports, createEvidence, createNetworkTracker } = await load()
    const browser = await chromium.launch({ headless: true })
    try {
      const context = await browser.newContext()
      await context.addCookies([{ name: 'fixture_session', value: 'public-fixture-session', domain: '127.0.0.1', path: '/api' }])
      const page = await context.newPage()
      await page.goto(fixture.origin)
      const tracker = createNetworkTracker(page, fixture.origin)
      const evidence = createEvidence()
      const action = () => qualifyTransports(page, context, { publicUrl: fixture.origin, sessionCookieName: 'fixture_session' }, 'public-fixture-csrf', tracker, evidence, 1)
      if (change) await assert.rejects(action(), { message: change === 'content' ? 'multipart_document_1' : change === 'ws' ? 'cookie_mcp_websocket_1' : 'hostile_inputs_1' })
      else {
        await action()
        assert.ok(Object.values(evidence.checks).every(check => check.passed))
        assert.deepEqual(fixture.observed(), { uploadSeen: true, streamClosed: true })
        assert.equal(tracker.failed(), false)
        await page.evaluate(() => fetch('/uncontrolled-failure').then(() => {}))
        assert.ok(!JSON.stringify(evidence).includes('public-fixture-session'))
      }
    } finally { await browser.close(); await fixture.close() }
  })
}

test('every mapped missing or false check refuses closure', async () => {
  const { createEvidence, completeEvidence, REQUIRED_CHECKS } = await load()
  for (const name of REQUIRED_CHECKS) {
    for (const missing of [true, false]) {
      const evidence = createEvidence()
      for (const required of REQUIRED_CHECKS) evidence.checks[required] = { passed: true }
      if (missing) delete evidence.checks[name]
      else evidence.checks[name].passed = false
      assert.throws(() => completeEvidence(evidence), { message: 'required_checks_failed' })
      assert.equal(evidence.passed, false)
    }
  }
})

test('controlled cancellation never exempts a second stream or other failed requests', async () => {
  const { EventEmitter } = await import('node:events')
  const { createNetworkTracker } = await load()
  const page = new EventEmitter()
  let failures = 0
  const tracker = createNetworkTracker(page, 'http://127.0.0.1:18081', () => { failures++ })
  const request = path => ({ url: () => 'http://127.0.0.1:18081' + path, failure: () => ({ errorText: 'net::ERR_ABORTED' }) })
  const first = request('/api/v1/notifications/stream')
  page.emit('request', first); tracker.armCancellation(); page.emit('requestfailed', first)
  assert.equal(tracker.failed(), false)
  const second = request('/api/v1/notifications/stream')
  page.emit('request', second); page.emit('requestfailed', second)
  assert.equal(tracker.failed(), true)
  page.emit('requestfailed', request('/api/v1/setup/first-run/state'))
  assert.equal(failures, 2)
})
