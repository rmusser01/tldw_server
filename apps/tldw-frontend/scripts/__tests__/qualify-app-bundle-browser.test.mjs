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
  { name: 'session_a', value: 'secret-session', domain: '127.0.0.1', path: '/', httpOnly: true, secure: false, sameSite: 'Lax', expires: 2000000000 },
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
  ['wrong path', x => { x[0].path = '/api' }],
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
          document.body.innerHTML = '<button>Continue</button>'
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
    const { createEvidence, createSetupResponseTracker, completeEvidence } = await load()
    assert.equal(typeof createSetupResponseTracker, 'function', 'setup response tracker is missing')
    const evidence = createEvidence()
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
  const { createEvidence, createSetupResponseTracker, completeEvidence } = await load()
  assert.equal(typeof createSetupResponseTracker, 'function', 'setup response tracker is missing')
  const evidence = createEvidence()
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

for (const [name, downExit, originalExit, expectedExit] of [
  ['cleanup succeeds', 0, 0, 0],
  ['cleanup fails after passing probe', 1, 0, 1],
  ['cleanup fails after original failure', 1, 7, 7],
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
        G2: false, G4: false, G12: false, checks: {} }))
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
      assert.equal(existsSync(join(owned, 'instance-1/instance/config.env')), downExit !== 0)
      const calls = readFileSync(join(root, 'calls.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line))
      assert.deepEqual(calls.map(args => args[args.indexOf('--project-name') + 1]), ['paired-1', 'paired-2'])
      assert.ok(calls.every(args => args.slice(-2).join() === 'down,--volumes'))
      assert.ok(!result.stderr.includes('secret-token'))
      const evidence = JSON.parse(readFileSync(evidencePath, 'utf8'))
      assert.ok(evidence.checks.owned_resources_removed, 'cleanup outcome is missing from evidence')
      assert.equal(evidence.checks.owned_resources_removed.passed, downExit === 0)
      if (downExit !== 0) {
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
