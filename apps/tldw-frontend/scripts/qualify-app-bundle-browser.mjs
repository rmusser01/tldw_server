import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const CHECKS = [
  'browser_launch', 'anonymous_profile_refused', 'setup_rendered', 'manual_master_key_absent',
  'setup_interaction', 'browser_managed_bootstrap', 'cookie_attributes',
  'cookie_only_profile', 'same_context_instances', 'missing_csrf_refused',
  'foreign_csrf_refused', 'foreign_session_refused', 'logout_isolated',
  'rebootstrap', 'live_errors_absent', 'setup_api_access',
]
const PROFILE = '/api/v1/users/me/profile'
const LOGOUT = '/api/v1/auth/single-user/session'

// This is the entire input contract; additional fields (including secrets) fail.
export function validateInput(input) {
  const fail = () => { throw new Error('invalid_public_input') }
  if (!input || Object.keys(input).join() !== 'instances' || !Array.isArray(input.instances) || input.instances.length !== 2) fail()
  const origins = new Set()
  const names = new Set()
  for (const instance of input.instances) {
    if (!instance || Object.keys(instance).sort().join() !== 'csrfCookieName,publicUrl,sessionCookieName') fail()
    let url
    try { url = new URL(instance.publicUrl) } catch { fail() }
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || instance.publicUrl !== url.origin || origins.has(url.origin)) fail()
    origins.add(url.origin)
    for (const name of [instance.sessionCookieName, instance.csrfCookieName]) {
      if (typeof name !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(name) || /^__/.test(name) || names.has(name)) fail()
      names.add(name)
    }
  }
  return input.instances
}

export function assertCookiePolicy(cookies, instance, now = Date.now() / 1000) {
  for (const [name, httpOnly] of [[instance.sessionCookieName, true], [instance.csrfCookieName, false]]) {
    const matches = cookies.filter(cookie => cookie.name === name)
    const cookie = matches[0]
    if (matches.length !== 1 || !cookie.value || cookie.httpOnly !== httpOnly || cookie.domain !== '127.0.0.1' || cookie.path !== '/' || cookie.secure || cookie.sameSite !== 'Lax' || (cookie.expires !== -1 && cookie.expires <= now)) {
      throw new Error('cookie_attributes_failed')
    }
  }
}

export function createEvidence() {
  return { schema_version: 1, passed: false, planned_setup_complete: false,
    setup_scope: 'managed_connection_and_initial_wizard_only',
    G2: false, G4: false, G12: false, checks: {} }
}

// Keep setup access live through wizard interaction, reload, and browser shutdown.
export function createSetupResponseTracker(evidence, index, publicUrl) {
  if (![1, 2].includes(index)) throw new Error('invalid_check')
  const name = `setup_api_access_${index}`
  const start = performance.now()
  let observed = false
  let refused = false
  evidence.checks[name] = { passed: false, duration_ms: 0 }
  return {
    observe(url, status) {
      const parsed = new URL(url)
      if (parsed.origin !== publicUrl || !parsed.pathname.startsWith('/api/v1/setup/')) return
      observed = true
      if (!Number.isInteger(status) || status < 200 || status >= 400) refused = true
      evidence.checks[name] = { passed: observed && !refused, duration_ms: Math.round(performance.now() - start) }
      if (refused) {
        evidence.passed = false
        evidence.failure_code ||= name
      }
    },
    assertSucceeded() {
      if (!observed || refused) throw new Error(name)
    },
  }
}

export function completeEvidence(evidence) {
  evidence.passed = false
  if (evidence.checks.setup_api_access_1?.passed !== true || evidence.checks.setup_api_access_2?.passed !== true || Object.values(evidence.checks).some(check => check.passed !== true)) {
    evidence.failure_code ||= 'required_checks_failed'
    throw new Error('required_checks_failed')
  }
  evidence.passed = true
}

export async function recordCheck(evidence, name, action) {
  const base = name.replace(/_[12]$/, '')
  if (!CHECKS.includes(base)) throw new Error('invalid_check')
  const start = performance.now()
  try {
    await action()
    evidence.checks[name] = { passed: true, duration_ms: Math.round(performance.now() - start) }
  } catch {
    evidence.checks[name] = { passed: false, duration_ms: Math.round(performance.now() - start) }
    evidence.failure_code = base === 'manual_master_key_absent' ? 'manual_master_key_required' : name
    // Never propagate Playwright errors: they may contain headers/body/token values.
    throw new Error(name)
  }
}

export async function inspectManagedSetup(page) {
  const masterKey = page.getByLabel('API Key', { exact: true })
  const start = page.getByRole('button', { name: 'Set up in WebUI', exact: true })
  await page.getByText('Loading setup...', { exact: true }).waitFor({ state: 'hidden' })
  await masterKey.or(start).first().waitFor({ state: 'visible' })
  if (await masterKey.isVisible()) throw new Error('manual_master_key_required')
  await start.click()
  await page.getByRole('heading', { name: 'First-time setup', exact: true }).waitFor()
  await page.getByRole('button', { name: /Solo, Docker/ }).click()
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  // Initial wizard progression is partial setup, not complete G2 qualification.
  await page.getByRole('heading', { name: /chat provider|provider setup|choose.*provider/i }).first().waitFor()
}

async function browserFetch(page, path, method = 'GET', csrf) {
  return page.evaluate(async ({ path, method, csrf }) => {
    const headers = csrf === undefined ? {} : { 'X-CSRF-Token': csrf }
    const response = await fetch(path, { method, headers, credentials: 'same-origin', cache: 'no-store' })
    return response.status
  }, { path, method, csrf })
}

export async function qualify(input, outputPath) {
  // Validate before importing/launching Playwright or creating an evidence file.
  const instances = validateInput(input)
  const evidence = createEvidence()
  let browser
  let completed = false
  const check = (name, action) => recordCheck(evidence, name, action)
  try {
    await check('browser_launch', async () => {
      const { chromium } = await import('playwright')
      browser = await chromium.launch({ headless: true })
    })
    const context = await browser.newContext() // One fresh context; no storage preseed.
    context.setDefaultTimeout(30_000)
    const pages = []
    let liveError = false
    await check('anonymous_profile_refused', async () => {
      for (const instance of instances) {
        const response = await context.request.get(instance.publicUrl + PROFILE)
        assert.ok([401, 403].includes(response.status()))
      }
    })
    for (const [index, instance] of instances.entries()) {
      const instanceCheck = (name, action) => check(`${name}_${index + 1}`, action)
      const page = await context.newPage()
      page.on('pageerror', () => { liveError = true })
      page.on('requestfailed', () => { liveError = true })
      page.on('response', response => {
        if (response.status() >= 500) liveError = true
      })
      const setupResponses = createSetupResponseTracker(evidence, index + 1, instance.publicUrl)
      page.on('response', response => setupResponses.observe(response.url(), response.status()))
      const bootstrap = page.waitForResponse(response => response.url() === instance.publicUrl + '/api/_tldw-webui/session' && response.request().method() === 'POST')
      // Consume rejections immediately so a navigation failure cannot leak a raw error.
      const bootstrapStatus = bootstrap.then(response => response.status(), () => 0)
      pages.push(page)
      await instanceCheck('setup_rendered', async () => {
        const response = await page.goto(instance.publicUrl + '/setup', { waitUntil: 'domcontentloaded' })
        assert.equal(response.status(), 200)
        assert.equal(new URL(page.url()).origin, instance.publicUrl)
        await page.getByRole('heading', { name: /Setup|Choose where to set up tldw/ }).first().waitFor()
      })
      await instanceCheck('browser_managed_bootstrap', async () => {
        assert.ok([200, 204].includes(await bootstrapStatus))
      })
      await instanceCheck('manual_master_key_absent', async () => {
        await page.getByText('Loading setup...', { exact: true }).waitFor({ state: 'hidden' })
        const masterKey = page.getByLabel('API Key', { exact: true })
        await masterKey.or(page.getByRole('button', { name: 'Set up in WebUI', exact: true })).first().waitFor({ state: 'visible' })
        assert.equal(await masterKey.isVisible(), false)
      })
      await instanceCheck('cookie_attributes', async () => assertCookiePolicy(await context.cookies(instance.publicUrl), instance))
      await instanceCheck('cookie_only_profile', async () => assert.equal(await browserFetch(page, PROFILE), 200))
      await instanceCheck('setup_interaction', async () => inspectManagedSetup(page))
      await instanceCheck('setup_api_access', async () => setupResponses.assertSucceeded())
    }
    await check('same_context_instances', async () => {
      assert.equal(context.pages().length, 2)
      for (const page of pages) assert.equal(await browserFetch(page, PROFILE), 200)
    })
    const cookies = await context.cookies()
    const csrf = instances.map(instance => cookies.find(cookie => cookie.name === instance.csrfCookieName).value)
    const sessions = instances.map(instance => cookies.find(cookie => cookie.name === instance.sessionCookieName).value)
    await check('missing_csrf_refused', async () => {
      for (const page of pages) assert.equal(await browserFetch(page, LOGOUT, 'DELETE'), 403)
    })
    await check('foreign_csrf_refused', async () => {
      for (let i = 0; i < 2; i++) assert.equal(await browserFetch(pages[i], LOGOUT, 'DELETE', csrf[1 - i]), 403)
    })
    await check('foreign_session_refused', async () => {
      for (let i = 0; i < 2; i++) {
        // Forge only the hostile request; never preseed or mutate browser storage.
        const response = await context.request.get(instances[i].publicUrl + PROFILE, {
          headers: { Cookie: `${instances[i].sessionCookieName}=${sessions[1 - i]}` },
        })
        assert.ok([401, 403].includes(response.status()))
      }
    })
    await check('logout_isolated', async () => {
      assert.equal(await browserFetch(pages[0], LOGOUT, 'DELETE', csrf[0]), 200)
      assert.ok([401, 403].includes(await browserFetch(pages[0], PROFILE)))
      assert.equal(await browserFetch(pages[1], PROFILE), 200)
    })
    await check('rebootstrap', async () => {
      const response = pages[0].waitForResponse(response => response.url() === instances[0].publicUrl + '/api/_tldw-webui/session' && response.request().method() === 'POST').then(response => response.status(), () => 0)
      await pages[0].reload({ waitUntil: 'domcontentloaded' })
      assert.ok([200, 204].includes(await response))
      assertCookiePolicy(await context.cookies(instances[0].publicUrl), instances[0])
      assert.equal(await browserFetch(pages[0], PROFILE), 200)
      assert.equal(await browserFetch(pages[1], PROFILE), 200)
    })
    await check('live_errors_absent', async () => assert.equal(liveError, false))
    completeEvidence(evidence)
    completed = true
    return evidence
  } finally {
    if (browser) await browser.close().catch(() => {})
    try {
      // A late setup failure during shutdown must still reject the completed run.
      if (completed) completeEvidence(evidence)
    } finally {
      writeFileSync(outputPath, JSON.stringify(evidence, null, 2) + '\n', { mode: 0o600 })
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    if (process.argv.length !== 4) throw new Error('invalid_public_input')
    await qualify(JSON.parse(readFileSync(process.argv[2], 'utf8')), process.argv[3])
    console.log('Live browser checklist passed; full G2/G4/G12 qualification remains open.')
  } catch {
    console.error('Live browser qualification failed; inspect the bounded public checklist.')
    process.exitCode = 1
  }
}
