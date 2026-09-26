import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
import { request } from 'node:http'

const CHECKS = [
  'browser_launch', 'anonymous_profile_refused', 'setup_rendered', 'manual_master_key_absent',
  'setup_interaction', 'browser_managed_bootstrap', 'cookie_attributes',
  'cookie_only_profile', 'same_context_instances', 'missing_csrf_refused',
  'foreign_csrf_refused', 'foreign_session_refused', 'logout_isolated',
  'rebootstrap', 'live_errors_absent', 'setup_api_access', 'browser_shutdown',
  'published_documentation', 'public_redirect', 'multipart_document',
  'notification_sse_cancel', 'cookie_mcp_websocket', 'hostile_inputs',
]
const INSTANCE_CHECKS = ['setup_rendered', 'manual_master_key_absent', 'setup_interaction',
  'browser_managed_bootstrap', 'cookie_attributes', 'cookie_only_profile', 'setup_api_access',
  'published_documentation', 'public_redirect', 'multipart_document',
  'notification_sse_cancel', 'cookie_mcp_websocket', 'hostile_inputs']
export const REQUIRED_CHECKS = CHECKS.flatMap(name => INSTANCE_CHECKS.includes(name) ? [`${name}_1`, `${name}_2`] : [name])
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
  for (const [name, httpOnly, path] of [[instance.sessionCookieName, true, '/api'], [instance.csrfCookieName, false, '/']]) {
    const matches = cookies.filter(cookie => cookie.name === name)
    const cookie = matches[0]
    if (matches.length !== 1 || !cookie.value || cookie.httpOnly !== httpOnly || cookie.domain !== '127.0.0.1' || cookie.path !== path || cookie.secure || cookie.sameSite !== 'Lax' || (cookie.expires !== -1 && cookie.expires <= now)) {
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
  if (evidence.failure_code || REQUIRED_CHECKS.some(name => evidence.checks[name]?.passed !== true) || Object.values(evidence.checks).some(check => check.passed !== true)) {
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
    evidence.passed = false
    evidence.failure_code ||= base === 'manual_master_key_absent' ? 'manual_master_key_required' : name
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
  await page.getByRole('heading', { name: 'Privacy and security', exact: true }).waitFor()
  await page.getByRole('checkbox', { name: 'I understand local or remote setup access and provider secret storage.', exact: true }).check()
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  // Initial wizard progression is partial setup, not complete G2 qualification.
  await page.getByRole('heading', { name: /chat provider|provider setup|choose.*provider/i }).first().waitFor()
}

async function browserFetch(page, path, method = 'GET', csrf) {
  return page.evaluate(async ({ path, method, csrf }) => {
    const headers = csrf === undefined ? {} : { 'X-CSRF-Token': csrf }
    const response = await fetch(path, { method, headers, credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(10_000) })
    await response.arrayBuffer()
    return response.status
  }, { path, method, csrf })
}

// Only this exact observed stream request may be cancelled after its first chunk.
export function createNetworkTracker(page, publicUrl, onFailure = () => {}) {
  let failed = false
  let streamRequest
  let controlledRequest
  const fail = () => { failed = true; onFailure() }
  page.on('pageerror', fail)
  page.on('request', req => {
    if (req.url() === publicUrl + '/api/v1/notifications/stream') streamRequest = req
  })
  page.on('requestfailed', req => {
    if (req === controlledRequest && req.failure()?.errorText === 'net::ERR_ABORTED') controlledRequest = undefined
    else fail()
  })
  page.on('response', response => { if (response.status() >= 500) fail() })
  return { failed: () => failed, armCancellation: () => {
    if (!streamRequest) throw new Error('notification_sse_cancel')
    controlledRequest = streamRequest
  } }
}

export async function closeBrowser(browser, evidence) {
  await recordCheck(evidence, 'browser_shutdown', async () => {
    let timer
    try {
      await Promise.race([browser.close(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('browser_shutdown')), 10_000)
      })])
    } finally { clearTimeout(timer) }
  })
}

// Native HTTP is required for adversarial Host/Origin/upgrade headers forbidden
// by browsers. Return only bounded status/boolean fields; never response data.
async function hostileRequest(url, headers, method = 'GET', upgrade = false) {
  return new Promise((resolve, reject) => {
    const req = request(url, { method, headers, agent: false }, res => {
      let size = 0
      const chunks = []
      res.on('data', chunk => {
        size += chunk.length
        chunks.push(chunk)
        if (size > 1_048_576) req.destroy(new Error('hostile_inputs'))
      })
      res.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8')
        const reflected = body.includes('public-forged-hop') || body.includes('hostile.invalid') ||
          /SINGLE_USER_API_KEY|TLDW_GATEWAY_HOP_SECRET/.test(body) ||
          Object.keys(res.headers).some(name => /^(x-api-key|authorization|x-tldw-)/i.test(name))
        resolve({ status: res.statusCode, cookie: Boolean(res.headers['set-cookie']), reflected })
      })
      res.on('error', () => reject(new Error('hostile_inputs')))
    })
    req.on('upgrade', (_res, socket) => { socket.destroy(); resolve({ status: 101, cookie: false }) })
    req.setTimeout(10_000, () => req.destroy(new Error('hostile_inputs')))
    req.on('error', () => reject(new Error('hostile_inputs')))
    if (upgrade) req.setHeader('Connection', 'Upgrade')
    req.end()
  })
}

export async function qualifyTransports(page, context, instance, csrf, tracker, evidence, index) {
  const check = (name, action) => recordCheck(evidence, `${name}_${index}`, action)
  await check('published_documentation', async () => {
    const result = await page.evaluate(async () => {
      const manifest = await fetch('/api/documentation/manifest', { signal: AbortSignal.timeout(10_000) })
      const data = await manifest.json()
      const content = await fetch('/api/documentation/content?source=server&relativePath=API-related%2FAuthNZ-API-Guide.md', { signal: AbortSignal.timeout(10_000) })
      const guide = await content.json()
      const traversal = await fetch('/api/documentation/content?source=server&relativePath=..%2FDesign%2Fprivate.md', { signal: AbortSignal.timeout(10_000) })
      await traversal.arrayBuffer()
      return manifest.status === 200 && Array.isArray(data.docsBySource?.server) && data.docsBySource.server.length > 0 &&
        data.docsBySource.server.some(entry => entry.source === 'server' && entry.relativePath === 'API-related/AuthNZ-API-Guide.md') &&
        content.status === 200 && guide.content?.startsWith('# AuthNZ API Guide\n') && traversal.status === 400
    })
    assert.equal(result, true)
  })
  await check('public_redirect', async () => {
    const response = await context.request.get(instance.publicUrl + '/api/v1/health/live/', { maxRedirects: 0 })
    assert.equal(response.status(), 307)
    assert.equal(response.headers().location, instance.publicUrl + '/api/v1/health/live')
    const final = await context.request.get(response.headers().location)
    assert.equal(final.status(), 200)
    assert.equal(new URL(final.url()).origin, instance.publicUrl)
  })
  await check('multipart_document', async () => {
    const passed = await page.evaluate(async csrf => {
      const sentinel = '# WP1 qualification\nA harmless public upload sentinel.\n'
      const data = new FormData()
      data.append('files', new File([sentinel], 'qualification.md', { type: 'text/markdown' }))
      data.append('perform_analysis', 'false'); data.append('perform_chunking', 'false')
      const response = await fetch('/api/v1/media/process-documents', {
        method: 'POST', body: data, headers: { 'X-CSRF-Token': csrf }, signal: AbortSignal.timeout(30_000),
      })
      const body = await response.json()
      const item = body.results?.[0]
      return response.status === 200 && body.results?.length === 1 && (!body.errors || body.errors.length === 0) &&
        ['success', 'warning'].includes(item?.status?.toLowerCase()) && typeof item.content === 'string' &&
        item.content.includes('A harmless public upload sentinel.')
    }, csrf)
    assert.equal(passed, true)
  })
  await check('notification_sse_cancel', async () => {
    await page.exposeFunction('__tldwQualificationCancel', tracker.armCancellation)
    const passed = await page.evaluate(async () => {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 10_000)
      let reader
      try {
        const response = await fetch('/api/v1/notifications/stream', { signal: controller.signal })
        if (response.status !== 200 || !response.headers.get('content-type')?.startsWith('text/event-stream')) return false
        reader = response.body.getReader()
        const first = await reader.read()
        if (first.done || !first.value?.length) return false
        await window.__tldwQualificationCancel()
        await reader.cancel()
        controller.abort()
        return (await reader.read()).done === true
      } finally { clearTimeout(timer); controller.abort(); reader?.releaseLock() }
    })
    assert.equal(passed, true)
  })
  await check('cookie_mcp_websocket', async () => {
    const passed = await page.evaluate(async () => {
      return new Promise(resolve => {
        const socket = new WebSocket(location.origin.replace('http:', 'ws:') + '/api/v1/mcp/ws?client_id=wp1-public-qualification')
        let result = false
        const timer = setTimeout(() => { socket.close(); resolve(false) }, 10_000)
        socket.onopen = () => socket.send(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize',
          params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'wp1-public-qualification', version: '1.0' } } }))
        socket.onmessage = event => {
          try {
            const body = JSON.parse(event.data)
            if (body.id !== 1) return
            result = body.jsonrpc === '2.0' && body.result?.protocolVersion === '2024-11-05' && body.result?.serverInfo?.name === 'tldw-mcp-unified'
          } catch { result = false }
          socket.close()
        }
        socket.onerror = () => { result = false; socket.close() }
        socket.onclose = () => { clearTimeout(timer); resolve(result) }
      })
    })
    assert.equal(passed, true)
  })
  await check('hostile_inputs', async () => {
    const origin = instance.publicUrl
    const cookies = await context.cookies(origin + '/api')
    const session = cookies.find(cookie => cookie.name === instance.sessionCookieName)
    assert.ok(session)
    const cookie = `${instance.sessionCookieName}=${session.value}`
    const forwarding = { Forwarded: 'for=203.0.113.9;host=hostile.invalid;proto=https',
      'X-Real-IP': '203.0.113.9', 'X-Forwarded-For': '203.0.113.9', 'X-Forwarded-Host': 'hostile.invalid',
      'X-Forwarded-Port': '443', 'X-Forwarded-Proto': 'https', 'X-Tldw-Gateway-Hop': 'public-forged-hop' }
    const legitimate = await hostileRequest(origin + '/api/v1/setup/first-run/state', { ...forwarding, Origin: origin, Cookie: cookie })
    assert.equal(legitimate.status, 200); assert.equal(legitimate.reflected, false)
    assert.equal((await hostileRequest(origin + '/_tldw/status', { Host: 'hostile.invalid' })).status, 403)
    const refused = await hostileRequest(origin + '/api/_tldw-webui/session', { ...forwarding, Origin: 'http://hostile.invalid' }, 'POST')
    assert.equal(refused.status, 403); assert.equal(refused.cookie, false); assert.equal(refused.reflected, false)
    const upgrade = { Upgrade: 'websocket', 'Sec-WebSocket-Version': '13', 'Sec-WebSocket-Key': 'cHVibGljLWZpeHR1cmUxNg==' }
    for (const headers of [{ ...upgrade, Origin: origin }, { ...upgrade, Origin: 'http://hostile.invalid', Cookie: cookie }]) {
      assert.ok([401, 403].includes((await hostileRequest(origin + '/api/v1/mcp/ws?client_id=wp1-public-refusal', headers, 'GET', true)).status))
    }
  })
}

export async function qualify(input, outputPath) {
  // Validate before importing/launching Playwright or creating an evidence file.
  const instances = validateInput(input)
  const evidence = createEvidence()
  let browser
  let completed = false
  let failure
  const check = (name, action) => recordCheck(evidence, name, action)
  try {
    await check('browser_launch', async () => {
      const { chromium } = await import('playwright')
      browser = await chromium.launch({ headless: true })
    })
    const context = await browser.newContext() // One fresh context; no storage preseed.
    context.setDefaultTimeout(30_000)
    const pages = []
    const trackers = []
    await check('anonymous_profile_refused', async () => {
      for (const instance of instances) {
        const response = await context.request.get(instance.publicUrl + PROFILE)
        assert.ok([401, 403].includes(response.status()))
      }
    })
    for (const [index, instance] of instances.entries()) {
      const instanceCheck = (name, action) => check(`${name}_${index + 1}`, action)
      const page = await context.newPage()
      trackers.push(createNetworkTracker(page, instance.publicUrl, () => {
        evidence.checks.live_errors_absent = { passed: false }
        evidence.failure_code ||= 'live_errors_absent'
        evidence.passed = false
      }))
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
      await instanceCheck('cookie_attributes', async () => assertCookiePolicy(await context.cookies(instance.publicUrl + '/api'), instance))
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
    for (let i = 0; i < 2; i++) await qualifyTransports(pages[i], context, instances[i], csrf[i], trackers[i], evidence, i + 1)
    await check('missing_csrf_refused', async () => {
      for (const page of pages) assert.equal(await browserFetch(page, LOGOUT, 'DELETE'), 403)
    })
    await check('foreign_csrf_refused', async () => {
      for (let i = 0; i < 2; i++) assert.equal(await browserFetch(pages[i], LOGOUT, 'DELETE', csrf[1 - i]), 403)
    })
    await check('foreign_session_refused', async () => {
      for (let i = 0; i < 2; i++) {
        // Forge only the hostile request; never preseed or mutate browser storage.
        // Native response cookies cannot replace the live browser's CSRF cookie.
        const response = await hostileRequest(instances[i].publicUrl + PROFILE, {
          Cookie: `${instances[i].sessionCookieName}=${sessions[1 - i]}`,
        })
        assert.ok([401, 403].includes(response.status))
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
      assertCookiePolicy(await context.cookies(instances[0].publicUrl + '/api'), instances[0])
      assert.equal(await browserFetch(pages[0], PROFILE), 200)
      assert.equal(await browserFetch(pages[1], PROFILE), 200)
    })
    await check('live_errors_absent', async () => assert.equal(trackers.some(tracker => tracker.failed()), false))
    completed = true
  } catch {
    evidence.failure_code ||= 'required_checks_failed'
    failure = new Error(evidence.failure_code)
  } finally {
    if (browser) {
      try { await closeBrowser(browser, evidence) } catch (error) { failure ||= error }
    }
    try {
      // Late setup or network failure during shutdown must reject completion.
      if (completed && !failure) completeEvidence(evidence)
    } catch (error) { failure ||= error }
    writeFileSync(outputPath, JSON.stringify(evidence, null, 2) + '\n', { mode: 0o600 })
  }
  if (failure) throw failure
  return evidence
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    if (process.argv.length !== 4) throw new Error('invalid_public_input')
    await qualify(JSON.parse(readFileSync(process.argv[2], 'utf8')), process.argv[3])
    console.log('Live browser transport checklist passed; fixture cleanup and candidate gates remain separate.')
  } catch {
    console.error('Live browser qualification failed; inspect the bounded public checklist.')
    process.exitCode = 1
  }
}
