/**
 * Extended Playwright test fixtures for workflow tests
 */
import { test as base, expect, type Page, type CDPSession, type TestInfo } from "@playwright/test"
import { readFileSync, realpathSync, lstatSync } from 'node:fs';
import path from "node:path"
import { tmpdir } from 'node:os';
import { createHash } from "node:crypto"
import {
  seedAuth,
  TEST_CONFIG,
  isBenign,
  fetchWithApiKey
} from "./helpers"
import { startApiCapture, getCapturedApiCalls } from "./api-assertions"

const deferred43Marker = "3485bd8cbd37b38ae9bfec6e7b3e9cb37164cfb53667f53bf78aa5530f6b1d82"
const deferred43MeasuredTitles = new Map([
  ["e2e/ux-regression/notes-p0.spec.ts", "NE-04: 'Print / Save as PDF' prints the note without a pop-up error"],
  ["e2e/workflows/notes-studio-derived-note.spec.ts", "selects markdown, derives a Cornell studio note, and prints it"],
])
type Deferred43Payload = {
  schema: number; realm: number; source: string; timeOrigin: number
  count: number; faults: number; first: number; counts: number[]; rows: number[][]; order: number
}
const deferred43Keys = "count,counts,faults,first,order,realm,rows,schema,source,timeOrigin"
function validateDeferred43Payload(value: unknown, realm: number): value is Deferred43Payload {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const data = value as Deferred43Payload
  return Object.keys(value).sort().join(",") === deferred43Keys && data.schema === 43 && data.realm === realm && data.source === deferred43Marker && Number.isFinite(data.timeOrigin) && Number.isSafeInteger(data.count) && data.count > 0 && data.count <= (realm === 1 ? 4096 : 2048) && Number.isSafeInteger(data.faults) && data.faults >= 0 && Number.isSafeInteger(data.first) && data.first >= 0 && data.first <= data.count && data.order === 0 && Array.isArray(data.counts) && data.counts.length === 4 && data.counts.every(value => Number.isSafeInteger(value) && value >= 0 && value <= 512) && Array.isArray(data.rows) && data.rows.length === data.count && data.rows.every((row, index) => [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,40,41,42,43,44,45,46,50,51,52,53,54,55,56].includes(row[0]) && Array.isArray(row) && row.length === 24 && row[1] === index + 1 && row.every(value => typeof value === "number" && Number.isFinite(value)))
}
async function deferred43Bound<T>(operation: Promise<T>, deadline = Date.now() + 4000): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error("DEFERRED43_OBSERVATION_DEADLINE")), Math.max(1, deadline - Date.now()))
  })
  try { return await Promise.race([operation, timeout]) }
  finally { if (timer) clearTimeout(timer) }
}
async function prepareDeferred43(page: Page, testInfo: TestInfo): Promise<null | (() => Promise<void>)> {
  if (process.env.TLDW_DEFERRED43_SOURCE !== deferred43Marker) return null
  const relative = path.relative(process.cwd(), testInfo.file).split(path.sep).join("/")
  if (deferred43MeasuredTitles.get(relative) !== testInfo.title) return null
  let faults = 0
  let session: CDPSession | undefined
  let frameId = ""
  let documents = 0
  let setupExpired = false
  const setupDeadline = Date.now() + 4000
  const guardSetup = () => { if (setupExpired || Date.now() >= setupDeadline) throw new Error("DEFERRED43_SETUP_DEADLINE") }
  let contextLost = false
  type Context = { id: number; frameId: string; uniqueId: string; default: boolean; isolated: boolean; destroyed: boolean }
  const contexts = new Map<number, Context>()
  let app: Deferred43Payload | undefined, utility: Deferred43Payload | undefined
  let calibration: { before: number; after: number; local: number; origin: number } | undefined
  const setup = async () => {
    const reporter = process.env.PLAYWRIGHT_JSON_OUTPUT_NAME;
    if (!reporter || path.basename(reporter) !== 'playwright-results.json') {
      faults |= 1024;
      return;
    }
    const frontend = realpathSync(process.cwd());
    const runId = process.env.TLDW_LIVE_TIER_UAT_RUN_ID;
    if (
      !runId ||
      !/^[A-Za-z0-9._-]{1,128}$/.test(runId) ||
      path.resolve(reporter) !==
        path.join(frontend, 'test-results/live-tier-uat', runId, 'playwright-results.json')
    ) {
      faults |= 1024;
      return;
    }
    const receiptDirectory = path.join(
      tmpdir(),
      `tldw-onboarding-uat-${runId}`,
      'reports/live-tier-uat'
    );
    for (
      let ancestor = path.resolve(receiptDirectory);
      ancestor !== path.resolve(tmpdir());
      ancestor = path.dirname(ancestor)
    ) {
      const stats = lstatSync(ancestor);
      if (
        !ancestor.startsWith(`${path.resolve(tmpdir())}${path.sep}`) ||
        !stats.isDirectory() ||
        stats.isSymbolicLink() ||
        stats.uid !== process.getuid?.()
      ) {
        faults |= 1024;
        return;
      }
    }
    if (
      realpathSync(receiptDirectory) !==
      path.join(realpathSync(tmpdir()), `tldw-onboarding-uat-${runId}`, 'reports/live-tier-uat')
    ) {
      faults |= 1024;
      return;
    }
    const receiptPath = path.join(realpathSync(receiptDirectory), 'deferred43-source-receipt.json');
    const receiptStats = lstatSync(receiptPath);
    if (
      !receiptStats.isFile() ||
      receiptStats.isSymbolicLink() ||
      receiptStats.nlink !== 1 ||
      (receiptStats.mode & 0o777) !== 0o600 ||
      receiptStats.uid !== process.getuid?.()
    ) {
      faults |= 1024;
      return;
    }
    if (receiptStats.size > 1048576) {
      faults |= 1;
      return;
    }
    const receiptBytes = readFileSync(receiptPath, 'utf8');
    const receipt = JSON.parse(receiptBytes) as {
      source?: unknown;
      installed?: unknown;
      appSource?: unknown;
      records?: Array<{ candidateHash: string; target: string }>;
    };
    if (
      receipt.source !== deferred43Marker ||
      typeof receipt.appSource !== 'string' ||
      createHash('sha256').update(receipt.appSource).digest('hex') !==
        '2e4fd3e7e678b72cd41e7ac2bf9d374c5510ab803c698021994c6ba64e39ebc3' ||
      receipt.installed !== true ||
      typeof receipt.appSource !== 'string' ||
      Buffer.byteLength(receipt.appSource) > 65536 ||
      !Array.isArray(receipt.records) ||
      receipt.records.length !== 13 ||
      receipt.records.some(
        (record) =>
          typeof record.target !== 'string' ||
          !realpathSync(record.target).startsWith(`${path.dirname(frontend)}${path.sep}`) ||
          createHash('sha256').update(readFileSync(record.target)).digest('hex') !==
            record.candidateHash
      )
    ) {
      faults |= 1024;
      return;
    }
    const receiptReadback = lstatSync(receiptPath);
    if (
      !receiptReadback.isFile() ||
      receiptReadback.isSymbolicLink() ||
      receiptReadback.ino !== receiptStats.ino ||
      receiptReadback.dev !== receiptStats.dev ||
      receiptReadback.nlink !== 1 ||
      (receiptReadback.mode & 0o777) !== 0o600 ||
      receiptReadback.uid !== process.getuid?.() ||
      createHash('sha256').update(readFileSync(receiptPath)).digest('hex') !==
        createHash('sha256').update(receiptBytes).digest('hex')
    ) {
      faults |= 1024;
      return;
    }
    session = await page.context().newCDPSession(page);
    guardSetup();
    session.on(
      'Runtime.executionContextCreated',
      (event: {
        context: {
          id: number;
          uniqueId?: string;
          auxData?: { frameId?: string; isDefault?: boolean; type?: string };
        };
      }) => {
        const context = event.context;
        if (frameId && context.auxData?.frameId !== frameId) return;
        if (contexts.has(context.id) && contexts.get(context.id)?.uniqueId !== context.uniqueId) {
          faults |= 256;
          return;
        }
        if (
          !Number.isSafeInteger(context.id) ||
          !context.auxData?.frameId ||
          Buffer.byteLength(context.auxData.frameId) > 128 ||
          typeof context.uniqueId !== 'string' ||
          Buffer.byteLength(context.uniqueId) > 128 ||
          contexts.size >= 8
        ) {
          faults |= 256;
          return;
        }
        contexts.set(context.id, {
          id: context.id,
          frameId: context.auxData.frameId,
          uniqueId: context.uniqueId,
          default: context.auxData.isDefault === true,
          isolated: context.auxData.type === 'isolated',
          destroyed: false,
        });
        if (
          frameId &&
          context.auxData.frameId === frameId &&
          context.auxData.isDefault === true &&
          ++documents > 2
        )
          faults |= 512;
      }
    );
    session.on('Runtime.executionContextDestroyed', (event: { executionContextId: number }) => {
      const context = contexts.get(event.executionContextId);
      if (context) {
        context.destroyed = true;
        if (documents >= 2 && context.frameId === frameId && context.isolated) {
          contextLost = true;
          faults |= 256;
        }
      }
    });
    session.on('Runtime.executionContextsCleared', () => {
      for (const context of contexts.values()) {
        context.destroyed = true;
        if (documents >= 2 && context.frameId === frameId && context.isolated) {
          contextLost = true;
          faults |= 256;
        }
      }
    });
    await session.send('Page.enable');
    guardSetup();
    const tree = (await session.send('Page.getFrameTree')) as {
      frameTree: { frame: { id: string } };
    };
    guardSetup();
    frameId = tree.frameTree.frame.id;
    if (typeof frameId !== 'string' || frameId.length > 128) {
      faults |= 256;
      return;
    }
    await session.send('Runtime.enable');
    guardSetup();
    documents = Array.from(contexts.values()).filter(
      (context) => context.frameId === frameId && context.default
    ).length;
    if (documents > 2) faults |= 512;
    if (faults) return;
    await page.addInitScript({ content: receipt.appSource });
    guardSetup();
    const before = Date.now();
    const clock = await page.evaluate(() => ({
      local: performance.now(),
      origin: performance.timeOrigin,
    }));
    guardSetup();
    calibration = { before, after: Date.now(), local: clock.local, origin: clock.origin };
  };
  try { await deferred43Bound(setup(), setupDeadline) } catch { setupExpired = true; faults |= 4 }
  return async () => {
    const deadline = Date.now() + 4000
    const selected: Context[] = []
    const clocks: Array<{ realm: number; before: number; after: number; origin: number; local: number }> = []
    const drain = async () => {
      if (!session || !frameId) { faults |= 256; return }
      const candidates = Array.from(contexts.values()).filter(context => context.frameId === frameId && !context.destroyed && (context.default || context.isolated))
      for (const context of candidates) {
        if (Date.now() >= deadline) throw new Error("DEFERRED43_OBSERVATION_DEADLINE")
        const response = await session.send("Runtime.evaluate", { contextId: context.id, expression: `(() => { const available = []; for (const [key, realm] of [["__df43App", 1], ["__df43Utility", 2]]) { const entry = Object.getOwnPropertyDescriptor(globalThis, key), observer = entry?.value, shape = observer && Object.getOwnPropertyDescriptors(observer); if (!entry || entry.writable !== false || entry.configurable !== false || !Object.isFrozen(observer) || shape?.source?.value !== ${JSON.stringify(deferred43Marker)} || shape.realm?.value !== realm || !Object.values(shape).every(field => Object.hasOwn(field, "value")) || typeof shape.drain?.value !== "function" || typeof shape.fault?.value !== "function") { available.push(false); continue; } available.push(true); } return { app: available[0], utility: available[1] }; })()`, returnByValue: true }) as { result?: { value?: { app?: boolean; utility?: boolean } }; exceptionDetails?: unknown }
        if (response.exceptionDetails) { faults |= 256; continue }
        if (context.default && response.result?.value?.app === true || context.isolated && response.result?.value?.utility === true) selected.push(context)
      }
      if (selected.length !== 2 || selected.filter(context => context.default).length !== 1 || selected.filter(context => context.isolated).length !== 1) { faults |= 256; return }
      for (const context of selected) {
        if (Date.now() >= deadline) throw new Error("DEFERRED43_OBSERVATION_DEADLINE")
        if (context.destroyed) { contextLost = true; faults |= 256; continue }
        const key = context.default ? "__df43App" : "__df43Utility"
        const before = Date.now()
        const response = await session.send("Runtime.evaluate", { contextId: context.id, expression: `(() => { const key = ${JSON.stringify(key)}, realm = ${context.default ? 1 : 2}; const entry = Object.getOwnPropertyDescriptor(globalThis, key), observer = entry?.value, shape = observer && Object.getOwnPropertyDescriptors(observer); if (!entry || entry.writable !== false || entry.configurable !== false || !Object.isFrozen(observer) || shape?.source?.value !== ${JSON.stringify(deferred43Marker)} || shape.realm?.value !== realm || !Object.values(shape).every(field => Object.hasOwn(field, "value")) || typeof shape.drain?.value !== "function" || typeof shape.fault?.value !== "function") return null; return {payload: observer.drain(), clock: {origin: performance.timeOrigin, local: performance.now()}}; })()`, returnByValue: true }) as { result?: { value?: { payload?: unknown; clock?: { origin: number; local: number } } }; exceptionDetails?: unknown }
        if (Date.now() >= deadline) throw new Error("DEFERRED43_OBSERVATION_DEADLINE")
        const clock = response.result?.value?.clock
        if (!clock || !Number.isFinite(clock.origin) || !Number.isFinite(clock.local)) { faults |= 256; continue }
        clocks.push({ realm: context.default ? 1 : 2, before, after: Date.now(), origin: clock.origin, local: clock.local })
        if (context.destroyed || response.exceptionDetails || !validateDeferred43Payload(response.result?.value?.payload, context.default ? 1 : 2)) { contextLost = context.destroyed; faults |= 256; continue }
        if (context.default) app = response.result.value.payload as Deferred43Payload
        else utility = response.result.value.payload as Deferred43Payload
      }
    }
    try { await deferred43Bound(drain(), deadline) } catch { faults |= 256 }
    try { if (session) await deferred43Bound(session.detach(), deadline) } catch { faults |= 256 }
    try {
      const metadata = { schema: 43, frameId, documents, contextLost, faults, calibration, clocks, contexts: selected.map(context => ({ id: context.id, frameId: context.frameId, uniqueId: context.uniqueId, default: context.default, isolated: context.isolated, destroyed: context.destroyed })), ordering: "ORDER_UNBOUND" }
      if (Buffer.byteLength(JSON.stringify(metadata)) > 65536) faults |= 1
      for (const [name, payload, bound] of [["app", app, 1048576], ["utility", utility, 524288]] as const) {
        if (!payload) { faults |= 256; continue }
        const bytes = JSON.stringify(payload)
        if (Buffer.byteLength(bytes) > bound) { faults |= 1; continue }
        faults |= payload.faults
        await deferred43Bound(testInfo.attach(`deferred43-${name}.json`, { body: bytes, contentType: "application/json" }), deadline)
      }
      await deferred43Bound(testInfo.attach("deferred43-contexts.json", { body: JSON.stringify({ ...metadata, faults }), contentType: "application/json" }), deadline)
    } catch { faults |= 4 }
    const owners = new Map<number, number[]>()
    for (const row of app?.rows ?? []) if (row[0] === 1) owners.set(row[3], row)
    const roots = Array.from(owners.values()).filter(row => row[6] > 0 && (row[4] === 1 || row[4] === 2))
    const dropdown = roots.filter(row => row[4] === 1), tooltip = roots.filter(row => row[4] === 2)
    const exported = Array.from(owners.values()).filter(row => row[4] === 3 && row[6] > 0 && row[11] === dropdown[0]?.[3])
    const rolesBound = dropdown.length === 1 && tooltip.length === 1 && dropdown[0][6] === tooltip[0][6] && dropdown[0][7] !== tooltip[0][7] && exported.length === 1
    if (!rolesBound) faults |= 8
    const summary = { schema: 43, rolesBound, available: faults === 0 && !!app && !!utility, faults, appRows: app?.count ?? 0, utilityRows: utility?.count ?? 0, first: utility?.first ?? 0, appDigest: app ? createHash("sha256").update(JSON.stringify(app)).digest("hex") : null, utilityDigest: utility ? createHash("sha256").update(JSON.stringify(utility)).digest("hex") : null, order: 0 }
    const line = `DEFERRED43 ${JSON.stringify(summary)}\n`
    if (Buffer.byteLength(line) <= 4096) process.stdout.write(line)
  }
}


/**
 * Diagnostics data collected during page visits
 */
export interface DiagnosticsData {
  console: Array<{ type: string; text: string; location?: { url: string; lineNumber: number } }>
  pageErrors: Array<{ message: string; stack: string }>
  requestFailures: Array<{ url: string; errorText: string }>
}

/**
 * Server info for preflight checks
 */
export interface ServerInfo {
  available: boolean
  version?: string
  models?: string[]
  modelSource?: "metadata" | "providers"
}

/**
 * Extended test fixtures
 */
export interface WorkflowFixtures {
  /** Diagnostics data collected during test */
  diagnostics: DiagnosticsData
  /** Page pre-seeded with auth config */
  authedPage: Page
  /** Server availability info */
  serverInfo: ServerInfo
}

/**
 * Extended test with workflow fixtures
 */
export const test = base.extend<WorkflowFixtures>({
  // Collect diagnostics automatically
  diagnostics: async ({ page }, use) => {
    const data: DiagnosticsData = {
      console: [],
      pageErrors: [],
      requestFailures: []
    }

    page.on("console", (msg) => {
      const location = msg.location()
      data.console.push({
        type: msg.type(),
        text: msg.text(),
        location: location.url ? { url: location.url, lineNumber: location.lineNumber } : undefined
      })
    })

    page.on("pageerror", (err) => {
      data.pageErrors.push({
        message: err.message,
        stack: err.stack || ""
      })
    })

    page.on("requestfailed", (req) => {
      data.requestFailures.push({
        url: req.url(),
        errorText: req.failure()?.errorText || ""
      })
    })

    await use(data)
  },

  // Pre-seeded authenticated page
  authedPage: async ({ page }, use, testInfo) => {
    const appOrigin = new URL(TEST_CONFIG.webUrl).origin
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], {
      origin: appOrigin,
    })
    await seedAuth(page)
    startApiCapture(page)
    const deferred43Eligible = process.env.TLDW_DEFERRED43_SOURCE === deferred43Marker && deferred43MeasuredTitles.get(path.relative(process.cwd(), testInfo.file).split(path.sep).join("/")) === testInfo.title
    const deferred43Drain = deferred43Eligible ? await prepareDeferred43(page, testInfo) : null
    try {
      await use(page)
    } finally {
      if (deferred43Drain) try { await deferred43Drain() } catch { /* Diagnostics cannot mask the original fixture outcome. */ }
    }
    // Teardown: attach API call log on test failure for debugging
    if (testInfo.status !== "passed") {
      const apiLog = getCapturedApiCalls(page)
      if (apiLog.length > 0) {
        await testInfo.attach("api-calls.json", {
          body: JSON.stringify(apiLog, null, 2),
          contentType: "application/json",
        })
      }
    }
  },

  // Server availability check
  serverInfo: async ({}, use) => {
    const info: ServerInfo = { available: false }

    try {
      // Check server health
      const healthUrl = `${TEST_CONFIG.serverUrl}/api/v1/health`
      const healthRes = await fetchWithApiKey(healthUrl).catch(() => null)

      if (healthRes?.ok) {
        info.available = true
        const healthData = await healthRes.json().catch(() => ({}))
        info.version = healthData.version
      } else {
        // Try alternative health check
        const rootRes = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}/`).catch(() => null)
        info.available = rootRes?.ok ?? false
      }

      // Check available models. Prefer the richer metadata endpoint because the
      // providers endpoint includes catalog-only models that may not be runnable.
      if (info.available) {
        const metadataUrl =
          `${TEST_CONFIG.serverUrl}/api/v1/llm/models/metadata?type=chat&output_modality=text`
        const metadataRes = await fetchWithApiKey(metadataUrl).catch(() => null)
        if (metadataRes?.ok) {
          const metadataData = await metadataRes.json().catch(() => ({}))
          info.models = extractUsableModelIds(metadataData)
          info.modelSource = "metadata"
        }

        if (!info.models || info.models.length === 0) {
          const modelsUrl = `${TEST_CONFIG.serverUrl}/api/v1/llm/providers`
          const modelsRes = await fetchWithApiKey(modelsUrl).catch(() => null)
          if (modelsRes?.ok) {
            const modelsData = await modelsRes.json().catch(() => ({}))
            info.models = extractModelIds(modelsData)
            info.modelSource = "providers"
          }
        }
      }
    } catch {
      info.available = false
    }

    await use(info)
  }
})

export { expect }

export interface ModerationReviewItemsFixture {
  populated: any[]
  empty: any[]
  permissionDenied: { status: number; body: Record<string, unknown> }
  backendError: { status: number; body: Record<string, unknown> }
  partialData: any[]
  expiredUndo: any[]
  redactedContent: any[]
}

export function loadModerationReviewItemsFixture(): ModerationReviewItemsFixture {
  return JSON.parse(
    readFileSync(path.resolve(__dirname, "../fixtures/moderation-review-items.json"), "utf8")
  ) as ModerationReviewItemsFixture
}

/**
 * Extract configured model IDs from the metadata response.
 */
export function extractUsableModelIds(payload: any): string[] {
  const models: string[] = []
  const entries = Array.isArray(payload?.models)
    ? payload.models
    : Array.isArray(payload)
      ? payload
      : []

  for (const model of entries) {
    if (!isRunnableModelDescriptor(model, { requireChatTextModel: true })) continue
    if (typeof model === "string") {
      models.push(model)
      continue
    }
    const provider = normalizeModelProvider(model)
    const id = normalizeModelField(model?.id ?? model?.model ?? model?.name)
    if (!id) continue
    models.push(formatModelId(id, provider))
  }

  return [...new Set(models)]
}

/**
 * Extract model IDs from provider response.
 */
export function extractModelIds(payload: any): string[] {
  const models: string[] = []

  // Handle { providers: [{ name, models: [...] }, ...] } shape (actual API response)
  const providers = Array.isArray(payload?.providers)
    ? payload.providers
    : Array.isArray(payload)
      ? payload
      : []

  for (const provider of providers) {
    if (Array.isArray(provider?.models)) {
      const providerUsable = isRunnableModelDescriptor(provider)
      const providerName =
        normalizeModelProvider(provider) ??
        normalizeModelField(provider?.name ?? provider?.id ?? provider?.provider)
      for (const model of provider.models) {
        if (
          !providerUsable ||
          !isRunnableModelDescriptor(model, { requireChatTextModel: true })
        ) {
          continue
        }
        if (typeof model === "string") {
          models.push(formatModelId(model, providerName))
        } else {
          const providerOverride = normalizeModelProvider(model)
          const id = normalizeModelField(model?.id ?? model?.model ?? model?.name)
          if (id) models.push(formatModelId(id, providerOverride ?? providerName))
        }
      }
    }
  }

  // Fallback: payload.models direct array
  if (models.length === 0 && Array.isArray(payload?.models)) {
    for (const model of payload.models) {
      if (!isRunnableModelDescriptor(model, { requireChatTextModel: true })) continue
      if (typeof model === "string") {
        models.push(model)
      } else {
        const provider = normalizeModelProvider(model)
        const id = normalizeModelField(model?.id ?? model?.model ?? model?.name)
        if (id) models.push(formatModelId(id, provider))
      }
    }
  }

  return [...new Set(models)]
}

function isRunnableModelDescriptor(
  value: any,
  options: { requireChatTextModel?: boolean } = {}
): boolean {
  if (!value || typeof value === "string") return true
  if (options.requireChatTextModel && !isChatTextModelDescriptor(value)) return false
  const catalogOnly = firstBoolean(value, ["catalog_only", "catalogOnly", "is_catalog_only"])
  if (catalogOnly === true) return false
  const configured = firstBoolean(value, ["is_configured", "isConfigured", "configured"])
  if (configured === false) return false
  const providerConfigured = firstBoolean(value, [
    "provider_is_configured",
    "providerIsConfigured",
    "provider_configured",
    "providerConfigured"
  ])
  if (providerConfigured === false) return false
  const providerEnabled = firstBoolean(value, [
    "provider_enabled",
    "providerEnabled",
    "enabled"
  ])
  if (providerEnabled === false) return false
  const availability = normalizeModelField(
    value?.availability ?? value?.status ?? value?.readiness
  )
  if (
    availability === "unavailable" ||
    availability === "not-configured" ||
    availability === "not_configured"
  ) {
    return false
  }
  const readinessReasonCode = normalizeModelField(
    value?.readiness_reason_code ?? value?.readinessReasonCode
  )
  if (
    [
      "egress_blocked",
      "missing_credentials",
      "provider_not_configured",
      "unsupported_chat_provider"
    ].includes(readinessReasonCode || "")
  ) {
    return false
  }
  const deprecated = firstBoolean(value, ["deprecated", "is_deprecated", "isDeprecated"])
  if (deprecated === true) return false
  return true
}

function isChatTextModelDescriptor(value: any): boolean {
  const modelTypes = firstStringList(value, ["type", "model_type", "modelType"])
  if (modelTypes.length > 0 && !modelTypes.includes("chat")) return false

  const outputModalities = firstStringList(value, [
    "output_modality",
    "outputModalities",
    "output_modalities",
    "modalities_output"
  ])
  if (outputModalities.length > 0 && !outputModalities.includes("text")) return false

  return true
}

function firstBoolean(value: any, keys: string[]): boolean | null {
  for (const key of keys) {
    const field = value?.[key] ?? value?.details?.[key] ?? value?.metadata?.[key]
    if (typeof field === "boolean") return field
  }
  return null
}

function firstStringList(value: any, keys: string[]): string[] {
  for (const key of keys) {
    const field = value?.[key] ?? value?.details?.[key] ?? value?.metadata?.[key]
    const normalized = normalizeStringList(field)
    if (normalized.length > 0) return normalized
  }
  return []
}

function normalizeStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => normalizeModelField(item)?.toLowerCase())
      .filter((item): item is string => Boolean(item))
  }
  const normalized = normalizeModelField(value)
  return normalized ? [normalized.toLowerCase()] : []
}

function normalizeModelField(value: unknown): string | null {
  if (typeof value !== "string" && typeof value !== "number") return null
  const trimmed = String(value).trim()
  return trimmed.length > 0 ? trimmed : null
}

function normalizeModelProvider(value: any): string | null {
  return normalizeModelField(
    value?.chat_provider ??
      value?.chatProvider ??
      value?.api_provider ??
      value?.apiProvider ??
      value?.provider_key ??
      value?.providerKey ??
      value?.provider
  )
}

function formatModelId(id: string, provider: string | null): string {
  if (!provider) return id
  const prefix = `${provider}:`
  return id.toLowerCase().startsWith(prefix.toLowerCase()) ? id : `${prefix}${id}`
}

/**
 * Get first available model ID
 */
export function getFirstModelId(serverInfo: ServerInfo): string | null {
  return serverInfo.models?.[0] ?? null
}

/**
 * Filter diagnostics to only critical issues
 */
export function getCriticalIssues(diagnostics: DiagnosticsData): {
  pageErrors: Array<{ message: string; stack: string }>
  consoleErrors: Array<{ type: string; text: string }>
  requestFailures: Array<{ url: string; errorText: string }>
} {
  return {
    pageErrors: diagnostics.pageErrors.filter((e) => !isBenign(e.message)),
    consoleErrors: diagnostics.console.filter(
      (c) => c.type === "error" && !isBenign(c.text)
    ),
    requestFailures: diagnostics.requestFailures.filter(
      (r) => !isBenign(r.url) && !isBenign(r.errorText)
    )
  }
}

/**
 * Assert no critical page errors occurred
 */
export async function assertNoCriticalErrors(diagnostics: DiagnosticsData): Promise<void> {
  const critical = getCriticalIssues(diagnostics)

  if (critical.pageErrors.length > 0) {
    const messages = critical.pageErrors.map((e) => e.message).join("\n")
    throw new Error(`Uncaught page errors:\n${messages}`)
  }
}

/**
 * Skip test if server is not available
 */
export function skipIfServerUnavailable(serverInfo: ServerInfo): void {
  if (!serverInfo.available) {
    test.skip(true, "Server is not available")
  }
}

/**
 * Skip test if no models are available
 */
export function skipIfNoModels(serverInfo: ServerInfo): void {
  if (!serverInfo.models || serverInfo.models.length === 0) {
    test.skip(true, "No LLM models available")
  }
}
