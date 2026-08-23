#!/usr/bin/env node

import crypto from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { pathToFileURL } from "node:url"
import { chromium } from "playwright"

const REQUIRED_ENV_KEYS = [
  "TLDW_E2E_SERVER_URL",
  "TLDW_WEB_URL",
  "TLDW_CDP_URL",
  "TLDW_SHARED_UAT_ADMIN_USERNAME",
  "TLDW_SHARED_UAT_ADMIN_PASSWORD",
  "TLDW_SHARED_UAT_FIXTURE_PASSWORD",
]

const PERSONAS = ["owner", "member", "nonmember"]
const DEFAULT_EVIDENCE_DIR = path.resolve(
  process.cwd(),
  "../../Docs/Reviews/assets/2026-08-21-shared-research-workspace-recipient-uat"
)
const SCREENSHOT_NAMES = {
  desktopGroundedAnswer: "desktop-grounded-answer.png",
  desktopSharedWorkspace: "desktop-shared-workspace.png",
  mobileSourcePreview: "mobile-source-preview.png",
  mobileSharedWorkspace: "mobile-shared-workspace.png",
  revokedShare: "revoked-share.png",
}
const ACCEPTANCE_NAMES = [
  "allSourcesGrounded",
  "blockedRevokedPreview",
  "citationPreview",
  "contextIsolation",
  "historyAfterReload",
  "malformedNeutralFailure",
  "memberSharedIsolation",
  "mobileResponsive",
  "noExtraBannerBars",
  "nonmemberNeutralFailure",
  "ownerRecipientView",
  "recipientChatVisibleInChats",
  "revocationFailClosed",
  "sentinelsExcluded",
  "subsetGrounded",
]
const OWNER_SENTINEL = "OWNER-UNRELATED-SENTINEL-7F3C9D"
const RECIPIENT_SENTINEL = "RECIPIENT-LOCAL-SENTINEL-4A8E2B"
const SOURCE_ONE_FACT = "Amber protocol token: AMBER-SIGNED-DATE-2024-03-17."
const SOURCE_TWO_FACT = "Cobalt trial token: COBALT-PARTICIPANTS-43."

export const buildAllSourcesQuestion = ({ amberTitle, cobaltTitle }) =>
  `From the source titled "${amberTitle}", report the protocol's signed-date token. ` +
  `From the source titled "${cobaltTitle}", report the trial's participant-count token. ` +
  "Return both exact values from the evidence and include a citation for each source."

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds))

export const ensureLocatorVisibleInViewport = async (locator, label) => {
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      await locator.waitFor({ state: "attached", timeout: 30_000 })
      await locator.scrollIntoViewIfNeeded()
      await locator.waitFor({ state: "visible", timeout: 30_000 })
      const inViewport = await locator.evaluate((element) => {
        const rect = element.getBoundingClientRect()
        return (
          rect.bottom > 0 &&
          rect.right > 0 &&
          rect.top < window.innerHeight &&
          rect.left < window.innerWidth
        )
      })
      if (!inViewport) throw new Error(`${label} is outside the viewport`)
      return
    } catch (error) {
      const detached = /not attached to the DOM/i.test(
        error instanceof Error ? error.message : String(error)
      )
      if (!detached || attempt === 3) throw error
    }
  }
}

export const waitForCheckboxState = async (
  locator,
  checked,
  { pollIntervalMs = 50, timeoutMs = 30_000 } = {}
) => {
  if ((await locator.isChecked()) === checked) return
  await locator.click()
  const deadline = Date.now() + timeoutMs
  do {
    if ((await locator.isChecked()) === checked) return
    await sleep(pollIntervalMs)
  } while (Date.now() <= deadline)
  throw new Error(`Checkbox did not become ${checked ? "checked" : "unchecked"}`)
}

const normalizeOrigin = (value, key) => {
  let parsed
  try {
    parsed = new URL(String(value || ""))
  } catch {
    throw new Error(`${key} must be an absolute HTTP URL`)
  }
  if (!/^https?:$/.test(parsed.protocol)) {
    throw new Error(`${key} must use http or https`)
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error(`${key} must not contain credentials, query, or fragment`)
  }
  return parsed.href.replace(/\/$/, "")
}

export const buildSharedUatConfig = ({ env = process.env } = {}) => {
  for (const key of REQUIRED_ENV_KEYS) {
    if (!String(env[key] || "").trim()) {
      throw new Error(`Missing required environment value: ${key}`)
    }
  }

  const runId = String(
    env.TLDW_SHARED_UAT_RUN_ID ||
      `${Date.now().toString(36)}-${crypto.randomBytes(4).toString("hex")}`
  ).replace(/[^a-zA-Z0-9_-]/g, "-")

  return {
    adminPassword: String(env.TLDW_SHARED_UAT_ADMIN_PASSWORD),
    adminUsername: String(env.TLDW_SHARED_UAT_ADMIN_USERNAME).trim(),
    apiUrl: normalizeOrigin(env.TLDW_E2E_SERVER_URL, "TLDW_E2E_SERVER_URL"),
    cdpUrl: normalizeOrigin(env.TLDW_CDP_URL, "TLDW_CDP_URL"),
    cleanupManifestPath: path.resolve(
      String(
        env.TLDW_SHARED_UAT_CLEANUP_MANIFEST ||
          path.join(os.tmpdir(), `tldw-shared-recipient-uat-${runId}-cleanup.json`)
      )
    ),
    evidenceDir: path.resolve(String(env.TLDW_SHARED_UAT_EVIDENCE_DIR || DEFAULT_EVIDENCE_DIR)),
    fixturePassword: String(env.TLDW_SHARED_UAT_FIXTURE_PASSWORD),
    llamaHealthUrl: "http://127.0.0.1:9099/health",
    personas: [...PERSONAS],
    providerProbeUrl: normalizeOrigin(
      env.TLDW_SHARED_UAT_PROVIDER_PROBE_URL || "http://127.0.0.1:19099",
      "TLDW_SHARED_UAT_PROVIDER_PROBE_URL"
    ),
    runId,
    usernamePrefixes: {
      member: String(env.TLDW_SHARED_UAT_MEMBER_PREFIX || "uat-member"),
      nonmember: String(env.TLDW_SHARED_UAT_NONMEMBER_PREFIX || "uat-nonmember"),
      owner: String(env.TLDW_SHARED_UAT_OWNER_PREFIX || "uat-owner"),
    },
    webUrl: normalizeOrigin(env.TLDW_WEB_URL, "TLDW_WEB_URL"),
  }
}

export const selectEffectiveTarget = ({ llamaHealthy, targets }) => {
  const ready = (Array.isArray(targets) ? targets : [])
    .filter((target) => target?.ready === true)
    .map((target) => ({
      model: String(target.model || "").trim(),
      provider: String(target.provider || "").trim(),
    }))
    .filter((target) => target.model && target.provider)
  if (llamaHealthy) {
    const llama = ready.find((target) => /^(?:llama(?:\.cpp)?|local-llm)$/i.test(target.provider))
    if (llama) return llama
  }
  return ready[0] || null
}

const requestPath = (entry) => {
  try {
    return new URL(entry.url).pathname.toLowerCase()
  } catch {
    return String(entry.url || "").toLowerCase()
  }
}

const AMBIENT_SAFE_GET_PATHS = new Set([
  "/api/_tldw-webui/runtime-config",
  "/api/v1/persona/profiles",
  "/api/v1/auth/me",
  "/api/v1/health",
  "/api/v1/health/live",
  "/api/v1/notifications",
  "/api/v1/notifications/stream",
  "/api/v1/notifications/unread-count",
  "/api/v1/rag/health",
  "/api/v1/llm/providers",
  "/api/v1/llm/models/metadata",
  "/api/v1/users/me/profile",
])

const allowedAmbientSafeGet = (entry) =>
  String(entry.method || "GET").toUpperCase() === "GET" &&
  AMBIENT_SAFE_GET_PATHS.has(requestPath(entry))

const allowedOwnerRevocationRequest = (entry) => {
  if (entry.context !== "owner-revocation") return false
  const pathname = requestPath(entry)
  const method = String(entry.method || "GET").toUpperCase()
  if (!pathname.startsWith("/api/")) return true
  if (allowedAmbientSafeGet(entry)) return true
  if (method === "GET") {
    return [
      /^\/api\/v1\/workspaces\/[^/]+\/context$/,
      /^\/api\/v1\/sharing\/workspaces\/[^/]+\/shares$/,
      /^\/api\/v1\/sharing\/tokens$/,
    ].some((pattern) => pattern.test(pathname))
  }
  return method === "DELETE" && /^\/api\/v1\/sharing\/shares\/[^/]+$/.test(pathname)
}

const forbiddenSharedRequest = (entry) => {
  const pathname = requestPath(entry)
  const method = String(entry.method || "GET").toUpperCase()
  if (allowedOwnerRevocationRequest(entry)) return null
  if (/\/api\/v1\/sharing\/shared-with-me\/[^/]+\/(?:media|full-media)(?:\/|$)/.test(pathname)) {
    return "removed_full_media"
  }
  if (/\/api\/v1\/(?:workspaces|research-workspace)(?:\/|$)/.test(pathname)) {
    return "local_workspace"
  }
  if (/\/api\/v1\/(?:prompt-)?studio(?:\/|$)/.test(pathname)) return "studio"
  if (/\/api\/v1\/notes?(?:\/|$)/.test(pathname)) return "notes"
  if (/\/api\/v1\/(?:mcp|acp|sandbox|artifacts?)(?:\/|$)/.test(pathname)) {
    return "local_tool"
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && /\/sources(?:\/|$)/.test(pathname)) {
    return "source_mutation"
  }
  if (
    !["GET", "HEAD", "OPTIONS"].includes(method) &&
    /\/api\/v1\/(?:media|ingestion|web-clips?|web-clipper|clips?|capture)(?:\/|$)/.test(pathname)
  ) {
    return "extension_writable_destination"
  }
  return null
}

const allowedSharedRequest = (entry) => {
  const pathname = requestPath(entry)
  if (entry.context === "member-chats") {
    if (!pathname.startsWith("/api/") && pathname !== "/openapi.json") return true
    if (allowedAmbientSafeGet(entry)) return true
    const method = String(entry.method || "GET").toUpperCase()
    if (method === "POST" && pathname === "/api/v1/rag/feedback/implicit") {
      return true
    }
    const safeReadPatterns = [
      /^\/openapi\.json$/,
      /^\/api\/v1\/audio\/health$/,
      /^\/api\/v1\/audio\/transcriptions\/health$/,
      /^\/api\/v1\/audio\/voices\/catalog$/,
      /^\/api\/v1\/characters\/$/,
      /^\/api\/v1\/chat\/conversations\/[^/]+\/share-links$/,
      /^\/api\/v1\/chats\/$/,
      /^\/api\/v1\/chats\/[^/]+\/(?:messages|research-runs|settings)$/,
      /^\/api\/v1\/config\/docs-info$/,
      /^\/api\/v1\/config\/providers$/,
      /^\/api\/v1\/ingestion-sources\/capabilities$/,
      /^\/api\/v1\/persona\/catalog$/,
      /^\/api\/v1\/prompts\/capabilities$/,
      /^\/api\/v1\/users\/me\/profile$/,
    ]
    return (
      ["GET", "OPTIONS"].includes(method) &&
      safeReadPatterns.some((pattern) => pattern.test(pathname))
    )
  }
  if (entry.context === "owner-revocation") return allowedOwnerRevocationRequest(entry)
  if (!pathname.startsWith("/api/")) return true
  if (allowedAmbientSafeGet(entry)) return true
  return /^\/api\/v1\/sharing\/shared-with-me\/[^/]+\/(?:workspace|sources(?:\/[^/]+\/preview)?|chat(?:\/messages)?)$/.test(
    pathname
  )
}

const MAX_CHAT_SETTINGS_REQUESTS = 2

const buildSettingsRequestProbe = (ledger) => {
  const requests = (ledger.requests || []).filter(
    (entry) =>
      entry.context === "member-chats" &&
      String(entry.method || "GET").toUpperCase() === "GET" &&
      /^\/api\/v1\/chats\/[^/]+\/settings$/.test(requestPath(entry))
  )
  return {
    count: requests.length,
    maximum: MAX_CHAT_SETTINGS_REQUESTS,
    statuses: requests.map((entry) => Number(entry.status)),
  }
}

export const classifyStrictLedger = (ledger) => {
  const failures = []
  const expectedEntries = ledger.expectedHttpFailures || []
  const usedExpected = new Set()
  const operationIds = new Set()
  for (const [index, entry] of expectedEntries.entries()) {
    const operationId = String(entry.operationId || "").trim()
    if (
      !operationId ||
      operationIds.has(operationId) ||
      !Number.isInteger(entry.consoleErrorCount) ||
      entry.consoleErrorCount < 0 ||
      entry.consoleErrorCount > 2
    ) {
      failures.push(`malformed_expected_http_failure:${index}`)
    }
    operationIds.add(operationId)
  }

  const matchesExpected = (expected, observed) =>
    expected.context === observed.context &&
    String(expected.method || "").toUpperCase() === String(observed.method || "").toUpperCase() &&
    Number(expected.status) === Number(observed.status) &&
    expected.url === observed.url &&
    (!("bodyHash" in expected) || expected.bodyHash === observed.bodyHash)

  for (const entry of ledger.requests || []) {
    const forbidden = forbiddenSharedRequest(entry)
    if (forbidden) {
      failures.push(`${forbidden}: ${entry.method} ${requestPath(entry)}`)
    }
    if (!allowedSharedRequest(entry)) {
      failures.push(`undeclared_api_request: ${entry.method} ${requestPath(entry)}`)
    }
    if (Number(entry.status) >= 400 || Number(entry.status) === 0) {
      const expectedIndex = expectedEntries.findIndex(
        (expectedEntry, index) => !usedExpected.has(index) && matchesExpected(expectedEntry, entry)
      )
      if (expectedIndex >= 0) usedExpected.add(expectedIndex)
      else failures.push(`unexpected_http_${entry.status}: ${entry.method} ${requestPath(entry)}`)
    }
  }
  for (const [index, entry] of expectedEntries.entries()) {
    if (!usedExpected.has(index)) {
      failures.push(`expected_http_failure_not_observed: ${entry.operationId || index}`)
    }
  }
  for (const entry of ledger.requestFailures || []) {
    failures.push(`request_failed: ${entry.method || ""} ${entry.url || ""}`)
  }
  for (const entry of ledger.pageErrors || []) {
    failures.push(`page_error: ${entry.message || "unknown"}`)
  }
  const usedConsoleCounts = new Map()
  for (const entry of ledger.consoleErrors || []) {
    const expectedIndex = expectedEntries.findIndex((expectedEntry, index) => {
      const used = usedConsoleCounts.get(index) || 0
      return (
        usedExpected.has(index) &&
        used < expectedEntry.consoleErrorCount &&
        Number(entry.status) === Number(expectedEntry.status) &&
        entry.context === expectedEntry.context &&
        entry.url === expectedEntry.url
      )
    })
    if (expectedIndex < 0) {
      failures.push(`console_error: ${entry.message || "unknown"}`)
      continue
    }
    usedConsoleCounts.set(expectedIndex, (usedConsoleCounts.get(expectedIndex) || 0) + 1)
  }
  for (const [index, entry] of expectedEntries.entries()) {
    if (
      usedExpected.has(index) &&
      (usedConsoleCounts.get(index) || 0) !== entry.consoleErrorCount
    ) {
      failures.push(`expected_console_error_count: ${entry.operationId || index}`)
    }
  }
  for (const entry of ledger.runtimeOverlays || []) {
    failures.push(`runtime_overlay: ${entry.text || "detected"}`)
  }
  return { failures, ok: failures.length === 0 }
}

const redactValue = (value, secrets) => {
  if (typeof value === "string") {
    let result = value
    for (const secret of secrets) {
      if (secret) result = result.split(secret).join("[REDACTED]")
    }
    return result
  }
  if (Array.isArray(value)) return value.map((item) => redactValue(item, secrets))
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => !/^(?:rawPrompt|rawAnswer|query|answer|content)$/i.test(key))
        .map(([key, item]) => [key, redactValue(item, secrets)])
    )
  }
  return value
}

export const createEvidenceRecord = ({
  acceptance,
  config,
  contextIsolationProof = [],
  failureMessageHash = null,
  finishedAt,
  fixture = null,
  ledger,
  provider,
  providerContextProof = null,
  providerReadiness = null,
  raceProbe,
  screenshots,
  startedAt,
  status = "passed",
  transitionProof = [],
}) => {
  const ledgerResult = classifyStrictLedger(ledger)
  const environment = {
    apiUrl: config.apiUrl,
    authMode: "multi_user",
    cdpUrl: config.cdpUrl,
    database: "sqlite",
    dockerAttempted: false,
    personas: config.personas,
    providerProbe: true,
    webUrl: config.webUrl,
  }
  const fixtureEvidence = fixture
    ? {
        canonicalStatusEnvelopeHash: sha256(JSON.stringify(fixture.statusEnvelope)),
        canonicalStatusQueryable: fixture.sourceDefs?.length === 2,
        provisioned: true,
        shareIdHash: sha256(fixture.shareId),
        sourceIds: fixture.sourceDefs.map((source) => sha256(source.id)),
        workspaceIdHash: sha256(fixture.workspaceId),
      }
    : { provisioned: false }
  const payload = redactValue(
    {
      acceptance,
      contextIsolationProof,
      environment,
      environmentHash: sha256(JSON.stringify(environment)),
      failureMessageHash,
      finishedAt,
      fixture: fixtureEvidence,
      ledger: {
        ...ledger,
        classification: ledgerResult,
      },
      provider,
      providerContextProof,
      providerReadiness: providerReadiness
        ? {
            model: String(providerReadiness.model || "").trim(),
            provider: String(providerReadiness.provider || "").trim(),
            ready: providerReadiness.ready === true,
          }
        : null,
      raceProbe,
      runIdHash: sha256(config.runId),
      settingsRequestProbe: buildSettingsRequestProbe(ledger),
      screenshots,
      startedAt,
      status,
      transitionProof,
    },
    [config.adminPassword, config.fixturePassword, config.adminUsername]
  )
  return { ...payload, validation: validateEvidencePayload(payload) }
}

const exactKeys = (value, keys) =>
  Boolean(value && typeof value === "object" && !Array.isArray(value)) &&
  JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort())

const isSha256 = (value) => /^sha256:[a-f0-9]{64}$/.test(String(value || ""))

const validateEvidencePayload = (evidence) => {
  const failures = []
  const payloadKeys = [
    "acceptance",
    "contextIsolationProof",
    "environment",
    "environmentHash",
    "failureMessageHash",
    "finishedAt",
    "fixture",
    "ledger",
    "provider",
    "providerContextProof",
    "providerReadiness",
    "raceProbe",
    "runIdHash",
    "screenshots",
    "settingsRequestProbe",
    "startedAt",
    "status",
    "transitionProof",
  ]
  if (!exactKeys(evidence, payloadKeys)) failures.push("evidence_shape")
  if (evidence.status !== "passed") failures.push("status")
  const startedAtMs = Date.parse(evidence.startedAt)
  const finishedAtMs = Date.parse(evidence.finishedAt)
  if (
    !Number.isFinite(startedAtMs) ||
    !Number.isFinite(finishedAtMs) ||
    new Date(startedAtMs).toISOString() !== evidence.startedAt ||
    new Date(finishedAtMs).toISOString() !== evidence.finishedAt ||
    finishedAtMs < startedAtMs ||
    evidence.failureMessageHash !== null
  ) {
    failures.push("run_metadata")
  }
  if (!exactKeys(evidence.acceptance, ACCEPTANCE_NAMES)) {
    failures.push("acceptance_shape")
  }
  for (const name of ACCEPTANCE_NAMES) {
    if (evidence.acceptance?.[name] !== true) failures.push(`acceptance:${name}`)
  }
  if (!exactKeys(evidence.screenshots, Object.keys(SCREENSHOT_NAMES))) {
    failures.push("screenshots_shape")
  }
  for (const [name, screenshot] of Object.entries(SCREENSHOT_NAMES)) {
    const recordedPath = String(evidence.screenshots?.[name] || "").trim()
    if (!recordedPath) {
      failures.push(`screenshot:${screenshot}`)
    } else if (recordedPath !== screenshot) {
      failures.push(`screenshot_path_not_repository_relative:${name}`)
    }
  }
  const environmentKeys = [
    "apiUrl",
    "authMode",
    "cdpUrl",
    "database",
    "dockerAttempted",
    "personas",
    "providerProbe",
    "webUrl",
  ]
  if (
    !exactKeys(evidence.environment, environmentKeys) ||
    evidence.environment?.authMode !== "multi_user" ||
    evidence.environment?.database !== "sqlite" ||
    evidence.environment?.dockerAttempted !== false ||
    evidence.environment?.providerProbe !== true ||
    JSON.stringify(evidence.environment?.personas) !== JSON.stringify(PERSONAS) ||
    evidence.environmentHash !== sha256(JSON.stringify(evidence.environment)) ||
    !isSha256(evidence.runIdHash)
  ) {
    failures.push("environment")
  }
  if (
    !exactKeys(evidence.provider, ["model", "provider"]) ||
    !evidence.provider?.provider ||
    !evidence.provider?.model
  ) {
    failures.push("provider")
  }
  if (
    !exactKeys(evidence.providerReadiness, ["model", "provider", "ready"]) ||
    evidence.providerReadiness?.ready !== true ||
    evidence.providerReadiness?.provider !== evidence.provider?.provider ||
    evidence.providerReadiness?.model !== evidence.provider?.model
  ) {
    failures.push("provider_readiness")
  }
  const contextProof = evidence.contextIsolationProof
  const contextProofKeys = [
    "configHash",
    "cookieHash",
    "markerCookieHash",
    "markerHash",
    "persona",
    "storageKeyHash",
  ]
  const distinctIdentityHashes = ["configHash", "cookieHash", "markerCookieHash", "markerHash"]
  if (
    !Array.isArray(contextProof) ||
    contextProof.length !== PERSONAS.length ||
    new Set(contextProof.map((entry) => entry?.persona)).size !== PERSONAS.length ||
    !PERSONAS.every((persona) => contextProof.some((entry) => entry?.persona === persona)) ||
    contextProof.some(
      (entry) =>
        !exactKeys(entry, contextProofKeys) ||
        ![entry.configHash, entry.cookieHash, entry.markerHash, entry.storageKeyHash].every(
          isSha256
        ) ||
        entry.markerHash !== entry.markerCookieHash
    ) ||
    distinctIdentityHashes.some(
      (field) => new Set(contextProof.map((entry) => entry?.[field])).size !== PERSONAS.length
    )
  ) {
    failures.push("context_isolation_proof")
  }
  if (
    !exactKeys(evidence.fixture, [
      "canonicalStatusEnvelopeHash",
      "canonicalStatusQueryable",
      "provisioned",
      "shareIdHash",
      "sourceIds",
      "workspaceIdHash",
    ]) ||
    evidence.fixture?.provisioned !== true ||
    evidence.fixture?.canonicalStatusQueryable !== true ||
    !isSha256(evidence.fixture?.canonicalStatusEnvelopeHash) ||
    !isSha256(evidence.fixture?.shareIdHash) ||
    !isSha256(evidence.fixture?.workspaceIdHash) ||
    evidence.fixture?.sourceIds?.length !== 2 ||
    !evidence.fixture.sourceIds.every(isSha256)
  ) {
    failures.push("fixture_proof")
  }
  const providerProof = evidence.providerContextProof
  const providerProofKeys = [
    "bodyUnchanged",
    "forwardedRequestCount",
    "inputBodyHashes",
    "maximumRequestCount",
    "mutationPayloadsAbsent",
    "outputBodyHashes",
    "ownerSentinelAbsent",
    "payloadJsonValid",
    "recipientSentinelAbsent",
    "toolPayloadsAbsent",
    "withinRequestBound",
  ]
  if (
    !exactKeys(providerProof, providerProofKeys) ||
    !Number.isInteger(providerProof?.forwardedRequestCount) ||
    providerProof.forwardedRequestCount < 1 ||
    !Number.isInteger(providerProof.maximumRequestCount) ||
    providerProof.maximumRequestCount < providerProof.forwardedRequestCount ||
    providerProof.maximumRequestCount > 32 ||
    providerProof.inputBodyHashes?.length !== providerProof.forwardedRequestCount ||
    providerProof.outputBodyHashes?.length !== providerProof.forwardedRequestCount ||
    !providerProof.inputBodyHashes?.every(isSha256) ||
    !providerProof.outputBodyHashes?.every(isSha256) ||
    providerProof.inputBodyHashes?.some(
      (hash, index) => hash !== providerProof.outputBodyHashes[index]
    ) ||
    ![
      "bodyUnchanged",
      "mutationPayloadsAbsent",
      "ownerSentinelAbsent",
      "payloadJsonValid",
      "recipientSentinelAbsent",
      "toolPayloadsAbsent",
      "withinRequestBound",
    ].every((key) => providerProof?.[key] === true)
  ) {
    failures.push("provider_context_proof")
  }
  const recomputedLedger = classifyStrictLedger(evidence.ledger || {})
  const expectedFailureKeys = [
    "bodyHash",
    "consoleErrorCount",
    "context",
    "method",
    "operationId",
    "status",
    "url",
  ]
  if (
    !exactKeys(evidence.ledger, [
      "classification",
      "closed",
      "consoleErrors",
      "expectedHttpFailures",
      "pageErrors",
      "requestFailures",
      "requests",
      "runtimeOverlays",
    ]) ||
    evidence.ledger?.closed !== true ||
    !exactKeys(evidence.ledger?.classification, ["failures", "ok"]) ||
    evidence.ledger.classification.ok !== recomputedLedger.ok ||
    JSON.stringify(evidence.ledger.classification.failures) !==
      JSON.stringify(recomputedLedger.failures) ||
    recomputedLedger.ok !== true ||
    evidence.ledger?.expectedHttpFailures?.some(
      (entry) =>
        !exactKeys(entry, expectedFailureKeys) ||
        !String(entry.operationId || "").trim() ||
        ![0, 1, 2].includes(entry.consoleErrorCount) ||
        ![404, 409].includes(entry.status) ||
        !["GET", "POST"].includes(entry.method) ||
        (entry.bodyHash !== null && !isSha256(entry.bodyHash))
    )
  ) {
    failures.push("ledger")
  }
  const statuses = evidence.raceProbe?.statuses || []
  const successCount = statuses.filter((status) => status === 200).length
  const conflictCount = statuses.filter((status) => status === 409).length
  if (
    successCount < 2 ||
    conflictCount !== 2 ||
    statuses.at(-1) !== 409 ||
    statuses.some((status) => ![200, 409].includes(status))
  ) {
    failures.push("race_statuses")
  }
  const turnHashes = evidence.raceProbe?.turnHashes || []
  if (turnHashes.length < 2 || turnHashes.at(-1) !== turnHashes.at(-2)) {
    failures.push("race_replay_equivalence")
  }
  if (
    !exactKeys(evidence.raceProbe, [
      "operations",
      "requestHashes",
      "requestIdHash",
      "responseHashes",
      "statuses",
      "timingsMs",
      "turnHashes",
    ]) ||
    evidence.raceProbe?.requestHashes?.length !== 2 ||
    !evidence.raceProbe.requestHashes.every(isSha256) ||
    !isSha256(evidence.raceProbe?.requestIdHash) ||
    evidence.raceProbe?.responseHashes?.length !== 2 ||
    !evidence.raceProbe.responseHashes.every(isSha256) ||
    evidence.raceProbe?.timingsMs?.length !== statuses.length ||
    evidence.raceProbe?.operations?.length !== 2 ||
    new Set(evidence.raceProbe.operations.map((entry) => entry.operationId)).size !== 2 ||
    !["race-concurrent-conflict", "race-fingerprint-conflict"].every((operationId) =>
      evidence.raceProbe.operations.some((operation) => operation.operationId === operationId)
    ) ||
    evidence.raceProbe.operations.some(
      (entry) =>
        !exactKeys(entry, ["bodyHash", "operationId", "status"]) ||
        entry.status !== 409 ||
        !isSha256(entry.bodyHash) ||
        !evidence.raceProbe.requestHashes.includes(entry.bodyHash) ||
        evidence.ledger.expectedHttpFailures.filter(
          (expected) =>
            expected.operationId === entry.operationId &&
            expected.bodyHash === entry.bodyHash &&
            expected.status === entry.status &&
            expected.consoleErrorCount === 1
        ).length !== 1
    )
  ) {
    failures.push("race_shape")
  }
  const settingsProbe = evidence.settingsRequestProbe
  if (
    !exactKeys(settingsProbe, ["count", "maximum", "statuses"]) ||
    settingsProbe.count < 1 ||
    settingsProbe.count > MAX_CHAT_SETTINGS_REQUESTS ||
    settingsProbe.statuses?.length !== settingsProbe.count ||
    settingsProbe.statuses.some((status) => status !== 200)
  ) {
    failures.push("settings_request_amplification")
  }
  const transitionProof = evidence.transitionProof
  const transitionProofKeys = [
    "allowedAbortCount",
    "allowedAborts",
    "consoleErrorCount",
    "context",
    "labelHash",
    "maximumOperationDeclarations",
    "maximumRequestCount",
    "observedRequests",
    "operations",
    "pageErrorCount",
    "registeredAbortCount",
    "registeredOperationCount",
    "requestCount",
    "runtimeOverlayCount",
    "unexpectedRequestCount",
    "withinRequestBound",
  ]
  const abortProofKeys = ["count", "id", "maximumCount", "method", "requestHash"]
  const observedRequestProofKeys = ["errorHash", "kind", "method", "requestHash", "status"]
  const operationProofKeys = [
    "allowedStatuses",
    "count",
    "maximumCount",
    "name",
    "observedStatuses",
  ]
  const invalidTransitionProof = (proof) => {
    if (!exactKeys(proof, transitionProofKeys)) return true
    const expectedOperations = transitionOperationContract(proof.context)
    const observedRequests = proof.observedRequests
    const operations = proof.operations
    const allowedAborts = proof.allowedAborts
    if (
      !expectedOperations ||
      !Array.isArray(observedRequests) ||
      !Array.isArray(operations) ||
      !Array.isArray(allowedAborts)
    ) {
      return true
    }
    const expectedByName = new Map(
      expectedOperations.map((operation) => [operation.name, operation])
    )
    const operationNames = operations.map((operation) => operation?.name)
    const responseRequests = observedRequests.filter((request) => request?.kind === "response")
    const operationStatuses = operations.flatMap((operation) => operation?.observedStatuses || [])
    return (
      !isSha256(proof.labelHash) ||
      proof.consoleErrorCount !== 0 ||
      proof.pageErrorCount !== 0 ||
      proof.runtimeOverlayCount !== 0 ||
      proof.unexpectedRequestCount !== 0 ||
      proof.withinRequestBound !== true ||
      proof.maximumOperationDeclarations !== MAX_TRANSITION_OPERATION_DECLARATIONS ||
      proof.maximumRequestCount !== MAX_TRANSITION_REQUESTS ||
      !Number.isInteger(proof.requestCount) ||
      proof.requestCount < 0 ||
      proof.requestCount > MAX_TRANSITION_REQUESTS ||
      observedRequests.length !== proof.requestCount ||
      observedRequests.some(
        (request) =>
          !exactKeys(request, observedRequestProofKeys) ||
          !["failure", "response"].includes(request.kind) ||
          !String(request.method || "").trim() ||
          !isSha256(request.requestHash) ||
          (request.kind === "failure"
            ? request.status !== null || !isSha256(request.errorHash)
            : !Number.isInteger(request.status) || request.errorHash !== null)
      ) ||
      proof.registeredOperationCount !== expectedOperations.length ||
      operations.length !== expectedOperations.length ||
      new Set(operationNames).size !== expectedOperations.length ||
      operations.some((operation) => {
        const expected = expectedByName.get(operation?.name)
        return (
          !exactKeys(operation, operationProofKeys) ||
          !expected ||
          JSON.stringify(operation.allowedStatuses) !==
            JSON.stringify(expected.allowedStatuses) ||
          operation.maximumCount !== expected.maximumCount ||
          !Array.isArray(operation.observedStatuses) ||
          operation.observedStatuses.some(
            (status) =>
              !Number.isInteger(status) || !operation.allowedStatuses.includes(status)
          ) ||
          operation.observedStatuses.length !== operation.count ||
          !Number.isInteger(operation.count) ||
          operation.count < 0 ||
          operation.count > operation.maximumCount
        )
      }) ||
      operations.reduce((total, operation) => total + operation.count, 0) !==
        responseRequests.length ||
      JSON.stringify(operationStatuses.sort((left, right) => left - right)) !==
        JSON.stringify(
          responseRequests.map((request) => request.status).sort((left, right) => left - right)
        ) ||
      proof.registeredAbortCount !== allowedAborts.length ||
      proof.registeredAbortCount > MAX_TRANSITION_ABORT_ALLOWANCES ||
      proof.allowedAbortCount !==
        allowedAborts.reduce((total, abort) => total + abort.count, 0) ||
      allowedAborts.some(
        (abort) =>
          !exactKeys(abort, abortProofKeys) ||
          !String(abort.id || "").trim() ||
          abort.method !== "GET" ||
          !Number.isInteger(abort.count) ||
          !Number.isInteger(abort.maximumCount) ||
          abort.count < 0 ||
          abort.maximumCount < 1 ||
          abort.count > abort.maximumCount ||
          abort.maximumCount > 2 ||
          !isSha256(abort.requestHash)
      ) ||
      allowedAborts.some(
        (abort) =>
          observedRequests.filter(
            (request) =>
              request.kind === "failure" &&
              request.method === abort.method &&
              request.requestHash === abort.requestHash
          ).length < abort.count
      )
    )
  }
  if (
    !Array.isArray(transitionProof) ||
    transitionProof.length !== 2 ||
    new Set(transitionProof.map((proof) => proof?.context)).size !== 2 ||
    !["owner-revocation", "member-chats"].every((context) =>
      transitionProof.some((proof) => proof?.context === context)
    ) ||
    transitionProof.some(invalidTransitionProof)
  ) {
    failures.push("transition_proof")
  }
  const serialized = JSON.stringify(evidence)
  if (/(?:\/Users\/|\/home\/|\.worktrees\/)/.test(serialized)) {
    failures.push("machine_path")
  }
  return { exitCode: failures.length ? 1 : 0, failures }
}

export const validateEvidenceRecord = (evidence) => {
  const { validation, ...payload } = evidence || {}
  const result = validateEvidencePayload(payload)
  if (
    !exactKeys(evidence, [...Object.keys(payload), "validation"]) ||
    !exactKeys(validation, ["exitCode", "failures"]) ||
    validation.exitCode !== result.exitCode ||
    JSON.stringify(validation.failures) !== JSON.stringify(result.failures)
  ) {
    return {
      exitCode: 1,
      failures: [...result.failures, "validation_record"],
    }
  }
  return result
}

const sha256 = (value) =>
  `sha256:${crypto.createHash("sha256").update(String(value)).digest("hex")}`

const boundedJson = (value, max = 2_000) => {
  const serialized = JSON.stringify(value)
  return serialized.length <= max ? value : { truncated: true, hash: sha256(serialized) }
}

const CLEANUP_ID_FIELDS = [
  "userIds",
  "organizationIds",
  "teamIds",
  "roleIds",
  "workspaceIds",
  "shareIds",
]

const registerCleanupMetadata = (config, additions) => {
  let current = {
    createdAt: new Date().toISOString(),
    runId: config.runId,
    version: 1,
  }
  if (fs.existsSync(config.cleanupManifestPath)) {
    current = JSON.parse(fs.readFileSync(config.cleanupManifestPath, "utf8"))
  }
  for (const field of CLEANUP_ID_FIELDS) {
    const existing = Array.isArray(current[field]) ? current[field] : []
    const incoming = Array.isArray(additions[field]) ? additions[field] : []
    current[field] = [...new Set([...existing, ...incoming].map(String))]
  }
  current.updatedAt = new Date().toISOString()
  fs.mkdirSync(path.dirname(config.cleanupManifestPath), { recursive: true })
  fs.writeFileSync(config.cleanupManifestPath, `${JSON.stringify(current, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  })
  fs.chmodSync(config.cleanupManifestPath, 0o600)
}

const apiCall = async (config, token, pathname, options = {}) => {
  const response = await fetch(`${config.apiUrl}${pathname}`, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  })
  const text = await response.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = { detail: text.slice(0, 500) }
  }
  if (!(options.expectedStatuses || [200, 201, 204]).includes(response.status)) {
    throw new Error(
      `${options.method || "GET"} ${pathname} failed with HTTP ${response.status}: ${JSON.stringify(boundedJson(body))}`
    )
  }
  return { body, status: response.status }
}

const resetProviderProbe = async (config) => {
  const response = await fetch(`${config.providerProbeUrl}/__tldw_provider_probe/reset`, {
    method: "POST",
  })
  if (response.status !== 204) {
    throw new Error(`Provider probe reset failed with HTTP ${response.status}`)
  }
}

const readProviderContextProof = async (config) => {
  const response = await fetch(`${config.providerProbeUrl}/__tldw_provider_probe/proof`)
  if (!response.ok) {
    throw new Error(`Provider probe proof failed with HTTP ${response.status}`)
  }
  return response.json()
}

const loginApi = async (config, username, password) => {
  const body = new URLSearchParams({ username, password })
  const response = await fetch(`${config.apiUrl}/api/v1/auth/login`, {
    body,
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    method: "POST",
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok || !payload.access_token) {
    throw new Error(`Admin login failed with HTTP ${response.status}`)
  }
  return payload.access_token
}

const provisionUser = async (config, adminToken, username) => {
  const response = await apiCall(config, adminToken, "/api/v1/admin/users", {
    body: JSON.stringify({
      email: `${username}@example.com`,
      is_active: true,
      is_verified: true,
      password: config.fixturePassword,
      role: "user",
      username,
    }),
    method: "POST",
  })
  const userId = Number(response.body?.id ?? response.body?.user_id)
  if (!Number.isSafeInteger(userId) || userId <= 0) {
    throw new Error(`User provisioning returned no valid ID for ${username}`)
  }
  return { id: userId, username }
}

const ingestDocument = async (config, token, title, content) => {
  const form = new FormData()
  form.append("media_type", "document")
  form.append("title", title)
  form.append("perform_analysis", "false")
  form.append("perform_chunking", "true")
  form.append("generate_embeddings", "true")
  form.append("embedding_dispatch_mode", "background")
  form.append("embedding_provider", "huggingface")
  form.append("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
  form.append("files", new Blob([content], { type: "text/plain" }), `${title}.txt`)
  const response = await apiCall(config, token, "/api/v1/media/add", {
    body: form,
    method: "POST",
  })
  const result = Array.isArray(response.body?.results)
    ? response.body.results[0]
    : response.body?.result || response.body
  const mediaId = Number(result?.db_id ?? result?.media_id ?? result?.id)
  if (!Number.isSafeInteger(mediaId) || mediaId <= 0) {
    throw new Error(`Ingest returned no media ID for ${title}`)
  }
  return mediaId
}

const addWorkspaceSource = async (config, token, workspaceId, source) => {
  await apiCall(config, token, `/api/v1/workspaces/${workspaceId}/sources`, {
    body: JSON.stringify(source),
    method: "POST",
  })
}

const pollQueryable = async (config, ownerToken, workspaceId, expectedIds) => {
  const deadline = Date.now() + 120_000
  let finalStatus = null
  while (Date.now() < deadline) {
    const response = await apiCall(
      config,
      ownerToken,
      `/api/v1/workspaces/${workspaceId}/sources/status`
    )
    finalStatus = response.body
    const statuses = new Map(
      (response.body?.sources || []).map((source) => [String(source.id), source.state])
    )
    if (expectedIds.every((id) => statuses.get(id) === "queryable")) {
      return boundedJson(finalStatus)
    }
    await sleep(1_000)
  }
  throw new Error(
    `Workspace sources did not become queryable: ${JSON.stringify(boundedJson(finalStatus))}`
  )
}

const provisionFixture = async (config) => {
  const adminToken = await loginApi(config, config.adminUsername, config.adminPassword)
  const usernames = Object.fromEntries(
    PERSONAS.map((persona) => [
      persona,
      `${config.usernamePrefixes[persona]}-${config.runId}`.slice(0, 50),
    ])
  )
  const owner = await provisionUser(config, adminToken, usernames.owner)
  registerCleanupMetadata(config, { userIds: [owner.id] })
  const member = await provisionUser(config, adminToken, usernames.member)
  registerCleanupMetadata(config, { userIds: [member.id] })
  const nonmember = await provisionUser(config, adminToken, usernames.nonmember)
  registerCleanupMetadata(config, { userIds: [nonmember.id] })

  const listedPermissions = await apiCall(
    config,
    adminToken,
    "/api/v1/admin/permissions?search=sharing.read"
  )
  let permission = (Array.isArray(listedPermissions.body) ? listedPermissions.body : []).find(
    (item) => item?.name === "sharing.read"
  )
  if (!permission) {
    const createdPermission = await apiCall(config, adminToken, "/api/v1/admin/permissions", {
      body: JSON.stringify({
        category: "sharing",
        description: "Read recipient shared research workspaces",
        name: "sharing.read",
      }),
      method: "POST",
    })
    permission = createdPermission.body
  }
  const permissionId = Number(permission?.id)
  if (!Number.isSafeInteger(permissionId) || permissionId <= 0) {
    throw new Error("Admin RBAC provisioning returned no sharing.read permission ID")
  }
  const recipientRole = await apiCall(config, adminToken, "/api/v1/admin/roles", {
    body: JSON.stringify({
      description: "Run-scoped shared research workspace recipient role",
      name: `shared-recipient-${config.runId}`.slice(0, 64),
    }),
    method: "POST",
  })
  const recipientRoleId = Number(recipientRole.body?.id)
  if (!Number.isSafeInteger(recipientRoleId) || recipientRoleId <= 0) {
    throw new Error("Admin RBAC provisioning returned no recipient role ID")
  }
  registerCleanupMetadata(config, { roleIds: [recipientRoleId] })
  await apiCall(
    config,
    adminToken,
    `/api/v1/admin/roles/${recipientRoleId}/permissions/${permissionId}`,
    { method: "POST" }
  )
  for (const user of [owner, member, nonmember]) {
    await apiCall(config, adminToken, `/api/v1/admin/users/${user.id}/roles/${recipientRoleId}`, {
      method: "POST",
    })
  }

  const org = await apiCall(config, adminToken, "/api/v1/admin/orgs", {
    body: JSON.stringify({ name: `UAT org ${config.runId}`, owner_user_id: owner.id }),
    method: "POST",
  })
  registerCleanupMetadata(config, { organizationIds: [org.body.id] })
  const team = await apiCall(config, adminToken, `/api/v1/admin/orgs/${org.body.id}/teams`, {
    body: JSON.stringify({ name: `UAT team ${config.runId}` }),
    method: "POST",
  })
  registerCleanupMetadata(config, { teamIds: [team.body.id] })
  for (const user of [owner, member]) {
    await apiCall(config, adminToken, `/api/v1/admin/orgs/${org.body.id}/members`, {
      body: JSON.stringify({ role: user === owner ? "owner" : "member", user_id: user.id }),
      method: "POST",
    })
    await apiCall(config, adminToken, `/api/v1/admin/teams/${team.body.id}/members`, {
      body: JSON.stringify({ role: user === owner ? "owner" : "member", user_id: user.id }),
      method: "POST",
    })
  }

  const ownerToken = await loginApi(config, owner.username, config.fixturePassword)
  const memberToken = await loginApi(config, member.username, config.fixturePassword)
  const workspaceName = `Recipient evidence ${config.runId}`

  const sourceDefs = [
    {
      content: `Amber protocol evidence. ${SOURCE_ONE_FACT} This fact is unique to the amber source.`,
      id: `source-amber-${config.runId}`,
      title: `Amber protocol ${config.runId}`,
    },
    {
      content: `Cobalt trial evidence. ${SOURCE_TWO_FACT} This fact is unique to the cobalt source.`,
      id: `source-cobalt-${config.runId}`,
      title: `Cobalt trial ${config.runId}`,
    },
  ]
  for (const [position, source] of sourceDefs.entries()) {
    const mediaId = await ingestDocument(config, ownerToken, source.title, source.content)
    source.mediaId = mediaId
    source.position = position
  }
  await ingestDocument(config, ownerToken, `Unrelated owner ${config.runId}`, OWNER_SENTINEL)

  const memberWorkspaceId = `recipient-local-${config.runId}`
  await apiCall(config, memberToken, `/api/v1/workspaces/${memberWorkspaceId}`, {
    body: JSON.stringify({ name: RECIPIENT_SENTINEL }),
    method: "PUT",
  })
  registerCleanupMetadata(config, { workspaceIds: [memberWorkspaceId] })
  const memberMediaId = await ingestDocument(
    config,
    memberToken,
    `Recipient local ${config.runId}`,
    RECIPIENT_SENTINEL
  )
  await addWorkspaceSource(config, memberToken, memberWorkspaceId, {
    id: `recipient-source-${config.runId}`,
    media_id: memberMediaId,
    position: 0,
    selected: true,
    source_type: "document",
    title: RECIPIENT_SENTINEL,
    url: `file://recipient-local-${config.runId}.txt`,
  })

  return {
    member,
    nonmember,
    owner,
    ownerToken,
    shareId: null,
    sourceDefs,
    statusEnvelope: null,
    teamId: Number(team.body.id),
    workspaceId: null,
    workspaceName,
  }
}

const finalizeFixtureWorkspace = async (config, fixture, workspaceId) => {
  registerCleanupMetadata(config, { workspaceIds: [workspaceId] })
  for (const source of fixture.sourceDefs) {
    await addWorkspaceSource(config, fixture.ownerToken, workspaceId, {
      id: source.id,
      media_id: source.mediaId,
      position: source.position,
      selected: true,
      source_type: "document",
      title: source.title,
      url: `file://${source.title}.txt`,
    })
  }
  const statusEnvelope = await pollQueryable(
    config,
    fixture.ownerToken,
    workspaceId,
    fixture.sourceDefs.map((source) => source.id)
  )
  const share = await apiCall(
    config,
    fixture.ownerToken,
    `/api/v1/sharing/workspaces/${workspaceId}/share`,
    {
      body: JSON.stringify({
        access_level: "view_chat",
        allow_clone: false,
        share_scope_id: fixture.teamId,
        share_scope_type: "team",
      }),
      method: "POST",
    }
  )
  const shareId = Number(share.body.id)
  if (!Number.isSafeInteger(shareId) || shareId <= 0) {
    throw new Error("Workspace share provisioning returned no share ID")
  }
  registerCleanupMetadata(config, { shareIds: [shareId] })
  return { ...fixture, shareId, statusEnvelope, workspaceId }
}

const makeLedger = () => ({
  closed: false,
  consoleErrors: [],
  expectedHttpFailures: [],
  pageErrors: [],
  requests: [],
  requestFailures: [],
  runtimeOverlays: [],
})

const attachLedger = (page, contextName, ledger) => {
  const pending = new Map()
  const inFlightApiRequests = new Set()
  let lastApiActivity = Date.now()
  const shouldSettle = (request) => {
    try {
      const pathname = new URL(request.url()).pathname
      return pathname.startsWith("/api/") && pathname !== "/api/v1/notifications/stream"
    } catch {
      return false
    }
  }
  const onRequest = (request) => {
    pending.set(request, {
      bodyHash: request.postData() ? sha256(request.postData()) : null,
      context: contextName,
      method: request.method(),
      status: 0,
      url: request.url(),
    })
    if (shouldSettle(request)) {
      inFlightApiRequests.add(request)
      lastApiActivity = Date.now()
    }
  }
  const onResponse = (response) => {
    const request = response.request()
    const entry = pending.get(request) || {
      context: contextName,
      method: request.method(),
      url: response.url(),
    }
    entry.status = response.status()
    ledger.requests.push(entry)
    pending.delete(request)
  }
  const onRequestFinished = (request) => {
    if (!inFlightApiRequests.delete(request)) return
    lastApiActivity = Date.now()
  }
  const onRequestFailed = (request) => {
    ledger.requestFailures.push({
      context: contextName,
      error: request.failure()?.errorText || "request failed",
      method: request.method(),
      url: request.url(),
    })
    pending.delete(request)
    if (inFlightApiRequests.delete(request)) lastApiActivity = Date.now()
  }
  const onPageError = (error) =>
    ledger.pageErrors.push({ context: contextName, message: error.message })
  const onConsole = (message) => {
    if (message.type() !== "error") return
    const statusMatch = message.text().match(/status of (\d{3})/i)
    ledger.consoleErrors.push({
      context: contextName,
      message: message.text(),
      status: statusMatch ? Number(statusMatch[1]) : null,
      url: message.location().url || null,
    })
  }
  page.on("request", onRequest)
  page.on("response", onResponse)
  page.on("requestfinished", onRequestFinished)
  page.on("requestfailed", onRequestFailed)
  page.on("pageerror", onPageError)
  page.on("console", onConsole)
  const dispose = () => {
    page.off("request", onRequest)
    page.off("response", onResponse)
    page.off("requestfinished", onRequestFinished)
    page.off("requestfailed", onRequestFailed)
    page.off("pageerror", onPageError)
    page.off("console", onConsole)
  }
  const waitForIdle = async (label, { quietMs = 500, timeoutMs = 70_000 } = {}) => {
    const deadline = Date.now() + timeoutMs
    while (inFlightApiRequests.size > 0 || Date.now() - lastApiActivity < quietMs) {
      if (Date.now() >= deadline) {
        const outstanding = [...inFlightApiRequests].map((request) => ({
          method: request.method(),
          path: requestPath({ url: request.url() }),
        }))
        throw new Error(
          `Browser API requests did not settle (${label}): ${JSON.stringify({ outstanding })}`
        )
      }
      await sleep(50)
    }
  }
  return { dispose, waitForIdle }
}

const MAX_TRANSITION_ABORT_ALLOWANCES = 12
const MAX_TRANSITION_OPERATION_DECLARATIONS = 64
const MAX_TRANSITION_REQUESTS = 64

const transitionOperationDeclaration = ({
  allowedStatuses = [200],
  maximumCount,
  name,
}) =>
  Object.freeze({
    allowedStatuses: Object.freeze([...allowedStatuses]),
    maximumCount,
    name,
  })

const COMMON_TRANSITION_OPERATION_DECLARATIONS = [
  ["destination-document", 1],
  ["webui-runtime-config", 2],
  ["webui-next-static", 32, [200, 304]],
  ["webui-next-font-geist", 2, [200, 304]],
  ["webui-font-arimo", 1, [200, 304]],
  ["webui-font-inter-semibold", 1, [200, 304]],
  ["webui-font-inter-medium", 1, [200, 304]],
  ["webui-font-inter-regular", 1, [200, 304]],
  ["ambient-persona-profiles", 4],
  ["ambient-auth-me", 4],
  ["ambient-health", 4],
  ["ambient-health-live", 4],
  ["ambient-notifications", 4],
  ["ambient-notification-stream", 4],
  ["ambient-notification-count", 4],
  ["ambient-rag-health", 4],
  ["ambient-llm-providers", 4],
  ["ambient-llm-models", 4],
  ["ambient-user-profile", 4],
].map(([name, maximumCount, allowedStatuses]) =>
  transitionOperationDeclaration({ allowedStatuses, maximumCount, name })
)

const OWNER_TRANSITION_OPERATION_DECLARATIONS = [
  ...COMMON_TRANSITION_OPERATION_DECLARATIONS,
  ...[
    ["owner-flashcard-decks", 1],
    ["owner-notes-search", 2],
    ["owner-slide-styles", 1],
    ["owner-user-storage", 2],
    ["owner-chat-commands", 1],
    ["owner-workspace-capabilities", 1],
    ["owner-workspace-context", 4],
    ["owner-workspace-source-views", 2],
    ["owner-workspace-sources", 2],
    ["owner-workspace-save", 2],
    ["owner-source-selection", 2],
    ["owner-migration-create", 1, [201]],
    ["owner-migration-finalize", 1],
    ["owner-migration-status", 1],
    ["owner-migration-delete-ack", 1],
    ["owner-migration-chunk-1", 1],
    ["owner-migration-chunk-2", 1],
    ["owner-migration-chunk-3", 1],
  ].map(([name, maximumCount, allowedStatuses]) =>
    transitionOperationDeclaration({ allowedStatuses, maximumCount, name })
  ),
]

const MEMBER_TRANSITION_OPERATION_DECLARATIONS = [
  ...COMMON_TRANSITION_OPERATION_DECLARATIONS,
  ...[
    ["chats-openapi", 2],
    ["chats-docs-info", 1],
    ["chats-ingestion-capabilities", 1],
    ["chats-audio-service-health", 1],
    ["chats-character-catalog", 1],
    ["chats-audio-health", 1],
    ["chats-voice-catalog", 1],
    ["chats-share-links", 1],
    ["chats-list", 2],
    ["chats-messages", 2],
    ["chats-research-runs", 1],
    ["chats-settings", 2],
    ["chats-provider-config", 1],
    ["chats-persona-catalog", 1],
    ["chats-prompt-capabilities", 1],
    ["chats-implicit-feedback", 1],
  ].map(([name, maximumCount, allowedStatuses]) =>
    transitionOperationDeclaration({ allowedStatuses, maximumCount, name })
  ),
]

const TRANSITION_OPERATION_CONTRACTS = Object.freeze({
  "member-chats": Object.freeze(MEMBER_TRANSITION_OPERATION_DECLARATIONS),
  "owner-revocation": Object.freeze(OWNER_TRANSITION_OPERATION_DECLARATIONS),
})

const transitionOperationContract = (contextName) =>
  TRANSITION_OPERATION_CONTRACTS[contextName] || null

export const getTransitionEvidenceOperationContract = (contextName) =>
  (transitionOperationContract(contextName) || []).map((operation) => ({
    allowedStatuses: [...operation.allowedStatuses],
    maximumCount: operation.maximumCount,
    name: operation.name,
  }))

const requireTransitionOperationDeclaration = (contextName, name) => {
  const declaration = transitionOperationContract(contextName)?.find(
    (operation) => operation.name === name
  )
  if (!declaration) {
    throw new Error(`Unknown ${contextName} transition operation: ${name}`)
  }
  return declaration
}

const transitionRequestOrigin = (entry) => {
  try {
    return new URL(entry.url).origin
  } catch {
    return ""
  }
}

const transitionOperation = ({
  allowedForbiddenKinds = [],
  allowedStatuses,
  maximumCount,
  method,
  name,
  origin,
  path: exactPath,
  pathPrefix,
}) => ({
  allowedForbiddenKinds,
  allowedStatuses: [...new Set(allowedStatuses)].sort((left, right) => left - right),
  maximumCount,
  method: String(method).toUpperCase(),
  name,
  origin: new URL(origin).origin,
  path: exactPath ? String(exactPath).toLowerCase() : null,
  pathPrefix: pathPrefix ? String(pathPrefix).toLowerCase() : null,
})

const declaredTransitionOperation = ({ contextName, name, ...matcher }) => {
  const declaration = requireTransitionOperationDeclaration(contextName, name)
  return transitionOperation({ ...declaration, ...matcher })
}

const commonTransitionOperations = ({ apiUrl, contextName, targetPath, webUrl }) => [
  declaredTransitionOperation({
    contextName,
    method: "GET",
    name: "destination-document",
    origin: webUrl,
    path: targetPath,
  }),
  declaredTransitionOperation({
    contextName,
    method: "GET",
    name: "webui-runtime-config",
    origin: webUrl,
    path: "/api/_tldw-webui/runtime-config",
  }),
  declaredTransitionOperation({
    contextName,
    method: "GET",
    name: "webui-next-static",
    origin: webUrl,
    pathPrefix: "/_next/",
  }),
  declaredTransitionOperation({
    contextName,
    method: "GET",
    name: "webui-next-font-geist",
    origin: webUrl,
    path: "/__nextjs_font/geist-latin.woff2",
  }),
  ...[
    ["webui-font-arimo", "/fonts/arimo.ttf"],
    ["webui-font-inter-semibold", "/fonts/inter-semibold.ttf"],
    ["webui-font-inter-medium", "/fonts/inter-medium.ttf"],
    ["webui-font-inter-regular", "/fonts/inter-regular.ttf"],
  ].map(([name, pathname]) =>
    declaredTransitionOperation({
      contextName,
      method: "GET",
      name,
      origin: webUrl,
      path: pathname,
    })
  ),
  ...[
    ["ambient-persona-profiles", "/api/v1/persona/profiles"],
    ["ambient-auth-me", "/api/v1/auth/me"],
    ["ambient-health", "/api/v1/health"],
    ["ambient-health-live", "/api/v1/health/live"],
    ["ambient-notifications", "/api/v1/notifications"],
    ["ambient-notification-stream", "/api/v1/notifications/stream"],
    ["ambient-notification-count", "/api/v1/notifications/unread-count"],
    ["ambient-rag-health", "/api/v1/rag/health"],
    ["ambient-llm-providers", "/api/v1/llm/providers"],
    ["ambient-llm-models", "/api/v1/llm/models/metadata"],
    ["ambient-user-profile", "/api/v1/users/me/profile"],
  ].map(([name, pathname]) =>
    declaredTransitionOperation({
      contextName,
      method: "GET",
      name,
      origin: apiUrl,
      path: pathname,
    })
  ),
]

const transitionMigrationIdentity = ({ apiUrl, ledger, workspaceId }) => {
  const apiOrigin = new URL(apiUrl).origin
  const migrationIds = new Set()
  const chunkIds = new Set()
  const entries = [...(ledger.requests || []), ...(ledger.requestFailures || [])]
  for (const entry of entries) {
    if (transitionRequestOrigin(entry) !== apiOrigin) continue
    const parts = requestPath(entry).split("/").filter(Boolean)
    if (
      parts[0] !== "api" ||
      parts[1] !== "v1" ||
      parts[2] !== "workspaces" ||
      parts[3] !== "migrations" ||
      !parts[4]
    ) {
      continue
    }
    const migrationId = decodeURIComponent(parts[4])
    migrationIds.add(migrationId)
    if (parts[5] === "chunks" && parts[6]) {
      chunkIds.add(decodeURIComponent(parts[6]))
    }
  }
  const normalizedWorkspaceId = String(workspaceId).toLowerCase()
  const expectedMigrationPrefix = `research-workspace-${normalizedWorkspaceId}-`
  const migrationId = [...migrationIds][0] || null
  const invalidReasons = []
  if (
    migrationIds.size > 1 ||
    (migrationId &&
      (!migrationId.startsWith(expectedMigrationPrefix) ||
        !/^[a-z0-9-]+$/.test(migrationId) ||
        !/^[a-f0-9]{16}$/.test(migrationId.slice(expectedMigrationPrefix.length))))
  ) {
    invalidReasons.push("owner_migration_identity")
  }
  const orderedChunks = [...chunkIds]
    .map((chunkId) => {
      const match = chunkId.match(/^chunk-(\d+)-[a-f0-9]{16}$/)
      return { chunkId, index: match ? Number(match[1]) : 0 }
    })
    .sort((left, right) => left.index - right.index)
  if (
    (migrationId && orderedChunks.length !== 3) ||
    orderedChunks.some(({ index }, position) => index !== position + 1)
  ) {
    invalidReasons.push("owner_migration_chunk_identity")
  }
  return { chunkIds: orderedChunks, invalidReasons, migrationId }
}

export const buildOwnerRevocationTransitionPolicy = ({ apiUrl, ledger, webUrl, workspaceId }) => {
  const workspacePath = `/api/v1/workspaces/${String(workspaceId).toLowerCase()}`
  const migration = transitionMigrationIdentity({ apiUrl, ledger, workspaceId })
  const ownerOperations = [
    ["owner-flashcard-decks", "GET", "/api/v1/flashcards/decks", []],
    ["owner-notes-search", "GET", "/api/v1/notes/search/", ["notes"]],
    ["owner-slide-styles", "GET", "/api/v1/slides/styles", []],
    ["owner-user-storage", "GET", "/api/v1/users/storage", []],
    ["owner-chat-commands", "GET", "/api/v1/chat/commands", []],
    [
      "owner-workspace-capabilities",
      "GET",
      "/api/v1/research-workspace/capabilities",
      ["local_workspace"],
    ],
    ["owner-workspace-context", "GET", `${workspacePath}/context`, ["local_workspace"]],
    [
      "owner-workspace-source-views",
      "GET",
      `${workspacePath}/source-views`,
      ["local_workspace"],
    ],
    ["owner-workspace-sources", "GET", `${workspacePath}/sources`, ["local_workspace"]],
    ["owner-workspace-save", "PUT", workspacePath, ["local_workspace"]],
    [
      "owner-source-selection",
      "PUT",
      `${workspacePath}/sources/selection`,
      ["local_workspace", "source_mutation"],
    ],
    ["owner-migration-create", "POST", "/api/v1/workspaces/migrations", ["local_workspace"]],
  ].map(([name, method, pathname, allowedForbiddenKinds]) =>
    declaredTransitionOperation({
      allowedForbiddenKinds,
      contextName: "owner-revocation",
      method,
      name,
      origin: apiUrl,
      path: pathname,
    })
  )
  if (migration.migrationId) {
    const migrationPath = `/api/v1/workspaces/migrations/${migration.migrationId}`
    ownerOperations.push(
      declaredTransitionOperation({
        allowedForbiddenKinds: ["local_workspace"],
        contextName: "owner-revocation",
        method: "POST",
        name: "owner-migration-finalize",
        origin: apiUrl,
        path: `${migrationPath}/finalize`,
      }),
      declaredTransitionOperation({
        allowedForbiddenKinds: ["local_workspace"],
        contextName: "owner-revocation",
        method: "GET",
        name: "owner-migration-status",
        origin: apiUrl,
        path: migrationPath,
      }),
      declaredTransitionOperation({
        allowedForbiddenKinds: ["local_workspace"],
        contextName: "owner-revocation",
        method: "POST",
        name: "owner-migration-delete-ack",
        origin: apiUrl,
        path: `${migrationPath}/client-delete-ack`,
      })
    )
    for (const { chunkId, index } of migration.chunkIds) {
      if (index < 1 || index > 3) continue
      ownerOperations.push(
        declaredTransitionOperation({
          allowedForbiddenKinds: ["local_workspace"],
          contextName: "owner-revocation",
          method: "PUT",
          name: `owner-migration-chunk-${index}`,
          origin: apiUrl,
          path: `${migrationPath}/chunks/${chunkId}`,
        })
      )
    }
  }
  return {
    allowedOrigins: [new URL(apiUrl).origin, new URL(webUrl).origin],
    invalidReasons: migration.invalidReasons,
    operations: [
      ...commonTransitionOperations({
        apiUrl,
        contextName: "owner-revocation",
        targetPath: "/research-workspace",
        webUrl,
      }),
      ...ownerOperations,
    ],
  }
}

export const buildMemberChatsTransitionPolicy = ({ apiUrl, conversationId, webUrl }) => {
  const encodedConversationId = encodeURIComponent(String(conversationId)).toLowerCase()
  const chatsPath = `/api/v1/chats/${encodedConversationId}`
  const operations = [
    ["chats-openapi", "GET", "/openapi.json"],
    ["chats-docs-info", "GET", "/api/v1/config/docs-info"],
    ["chats-ingestion-capabilities", "GET", "/api/v1/ingestion-sources/capabilities"],
    ["chats-audio-service-health", "GET", "/api/v1/audio/health"],
    ["chats-character-catalog", "GET", "/api/v1/characters/"],
    ["chats-audio-health", "GET", "/api/v1/audio/transcriptions/health"],
    ["chats-voice-catalog", "GET", "/api/v1/audio/voices/catalog"],
    [
      "chats-share-links",
      "GET",
      `/api/v1/chat/conversations/${encodedConversationId}/share-links`,
    ],
    ["chats-list", "GET", "/api/v1/chats/"],
    ["chats-messages", "GET", `${chatsPath}/messages`],
    ["chats-research-runs", "GET", `${chatsPath}/research-runs`],
    ["chats-settings", "GET", `${chatsPath}/settings`],
    ["chats-provider-config", "GET", "/api/v1/config/providers"],
    ["chats-persona-catalog", "GET", "/api/v1/persona/catalog"],
    ["chats-prompt-capabilities", "GET", "/api/v1/prompts/capabilities"],
    ["chats-implicit-feedback", "POST", "/api/v1/rag/feedback/implicit"],
  ].map(([name, method, pathname]) =>
    declaredTransitionOperation({
      contextName: "member-chats",
      method,
      name,
      origin: apiUrl,
      path: pathname,
    })
  )
  return {
    allowedOrigins: [new URL(apiUrl).origin, new URL(webUrl).origin],
    invalidReasons: [],
    operations: [
      ...commonTransitionOperations({
        apiUrl,
        contextName: "member-chats",
        targetPath: "/chat",
        webUrl,
      }),
      ...operations,
    ],
  }
}

const forbiddenTransitionRequest = (entry) => {
  const pathname = requestPath(entry)
  const method = String(entry.method || "GET").toUpperCase()
  if (/\/api\/v1\/sharing\/shared-with-me\/[^/]+\/(?:media|full-media)(?:\/|$)/.test(pathname)) {
    return "removed_full_media"
  }
  if (/\/api\/v1\/(?:workspaces|research-workspace)(?:\/|$)/.test(pathname)) {
    return "local_workspace"
  }
  if (/\/api\/v1\/(?:prompt-)?studio(?:\/|$)/.test(pathname)) return "studio"
  if (/\/api\/v1\/notes?(?:\/|$)/.test(pathname)) return "notes"
  if (/\/api\/v1\/(?:mcp|acp|sandbox|artifacts?)(?:\/|$)/.test(pathname)) {
    return "local_tool"
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && /\/sources(?:\/|$)/.test(pathname)) {
    return "source_mutation"
  }
  if (
    !["GET", "HEAD", "OPTIONS"].includes(method) &&
    /\/api\/v1\/(?:media|ingestion|web-clips?|web-clipper|clips?|capture)(?:\/|$)/.test(pathname)
  ) {
    return "extension_writable_destination"
  }
  return null
}

const validateTransitionOperationPolicy = (policy) => {
  const operations = Array.isArray(policy?.operations) ? policy.operations : []
  const allowedOrigins = Array.isArray(policy?.allowedOrigins) ? policy.allowedOrigins : []
  return (
    Array.isArray(policy?.invalidReasons) &&
    policy.invalidReasons.length === 0 &&
    operations.length > 0 &&
    operations.length <= MAX_TRANSITION_OPERATION_DECLARATIONS &&
    allowedOrigins.length > 0 &&
    new Set(allowedOrigins).size === allowedOrigins.length &&
    new Set(operations.map((operation) => operation.name)).size === operations.length &&
    operations.every(
      (operation) =>
        String(operation.name || "").trim() &&
        allowedOrigins.includes(operation.origin) &&
        Number.isInteger(operation.maximumCount) &&
        operation.maximumCount >= 1 &&
        operation.maximumCount <= MAX_TRANSITION_REQUESTS &&
        Array.isArray(operation.allowedStatuses) &&
        operation.allowedStatuses.length > 0 &&
        operation.allowedStatuses.every(
          (status) => Number.isInteger(status) && status >= 100 && status <= 599
        ) &&
        new Set(operation.allowedStatuses).size === operation.allowedStatuses.length &&
        operation.allowedStatuses.every(
          (status, index) => index === 0 || operation.allowedStatuses[index - 1] < status
        ) &&
        Boolean(operation.path) !== Boolean(operation.pathPrefix)
    )
  )
}

const transitionOperationMatches = (entry, operation) => {
  if (String(entry.method || "").toUpperCase() !== operation.method) return false
  if (transitionRequestOrigin(entry) !== operation.origin) return false
  const pathname = requestPath(entry)
  return operation.path ? pathname === operation.path : pathname.startsWith(operation.pathPrefix)
}

export const classifyTransitionLedger = (
  ledger,
  { abortAllowances = [], contextName = "", operationPolicy, transitionLabel = "" } = {}
) => {
  const failures = []
  const normalizedEntry = (entry) => ({ ...entry, context: contextName })
  const allowanceUsage = new Map()
  const operationUsage = new Map()
  const operationStatuses = new Map()
  const operationPolicyValid = validateTransitionOperationPolicy(operationPolicy)
  if (!operationPolicyValid) failures.push("malformed_transition_operation_policy")
  if (
    abortAllowances.length > MAX_TRANSITION_ABORT_ALLOWANCES ||
    abortAllowances.some(
      (allowance) =>
        !String(allowance.id || "").trim() ||
        String(allowance.method || "").toUpperCase() !== "GET" ||
        !String(allowance.url || "").startsWith("http") ||
        !Number.isInteger(allowance.count) ||
        allowance.count < 1 ||
        allowance.count > 2
    ) ||
    new Set(abortAllowances.map((allowance) => allowance.id)).size !== abortAllowances.length
  ) {
    failures.push("malformed_transition_abort_allowance")
  }
  for (const entry of ledger.requests || []) {
    const classifiedEntry = normalizedEntry(entry)
    const origin = transitionRequestOrigin(classifiedEntry)
    if (!operationPolicy?.allowedOrigins?.includes(origin)) {
      failures.push(`unknown_origin: ${entry.method} ${requestPath(entry)}`)
    }
    const matchingOperation = operationPolicy?.operations?.find((operation) =>
      transitionOperationMatches(classifiedEntry, operation)
    )
    if (!matchingOperation) {
      failures.push(`undeclared_transition_operation: ${entry.method} ${requestPath(entry)}`)
    } else {
      const used = (operationUsage.get(matchingOperation.name) || 0) + 1
      operationUsage.set(matchingOperation.name, used)
      const statuses = operationStatuses.get(matchingOperation.name) || []
      statuses.push(entry.status)
      operationStatuses.set(matchingOperation.name, statuses)
      if (used > matchingOperation.maximumCount) {
        failures.push(`transition_operation_bound: ${matchingOperation.name}`)
      }
      if (
        !Number.isInteger(entry.status) ||
        !Array.isArray(matchingOperation.allowedStatuses) ||
        !matchingOperation.allowedStatuses.includes(entry.status)
      ) {
        failures.push(
          `unexpected_transition_status: ${matchingOperation.name} ${String(entry.status)}`
        )
      }
    }
    const forbidden = forbiddenTransitionRequest(classifiedEntry)
    if (forbidden && !matchingOperation?.allowedForbiddenKinds?.includes(forbidden)) {
      failures.push(`${forbidden}: ${entry.method} ${requestPath(entry)}`)
    }
  }
  for (const entry of ledger.requestFailures || []) {
    const classifiedEntry = normalizedEntry(entry)
    const origin = transitionRequestOrigin(classifiedEntry)
    if (!operationPolicy?.allowedOrigins?.includes(origin)) {
      failures.push(`unknown_origin: ${entry.method} ${requestPath(entry)}`)
      continue
    }
    const allowanceIndex = abortAllowances.findIndex((allowance, index) => {
      const used = allowanceUsage.get(index) || 0
      return (
        used < allowance.count &&
        String(entry.error || "") === "net::ERR_ABORTED" &&
        String(entry.method || "").toUpperCase() === allowance.method &&
        entry.url === allowance.url
      )
    })
    if (allowanceIndex >= 0) {
      allowanceUsage.set(allowanceIndex, (allowanceUsage.get(allowanceIndex) || 0) + 1)
      continue
    }
    const forbidden = forbiddenTransitionRequest(classifiedEntry)
    if (forbidden) {
      failures.push(`${forbidden}: ${entry.method} ${requestPath(entry)}`)
    }
    failures.push(`request_failed: ${entry.method || ""} ${requestPath(entry)}`)
  }
  for (const entry of ledger.pageErrors || []) {
    failures.push(`page_error: ${entry.message || "unknown"}`)
  }
  for (const entry of ledger.consoleErrors || []) {
    failures.push(`console_error: ${entry.message || "unknown"}`)
  }
  for (const entry of ledger.runtimeOverlays || []) {
    failures.push(`runtime_overlay: ${entry.text || "detected"}`)
  }
  const observedRequests = [
    ...(ledger.requests || []).map((entry) => ({
      errorHash: null,
      kind: "response",
      method: String(entry.method || "").toUpperCase(),
      requestHash: sha256(`${entry.method || ""} ${entry.url || ""}`),
      status: Number(entry.status),
    })),
    ...(ledger.requestFailures || []).map((entry) => ({
      errorHash: sha256(entry.error || "request failed"),
      kind: "failure",
      method: String(entry.method || "").toUpperCase(),
      requestHash: sha256(`${entry.method || ""} ${entry.url || ""}`),
      status: null,
    })),
  ]
  if (observedRequests.length > MAX_TRANSITION_REQUESTS) {
    failures.push("transition_request_bound_exceeded")
  }
  const proof = {
    allowedAbortCount: [...allowanceUsage.values()].reduce((total, count) => total + count, 0),
    allowedAborts: abortAllowances.map((allowance, index) => ({
      count: allowanceUsage.get(index) || 0,
      id: allowance.id,
      maximumCount: allowance.count,
      method: allowance.method,
      requestHash: sha256(`${allowance.method} ${allowance.url}`),
    })),
    consoleErrorCount: (ledger.consoleErrors || []).length,
    context: contextName,
    labelHash: sha256(transitionLabel),
    maximumRequestCount: MAX_TRANSITION_REQUESTS,
    observedRequests: observedRequests.slice(0, MAX_TRANSITION_REQUESTS),
    maximumOperationDeclarations: MAX_TRANSITION_OPERATION_DECLARATIONS,
    operations: (operationPolicy?.operations || []).map((operation) => ({
      allowedStatuses: [...(operation.allowedStatuses || [])],
      count: operationUsage.get(operation.name) || 0,
      maximumCount: operation.maximumCount,
      name: operation.name,
      observedStatuses: [...(operationStatuses.get(operation.name) || [])],
    })),
    pageErrorCount: (ledger.pageErrors || []).length,
    registeredAbortCount: abortAllowances.length,
    registeredOperationCount: (operationPolicy?.operations || []).length,
    requestCount: (ledger.requests || []).length + (ledger.requestFailures || []).length,
    runtimeOverlayCount: (ledger.runtimeOverlays || []).length,
    unexpectedRequestCount: failures.filter((failure) =>
      /(?:_operation|_origin|_status|request_failed|removed_full_media|local_workspace|local_tool|source_mutation)/.test(
        failure
      )
    ).length,
    withinRequestBound: observedRequests.length <= MAX_TRANSITION_REQUESTS,
  }
  return { failures, ok: failures.length === 0, proof }
}

export const beginStrictLedgerAfterTransition = async ({
  attach = attachLedger,
  contextName,
  ledger,
  page,
  transition,
  transitionAbortAllowances = [],
  transitionLabel,
  transitionLedger,
  transitionOperationPolicy,
  transitionProof = [],
}) => {
  const transitionController = attach(page, `${contextName}-transition`, transitionLedger)
  try {
    await transition()
    await transitionController.waitForIdle(transitionLabel)
    const transitionResult = classifyTransitionLedger(transitionLedger, {
      abortAllowances: transitionAbortAllowances,
      contextName,
      operationPolicy:
        typeof transitionOperationPolicy === "function"
          ? transitionOperationPolicy(transitionLedger)
          : transitionOperationPolicy,
      transitionLabel,
    })
    transitionProof.push(transitionResult.proof)
    if (!transitionResult.ok) {
      throw new Error(
        `Transition observation failed (${transitionLabel}): ${JSON.stringify(transitionResult.failures)}`
      )
    }
  } finally {
    transitionController.dispose()
  }
  return attach(page, contextName, ledger)
}

const detectRuntimeOverlay = async (page, contextName, ledger) => {
  const overlay = await page
    .locator("nextjs-portal, [data-nextjs-dialog-overlay], [data-next-badge-root]")
    .filter({ hasText: /Unhandled Runtime Error|Build Error|Application error/i })
    .first()
    .textContent()
    .catch(() => "")
  if (overlay) ledger.runtimeOverlays.push({ context: contextName, text: overlay.slice(0, 300) })
}

const prepareContext = async (context, config, marker) => {
  await context.addInitScript(
    ({ apiUrl, markerValue }) => {
      const current = (() => {
        try {
          return JSON.parse(localStorage.getItem("tldwConfig") || "{}")
        } catch {
          return {}
        }
      })()
      const next = {
        ...current,
        apiKey: "",
        authMode: "multi-user",
        serverUrl: apiUrl,
      }
      localStorage.setItem("tldwConfig", JSON.stringify(next))
      localStorage.setItem("serverUrl", apiUrl)
      localStorage.setItem("tldwServerUrl", apiUrl)
      localStorage.setItem("tldw-api-host", apiUrl)
      localStorage.setItem("authMode", "multi-user")
      localStorage.setItem("isMigrated", "true")
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("shared-uat-context-marker", markerValue)
    },
    { apiUrl: config.apiUrl, markerValue: marker }
  )
  await context.addCookies([
    {
      name: "shared-uat-context",
      sameSite: "Strict",
      url: config.webUrl,
      value: marker,
    },
  ])
}

const loginThroughWebUi = async (page, config, username) => {
  await page.goto(`${config.webUrl}/settings/tldw`, { waitUntil: "domcontentloaded" })
  const serverInput = page.getByLabel(/server url/i)
  await serverInput.waitFor({ state: "visible", timeout: 60_000 })
  await serverInput.fill(config.apiUrl)

  const authControl = page.getByText("Multi User (Login)", { exact: true })
  if (await authControl.isVisible().catch(() => false)) await authControl.click()
  const passwordMode = page.getByText("Password", { exact: true })
  if (await passwordMode.isVisible().catch(() => false)) await passwordMode.click()
  await page.getByLabel(/^username$/i).fill(username)
  await page.locator("input#password").fill(config.fixturePassword)
  const unreadCountSettled = page.waitForResponse(
    (response) => {
      const pathname = new URL(response.url()).pathname
      return (
        response.request().method() === "GET" && pathname === "/api/v1/notifications/unread-count"
      )
    },
    { timeout: 30_000 }
  )
  await page.getByRole("button", { name: /^login$/i }).click()
  await page.getByText("Logged In", { exact: true }).waitFor({ timeout: 30_000 })
  await unreadCountSettled
}

const createOwnerFixtureWorkspaceThroughUi = async (ownerPage, config, workspaceName) => {
  await ownerPage.goto(`${config.webUrl}/research-workspace`, {
    waitUntil: "domcontentloaded",
  })
  const workspaceSwitcher = ownerPage.getByTestId("workspace-workspaces-button")
  await workspaceSwitcher.waitFor({ state: "visible", timeout: 60_000 })
  await ownerPage.getByRole("button", { name: "Rename workspace" }).waitFor({
    state: "visible",
    timeout: 60_000,
  })
  await ownerPage.waitForTimeout(1_500)

  const createdWorkspaceResponsePromise = ownerPage.waitForResponse(
    (response) => {
      if (response.request().method() !== "PUT") return false
      let pathname
      let body
      try {
        pathname = new URL(response.url()).pathname
        body = response.request().postDataJSON()
      } catch {
        return false
      }
      return /^\/api\/v1\/workspaces\/[^/]+$/.test(pathname) && body?.name === "New Research"
    },
    { timeout: 30_000 }
  )
  await workspaceSwitcher.click()
  await ownerPage.getByRole("menuitem", { name: "New Workspace" }).click()
  const createdWorkspaceResponse = await createdWorkspaceResponsePromise
  if (!createdWorkspaceResponse.ok()) {
    throw new Error(
      `Owner UI workspace creation failed status=${createdWorkspaceResponse.status()}`
    )
  }
  const createdPath = new URL(createdWorkspaceResponse.url()).pathname
  const workspaceId = decodeURIComponent(createdPath.split("/").at(-1) || "")
  if (!workspaceId) throw new Error("Owner UI workspace creation returned no workspace ID")

  await ownerPage.getByRole("heading", { name: "New Research", exact: true }).waitFor()
  await ownerPage.getByRole("button", { name: "Rename workspace" }).click()
  const workspaceNameInput = ownerPage.getByRole("textbox", { name: "Workspace name" })
  await workspaceNameInput.fill(workspaceName)
  const renamedWorkspaceResponsePromise = ownerPage.waitForResponse(
    (response) => {
      if (response.request().method() !== "PUT") return false
      let pathname
      let body
      try {
        pathname = new URL(response.url()).pathname
        body = response.request().postDataJSON()
      } catch {
        return false
      }
      return (
        pathname === `/api/v1/workspaces/${encodeURIComponent(workspaceId)}` &&
        body?.name === workspaceName
      )
    },
    { timeout: 30_000 }
  )
  await ownerPage.getByRole("button", { name: "Save", exact: true }).click()
  const renamedWorkspaceResponse = await renamedWorkspaceResponsePromise
  if (!renamedWorkspaceResponse.ok()) {
    throw new Error(`Owner UI workspace rename failed status=${renamedWorkspaceResponse.status()}`)
  }
  await ownerPage.getByRole("heading", { name: workspaceName, exact: true }).waitFor()
  return workspaceId
}

const settleOwnerFixtureWorkspaceForRevocation = async (ownerPage, config, fixture) => {
  const tracked = new WeakSet()
  let pending = 0
  let lastActivity = Date.now()
  const shouldTrack = (request) => {
    try {
      const url = new URL(request.url())
      return url.origin === config.apiUrl && url.pathname !== "/api/v1/notifications/stream"
    } catch {
      return false
    }
  }
  const onRequest = (request) => {
    if (!shouldTrack(request)) return
    tracked.add(request)
    pending += 1
    lastActivity = Date.now()
  }
  const onSettled = (request) => {
    if (!tracked.has(request)) return
    tracked.delete(request)
    pending = Math.max(0, pending - 1)
    lastActivity = Date.now()
  }
  ownerPage.on("request", onRequest)
  ownerPage.on("requestfinished", onSettled)
  ownerPage.on("requestfailed", onSettled)
  try {
    await ownerPage.goto(`${config.webUrl}/research-workspace`, {
      waitUntil: "domcontentloaded",
    })
    await ownerPage
      .getByRole("heading", { name: fixture.workspaceName, exact: true })
      .waitFor({ state: "visible", timeout: 30_000 })
    await ownerPage
      .getByTestId("workspace-share-button")
      .waitFor({ state: "visible", timeout: 30_000 })
    const deadline = Date.now() + 30_000
    while (pending > 0 || Date.now() - lastActivity < 750) {
      if (Date.now() >= deadline) {
        throw new Error(
          `Owner workspace did not settle before revocation: ${JSON.stringify({ pending })}`
        )
      }
      await sleep(100)
    }
    await ownerPage.evaluate(async () => {
      await document.fonts?.ready
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
    })
  } finally {
    ownerPage.off("request", onRequest)
    ownerPage.off("requestfinished", onSettled)
    ownerPage.off("requestfailed", onSettled)
  }
}

const contextStorageProof = async (pages) => {
  const snapshots = []
  for (const [persona, page] of Object.entries(pages)) {
    const context = page.context()
    const cookies = await context.cookies()
    const state = await page.evaluate(() => ({
      config: localStorage.getItem("tldwConfig") || "",
      marker: localStorage.getItem("shared-uat-context-marker"),
      storageKeys: Object.keys(localStorage).sort(),
    }))
    snapshots.push({
      configHash: sha256(state.config),
      cookieHash: sha256(JSON.stringify(cookies)),
      markerHash: sha256(state.marker),
      markerCookieHash: sha256(
        cookies.find((cookie) => cookie.name === "shared-uat-context")?.value || ""
      ),
      persona,
      storageKeyHash: sha256(state.storageKeys.join("\n")),
    })
  }
  const markers = new Set(snapshots.map((snapshot) => snapshot.markerHash))
  const cookieMarkers = new Set(snapshots.map((snapshot) => snapshot.markerCookieHash))
  const tokenHashes = new Set(snapshots.map((snapshot) => snapshot.configHash))
  const matchingMarkers = snapshots.every(
    (snapshot) => snapshot.markerHash === snapshot.markerCookieHash
  )
  if (
    markers.size !== 3 ||
    cookieMarkers.size !== 3 ||
    tokenHashes.size !== 3 ||
    !matchingMarkers
  ) {
    throw new Error("CDP browser contexts did not preserve isolated cookies and storage")
  }
  return snapshots
}

const addExpectedFailure = (
  ledger,
  context,
  method,
  status,
  url,
  { bodyHash = null, consoleErrorCount = 1, operationId }
) => {
  ledger.expectedHttpFailures.push({
    bodyHash,
    consoleErrorCount,
    context,
    method,
    operationId,
    status,
    url,
  })
}

const assertNoSentinel = async (page) => {
  const body = await page.locator("body").innerText()
  if (body.includes(OWNER_SENTINEL) || body.includes(RECIPIENT_SENTINEL)) {
    throw new Error("A fixture sentinel leaked into the shared workspace surface")
  }
}

const waitForSharedShell = async (page, phase, ledger) => {
  try {
    await page.locator('[data-testid="shared-workspace-shell"]').waitFor({ timeout: 30_000 })
  } catch (error) {
    const pageUrl = new URL(page.url())
    const diagnostics = {
      consoleErrorHashes: ledger.consoleErrors.slice(-5).map((entry) => sha256(entry.message)),
      pageErrorHashes: ledger.pageErrors.slice(-5).map((entry) => sha256(entry.message)),
      phase,
      requestStatusTail: ledger.requests.slice(-12).map((entry) => ({
        context: entry.context,
        method: entry.method,
        path: requestPath(entry),
        status: entry.status,
      })),
      route: `${pageUrl.pathname}${pageUrl.search}`,
      routePending:
        (await page.locator('[data-testid="research-workspace-route-pending"]').count()) > 0,
      titleHash: sha256(await page.title()),
      unavailable:
        (await page
          .getByRole("heading", { name: "This shared workspace isn't available." })
          .count()) > 0,
    }
    throw new Error(`Shared shell timeout: ${JSON.stringify(diagnostics)}`, { cause: error })
  }
}

const waitForNewAssistant = async (page, priorCount) => {
  const messages = page.locator('[data-testid="shared-workspace-chat-pane"] article')
  await page.waitForFunction(
    ({ selector, prior }) => document.querySelectorAll(selector).length >= prior + 2,
    { prior: priorCount, selector: '[data-testid="shared-workspace-chat-pane"] article' },
    { timeout: 120_000 }
  )
  const latest = messages.last()
  await latest.getByText(/.+/).first().waitFor({ state: "visible" })
  return latest
}

const askQuestion = async (page, question) => {
  const messages = page.locator('[data-testid="shared-workspace-chat-pane"] article')
  const prior = await messages.count()
  const responsePromise = page.waitForResponse((response) => {
    if (response.request().method() !== "POST") return false
    try {
      return /^\/api\/v1\/sharing\/shared-with-me\/[^/]+\/chat$/.test(
        new URL(response.url()).pathname
      )
    } catch {
      return false
    }
  })
  await page.getByLabel("Ask about shared sources").fill(question)
  await page.getByLabel("Ask shared workspace").click()
  const response = await responsePromise
  if (!response.ok()) {
    const body = await response.json().catch(() => ({}))
    const rawCode = String(body?.detail?.code || "unknown")
    const code = /^[a-z0-9_]{1,80}$/.test(rawCode) ? rawCode : "unknown"
    throw new Error(`Shared chat failed status=${response.status()} code=${code}`)
  }
  return waitForNewAssistant(page, prior)
}

const inspectLayout = async (page, mode) => {
  const metrics = await page.evaluate(() => {
    const visible = (element) => {
      const rect = element.getBoundingClientRect()
      const style = getComputedStyle(element)
      return (
        rect.width > 0 &&
        rect.height > 0 &&
        style.visibility !== "hidden" &&
        style.display !== "none"
      )
    }
    const controls = Array.from(document.querySelectorAll("button,input,select,textarea,a"))
      .filter(visible)
      .map((element) => {
        const rect = element.getBoundingClientRect()
        return {
          bottom: Math.round(rect.bottom),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          tag: element.tagName.toLowerCase(),
          testId: element.getAttribute("data-testid"),
          top: Math.round(rect.top),
        }
      })
    const intersectsViewport = (rect) =>
      rect.right > 0 && rect.left < innerWidth && rect.bottom > 0 && rect.top < innerHeight
    const horizontalOffenders = controls
      .filter(intersectsViewport)
      .filter((rect) => rect.left < -1 || rect.right > innerWidth + 1)
    const verticalScrollContainers = Array.from(
      document.querySelectorAll('[data-testid^="shared-workspace-"] *')
    ).filter((element) => {
      if (!visible(element)) return false
      const overflowY = getComputedStyle(element).overflowY
      return overflowY === "auto" || overflowY === "scroll"
    }).length
    const tabs = Array.from(document.querySelectorAll('[role="tab"]')).filter(visible)
    return {
      activeTabs: tabs.filter((element) => element.getAttribute("aria-selected") === "true").length,
      bodyOverflowX: document.body.scrollWidth - document.body.clientWidth,
      documentOverflowX:
        document.documentElement.scrollWidth - document.documentElement.clientWidth,
      horizontalOffenderCount: horizontalOffenders.length,
      horizontalOffenders: horizontalOffenders.slice(0, 8),
      panes: Array.from(document.querySelectorAll('[data-testid^="shared-workspace-"]'))
        .filter(visible)
        .map((element) => element.getAttribute("data-testid")),
      tabs: tabs.length,
      verticalScrollContainers,
    }
  })
  if (
    metrics.documentOverflowX !== 0 ||
    metrics.bodyOverflowX !== 0 ||
    metrics.horizontalOffenderCount > 0
  ) {
    throw new Error(
      `${mode} shared workspace layout overflowed horizontally: ${JSON.stringify(metrics)}`
    )
  }
  if (
    mode === "mobile" &&
    (metrics.tabs !== 2 || metrics.activeTabs !== 1 || metrics.verticalScrollContainers < 1)
  ) {
    throw new Error(`Mobile shared workspace controls are incomplete: ${JSON.stringify(metrics)}`)
  }
  return metrics
}

const stabilizeInitialDesktopCapture = async (page) => {
  const headerToggle = page.getByTestId("chat-header-sidebar-toggle")
  await headerToggle.waitFor({ state: "visible", timeout: 30_000 })
  const neutralSurface = page.locator('[data-testid="shared-workspace-shell"] header p').first()
  await neutralSurface.waitFor({ state: "visible", timeout: 30_000 })
  await page.evaluate(async () => {
    await document.fonts?.ready
    window.scrollTo({ behavior: "instant", left: 0, top: 0 })

    const headerControl = document.querySelector('[data-testid="chat-header-sidebar-toggle"]')
    for (
      let scrollAncestor = headerControl?.parentElement;
      scrollAncestor;
      scrollAncestor = scrollAncestor.parentElement
    ) {
      if (scrollAncestor.scrollLeft !== 0 || scrollAncestor.scrollTop !== 0) {
        scrollAncestor.scrollTo({ behavior: "instant", left: 0, top: 0 })
      }
    }

    const snapshot = () => {
      const shell = document.querySelector('[data-testid="shared-workspace-shell"]')
      if (!headerControl || !shell) return null
      const headerRect = headerControl.getBoundingClientRect()
      const shellRect = shell.getBoundingClientRect()
      return JSON.stringify([
        Math.round(headerRect.top),
        Math.round(headerRect.bottom),
        Math.round(shellRect.top),
        Math.round(shellRect.bottom),
        Math.round(window.scrollX),
        Math.round(window.scrollY),
      ])
    }

    let previous = null
    for (let frame = 0; frame < 8; frame += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve))
      const current = snapshot()
      if (current && current === previous) return
      previous = current
    }
    throw new Error("Hosted header and shared workspace layout did not stabilize")
  })
  await neutralSurface.click()
  await page.waitForFunction(() => !document.activeElement?.matches('h1[tabindex="-1"]'))
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  )
}

const stabilizeMobilePreviewCapture = async (page, fixture) => {
  const previewDialog = page
    .locator('[role="dialog"]:visible')
    .filter({ hasText: fixture.sourceDefs[0].title })
    .last()
  await previewDialog.waitFor({ state: "visible", timeout: 30_000 })
  await previewDialog
    .getByText(fixture.sourceDefs[0].title, { exact: true })
    .waitFor({ state: "visible", timeout: 30_000 })
  await previewDialog
    .getByText("AMBER-SIGNED-DATE-2024-03-17", { exact: false })
    .waitFor({ state: "visible", timeout: 30_000 })
  await previewDialog.evaluate(async (element) => {
    await document.fonts?.ready
    let previous = null
    let stableFrames = 0
    let geometry = null
    for (let frame = 0; frame < 45; frame += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve))
      const rect = element.getBoundingClientRect()
      geometry = {
        bottom: Math.round(rect.bottom),
        height: Math.round(rect.height),
        left: Math.round(rect.left),
        right: Math.round(rect.right),
        top: Math.round(rect.top),
        viewportHeight: window.innerHeight,
        viewportWidth: window.innerWidth,
        width: Math.round(rect.width),
      }
      const current = JSON.stringify(geometry)
      stableFrames = current === previous ? stableFrames + 1 : 0
      previous = current
      if (stableFrames >= 2) break
    }
    if (
      !geometry ||
      geometry.width < window.innerWidth - 2 ||
      geometry.left < -1 ||
      geometry.right > window.innerWidth + 1 ||
      geometry.top < -1 ||
      geometry.bottom > window.innerHeight + 1 ||
      stableFrames < 2
    ) {
      throw new Error(
        `Mobile source preview did not stabilize in viewport: ${JSON.stringify(geometry)}`
      )
    }
  })
}

const stabilizeRevokedCapture = async (page) => {
  const neutralSurface = page.locator("main").first()
  await neutralSurface.waitFor({ state: "visible", timeout: 30_000 })
  await neutralSurface.click({ position: { x: 8, y: 8 } })
  await page.waitForFunction(() => !document.activeElement?.matches(`h1[tabindex='-1']`))
  await page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  )
}

const fetchFromPage = async (page, url, init = {}) =>
  page.evaluate(
    async ({ requestInit, requestUrl }) => {
      const storedConfig = localStorage.getItem("tldwConfig")
      const token = storedConfig ? JSON.parse(storedConfig).accessToken : null
      if (!token) throw new Error("Authenticated browser context has no access token")
      const headers = new Headers(requestInit.headers || {})
      headers.set("Authorization", `Bearer ${token}`)
      const response = await fetch(requestUrl, { ...requestInit, headers })
      return { body: await response.text(), status: response.status }
    },
    { requestInit: init, requestUrl: url }
  )

const raceProbe = async (memberPage, config, fixture, ledger, target) => {
  const requestId = crypto.randomUUID()
  const baseBody = {
    model: target.model,
    provider: target.provider,
    query: `Return the exact AMBER-SIGNED-DATE-2024-03-17 token from the Amber source. Race ${config.runId}`,
    request_id: requestId,
    source_scope: { mode: "include", source_ids: [fixture.sourceDefs[0].id] },
  }
  const url = `${config.apiUrl}/api/v1/sharing/shared-with-me/${fixture.shareId}/chat`
  const changedBody = { ...baseBody, query: `${baseBody.query} changed fingerprint` }
  const baseBodyHash = sha256(JSON.stringify(baseBody))
  const changedBodyHash = sha256(JSON.stringify(changedBody))
  addExpectedFailure(ledger, "member", "POST", 409, url, {
    bodyHash: baseBodyHash,
    operationId: "race-concurrent-conflict",
  })
  addExpectedFailure(ledger, "member", "POST", 409, url, {
    bodyHash: changedBodyHash,
    operationId: "race-fingerprint-conflict",
  })
  const call = (body) => {
    const started = performance.now()
    return fetchFromPage(memberPage, url, {
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    }).then((result) => ({ ...result, timingMs: Math.round(performance.now() - started) }))
  }
  const raced = await Promise.all([call(baseBody), call(baseBody)])
  const successful = raced.find((result) => result.status === 200)
  if (!successful)
    throw new Error(`Matching request race had no successful writer: ${raced.map((r) => r.status)}`)
  const replay = await call(baseBody)
  const conflict = await call(changedBody)
  if (replay.status !== 200 || conflict.status !== 409) {
    throw new Error(
      `Race replay/conflict contract failed: replay=${replay.status} conflict=${conflict.status}`
    )
  }
  const canonicalSuccess = JSON.parse(successful.body)
  const canonicalReplay = JSON.parse(replay.body)
  if (
    canonicalSuccess.turn?.assistant_message?.message_id !==
      canonicalReplay.turn?.assistant_message?.message_id ||
    canonicalReplay.replay?.replayed !== true
  ) {
    throw new Error("Matching request replay did not return the stored turn")
  }
  const conflictBody = JSON.parse(conflict.body || "{}")
  if (conflictBody?.detail?.code !== "request_id_conflict") {
    throw new Error("Changed request fingerprint did not return typed request_id_conflict")
  }
  return {
    conversationId: canonicalSuccess.conversation_id,
    evidence: {
      operations: [
        {
          bodyHash: baseBodyHash,
          operationId: "race-concurrent-conflict",
          status: 409,
        },
        {
          bodyHash: changedBodyHash,
          operationId: "race-fingerprint-conflict",
          status: 409,
        },
      ],
      requestHashes: [baseBodyHash, changedBodyHash],
      requestIdHash: sha256(requestId),
      responseHashes: [sha256(successful.body), sha256(replay.body)],
      statuses: [...raced.map((result) => result.status), replay.status, conflict.status],
      timingsMs: [...raced.map((result) => result.timingMs), replay.timingMs, conflict.timingMs],
      turnHashes: [
        sha256(canonicalSuccess.turn.assistant_message.message_id),
        sha256(canonicalReplay.turn.assistant_message.message_id),
      ],
    },
  }
}

const writeEvidence = (config, evidence) => {
  fs.mkdirSync(config.evidenceDir, { recursive: true })
  fs.writeFileSync(
    path.join(config.evidenceDir, "evidence.json"),
    `${JSON.stringify(evidence, null, 2)}\n`
  )
}

const screenshotEvidence = (config) =>
  Object.fromEntries(
    Object.entries(SCREENSHOT_NAMES).map(([key, filename]) => {
      const screenshotPath = path.join(config.evidenceDir, filename)
      return [key, fs.existsSync(screenshotPath) ? filename : ""]
    })
  )

export const runLiveUat = async (config) => {
  const startedAt = new Date().toISOString()
  fs.mkdirSync(config.evidenceDir, { recursive: true })
  for (const filename of [...Object.values(SCREENSHOT_NAMES), "evidence.json"]) {
    fs.rmSync(path.join(config.evidenceDir, filename), { force: true })
  }
  const ledger = makeLedger()
  const acceptance = Object.fromEntries(ACCEPTANCE_NAMES.map((name) => [name, false]))
  let contextProof = []
  let provider = null
  let providerContextProof = null
  let providerReadiness = null
  let race = null
  let recipientConversationId = null
  let fixture = null
  const transitionProof = []
  let browser = null
  let ownerContext = null
  let memberContext = null
  let nonmemberContext = null
  const ledgerControllers = []
  const disposeLedgers = () => {
    while (ledgerControllers.length > 0) ledgerControllers.pop()?.dispose()
    ledger.closed = true
  }

  try {
    await resetProviderProbe(config)
    fixture = await provisionFixture(config)
    browser = await chromium.connectOverCDP(config.cdpUrl)
    ownerContext = await browser.newContext({ viewport: { height: 900, width: 1440 } })
    memberContext = await browser.newContext({ viewport: { height: 900, width: 1440 } })
    nonmemberContext = await browser.newContext({ viewport: { height: 900, width: 1440 } })
    const ownerPage = await ownerContext.newPage()
    const memberPage = await memberContext.newPage()
    const nonmemberPage = await nonmemberContext.newPage()
    await prepareContext(ownerContext, config, `owner-${config.runId}`)
    await prepareContext(memberContext, config, `member-${config.runId}`)
    await prepareContext(nonmemberContext, config, `nonmember-${config.runId}`)
    await loginThroughWebUi(ownerPage, config, fixture.owner.username)
    await loginThroughWebUi(memberPage, config, fixture.member.username)
    await loginThroughWebUi(nonmemberPage, config, fixture.nonmember.username)
    contextProof = await contextStorageProof({
      member: memberPage,
      nonmember: nonmemberPage,
      owner: ownerPage,
    })
    acceptance.contextIsolation = contextProof.length === 3
    const ownerWorkspaceId = await createOwnerFixtureWorkspaceThroughUi(
      ownerPage,
      config,
      fixture.workspaceName
    )
    await ownerPage.goto("about:blank", { waitUntil: "load" })
    fixture = await finalizeFixtureWorkspace(config, fixture, ownerWorkspaceId)
    await Promise.all(
      [memberPage, nonmemberPage].map((page) => page.goto("about:blank", { waitUntil: "load" }))
    )

    const ownerRecipientPage = await ownerContext.newPage()
    const ownerLedger = attachLedger(ownerRecipientPage, "owner", ledger)
    const memberLedger = attachLedger(memberPage, "member", ledger)
    const nonmemberLedger = attachLedger(nonmemberPage, "nonmember", ledger)
    ledgerControllers.push(ownerLedger, memberLedger, nonmemberLedger)
    const sharedUrl = `${config.webUrl}/research-workspace?shared=${fixture.shareId}`
    const memberBootstrapResponsePromise = memberPage.waitForResponse(
      (response) => {
        if (response.request().method() !== "GET") return false
        try {
          return (
            new URL(response.url()).pathname ===
            `/api/v1/sharing/shared-with-me/${fixture.shareId}/workspace`
          )
        } catch {
          return false
        }
      },
      { timeout: 30_000 }
    )
    await memberPage.goto(sharedUrl, { waitUntil: "domcontentloaded" })
    const memberBootstrapResponse = await memberBootstrapResponsePromise
    if (!memberBootstrapResponse.ok()) {
      throw new Error(
        `Canonical member bootstrap failed status=${memberBootstrapResponse.status()}`
      )
    }
    const bootstrap = await memberBootstrapResponse.json()
    providerReadiness = bootstrap.generation_default
    await waitForSharedShell(memberPage, "member-initial", ledger)
    await memberPage.getByText(fixture.workspaceName, { exact: true }).waitFor()
    for (const source of fixture.sourceDefs) {
      await memberPage.getByText(source.title, { exact: true }).waitFor()
    }
    await assertNoSentinel(memberPage)
    acceptance.memberSharedIsolation = true
    acceptance.sentinelsExcluded = true
    await stabilizeInitialDesktopCapture(memberPage)
    await inspectLayout(memberPage, "desktop")
    await memberPage.screenshot({
      path: path.join(config.evidenceDir, SCREENSHOT_NAMES.desktopSharedWorkspace),
    })

    const llamaHealthy = await fetch(config.llamaHealthUrl)
      .then((response) => response.ok)
      .catch(() => false)
    provider = selectEffectiveTarget({
      llamaHealthy,
      targets: [bootstrap.generation_default],
    })
    if (!provider) throw new Error("No actually configured recipient generation target is ready")

    for (const source of fixture.sourceDefs) {
      if (!(await memberPage.getByLabel(`Select ${source.title}`).isChecked())) {
        throw new Error("All-source mode did not start with both queryable sources selected")
      }
    }
    const firstQuestion = buildAllSourcesQuestion({
      amberTitle: fixture.sourceDefs[0].title,
      cobaltTitle: fixture.sourceDefs[1].title,
    })
    const firstAssistant = await askQuestion(memberPage, firstQuestion)
    const firstAnswer = await firstAssistant.locator(":scope > div").first().innerText()
    const firstAnswerShape = {
      amberTokenPresent: firstAnswer.includes("AMBER-SIGNED-DATE-2024-03-17"),
      characterCount: firstAnswer.length,
      cobaltTokenPresent: firstAnswer.includes("COBALT-PARTICIPANTS-43"),
    }
    if (!firstAnswerShape.amberTokenPresent || !firstAnswerShape.cobaltTokenPresent) {
      throw new Error(
        `All-source answer did not copy both exact fixture tokens: ${JSON.stringify(firstAnswerShape)}`
      )
    }
    const firstCitations = firstAssistant.getByRole("button", { name: /Open citation/i })
    const firstCitationText = await firstCitations.allInnerTexts()
    if (
      firstCitationText.length < 2 ||
      !firstCitationText.some((text) => text.includes(fixture.sourceDefs[0].title)) ||
      !firstCitationText.some((text) => text.includes(fixture.sourceDefs[1].title))
    ) {
      throw new Error("All-source answer did not cite both fixture sources")
    }
    acceptance.allSourcesGrounded = true
    await firstCitations.filter({ hasText: fixture.sourceDefs[1].title }).first().click()
    await memberPage.getByRole("dialog", { name: "Source preview" }).waitFor()
    await memberPage
      .getByRole("dialog", { name: "Source preview" })
      .getByText("COBALT-PARTICIPANTS-43", { exact: false })
      .waitFor()
    const previewText = await memberPage.getByRole("dialog", { name: "Source preview" }).innerText()
    if (!previewText.includes("COBALT-PARTICIPANTS-43")) {
      throw new Error("Citation preview did not expose bounded supporting evidence")
    }
    acceptance.citationPreview = true
    await memberPage.getByLabel("Close source preview").click()

    const selectedToRemove = memberPage.getByLabel(`Select ${fixture.sourceDefs[1].title}`)
    await waitForCheckboxState(selectedToRemove, false)
    if (
      !(await memberPage.getByLabel(`Select ${fixture.sourceDefs[0].title}`).isChecked()) ||
      (await selectedToRemove.isChecked())
    ) {
      throw new Error("Subset mode did not retain exactly the Amber source")
    }
    const subsetAssistant = await askQuestion(
      memberPage,
      "Return the exact Amber protocol token from the selected source."
    )
    const subsetText = await subsetAssistant.innerText()
    if (
      !subsetText.includes("AMBER-SIGNED-DATE-2024-03-17") ||
      subsetText.includes("COBALT-PARTICIPANTS-43")
    ) {
      throw new Error("Subset answer escaped the selected source scope")
    }
    const subsetCitationText = await subsetAssistant
      .getByRole("button", { name: /Open citation/i })
      .allInnerTexts()
    if (
      subsetCitationText.length < 1 ||
      subsetCitationText.some((text) => text.includes(fixture.sourceDefs[1].title)) ||
      subsetCitationText.some((text) => !text.includes(fixture.sourceDefs[0].title))
    ) {
      throw new Error("Subset citations included an unselected source")
    }
    acceptance.subsetGrounded = true
    await assertNoSentinel(memberPage)
    await memberPage.screenshot({
      path: path.join(config.evidenceDir, SCREENSHOT_NAMES.desktopGroundedAnswer),
    })

    const messageIds = await memberPage
      .locator("[data-message-id]")
      .evaluateAll((nodes) =>
        nodes.map((node) => node.getAttribute("data-message-id")).filter(Boolean)
      )
    await memberLedger.waitForIdle("before member reload")
    await memberPage.reload({ waitUntil: "domcontentloaded" })
    await waitForSharedShell(memberPage, "member-reload", ledger)
    for (const messageId of messageIds) {
      await memberPage.locator(`[data-message-id="${messageId}"]`).waitFor()
    }
    acceptance.historyAfterReload = true

    await ownerRecipientPage.goto(sharedUrl, { waitUntil: "domcontentloaded" })
    await waitForSharedShell(ownerRecipientPage, "owner-recipient", ledger)
    await ownerRecipientPage.getByText(fixture.workspaceName, { exact: true }).waitFor()
    if (
      await ownerRecipientPage
        .getByText(/Add source|Studio|General Chat/i)
        .isVisible()
        .catch(() => false)
    ) {
      throw new Error("Owner recipient-style view exposed local mutation controls")
    }
    acceptance.ownerRecipientView = true

    await memberLedger.waitForIdle("before malformed route")
    await memberPage.goto(`${config.webUrl}/research-workspace?shared=invalid`, {
      waitUntil: "domcontentloaded",
    })
    await memberPage
      .getByRole("heading", { name: "This shared workspace isn't available." })
      .waitFor()
    await assertNoSentinel(memberPage)
    acceptance.malformedNeutralFailure = true

    const nonmemberApiUrl = `${config.apiUrl}/api/v1/sharing/shared-with-me/${fixture.shareId}/workspace`
    addExpectedFailure(ledger, "nonmember", "GET", 404, nonmemberApiUrl, {
      operationId: "nonmember-neutral-bootstrap",
    })
    await nonmemberPage.goto(sharedUrl, { waitUntil: "domcontentloaded" })
    await nonmemberPage
      .getByRole("heading", { name: "This shared workspace isn't available." })
      .waitFor()
    await assertNoSentinel(nonmemberPage)
    acceptance.nonmemberNeutralFailure = true

    await memberLedger.waitForIdle("before member race route")
    await memberPage.goto(sharedUrl, { waitUntil: "domcontentloaded" })
    await waitForSharedShell(memberPage, "member-before-race", ledger)
    const raceResult = await raceProbe(memberPage, config, fixture, ledger, provider)
    race = raceResult.evidence
    recipientConversationId = String(raceResult.conversationId || "").trim()
    if (!recipientConversationId) {
      throw new Error("Race response did not identify the persisted recipient conversation")
    }

    await memberPage.setViewportSize({ height: 844, width: 390 })
    await inspectLayout(memberPage, "mobile")
    await memberPage.getByRole("tab", { name: "Chat" }).click()
    await memberPage.getByLabel("Ask about shared sources").waitFor()
    await memberPage.getByRole("tab", { name: "Sources" }).click()
    await memberPage.getByLabel(`Preview ${fixture.sourceDefs[0].title}`).waitFor()
    await inspectLayout(memberPage, "mobile")
    await memberPage.screenshot({
      path: path.join(config.evidenceDir, SCREENSHOT_NAMES.mobileSharedWorkspace),
    })
    await memberPage.getByLabel(`Preview ${fixture.sourceDefs[0].title}`).click()
    await stabilizeMobilePreviewCapture(memberPage, fixture)
    await inspectLayout(memberPage, "mobile")
    await memberPage.screenshot({
      path: path.join(config.evidenceDir, SCREENSHOT_NAMES.mobileSourcePreview),
    })
    await memberPage.getByLabel("Close source preview").click()
    acceptance.mobileResponsive = true
    await memberPage.setViewportSize({ height: 900, width: 1440 })

    const bodyText = await memberPage.locator("body").innerText()
    if (/trust banner|migration banner|workspace banner|local storage status/i.test(bodyText)) {
      throw new Error("Shared surface rendered an extra banner bar")
    }
    acceptance.noExtraBannerBars = true

    await detectRuntimeOverlay(ownerRecipientPage, "owner", ledger)
    await ownerLedger.waitForIdle("before owner ledger disposal")
    ownerLedger.dispose()
    const ownerRevocationLedger = await beginStrictLedgerAfterTransition({
      contextName: "owner-revocation",
      ledger,
      page: ownerPage,
      transition: () => settleOwnerFixtureWorkspaceForRevocation(ownerPage, config, fixture),
      transitionAbortAllowances: [
        {
          count: 1,
          id: "owner-workspace-context-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/workspaces/${fixture.workspaceId}/context`,
        },
      ],
      transitionLabel: "owner revocation preparation",
      transitionLedger: makeLedger(),
      transitionOperationPolicy: (transitionLedger) =>
        buildOwnerRevocationTransitionPolicy({
          apiUrl: config.apiUrl,
          ledger: transitionLedger,
          webUrl: config.webUrl,
          workspaceId: fixture.workspaceId,
        }),
      transitionProof,
    })
    const shareButton = ownerPage.getByTestId("workspace-share-button")
    ledgerControllers.push(ownerRevocationLedger)
    await shareButton.click()
    await ownerPage.getByRole("tab", { name: "Active Shares" }).click()
    const teamScopeLabel = `Team #${fixture.teamId}`
    const teamShareRow = ownerPage.getByRole("row").filter({ hasText: teamScopeLabel })
    await teamShareRow.waitFor({ state: "visible", timeout: 30_000 })
    const revokeShareButton = teamShareRow.getByRole("button", {
      name: `Revoke team share Team #${fixture.teamId}`,
    })
    await revokeShareButton.waitFor({ state: "visible", timeout: 30_000 })
    const revokeResponsePromise = ownerPage.waitForResponse((response) => {
      if (response.request().method() !== "DELETE") return false
      try {
        return new URL(response.url()).pathname === `/api/v1/sharing/shares/${fixture.shareId}`
      } catch {
        return false
      }
    })
    await revokeShareButton.click()
    await ownerPage.getByRole("button", { name: "Revoke", exact: true }).click()
    const revokeResponse = await revokeResponsePromise
    if (![200, 204].includes(revokeResponse.status())) {
      throw new Error(`Owner UI revocation failed status=${revokeResponse.status()}`)
    }
    await ownerPage.getByText("Share revoked", { exact: true }).waitFor({ timeout: 30_000 })
    await revokeShareButton.waitFor({ state: "hidden", timeout: 30_000 })
    await detectRuntimeOverlay(ownerPage, "owner-revocation", ledger)
    await ownerRevocationLedger.waitForIdle("before owner revocation disposal")
    ownerRevocationLedger.dispose()
    const revokedWorkspaceUrl = `${config.apiUrl}/api/v1/sharing/shared-with-me/${fixture.shareId}/workspace`
    addExpectedFailure(ledger, "member", "GET", 404, revokedWorkspaceUrl, {
      operationId: "revoked-workspace-bootstrap",
    })
    await memberLedger.waitForIdle("before revoked member route")
    await memberPage.goto(sharedUrl, { waitUntil: "domcontentloaded" })
    await memberPage
      .getByRole("heading", { name: "This shared workspace isn't available." })
      .waitFor()
    await assertNoSentinel(memberPage)
    acceptance.revocationFailClosed = true

    const revokedPreviewUrl = `${config.apiUrl}/api/v1/sharing/shared-with-me/${fixture.shareId}/sources/${fixture.sourceDefs[0].id}/preview`
    addExpectedFailure(ledger, "member", "GET", 404, revokedPreviewUrl, {
      operationId: "revoked-source-preview",
    })
    const blockedPreview = await fetchFromPage(memberPage, revokedPreviewUrl)
    if (blockedPreview.status !== 404)
      throw new Error("Revoked citation preview did not fail closed")
    acceptance.blockedRevokedPreview = true
    await stabilizeRevokedCapture(memberPage)
    await memberPage.screenshot({
      path: path.join(config.evidenceDir, SCREENSHOT_NAMES.revokedShare),
    })

    await detectRuntimeOverlay(memberPage, "member", ledger)
    await memberLedger.waitForIdle("before member ledger disposal")
    memberLedger.dispose()
    const sidebarToggle = memberPage.getByTestId("chat-header-sidebar-toggle")
    const memberChatBaseUrl = `${config.apiUrl}/api/v1/chats/${encodeURIComponent(recipientConversationId)}`
    const memberChatsLedger = await beginStrictLedgerAfterTransition({
      contextName: "member-chats",
      ledger,
      page: memberPage,
      transition: async () => {
        await memberPage.goto(`${config.webUrl}/chat`, { waitUntil: "domcontentloaded" })
        await sidebarToggle.waitFor({ timeout: 30_000 })
      },
      transitionAbortAllowances: [
        {
          count: 1,
          id: "chat-share-links-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/chat/conversations/${encodeURIComponent(recipientConversationId)}/share-links?scope_type=global`,
        },
        {
          count: 2,
          id: "chat-openapi-teardown",
          method: "GET",
          url: `${config.apiUrl}/openapi.json`,
        },
        {
          count: 1,
          id: "chat-profile-preferences-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/users/me/profile?sections=preferences`,
        },
        {
          count: 1,
          id: "chat-provider-config-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/config/providers`,
        },
        {
          count: 1,
          id: "chat-research-runs-teardown",
          method: "GET",
          url: `${memberChatBaseUrl}/research-runs`,
        },
        {
          count: 1,
          id: "chat-persona-catalog-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/persona/catalog`,
        },
        {
          count: 1,
          id: "chat-prompt-capabilities-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/prompts/capabilities`,
        },
        {
          count: 1,
          id: "chat-voice-catalog-teardown",
          method: "GET",
          url: `${config.apiUrl}/api/v1/audio/voices/catalog?provider=kitten_tts`,
        },
      ],
      transitionLabel: "Chats navigation",
      transitionLedger: makeLedger(),
      transitionOperationPolicy: buildMemberChatsTransitionPolicy({
        apiUrl: config.apiUrl,
        conversationId: recipientConversationId,
        webUrl: config.webUrl,
      }),
      transitionProof,
    })
    ledgerControllers.push(memberChatsLedger)
    const sidebarLabel = await sidebarToggle.getAttribute("aria-label")
    const waitForHistoryResponse = () =>
      memberPage.waitForResponse((response) => {
        if (response.request().method() !== "GET") return false
        try {
          return new URL(response.url()).pathname === "/api/v1/chats/"
        } catch {
          return false
        }
      })
    let historyResponse = null
    if (sidebarLabel === "Expand sidebar") {
      historyResponse = waitForHistoryResponse()
      await sidebarToggle.click()
    } else if (sidebarLabel !== "Collapse sidebar") {
      throw new Error("Chats sidebar toggle exposed an unexpected accessible state")
    }
    const recentConversations = memberPage.getByRole("button", {
      name: "Recent conversations",
    })
    const legacyHistoryDrawer = memberPage.locator(".ant-drawer-content:visible").first()
    let historySurface = null
    let persistentHistoryVisible = false
    for (let attempt = 0; attempt < 150; attempt += 1) {
      if (await recentConversations.isVisible().catch(() => false)) {
        historySurface = memberPage.getByTestId("chat-sidebar")
        persistentHistoryVisible = true
        break
      }
      if (await legacyHistoryDrawer.isVisible().catch(() => false)) {
        historySurface = legacyHistoryDrawer
        break
      }
      await sleep(200)
    }
    if (!historySurface) {
      throw new Error("Chats sidebar did not expose persistent or legacy history")
    }
    if (
      persistentHistoryVisible &&
      (await recentConversations.getAttribute("aria-expanded")) !== "true"
    ) {
      historyResponse ||= waitForHistoryResponse()
      await recentConversations.click()
    }
    if (historyResponse) {
      const response = await historyResponse
      if (!response.ok()) {
        throw new Error(`Chats history failed status=${response.status()}`)
      }
    }
    const persistedConversation = historySurface.getByText(fixture.workspaceName, { exact: true })
    await persistedConversation.waitFor({ timeout: 30_000 })
    const persistedMessagesResponsePromise = memberPage.waitForResponse(
      (response) => {
        if (response.request().method() !== "GET") return false
        try {
          return /\/api\/v1\/chats\/[^/]+\/messages$/.test(new URL(response.url()).pathname)
        } catch {
          return false
        }
      },
      { timeout: 30_000 }
    )
    await persistedConversation.click()
    const persistedMessagesResponse = await persistedMessagesResponsePromise
    if (!persistedMessagesResponse.ok()) {
      throw new Error(`Chats messages failed status=${persistedMessagesResponse.status()}`)
    }
    const persistedQuestion = memberPage.getByText(firstQuestion, { exact: true })
    await ensureLocatorVisibleInViewport(persistedQuestion, "Persisted question")
    const persistedAnswer = memberPage.getByText(firstAnswer, { exact: true })
    await ensureLocatorVisibleInViewport(persistedAnswer, "Persisted answer")
    acceptance.recipientChatVisibleInChats = true

    await detectRuntimeOverlay(memberPage, "member-chats", ledger)
    await memberChatsLedger.waitForIdle("before Chats ledger disposal")
    memberChatsLedger.dispose()
    await detectRuntimeOverlay(nonmemberPage, "nonmember", ledger)
    await nonmemberLedger.waitForIdle("before nonmember ledger disposal")
    disposeLedgers()
    providerContextProof = await readProviderContextProof(config)
    const evidence = createEvidenceRecord({
      acceptance,
      config,
      contextIsolationProof: contextProof,
      finishedAt: new Date().toISOString(),
      fixture,
      ledger,
      provider,
      providerContextProof,
      providerReadiness,
      raceProbe: race,
      screenshots: screenshotEvidence(config),
      startedAt,
      transitionProof,
    })
    const validation = validateEvidenceRecord(evidence)
    writeEvidence(config, evidence)
    if (validation.exitCode !== 0) {
      throw new Error(`Live evidence failed validation: ${validation.failures.join(", ")}`)
    }
    return evidence
  } catch (error) {
    disposeLedgers()
    providerContextProof ||= await readProviderContextProof(config).catch(() => null)
    const message = error instanceof Error ? error.message : String(error)
    const evidence = createEvidenceRecord({
      acceptance,
      config,
      contextIsolationProof: contextProof,
      failureMessageHash: sha256(message),
      finishedAt: new Date().toISOString(),
      fixture,
      ledger,
      provider,
      providerContextProof,
      providerReadiness,
      raceProbe: race,
      screenshots: screenshotEvidence(config),
      startedAt,
      status: "failed",
      transitionProof,
    })
    writeEvidence(config, evidence)
    throw error
  } finally {
    disposeLedgers()
    await Promise.allSettled(
      [ownerContext, memberContext, nonmemberContext]
        .filter(Boolean)
        .map((context) => context.close())
    )
    await browser?.close().catch(() => {})
  }
}

export const main = async ({ env = process.env } = {}) => {
  let config
  try {
    config = buildSharedUatConfig({ env })
    const evidence = await runLiveUat(config)
    console.log(
      `[shared-recipient-uat] passed provider=${evidence.provider.provider} model=${evidence.provider.model} evidence=${path.join(config.evidenceDir, "evidence.json")}`
    )
    return 0
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    const safeMessage = config
      ? redactValue(message, [config.adminPassword, config.fixturePassword, config.adminUsername])
      : message
    console.error(`[shared-recipient-uat] failed: ${safeMessage}`)
    return 1
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  process.exitCode = await main()
}
