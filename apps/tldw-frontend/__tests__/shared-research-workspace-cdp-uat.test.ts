import crypto from "node:crypto"
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  buildAllSourcesQuestion,
  beginStrictLedgerAfterTransition,
  buildMemberChatsTransitionPolicy,
  buildOwnerRevocationTransitionPolicy,
  buildSharedUatConfig,
  classifyTransitionLedger,
  classifyStrictLedger,
  createEvidenceRecord,
  ensureLocatorVisibleInViewport,
  selectEffectiveTarget,
  validateEvidenceRecord,
} from "../scripts/shared-research-workspace-cdp-uat.mjs"

const REQUIRED_ENV = {
  TLDW_CDP_URL: "http://127.0.0.1:9222",
  TLDW_E2E_SERVER_URL: "http://127.0.0.1:18001",
  TLDW_SHARED_UAT_ADMIN_PASSWORD: "admin-secret-value",
  TLDW_SHARED_UAT_ADMIN_USERNAME: "uat-admin",
  TLDW_SHARED_UAT_FIXTURE_PASSWORD: "fixture-secret-value",
  TLDW_WEB_URL: "http://127.0.0.1:18082",
}

const hash = (value: string) => `sha256:${crypto.createHash("sha256").update(value).digest("hex")}`

const cleanLedger = {
  closed: true,
  consoleErrors: [],
  expectedHttpFailures: [],
  pageErrors: [],
  requests: [
    {
      context: "member",
      method: "GET",
      status: 200,
      url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/workspace",
    },
    {
      context: "member-chats",
      method: "GET",
      status: 200,
      url: "http://127.0.0.1:18001/api/v1/chats/chat-1/settings?scope_type=global",
    },
  ],
  requestFailures: [],
  runtimeOverlays: [],
}

const completeAcceptance = {
  allSourcesGrounded: true,
  blockedRevokedPreview: true,
  citationPreview: true,
  contextIsolation: true,
  historyAfterReload: true,
  malformedNeutralFailure: true,
  memberSharedIsolation: true,
  mobileResponsive: true,
  noExtraBannerBars: true,
  nonmemberNeutralFailure: true,
  ownerRecipientView: true,
  recipientChatVisibleInChats: true,
  revocationFailClosed: true,
  sentinelsExcluded: true,
  subsetGrounded: true,
}

const completeScreenshots = {
  desktopGroundedAnswer: "desktop-grounded-answer.png",
  desktopSharedWorkspace: "desktop-shared-workspace.png",
  mobileSourcePreview: "mobile-source-preview.png",
  mobileSharedWorkspace: "mobile-shared-workspace.png",
  revokedShare: "revoked-share.png",
}

const completeContextIsolationProof = ["owner", "member", "nonmember"].map((persona) => ({
  configHash: hash(`${persona}-config`),
  cookieHash: hash(`${persona}-cookie`),
  markerCookieHash: hash(`${persona}-marker`),
  markerHash: hash(`${persona}-marker`),
  persona,
  storageKeyHash: hash(`${persona}-storage`),
}))

const completeProviderContextProof = {
  bodyUnchanged: true,
  forwardedRequestCount: 2,
  inputBodyHashes: [hash("provider-request-a"), hash("provider-request-b")],
  maximumRequestCount: 16,
  mutationPayloadsAbsent: true,
  outputBodyHashes: [hash("provider-request-a"), hash("provider-request-b")],
  ownerSentinelAbsent: true,
  payloadJsonValid: true,
  recipientSentinelAbsent: true,
  toolPayloadsAbsent: true,
  withinRequestBound: true,
}

const completeTransitionProof = [
  {
    allowedAbortCount: 1,
    allowedAborts: [
      {
        count: 1,
        id: "owner-workspace-context-teardown",
        maximumCount: 1,
        method: "GET",
        requestHash: hash("owner-transition-request"),
      },
    ],
    consoleErrorCount: 0,
    context: "owner-revocation",
    labelHash: hash("owner-transition"),
    maximumOperationDeclarations: 64,
    maximumRequestCount: 64,
    observedRequests: [
      {
        errorHash: hash("net::ERR_ABORTED"),
        kind: "failure",
        method: "GET",
        requestHash: hash("owner-transition-request"),
        status: null,
      },
    ],
    operations: [
      {
        count: 0,
        maximumCount: 1,
        name: "owner-workspace-context",
      },
    ],
    pageErrorCount: 0,
    registeredAbortCount: 1,
    registeredOperationCount: 1,
    requestCount: 1,
    runtimeOverlayCount: 0,
    unexpectedRequestCount: 0,
    withinRequestBound: true,
  },
  {
    allowedAbortCount: 0,
    allowedAborts: [],
    consoleErrorCount: 0,
    context: "member-chats",
    labelHash: hash("member-transition"),
    maximumOperationDeclarations: 64,
    maximumRequestCount: 64,
    observedRequests: [],
    operations: [
      {
        count: 0,
        maximumCount: 1,
        name: "chats-openapi",
      },
    ],
    pageErrorCount: 0,
    registeredAbortCount: 0,
    registeredOperationCount: 1,
    requestCount: 0,
    runtimeOverlayCount: 0,
    unexpectedRequestCount: 0,
    withinRequestBound: true,
  },
]

const makeCompleteEvidence = () => {
  const config = buildSharedUatConfig({ env: REQUIRED_ENV })
  const raceUrl = "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/chat"
  const requestAHash = hash("request-a")
  const requestBHash = hash("request-b")
  const expectedRaceFailures = [
    {
      bodyHash: requestAHash,
      consoleErrorCount: 1,
      context: "member",
      method: "POST",
      operationId: "race-concurrent-conflict",
      status: 409,
      url: raceUrl,
    },
    {
      bodyHash: requestBHash,
      consoleErrorCount: 1,
      context: "member",
      method: "POST",
      operationId: "race-fingerprint-conflict",
      status: 409,
      url: raceUrl,
    },
  ]
  const completeLedger = {
    ...cleanLedger,
    consoleErrors: expectedRaceFailures.map((entry) => ({
      context: entry.context,
      message: "Failed to load resource: the server responded with a status of 409 (Conflict)",
      status: entry.status,
      url: entry.url,
    })),
    expectedHttpFailures: expectedRaceFailures,
    requests: [
      ...cleanLedger.requests,
      ...expectedRaceFailures.map(
        ({ operationId: _operationId, consoleErrorCount: _count, ...entry }) => entry
      ),
    ],
  }
  return createEvidenceRecord({
    acceptance: completeAcceptance,
    config,
    contextIsolationProof: completeContextIsolationProof,
    finishedAt: "2026-08-22T12:01:00.000Z",
    fixture: {
      shareId: 42,
      sourceDefs: [{ id: "source-a" }, { id: "source-b" }],
      statusEnvelope: { queryable: 2, total: 2 },
      workspaceId: "workspace-42",
    },
    ledger: completeLedger,
    provider: { model: "configured-model", provider: "local-llm" },
    providerContextProof: completeProviderContextProof,
    providerReadiness: {
      model: "configured-model",
      provider: "local-llm",
      ready: true,
    },
    raceProbe: {
      operations: [
        {
          bodyHash: requestAHash,
          operationId: "race-concurrent-conflict",
          status: 409,
        },
        {
          bodyHash: requestBHash,
          operationId: "race-fingerprint-conflict",
          status: 409,
        },
      ],
      requestHashes: [requestAHash, requestBHash],
      requestIdHash: hash("request-id"),
      responseHashes: [hash("writer-response"), hash("replay-response")],
      statuses: [409, 200, 200, 409],
      timingsMs: [120, 124, 12, 8],
      turnHashes: [hash("turn-a"), hash("turn-a")],
    },
    screenshots: completeScreenshots,
    startedAt: "2026-08-22T12:00:00.000Z",
    transitionProof: completeTransitionProof,
  })
}

describe("shared-research-workspace-cdp-uat runner contract", () => {
  const transitionLedger = () => ({
    consoleErrors: [],
    expectedHttpFailures: [],
    pageErrors: [],
    requests: [],
    requestFailures: [],
    runtimeOverlays: [],
  })
  const memberChatsPolicy = () =>
    buildMemberChatsTransitionPolicy({
      apiUrl: "http://127.0.0.1:18001",
      conversationId: "conversation-17",
      webUrl: "http://127.0.0.1:18082",
    })

  it("installs the strict ledger only after a transition observer settles", async () => {
    const calls: string[] = []
    const transitionLedgerRecord = transitionLedger()
    const strictLedger = { strict: true }
    const attach = (_page: unknown, contextName: string, ledger: unknown) => {
      calls.push(
        `attach:${contextName}:${ledger === transitionLedgerRecord ? "transition" : "strict"}`
      )
      return {
        dispose: () => calls.push(`dispose:${contextName}`),
        waitForIdle: async (label: string) => calls.push(`idle:${contextName}:${label}`),
      }
    }

    await beginStrictLedgerAfterTransition({
      attach,
      contextName: "member-chats",
      ledger: strictLedger,
      page: {},
      transition: async () => calls.push("transition"),
      transitionLabel: "Chats navigation",
      transitionLedger: transitionLedgerRecord,
      transitionOperationPolicy: memberChatsPolicy(),
    })

    expect(calls).toEqual([
      "attach:member-chats-transition:transition",
      "transition",
      "idle:member-chats-transition:Chats navigation",
      "dispose:member-chats-transition",
      "attach:member-chats:strict",
    ])
  })

  it("disposes the transition observer without attaching a strict ledger when navigation fails", async () => {
    const calls: string[] = []
    const transitionFailure = new Error("navigation failed")
    const attach = (_page: unknown, contextName: string) => {
      calls.push(`attach:${contextName}`)
      return {
        dispose: () => calls.push(`dispose:${contextName}`),
        waitForIdle: async () => calls.push(`idle:${contextName}`),
      }
    }

    await expect(
      beginStrictLedgerAfterTransition({
        attach,
        contextName: "owner-revocation",
        ledger: {},
        page: {},
        transition: async () => {
          throw transitionFailure
        },
        transitionLabel: "owner revocation preparation",
        transitionLedger: transitionLedger(),
      })
    ).rejects.toThrow(transitionFailure)
    expect(calls).toEqual([
      "attach:owner-revocation-transition",
      "dispose:owner-revocation-transition",
    ])
  })

  it.each([
    [
      "console",
      (ledger: ReturnType<typeof transitionLedger>) =>
        ledger.consoleErrors.push({ message: "transition console error" }),
    ],
    [
      "page",
      (ledger: ReturnType<typeof transitionLedger>) =>
        ledger.pageErrors.push({ message: "transition page error" }),
    ],
    [
      "HTTP",
      (ledger: ReturnType<typeof transitionLedger>) =>
        ledger.requests.push({
          method: "GET",
          status: 500,
          url: "http://127.0.0.1:18001/api/v1/health",
        }),
    ],
  ])("rejects a transition %s error before strict ledger attachment", async (_kind, addFailure) => {
    const calls: string[] = []
    const transient = transitionLedger()
    const attach = (_page: unknown, contextName: string) => {
      calls.push(`attach:${contextName}`)
      return {
        dispose: () => calls.push(`dispose:${contextName}`),
        waitForIdle: async () => undefined,
      }
    }

    await expect(
      beginStrictLedgerAfterTransition({
        attach,
        contextName: "member-chats",
        ledger: transitionLedger(),
        page: {},
        transition: async () => addFailure(transient),
        transitionLabel: "Chats navigation",
        transitionLedger: transient,
        transitionOperationPolicy: memberChatsPolicy(),
      })
    ).rejects.toThrow("Transition observation failed")
    expect(calls).toEqual(["attach:member-chats-transition", "dispose:member-chats-transition"])
  })

  it("allows only explicitly registered and bounded route-teardown GET aborts", () => {
    const abort = {
      context: "member-chats-transition",
      error: "net::ERR_ABORTED",
      method: "GET",
      url: "http://127.0.0.1:18001/api/v1/config/providers",
    }
    const allowance = {
      count: 1,
      id: "chat-config-provider-teardown",
      method: "GET",
      url: abort.url,
    }

    const accepted = classifyTransitionLedger(
      {
        ...transitionLedger(),
        requestFailures: [abort],
      },
      {
        abortAllowances: [allowance],
        contextName: "member-chats",
        operationPolicy: memberChatsPolicy(),
        transitionLabel: "Chats navigation",
      }
    )
    expect(accepted.ok).toBe(true)
    expect(accepted.proof.allowedAborts).toEqual([
      expect.objectContaining({ count: 1, id: allowance.id, maximumCount: 1, method: "GET" }),
    ])
    expect(accepted.proof.observedRequests).toEqual([
      expect.objectContaining({
        errorHash: expect.stringMatching(/^sha256:[a-f0-9]{64}$/),
        kind: "failure",
        method: "GET",
        requestHash: expect.stringMatching(/^sha256:[a-f0-9]{64}$/),
        status: null,
      }),
    ])
    expect(JSON.stringify(accepted.proof)).not.toContain(abort.url)
    expect(
      classifyTransitionLedger(
        { ...transitionLedger(), requestFailures: [abort] },
        {
          abortAllowances: [],
          contextName: "member-chats",
          operationPolicy: memberChatsPolicy(),
          transitionLabel: "Chats navigation",
        }
      ).ok
    ).toBe(false)
    expect(
      classifyTransitionLedger(
        { ...transitionLedger(), requestFailures: [abort, abort] },
        {
          abortAllowances: [allowance],
          contextName: "member-chats",
          operationPolicy: memberChatsPolicy(),
          transitionLabel: "Chats navigation",
        }
      ).ok
    ).toBe(false)
  })

  it.each([
    ["unknown origin", "http://example.test/api/v1/config/providers"],
    ["old media route", "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/media/7"],
    ["local workspace", "http://127.0.0.1:18001/api/v1/workspaces/42/context"],
    ["tool route", "http://127.0.0.1:18001/api/v1/mcp/tools"],
  ])("rejects an aborted GET for an unregistered %s", (_label, url) => {
    expect(
      classifyTransitionLedger(
        {
          ...transitionLedger(),
          requestFailures: [
            {
              context: "member-chats-transition",
              error: "net::ERR_ABORTED",
              method: "GET",
              url,
            },
          ],
        },
        {
          abortAllowances: [],
          contextName: "member-chats",
          operationPolicy: memberChatsPolicy(),
          transitionLabel: "Chats navigation",
        }
      ).ok
    ).toBe(false)
  })

  it("applies the strict successful-request allowlist during transitions", () => {
    expect(
      classifyTransitionLedger(
        {
          ...transitionLedger(),
          requests: [
            {
              context: "member-chats-transition",
              method: "GET",
              status: 200,
              url: "http://127.0.0.1:18001/api/v1/unknown-bootstrap",
            },
          ],
        },
        {
          abortAllowances: [],
          contextName: "member-chats",
          operationPolicy: memberChatsPolicy(),
          transitionLabel: "Chats navigation",
        }
      ).ok
    ).toBe(false)
  })

  it("allows only the exact bounded owner-management transition operations", async () => {
    const runnerModule =
      (await import("../scripts/shared-research-workspace-cdp-uat.mjs")) as unknown as {
        buildOwnerRevocationTransitionPolicy: (input: {
          apiUrl: string
          ledger: ReturnType<typeof transitionLedger>
          webUrl: string
          workspaceId: string
        }) => unknown
      }
    expect(runnerModule.buildOwnerRevocationTransitionPolicy).toBeTypeOf("function")

    const apiUrl = "http://127.0.0.1:18001"
    const webUrl = "http://127.0.0.1:18082"
    const workspaceId = "644c57b8-f897-4dd0-945b-405220ea31a0"
    const migrationId = `research-workspace-${workspaceId}-a5350df9032d49e1`
    const requests = [
      { method: "GET", status: 200, url: `${webUrl}/research-workspace` },
      { method: "GET", status: 200, url: `${webUrl}/fonts/arimo.ttf` },
      { method: "GET", status: 200, url: `${webUrl}/fonts/inter-semibold.ttf` },
      { method: "GET", status: 200, url: `${webUrl}/fonts/inter-medium.ttf` },
      { method: "GET", status: 200, url: `${webUrl}/fonts/inter-regular.ttf` },
      { method: "PUT", status: 200, url: `${apiUrl}/api/v1/workspaces/${workspaceId}` },
      { method: "GET", status: 200, url: `${apiUrl}/api/v1/notes/search/` },
      {
        method: "PUT",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/${workspaceId}/sources/selection`,
      },
      { method: "POST", status: 201, url: `${apiUrl}/api/v1/workspaces/migrations` },
      {
        method: "PUT",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}/chunks/chunk-1-a390a9d0560eab4c`,
      },
      {
        method: "PUT",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}/chunks/chunk-2-a783b694e4c89bdd`,
      },
      {
        method: "PUT",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}/chunks/chunk-3-d680369f3a606b35`,
      },
      {
        method: "POST",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}/finalize`,
      },
      { method: "GET", status: 200, url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}` },
      {
        method: "POST",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}/client-delete-ack`,
      },
    ]
    const ledger = { ...transitionLedger(), requests }
    const operationPolicy = runnerModule.buildOwnerRevocationTransitionPolicy({
      apiUrl,
      ledger,
      webUrl,
      workspaceId,
    })
    const result = classifyTransitionLedger(ledger, {
      contextName: "owner-revocation",
      operationPolicy,
      transitionLabel: "owner revocation preparation",
    })

    expect(result.ok).toBe(true)
    expect(result.proof.operations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ count: 1, name: "owner-workspace-save" }),
        expect.objectContaining({ count: 1, name: "owner-migration-chunk-1" }),
      ])
    )
    expect(JSON.stringify(result.proof.operations)).not.toContain(workspaceId)
    expect(JSON.stringify(result.proof.operations)).not.toContain(migrationId)

    const unknownFont = {
      ...transitionLedger(),
      requests: [{ method: "GET", status: 200, url: `${webUrl}/fonts/unknown.ttf` }],
    }
    expect(
      classifyTransitionLedger(unknownFont, {
        contextName: "owner-revocation",
        operationPolicy: buildOwnerRevocationTransitionPolicy({
          apiUrl,
          ledger: unknownFont,
          webUrl,
          workspaceId,
        }),
        transitionLabel: "owner revocation preparation",
      }).ok
    ).toBe(false)
  })

  it.each([
    [
      "wrong workspace",
      (apiUrl: string, workspaceId: string) => ({
        method: "PUT",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/${workspaceId}-wrong`,
      }),
    ],
    [
      "wrong migration",
      (apiUrl: string) => ({
        method: "POST",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/migrations/research-workspace-wrong-a5350df9032d49e1/finalize`,
      }),
    ],
    [
      "wrong method",
      (apiUrl: string, workspaceId: string) => ({
        method: "PATCH",
        status: 200,
        url: `${apiUrl}/api/v1/workspaces/${workspaceId}`,
      }),
    ],
    [
      "wrong path",
      (apiUrl: string) => ({ method: "POST", status: 200, url: `${apiUrl}/api/v1/notes/search/` }),
    ],
    [
      "unknown origin",
      (_apiUrl: string, workspaceId: string) => ({
        method: "PUT",
        status: 200,
        url: `http://example.test/api/v1/workspaces/${workspaceId}`,
      }),
    ],
  ])("rejects an owner transition operation with a %s", async (_label, makeRequest) => {
    const runnerModule =
      (await import("../scripts/shared-research-workspace-cdp-uat.mjs")) as unknown as {
        buildOwnerRevocationTransitionPolicy: (input: Record<string, unknown>) => unknown
      }
    const apiUrl = "http://127.0.0.1:18001"
    const webUrl = "http://127.0.0.1:18082"
    const workspaceId = "644c57b8-f897-4dd0-945b-405220ea31a0"
    const ledger = {
      ...transitionLedger(),
      requests: [makeRequest(apiUrl, workspaceId)],
    }
    const operationPolicy = runnerModule.buildOwnerRevocationTransitionPolicy({
      apiUrl,
      ledger,
      webUrl,
      workspaceId,
    })

    expect(
      classifyTransitionLedger(ledger, {
        contextName: "owner-revocation",
        operationPolicy,
        transitionLabel: "owner revocation preparation",
      }).ok
    ).toBe(false)
  })

  it("rejects owner transition operations beyond their declared count", async () => {
    const runnerModule =
      (await import("../scripts/shared-research-workspace-cdp-uat.mjs")) as unknown as {
        buildOwnerRevocationTransitionPolicy: (input: Record<string, unknown>) => unknown
      }
    const apiUrl = "http://127.0.0.1:18001"
    const webUrl = "http://127.0.0.1:18082"
    const workspaceId = "644c57b8-f897-4dd0-945b-405220ea31a0"
    const repeated = {
      method: "PUT",
      status: 200,
      url: `${apiUrl}/api/v1/workspaces/${workspaceId}`,
    }
    const ledger = { ...transitionLedger(), requests: [repeated, repeated, repeated] }
    const operationPolicy = runnerModule.buildOwnerRevocationTransitionPolicy({
      apiUrl,
      ledger,
      webUrl,
      workspaceId,
    })

    const result = classifyTransitionLedger(ledger, {
      contextName: "owner-revocation",
      operationPolicy,
      transitionLabel: "owner revocation preparation",
    })
    expect(result.ok).toBe(false)
    expect(result.failures).toContain("transition_operation_bound: owner-workspace-save")
  })

  it("rejects an extra owner migration chunk identity", () => {
    const apiUrl = "http://127.0.0.1:18001"
    const webUrl = "http://127.0.0.1:18082"
    const workspaceId = "644c57b8-f897-4dd0-945b-405220ea31a0"
    const migrationId = `research-workspace-${workspaceId}-a5350df9032d49e1`
    const requests = [1, 2, 3, 4].map((index) => ({
      method: "PUT",
      status: 200,
      url: `${apiUrl}/api/v1/workspaces/migrations/${migrationId}/chunks/chunk-${index}-${String(index).repeat(16)}`,
    }))
    const ledger = { ...transitionLedger(), requests }
    const operationPolicy = buildOwnerRevocationTransitionPolicy({
      apiUrl,
      ledger,
      webUrl,
      workspaceId,
    })

    expect(
      classifyTransitionLedger(ledger, {
        contextName: "owner-revocation",
        operationPolicy,
        transitionLabel: "owner revocation preparation",
      }).ok
    ).toBe(false)
  })

  it("partitions the Chats transition allowlist by the exact conversation", async () => {
    const runnerModule =
      (await import("../scripts/shared-research-workspace-cdp-uat.mjs")) as unknown as {
        buildMemberChatsTransitionPolicy: (input: Record<string, unknown>) => unknown
      }
    expect(runnerModule.buildMemberChatsTransitionPolicy).toBeTypeOf("function")
    const apiUrl = "http://127.0.0.1:18001"
    const webUrl = "http://127.0.0.1:18082"
    const conversationId = "conversation-17"
    const operationPolicy = runnerModule.buildMemberChatsTransitionPolicy({
      apiUrl,
      conversationId,
      webUrl,
    })
    const acceptedLedger = {
      ...transitionLedger(),
      requests: [
        { method: "GET", status: 200, url: `${webUrl}/chat` },
        { method: "GET", status: 200, url: `${webUrl}/__nextjs_font/geist-latin.woff2` },
        { method: "GET", status: 200, url: `${webUrl}/__nextjs_font/geist-latin.woff2` },
        { method: "GET", status: 200, url: `${apiUrl}/api/v1/config/docs-info` },
        { method: "GET", status: 200, url: `${apiUrl}/api/v1/ingestion-sources/capabilities` },
        { method: "GET", status: 200, url: `${apiUrl}/api/v1/audio/health` },
        { method: "GET", status: 200, url: `${apiUrl}/api/v1/characters/` },
        {
          method: "GET",
          status: 200,
          url: `${apiUrl}/api/v1/chats/${conversationId}/research-runs`,
        },
      ],
    }
    expect(
      classifyTransitionLedger(acceptedLedger, {
        contextName: "member-chats",
        operationPolicy,
        transitionLabel: "Chats navigation",
      }).ok
    ).toBe(true)

    for (const request of [
      {
        method: "GET",
        status: 200,
        url: `${apiUrl}/api/v1/chats/wrong-conversation/research-runs`,
      },
      { method: "POST", status: 200, url: `${apiUrl}/api/v1/config/providers` },
      { method: "GET", status: 200, url: "http://example.test/openapi.json" },
    ]) {
      expect(
        classifyTransitionLedger(
          { ...transitionLedger(), requests: [request] },
          {
            contextName: "member-chats",
            operationPolicy,
            transitionLabel: "Chats navigation",
          }
        ).ok
      ).toBe(false)
    }
  })

  it("re-resolves a message locator after one React DOM-detachment race", async () => {
    let scrollAttempts = 0
    const locator = {
      evaluate: async () => true,
      scrollIntoViewIfNeeded: async () => {
        scrollAttempts += 1
        if (scrollAttempts === 1) {
          throw new Error("Element is not attached to the DOM")
        }
      },
      waitFor: async () => undefined,
    }

    await expect(
      ensureLocatorVisibleInViewport(locator, "persisted question")
    ).resolves.toBeUndefined()
    expect(scrollAttempts).toBe(2)
  })

  it("does not retry non-detachment locator failures", async () => {
    let scrollAttempts = 0
    const locator = {
      evaluate: async () => true,
      scrollIntoViewIfNeeded: async () => {
        scrollAttempts += 1
        throw new Error("Target page, context or browser has been closed")
      },
      waitFor: async () => undefined,
    }

    await expect(ensureLocatorVisibleInViewport(locator, "persisted answer")).rejects.toThrow(
      "Target page, context or browser has been closed"
    )
    expect(scrollAttempts).toBe(1)
  })

  it("fails after three consecutive DOM-detachment races", async () => {
    let scrollAttempts = 0
    const locator = {
      evaluate: async () => true,
      scrollIntoViewIfNeeded: async () => {
        scrollAttempts += 1
        throw new Error("Element is not attached to the DOM")
      },
      waitFor: async () => undefined,
    }

    await expect(ensureLocatorVisibleInViewport(locator, "persisted answer")).rejects.toThrow(
      "Element is not attached to the DOM"
    )
    expect(scrollAttempts).toBe(3)
  })

  it("asks for both source facts without leaking either expected value", () => {
    const question = buildAllSourcesQuestion({
      amberTitle: "Amber protocol final18",
      cobaltTitle: "Cobalt trial final18",
    })

    expect(question).toContain("Amber protocol final18")
    expect(question).toContain("signed-date token")
    expect(question).toContain("Cobalt trial final18")
    expect(question).toContain("participant-count token")
    expect(question).toContain("citation for each source")
    expect(question).not.toContain("AMBER-SIGNED-DATE-2024-03-17")
    expect(question).not.toContain("COBALT-PARTICIPANTS-43")
  })

  it.each([
    "TLDW_E2E_SERVER_URL",
    "TLDW_WEB_URL",
    "TLDW_CDP_URL",
    "TLDW_SHARED_UAT_ADMIN_USERNAME",
    "TLDW_SHARED_UAT_ADMIN_PASSWORD",
    "TLDW_SHARED_UAT_FIXTURE_PASSWORD",
  ])("rejects a missing required environment value: %s", (key) => {
    const env = { ...REQUIRED_ENV }
    delete env[key as keyof typeof env]

    expect(() => buildSharedUatConfig({ env })).toThrow(key)
  })

  it("requires explicit local API, WebUI, and Chrome debugging URLs", () => {
    const config = buildSharedUatConfig({
      env: {
        ...REQUIRED_ENV,
        TLDW_SHARED_UAT_EVIDENCE_DIR: "/tmp/shared-recipient-uat",
      },
    })

    expect(config).toMatchObject({
      apiUrl: "http://127.0.0.1:18001",
      cdpUrl: "http://127.0.0.1:9222",
      evidenceDir: "/tmp/shared-recipient-uat",
      webUrl: "http://127.0.0.1:18082",
    })
    expect(config.personas).toEqual(["owner", "member", "nonmember"])
  })

  it("prefers a healthy configured llama.cpp target and otherwise records a real ready target", () => {
    const targets = [
      { model: "fallback-model", provider: "openai", ready: true },
      { model: "local-model", provider: "llama.cpp", ready: true },
    ]

    expect(selectEffectiveTarget({ llamaHealthy: true, targets })).toEqual({
      model: "local-model",
      provider: "llama.cpp",
    })
    expect(selectEffectiveTarget({ llamaHealthy: false, targets })).toEqual({
      model: "fallback-model",
      provider: "openai",
    })
    expect(
      selectEffectiveTarget({
        llamaHealthy: true,
        targets: [{ model: "configured-model", provider: "openai", ready: true }],
      })
    ).toEqual({ model: "configured-model", provider: "openai" })
    expect(selectEffectiveTarget({ llamaHealthy: false, targets: [] })).toBeNull()
  })

  it("fails the strict ledger on every undeclared browser or request error", () => {
    const scenarios = [
      {
        requests: [
          {
            context: "member",
            method: "GET",
            status: 500,
            url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/workspace",
          },
        ],
      },
      {
        requestFailures: [
          {
            context: "member",
            error: "net::ERR_FAILED",
            method: "GET",
            url: "http://127.0.0.1:18001/api/v1/health",
          },
        ],
      },
      { pageErrors: [{ context: "owner", message: "render failed" }] },
      { consoleErrors: [{ context: "member", message: "uncaught error" }] },
      { runtimeOverlays: [{ context: "member", text: "Unhandled Runtime Error" }] },
      {
        requests: [
          {
            context: "member",
            method: "GET",
            status: 200,
            url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/media/7",
          },
        ],
      },
      {
        requests: [
          {
            context: "member",
            method: "GET",
            status: 200,
            url: "http://127.0.0.1:18001/api/v1/research/workspaces",
          },
        ],
      },
      {
        requests: [
          {
            context: "member",
            method: "POST",
            status: 200,
            url: "http://127.0.0.1:18001/api/v1/workspaces/1/sources",
          },
        ],
      },
    ]

    for (const scenario of scenarios) {
      const classification = classifyStrictLedger({
        ...cleanLedger,
        ...scenario,
      })
      expect(classification.ok).toBe(false)
      expect(classification.failures.length).toBeGreaterThan(0)
    }
  })

  it("allows only the exact owner-revocation UI request surface", () => {
    const ownerRequests = [
      ["GET", "/api/v1/auth/me"],
      ["GET", "/api/v1/health"],
      ["GET", "/api/v1/notifications/unread-count"],
      ["GET", "/api/v1/llm/models/metadata"],
      ["GET", "/api/v1/workspaces/shared-uat-run/context"],
      ["GET", "/api/v1/sharing/workspaces/shared-uat-run/shares"],
      ["GET", "/api/v1/sharing/tokens"],
      ["DELETE", "/api/v1/sharing/shares/42"],
    ].map(([method, pathname]) => ({
      context: "owner-revocation",
      method,
      status: 200,
      url: `http://127.0.0.1:18001${pathname}`,
    }))

    expect(classifyStrictLedger({ ...cleanLedger, requests: ownerRequests }).ok).toBe(true)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        requests: [
          {
            context: "owner-revocation",
            method: "GET",
            status: 200,
            url: "http://127.0.0.1:18001/api/v1/workspaces/shared-uat-run",
          },
        ],
      }).ok
    ).toBe(false)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        requests: [
          {
            context: "member",
            method: "GET",
            status: 200,
            url: "http://127.0.0.1:18001/api/v1/workspaces/shared-uat-run/context",
          },
        ],
      }).ok
    ).toBe(false)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        requests: [
          {
            context: "owner-revocation",
            method: "POST",
            status: 200,
            url: "http://127.0.0.1:18001/api/v1/workspaces/shared-uat-run/sources",
          },
        ],
      }).ok
    ).toBe(false)

    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )
    expect(source).toContain("const allowedAmbientSafeGet = (entry) =>")
    expect(source).toContain("allowedAmbientSafeGet(entry)")
  })

  it("accepts only an exactly declared neutral HTTP failure", () => {
    const request = {
      bodyHash: null,
      context: "nonmember",
      method: "GET",
      status: 404,
      url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/workspace",
    }
    const exact = {
      bodyHash: null,
      consoleErrorCount: 0,
      context: "nonmember",
      method: "GET",
      operationId: "nonmember-neutral-bootstrap",
      status: 404,
      url: request.url,
    }

    expect(
      classifyStrictLedger({
        ...cleanLedger,
        expectedHttpFailures: [exact],
        requests: [request],
      }).ok
    ).toBe(true)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        expectedHttpFailures: [{ ...exact, context: "member" }],
        requests: [request],
      }).ok
    ).toBe(false)
  })

  it("correlates expected failures as one-shot operations instead of a Set", () => {
    const request = {
      bodyHash: "sha256:matching-race-body",
      context: "member",
      method: "POST",
      status: 409,
      url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/chat",
    }
    const expected = {
      ...request,
      consoleErrorCount: 0,
      operationId: "race-concurrent-conflict",
    }

    expect(
      classifyStrictLedger({
        ...cleanLedger,
        expectedHttpFailures: [expected],
        requests: [request, request],
      }).ok
    ).toBe(false)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        expectedHttpFailures: [
          expected,
          {
            ...expected,
            bodyHash: "sha256:changed-race-body",
            operationId: "race-fingerprint-conflict",
          },
        ],
        requests: [request, { ...request, bodyHash: "sha256:changed-race-body" }],
      }).ok
    ).toBe(true)
  })

  it("requires exact console-error multiplicity for each declared operation", () => {
    const request = {
      bodyHash: "sha256:race-body",
      context: "member",
      method: "POST",
      status: 409,
      url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/chat",
    }
    const consoleError = {
      context: request.context,
      message: "Failed to load resource: the server responded with a status of 409 (Conflict)",
      status: request.status,
      url: request.url,
    }
    const expected = {
      ...request,
      consoleErrorCount: 1,
      operationId: "race-concurrent-conflict",
    }

    expect(
      classifyStrictLedger({
        ...cleanLedger,
        consoleErrors: [consoleError],
        expectedHttpFailures: [expected],
        requests: [request],
      }).ok
    ).toBe(true)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        consoleErrors: [consoleError, consoleError],
        expectedHttpFailures: [expected],
        requests: [request],
      }).ok
    ).toBe(false)
    expect(
      classifyStrictLedger({
        ...cleanLedger,
        consoleErrors: [],
        expectedHttpFailures: [expected],
        requests: [request],
      }).ok
    ).toBe(false)
  })

  it("accepts console noise only for an exact declared request and rejects every request failure", () => {
    const conflict = {
      context: "member",
      method: "POST",
      status: 409,
      url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/chat",
    }
    const ambientAbort = {
      context: "member",
      error: "net::ERR_ABORTED",
      method: "GET",
      url: "http://127.0.0.1:18001/api/v1/notifications/unread-count",
    }
    const expectedConsole = {
      context: "member",
      message: "Failed to load resource: the server responded with a status of 409 (Conflict)",
      status: 409,
      url: conflict.url,
    }
    const accepted = {
      ...cleanLedger,
      consoleErrors: [expectedConsole],
      expectedHttpFailures: [
        {
          ...conflict,
          consoleErrorCount: 1,
          operationId: "race-concurrent-conflict",
        },
      ],
      requests: [conflict],
    }

    expect(classifyStrictLedger(accepted).ok).toBe(true)
    expect(
      classifyStrictLedger({
        ...accepted,
        requestFailures: [ambientAbort],
      }).ok
    ).toBe(false)
    expect(
      classifyStrictLedger({
        ...accepted,
        requests: [{ ...conflict, method: "GET" }],
      }).ok
    ).toBe(false)
    expect(
      classifyStrictLedger({
        ...accepted,
        consoleErrors: [{ ...expectedConsole, status: 500 }],
      }).ok
    ).toBe(false)
  })

  it("allows ambient read-only health and notification traffic in shared mode", () => {
    const requests = [
      "/api/v1/auth/me",
      "/api/v1/health",
      "/api/v1/rag/health",
      "/api/v1/health/live",
      "/api/v1/notifications/stream",
    ].map((pathname) => ({
      context: "member",
      method: "GET",
      status: 200,
      url: `http://127.0.0.1:18001${pathname}`,
    }))

    expect(classifyStrictLedger({ ...cleanLedger, requests }).ok).toBe(true)
  })

  it("redacts credentials from nested evidence and never writes raw prompts or answers", () => {
    const config = buildSharedUatConfig({ env: REQUIRED_ENV })
    const evidence = createEvidenceRecord({
      acceptance: completeAcceptance,
      config,
      finishedAt: "2026-08-22T12:01:00.000Z",
      ledger: cleanLedger,
      provider: { model: "configured-model", provider: "openai" },
      raceProbe: {
        requestHashes: ["sha256:request-a", "sha256:request-b"],
        responseHashes: ["sha256:writer-response", "sha256:replay-response"],
        statuses: [200, 200, 409],
        timingsMs: [120, 124, 12],
        turnHashes: ["sha256:turn-a", "sha256:turn-a"],
      },
      screenshots: completeScreenshots,
      startedAt: "2026-08-22T12:00:00.000Z",
    })
    const serialized = JSON.stringify(evidence)

    expect(serialized).not.toContain(REQUIRED_ENV.TLDW_SHARED_UAT_ADMIN_PASSWORD)
    expect(serialized).not.toContain(REQUIRED_ENV.TLDW_SHARED_UAT_FIXTURE_PASSWORD)
    expect(serialized).not.toContain(REQUIRED_ENV.TLDW_SHARED_UAT_ADMIN_USERNAME)
    expect(serialized).not.toContain("rawPrompt")
    expect(serialized).not.toContain("rawAnswer")
  })

  it("returns a nonzero validation result when required live evidence is incomplete", () => {
    const complete = makeCompleteEvidence()

    expect(validateEvidenceRecord(complete)).toEqual({ exitCode: 0, failures: [] })
    expect(
      validateEvidenceRecord({
        ...complete,
        acceptance: { ...complete.acceptance, sentinelsExcluded: false },
      }).exitCode
    ).toBe(1)
    expect(
      validateEvidenceRecord({
        ...complete,
        screenshots: { ...complete.screenshots, revokedShare: "" },
      }).exitCode
    ).toBe(1)
    expect(
      validateEvidenceRecord({
        ...complete,
        raceProbe: { ...complete.raceProbe, statuses: [200, 500, 409] },
      }).exitCode
    ).toBe(1)
    expect(
      validateEvidenceRecord({
        ...complete,
        raceProbe: {
          ...complete.raceProbe,
          turnHashes: ["sha256:turn-a", "sha256:turn-b"],
        },
      }).exitCode
    ).toBe(1)
    expect(
      validateEvidenceRecord({
        ...complete,
        screenshots: {
          ...complete.screenshots,
          desktopSharedWorkspace:
            "/Users/example/project/.worktrees/task/desktop-shared-workspace.png",
        },
      }).failures
    ).toContain("screenshot_path_not_repository_relative:desktopSharedWorkspace")
  })

  it("requires the exact canonical acceptance and screenshot keys", () => {
    const complete = makeCompleteEvidence()
    const { sentinelsExcluded: _missing, ...missingAcceptance } = complete.acceptance

    expect(validateEvidenceRecord(complete)).toEqual({ exitCode: 0, failures: [] })
    expect(validateEvidenceRecord({ ...complete, acceptance: missingAcceptance }).exitCode).toBe(1)
    expect(
      validateEvidenceRecord({
        ...complete,
        acceptance: { ...complete.acceptance, unexpectedAcceptance: true },
      }).exitCode
    ).toBe(1)
    expect(
      validateEvidenceRecord({
        ...complete,
        screenshots: {
          ...complete.screenshots,
          unexpectedScreenshot: "unexpected.png",
        },
      }).exitCode
    ).toBe(1)
  })

  it.each([
    [
      "failed status",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({ ...evidence, status: "failed" }),
    ],
    [
      "unready provider",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        providerReadiness: { ...evidence.providerReadiness, ready: false },
      }),
    ],
    [
      "missing isolation proof",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        contextIsolationProof: [],
      }),
    ],
    [
      "missing provider probe proof",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        providerContextProof: null,
      }),
    ],
    [
      "zero provider requests",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        providerContextProof: {
          ...evidence.providerContextProof,
          forwardedRequestCount: 0,
          inputBodyHashes: [],
          outputBodyHashes: [],
        },
      }),
    ],
    [
      "sentinel in provider context",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        providerContextProof: { ...evidence.providerContextProof, ownerSentinelAbsent: false },
      }),
    ],
    [
      "unexpected transition traffic",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        transitionProof: [{ ...evidence.transitionProof[0], unexpectedRequestCount: 1 }],
      }),
    ],
    [
      "malformed timestamp",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        finishedAt: "not-a-timestamp",
      }),
    ],
    [
      "duplicate expected failure",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({
        ...evidence,
        ledger: {
          ...evidence.ledger,
          expectedHttpFailures: [
            ...evidence.ledger.expectedHttpFailures,
            evidence.ledger.expectedHttpFailures[0],
          ],
        },
      }),
    ],
    [
      "unknown top-level field",
      (evidence: ReturnType<typeof makeCompleteEvidence>) => ({ ...evidence, weakOverride: true }),
    ],
  ])("rejects malformed evidence with %s", (_label, mutate) => {
    expect(validateEvidenceRecord(mutate(makeCompleteEvidence())).exitCode).toBe(1)
  })

  it("records and bounds Chats settings requests without hiding amplification", () => {
    const complete = makeCompleteEvidence()

    expect(complete.settingsRequestProbe).toEqual({
      count: 1,
      maximum: 2,
      statuses: [200],
    })
    expect(validateEvidenceRecord(complete)).toEqual({ exitCode: 0, failures: [] })
    expect(
      validateEvidenceRecord({
        ...complete,
        settingsRequestProbe: { count: 3, maximum: 2, statuses: [200, 200, 200] },
      }).failures
    ).toContain("settings_request_amplification")
  })

  it("uses only connectOverCDP and creates three isolated browser contexts", () => {
    const sourcePath = path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("chromium.connectOverCDP(config.cdpUrl)")
    expect(source.match(/browser\.newContext\(/g)).toHaveLength(3)
    expect(source).not.toMatch(/chromium\.launch\s*\(/)
    expect(source).not.toMatch(/computer[ -]?control|cua-driver|osascript/i)
    expect(source).not.toMatch(/page\.route\s*\(|Fetch\.fulfillRequest/)
  })

  it("records cookie and storage isolation and scopes shared-mode ledger listeners", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("context.cookies()")
    expect(source).toContain("cookieHash")
    expect(source).toContain("contextIsolationProof: contextProof")
    expect(source).toContain('page.goto("about:blank", { waitUntil: "load" })')
    expect(source).toContain('operationId: "race-concurrent-conflict"')
    expect(source).toContain('operationId: "race-fingerprint-conflict"')
    expect(source).toContain("ownerLedger.dispose()")
    expect(source).toContain("memberLedger.dispose()")
  })

  it("settles tracked API work before navigation and ledger disposal", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain('page.on("requestfinished", onRequestFinished)')
    expect(source).toContain('pathname !== "/api/v1/notifications/stream"')
    expect(source).toContain('await memberLedger.waitForIdle("before member reload")')
    expect(source).toContain('await memberLedger.waitForIdle("before malformed route")')
    expect(source).toContain('await memberLedger.waitForIdle("before member race route")')
    expect(source).toContain('await ownerLedger.waitForIdle("before owner ledger disposal")')
    expect(source).toContain(
      'await ownerRevocationLedger.waitForIdle("before owner revocation disposal")'
    )
    expect(source).toContain('await memberChatsLedger.waitForIdle("before Chats ledger disposal")')
    expect(source).not.toContain("navigationAbortedAmbientRead")
  })

  it("uses fixture email addresses accepted by the admin API validator", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("@example.com")
    expect(source).not.toContain("@example.invalid")
  })

  it("uses ordinary fixture-user authentication after admin provisioning", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("loginApi(config, owner.username, config.fixturePassword)")
    expect(source).toContain("loginApi(config, member.username, config.fixturePassword)")
    expect(source).not.toContain("/api/v1/admin/impersonate/")
  })

  it("provisions organization and team membership before fixture-user login", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("`/api/v1/admin/orgs/${org.body.id}/members`")
    expect(source).toContain("`/api/v1/admin/teams/${team.body.id}/members`")
  })

  it("requests real locally cached embeddings for canonical queryable status", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain('form.append("generate_embeddings", "true")')
    expect(source).toContain('form.append("embedding_dispatch_mode", "background")')
    expect(source).toContain('form.append("embedding_provider", "huggingface")')
    expect(source).toContain(
      'form.append("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")'
    )
  })

  it("targets the password textbox without matching the auth-mode radio", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain('locator("input#password")')
    expect(source).toContain("const unreadCountSettled = page.waitForResponse(")
    expect(source).toContain('pathname === "/api/v1/notifications/unread-count"')
    expect(source).toContain("await unreadCountSettled")
    expect(source).not.toContain("getByLabel(/^password$/i)")
  })

  it("reports redacted route and request diagnostics when the shared shell times out", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const waitForSharedShell = async")
    expect(source).toContain("requestStatusTail")
    expect(source).toContain("consoleErrorHashes")
    expect(source).toContain("pageErrorHashes")
    expect(source).not.toContain(
      "memberPage.locator('[data-testid=\"shared-workspace-shell\"]').waitFor"
    )
  })

  it("provisions sharing.read through ordinary admin RBAC APIs before user login", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain('"/api/v1/admin/permissions?search=sharing.read"')
    expect(source).toContain('"/api/v1/admin/permissions"')
    expect(source).toContain('"/api/v1/admin/roles"')
    expect(source).toContain("`/api/v1/admin/roles/${recipientRoleId}/permissions/${permissionId}`")
    expect(source).toContain("`/api/v1/admin/users/${user.id}/roles/${recipientRoleId}`")
  })

  it("authenticates CDP page probes without returning or logging the token", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain('localStorage.getItem("tldwConfig")')
    expect(source).toContain('headers.set("Authorization", `Bearer ${token}`)')
    expect(source).not.toMatch(/return\s+\{[^}]*token/u)
  })

  it("fails shared-chat waits immediately on a typed HTTP error", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("page.waitForResponse(")
    expect(source).toContain("Shared chat failed status=")
    expect(source).toContain("code=")
  })

  it("writes redacted partial evidence on live failure and removes stale artifacts", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("fs.rmSync(path.join(config.evidenceDir, filename), { force: true })")
    expect(source).toContain('status: "failed"')
    expect(source).toContain("failureMessageHash")
    expect(source).toContain("providerReadiness = bootstrap.generation_default")
    expect(source).toContain("providerReadiness,")
    expect(source).toContain("providerContextProof,")
    expect(source).toContain("const disposeLedgers = () =>")
    expect(source).toMatch(
      /try \{\s*await resetProviderProbe\(config\)\s*fixture = await provisionFixture\(config\)/u
    )
    expect(source).toMatch(/catch \(error\) \{\s*disposeLedgers\(\)/u)
    expect(source).toMatch(/catch \(error\) \{[\s\S]*writeEvidence\(config, evidence\)/u)
  })

  it("records stable screenshot artifact names without home or worktree paths", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain('return [key, fs.existsSync(screenshotPath) ? filename : ""]')
    expect(source).not.toMatch(
      /return \[key, fs\.existsSync\(screenshotPath\) \? screenshotPath : ""\]/u
    )
  })

  it("registers actionable cleanup metadata outside committed evidence with mode 0600", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("TLDW_SHARED_UAT_CLEANUP_MANIFEST")
    expect(source).toContain("cleanupManifestPath")
    expect(source).toContain("os.tmpdir()")
    expect(source).toContain("const registerCleanupMetadata =")
    expect(source).toContain("mode: 0o600")
    expect(source).toContain("fs.chmodSync(config.cleanupManifestPath, 0o600)")
    expect(source).toContain("userIds")
    expect(source).toContain("organizationIds")
    expect(source).toContain("teamIds")
    expect(source).toContain("roleIds")
    expect(source).toContain("workspaceIds")
    expect(source).toContain("shareIds")
    expect(source).not.toMatch(/cleanup[^\n]*(?:password|token)/iu)
  })

  it("uses the UI bootstrap response and reserves hidden fetches for explicit probes", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const memberBootstrapResponsePromise = memberPage.waitForResponse")
    expect(source).toContain("const memberBootstrapResponse = await memberBootstrapResponsePromise")
    expect(source).toContain("const bootstrap = await memberBootstrapResponse.json()")
    expect(source).not.toMatch(
      /fetchFromPage\([\s\S]{0,250}shared-with-me\/\$\{fixture\.shareId\}\/workspace/u
    )
    expect(source).not.toMatch(
      /fetchFromPage\([\s\S]{0,250}sharing\/shares\/\$\{fixture\.shareId\}/u
    )
    expect(source).toContain("const raceProbe = async")
    expect(source).toContain("const blockedPreview = await fetchFromPage")
  })

  it("uses deterministic fixture tokens without leaking them into the all-source question", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain(
      'const SOURCE_ONE_FACT = "Amber protocol token: AMBER-SIGNED-DATE-2024-03-17."'
    )
    expect(source).toContain(
      'const SOURCE_TWO_FACT = "Cobalt trial token: COBALT-PARTICIPANTS-43."'
    )
    expect(source).not.toContain(
      '"Amber: return both exact tokens AMBER-SIGNED-DATE-2024-03-17 and COBALT-PARTICIPANTS-43 with one citation for Amber and one for Cobalt."'
    )
    expect(source).toContain('"Return the exact Amber protocol token from the selected source."')
    expect(source).toContain('firstAnswer.includes("COBALT-PARTICIPANTS-43")')
    expect(source).toContain('firstAnswer.includes("AMBER-SIGNED-DATE-2024-03-17")')
    expect(source).toContain('subsetText.includes("AMBER-SIGNED-DATE-2024-03-17")')
    expect(source).toMatch(
      /\.getByText\(\s*"COBALT-PARTICIPANTS-43"\s*,\s*\{\s*exact:\s*false\s*\}\s*\)\s*\.waitFor\(\)/u
    )
    expect(source).toContain("text.includes(fixture.sourceDefs[0].title)")
    expect(source).toContain("text.includes(fixture.sourceDefs[1].title)")
    expect(source).toContain("firstCitationText.some")
    expect(source).toContain("firstAnswerShape")
    expect(source).not.toContain("firstAnswer, firstCitationText")
  })

  it("permits vertical scrolling while keeping bounded mobile layout diagnostics", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).not.toContain("rect.top < -1 || rect.bottom > innerHeight + 1")
    expect(source).toContain("horizontalOffenderCount")
    expect(source).toContain("horizontalOffenders")
    expect(source).toContain("const intersectsViewport")
    expect(source).toMatch(/controls\s*\.filter\(intersectsViewport\)/u)
    expect(source).toContain("verticalScrollContainers")
    expect(source).toContain("activeTabs")
    expect(source).toContain(".slice(0, 8)")
    expect(source).toContain("JSON.stringify(metrics)")
  })

  it("stabilizes the hosted header and neutral focus before the initial desktop capture", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const stabilizeInitialDesktopCapture = async (page) =>")
    expect(source).toContain('getByTestId("chat-header-sidebar-toggle")')
    expect(source).toContain('waitFor({ state: "visible", timeout: 30_000 })')
    expect(source).toContain('window.scrollTo({ behavior: "instant", left: 0, top: 0 })')
    expect(source).toContain("document.fonts?.ready")
    expect(source).toContain("requestAnimationFrame")
    expect(source).toContain("locator('[data-testid=\"shared-workspace-shell\"] header p')")
    expect(source).toContain("await neutralSurface.click()")
    expect(source).not.toContain("addStyleTag")
    expect(source).toMatch(
      /await stabilizeInitialDesktopCapture\(memberPage\)[\s\S]*await inspectLayout\(memberPage, "desktop"\)[\s\S]*SCREENSHOT_NAMES\.desktopSharedWorkspace/u
    )
  })

  it("revokes the team share through the canonical owner workspace UI", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("teamId: Number(team.body.id)")
    expect(source).toContain("const createOwnerFixtureWorkspaceThroughUi = async")
    expect(source).toContain("const settleOwnerFixtureWorkspaceForRevocation = async")
    expect(source).toContain('getByTestId("workspace-workspaces-button")')
    expect(source).toContain('getByRole("menuitem", { name: "New Workspace" })')
    expect(source).toContain('getByRole("button", { name: "Rename workspace" })')
    expect(source).toContain('getByRole("textbox", { name: "Workspace name" })')
    expect(source).toContain('getByRole("button", { name: "Save", exact: true })')
    expect(source).toContain("ownerPage.waitForResponse")
    expect(source).toContain('response.request().method() !== "PUT"')
    expect(source).toContain("const ownerRecipientPage = await ownerContext.newPage()")
    expect(source).toContain('getByTestId("workspace-share-button")')
    expect(source).toContain('getByRole("tab", { name: "Active Shares" })')
    expect(source).toContain('getByRole("row").filter({ hasText: teamScopeLabel })')
    expect(source).toContain("name: `Revoke team share Team #${fixture.teamId}`")
    expect(source).toContain('name: "Revoke", exact: true')
    expect(source).toContain('getByText("Share revoked", { exact: true })')
    expect(source).toContain("ownerPage.waitForResponse")
    expect(source).toContain('response.request().method() !== "DELETE"')
    expect(source).not.toMatch(
      /fetchFromPage\([\s\S]{0,400}\/api\/v1\/sharing\/shares\/\$\{fixture\.shareId\}[\s\S]{0,100}method: "DELETE"/u
    )

    const parkedOwnerIndex = source.indexOf(
      'await ownerPage.goto("about:blank", { waitUntil: "load" })'
    )
    const finalizeFixtureIndex = source.indexOf(
      "fixture = await finalizeFixtureWorkspace(config, fixture, ownerWorkspaceId)"
    )
    const restoreOwnerIndex = source.indexOf(
      "settleOwnerFixtureWorkspaceForRevocation(ownerPage, config, fixture)"
    )
    const revocationLedgerIndex = source.indexOf(
      "const ownerRevocationLedger = await beginStrictLedgerAfterTransition({"
    )
    expect(parkedOwnerIndex).toBeGreaterThan(0)
    expect(parkedOwnerIndex).toBeLessThan(finalizeFixtureIndex)
    expect(restoreOwnerIndex).toBeGreaterThan(finalizeFixtureIndex)
    expect(revocationLedgerIndex).toBeGreaterThan(finalizeFixtureIndex)
    expect(revocationLedgerIndex).toBeLessThan(restoreOwnerIndex)
  })

  it("waits for a stable full-width mobile source preview with visible evidence", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const stabilizeMobilePreviewCapture = async")
    expect(source).toContain("'[role=\"dialog\"]:visible'")
    expect(source).toContain("fixture.sourceDefs[0].title")
    expect(source).toContain('getByText("AMBER-SIGNED-DATE-2024-03-17"')
    expect(source).toContain("requestAnimationFrame")
    expect(source).toContain("geometry.width < window.innerWidth - 2")
    expect(source).toContain("geometry.left < -1")
    expect(source).toContain("geometry.right > window.innerWidth + 1")
    expect(source).toMatch(
      /SCREENSHOT_NAMES\.mobileSharedWorkspace[\s\S]*await stabilizeMobilePreviewCapture\(memberPage, fixture\)[\s\S]*SCREENSHOT_NAMES\.mobileSourcePreview/u
    )
  })

  it("moves focus through a neutral click before revoked-state evidence capture", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const stabilizeRevokedCapture = async (page) =>")
    expect(source).toContain("await neutralSurface.click()")
    expect(source).toContain("!document.activeElement?.matches(`h1[tabindex='-1']`)")
    expect(source).toContain("requestAnimationFrame(() => requestAnimationFrame(resolve))")
    expect(source).not.toContain("addStyleTag")
    expect(source).toMatch(
      /await stabilizeRevokedCapture\(memberPage\)[\s\S]*SCREENSHOT_NAMES\.revokedShare/u
    )
  })

  it("opens Chats and proves the persisted question and generated answer text", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const firstQuestion =")
    expect(source).toContain("const firstAnswer =")
    expect(source).toContain('getByTestId("chat-header-sidebar-toggle")')
    expect(source).toContain('getAttribute("aria-label")')
    expect(source).toContain('sidebarLabel === "Expand sidebar"')
    expect(source).toContain("const recentConversations = memberPage.getByRole")
    expect(source).toContain('name: "Recent conversations"')
    expect(source).toContain('memberPage.locator(".ant-drawer-content:visible")')
    expect(source).toContain("let historySurface")
    expect(source).toContain("historySurface.getByText(fixture.workspaceName")
    expect(source).toContain('getAttribute("aria-expanded")')
    expect(source).toContain("const persistedMessagesResponsePromise = memberPage.waitForResponse")
    expect(source).toContain(
      "const persistedMessagesResponse = await persistedMessagesResponsePromise"
    )
    expect(source).toContain("/\\/api\\/v1\\/chats\\/[^/]+\\/messages$/")
    expect(source).toContain(
      'await ensureLocatorVisibleInViewport(persistedQuestion, "Persisted question")'
    )
    expect(source).toContain(
      'await ensureLocatorVisibleInViewport(persistedAnswer, "Persisted answer")'
    )
  })
})
