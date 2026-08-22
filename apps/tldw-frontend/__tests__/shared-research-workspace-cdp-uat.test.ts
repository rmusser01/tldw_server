import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  buildAllSourcesQuestion,
  beginStrictLedgerAfterTransition,
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

const cleanLedger = {
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
  mobileSharedWorkspace: "mobile-shared-workspace.png",
  revokedShare: "revoked-share.png",
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

  it("installs the strict ledger only after a transition observer settles", async () => {
    const calls: string[] = []
    const transitionLedgerRecord = transitionLedger()
    const strictLedger = { strict: true }
    const attach = (_page: unknown, contextName: string, ledger: unknown) => {
      calls.push(`attach:${contextName}:${ledger === transitionLedgerRecord ? "transition" : "strict"}`)
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
    ["console", (ledger: ReturnType<typeof transitionLedger>) => ledger.consoleErrors.push({ message: "transition console error" })],
    ["page", (ledger: ReturnType<typeof transitionLedger>) => ledger.pageErrors.push({ message: "transition page error" })],
    ["HTTP", (ledger: ReturnType<typeof transitionLedger>) => ledger.requests.push({ method: "GET", status: 500, url: "http://127.0.0.1:18001/api/v1/health" })],
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
      })
    ).rejects.toThrow("Transition observation failed")
    expect(calls).toEqual([
      "attach:member-chats-transition",
      "dispose:member-chats-transition",
    ])
  })

  it("allows only route-teardown GET aborts during a transition", () => {
    const abort = {
      error: "net::ERR_ABORTED",
      method: "GET",
      url: "http://127.0.0.1:18001/api/v1/config/providers",
    }

    expect(
      classifyTransitionLedger({
        ...transitionLedger(),
        requestFailures: [abort],
      }).ok
    ).toBe(true)
    expect(
      classifyTransitionLedger({
        ...transitionLedger(),
        requestFailures: [{ ...abort, method: "POST" }],
      }).ok
    ).toBe(false)
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

    await expect(
      ensureLocatorVisibleInViewport(locator, "persisted answer")
    ).rejects.toThrow("Target page, context or browser has been closed")
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

    await expect(
      ensureLocatorVisibleInViewport(locator, "persisted answer")
    ).rejects.toThrow("Element is not attached to the DOM")
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
      { requests: [{ context: "member", method: "GET", status: 500, url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/workspace" }] },
      { requestFailures: [{ context: "member", error: "net::ERR_FAILED", method: "GET", url: "http://127.0.0.1:18001/api/v1/health" }] },
      { pageErrors: [{ context: "owner", message: "render failed" }] },
      { consoleErrors: [{ context: "member", message: "uncaught error" }] },
      { runtimeOverlays: [{ context: "member", text: "Unhandled Runtime Error" }] },
      { requests: [{ context: "member", method: "GET", status: 200, url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/media/7" }] },
      { requests: [{ context: "member", method: "GET", status: 200, url: "http://127.0.0.1:18001/api/v1/research/workspaces" }] },
      { requests: [{ context: "member", method: "POST", status: 200, url: "http://127.0.0.1:18001/api/v1/workspaces/1/sources" }] },
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

    expect(
      classifyStrictLedger({ ...cleanLedger, requests: ownerRequests }).ok
    ).toBe(true)
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
      context: "nonmember",
      method: "GET",
      status: 404,
      url: "http://127.0.0.1:18001/api/v1/sharing/shared-with-me/42/workspace",
    }
    const exact = {
      context: "nonmember",
      method: "GET",
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
      expectedHttpFailures: [conflict],
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
    const config = buildSharedUatConfig({ env: REQUIRED_ENV })
    const complete = createEvidenceRecord({
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
      })
    ).toEqual({
      exitCode: 1,
      failures: ["screenshot_path_not_repository_relative:desktopSharedWorkspace"],
    })
  })

  it("records and bounds Chats settings requests without hiding amplification", () => {
    const config = buildSharedUatConfig({ env: REQUIRED_ENV })
    const complete = createEvidenceRecord({
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
      })
    ).toEqual({ exitCode: 1, failures: ["settings_request_amplification"] })
  })

  it("uses only connectOverCDP and creates three isolated browser contexts", () => {
    const sourcePath = path.resolve(
      process.cwd(),
      "scripts/shared-research-workspace-cdp-uat.mjs"
    )
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
    expect(source).toContain("evidence.contextIsolationProof = contextProof")
    expect(source).toContain('page.goto("about:blank", { waitUntil: "load" })')
    expect(source).toContain('addExpectedFailure(ledger, "member", "POST", 409, raceUrl)')
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
    expect(source).toContain('await ownerRevocationLedger.waitForIdle("before owner revocation disposal")')
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
    expect(source).not.toContain('getByLabel(/^password$/i)')
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
      'memberPage.locator(\'[data-testid="shared-workspace-shell"]\').waitFor'
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
    expect(source).toContain("evidence.providerReadiness = providerReadiness")
    expect(source).toContain("const disposeLedgers = () =>")
    expect(source).toMatch(/try \{\s*fixture = await provisionFixture\(config\)/u)
    expect(source).toMatch(/catch \(error\) \{\s*disposeLedgers\(\)/u)
    expect(source).toMatch(/catch \(error\) \{[\s\S]*writeEvidence\(config, evidence\)/u)
  })

  it("records stable screenshot artifact names without home or worktree paths", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain(
      'return [key, fs.existsSync(screenshotPath) ? filename : ""]'
    )
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
    expect(source).toContain(
      '"Return the exact Amber protocol token from the selected source."'
    )
    expect(source).toContain('firstAnswer.includes("COBALT-PARTICIPANTS-43")')
    expect(source).toContain('firstAnswer.includes("AMBER-SIGNED-DATE-2024-03-17")')
    expect(source).toContain('subsetText.includes("AMBER-SIGNED-DATE-2024-03-17")')
    expect(source).toContain(
      'getByText("COBALT-PARTICIPANTS-43", { exact: false }).waitFor()'
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
    expect(source).toContain("controls.filter(intersectsViewport)")
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
    expect(source).toContain(
      'locator(\'[data-testid="shared-workspace-shell"] header p\')'
    )
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
    expect(source).toContain(
      'name: `Revoke team share Team #${fixture.teamId}`'
    )
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
      /await stabilizeMobilePreviewCapture\(memberPage, fixture\)[\s\S]*SCREENSHOT_NAMES\.mobileSharedWorkspace/u
    )
  })

  it("moves focus through a neutral click before revoked-state evidence capture", () => {
    const source = fs.readFileSync(
      path.resolve(process.cwd(), "scripts/shared-research-workspace-cdp-uat.mjs"),
      "utf8"
    )

    expect(source).toContain("const stabilizeRevokedCapture = async (page) =>")
    expect(source).toContain("await neutralSurface.click()")
    expect(source).toContain(
      "!document.activeElement?.matches(`h1[tabindex='-1']`)"
    )
    expect(source).toContain(
      "requestAnimationFrame(() => requestAnimationFrame(resolve))"
    )
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
    expect(source).toContain("const persistedMessagesResponse = await persistedMessagesResponsePromise")
    expect(source).toContain("/\\/api\\/v1\\/chats\\/[^/]+\\/messages$/")
    expect(source).toContain(
      'await ensureLocatorVisibleInViewport(persistedQuestion, "Persisted question")'
    )
    expect(source).toContain(
      'await ensureLocatorVisibleInViewport(persistedAnswer, "Persisted answer")'
    )
  })
})
