import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import { EventEmitter } from "node:events"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it, vi } from "vitest"
import {
  buildLiveTierBackendEnv,
  buildLiveTierProfile,
} from "../live-tier-uat/profile.mjs"
import { inventorySource } from "../live-tier-uat/inventory-api-mocks.mjs"
import { shouldInstallTitleSettingsStub } from "../../e2e/utils/helpers"
import { shouldInstallSmokeApiStubs } from "../../e2e/smoke/smoke.setup"
import {
  collectSkippedTests,
  parseListOutput,
  renderMarkdownReport,
  summarizePlaywrightReport,
} from "../live-tier-uat/report.mjs"
import {
  assertFreshRunTargets,
  assertServicesStopped,
  buildCommands,
  assertNoMutableRepoDatabasePaths,
  assertOnlyLoopbackHttpRequests,
  isCertificationRun,
  installTerminationHandlers,
  parseArgs,
  readPlaywrightReport,
  removeGeneratedPath,
  runCommand,
  runLiveTierUat,
  stopSpawnedProcesses,
} from "../live-tier-uat/run.mjs"
import { stopProcessTree } from "../onboarding-uat/processes.mjs"
import { assertProjectAccounting } from "../live-tier-uat/report.mjs"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(testDir, "../..")
const repoRoot = path.resolve(frontendRoot, "../..")

describe("live Tier UAT profile", () => {
  it("places every mutable backend path under the disposable run directory", () => {
    const baseTmpDir = mkdtempSync(path.join(tmpdir(), "tldw-live-tier-test-"))

    try {
      const profile = buildLiveTierProfile({
        repoRoot,
        frontendRoot,
        runId: "profile-isolation",
        mockPort: 18183,
        baseTmpDir,
        pythonCommand: "/test/venv/bin/python",
      })

      expect(Object.values(profile.databasePaths).every((databasePath) =>
        path.resolve(databasePath).startsWith(`${path.resolve(profile.runDir)}${path.sep}`)
      )).toBe(true)
      expect(profile.databasePaths.jobs).toBe(path.join(profile.databaseDir, "jobs.db"))
      expect(profile.databasePaths.evaluations.startsWith(
        `${profile.userDatabasesDir}${path.sep}`
      )).toBe(true)
      expect(profile.databasePaths.acpSessions).toBe(
        path.join(profile.userDatabasesDir, "1", "acp_sessions.db")
      )
      expect(profile.acpWorkspaceRootBase.startsWith(
        `${profile.runDir}${path.sep}`
      )).toBe(true)
      expect(profile.databasePaths.mcpMedia).toBe(
        path.join(profile.userDatabasesDir, "1", "Media_DB_v2.db")
      )
      expect(profile.databasePaths.mcpDocs).toBe(path.join(profile.databaseDir, "mcp_docs.db"))
      const mcpModules = readFileSync(profile.mcpModulesConfigPath, "utf8")
      expect(mcpModules).toContain(`db_path: ${profile.databasePaths.mcpMedia}`)
      expect(mcpModules).toContain(`db_path: ${profile.databasePaths.mcpDocs}`)
      expect(mcpModules).not.toContain("db_path: Databases/")
      const acpRunnerConfig = readFileSync(profile.acpRunnerConfigPath, "utf8")
      expect(acpRunnerConfig).toContain("default: opencode")
      expect(acpRunnerConfig).toContain('command: "/test/venv/bin/python"')
      expect(acpRunnerConfig).toContain(
        `- ${JSON.stringify(path.join(repoRoot, "Helper_Scripts/acp_stub_agent.py"))}`
      )
      expect(profile.fixtureRoot.startsWith(frontendRoot)).toBe(true)
    } finally {
      rmSync(baseTmpDir, { recursive: true, force: true })
    }
  })

  it("points every model alias and evaluations database at isolated services", () => {
    const baseTmpDir = mkdtempSync(path.join(tmpdir(), "tldw-live-tier-env-"))

    try {
      const profile = buildLiveTierProfile({
        repoRoot,
        frontendRoot,
        runId: "backend-env",
        mockPort: 18184,
        baseTmpDir,
        pythonCommand: "/test/venv/bin/python",
      })
      const env = buildLiveTierBackendEnv({
        profile,
        mockPort: 18184,
        baseEnv: { NODE_ENV: "test", PATH: process.env.PATH },
      })

      expect(env.CUSTOM_OPENAI_API_IP).toBe("http://127.0.0.1:18184/v1")
      expect(env.CUSTOM_OPENAI_API_KEY).toBe(env.OPENAI_API_KEY)
      expect(env.CUSTOM_OPENAI_API_MODEL).toBe("local-uat-chat")
      expect(env.EVALUATIONS_TEST_DB_PATH).toBe(profile.databasePaths.evaluations)
      expect(env.JOBS_DB_PATH).toBe(profile.databasePaths.jobs)
      expect(env.USER_DB_BASE_DIR).toBe(profile.userDatabasesDir)
      expect(env.ACP_SESSIONS_DB_PATH).toBe(profile.databasePaths.acpSessions)
      expect(env.ACP_WORKSPACE_ALLOWED_BASE_PATHS).toBe(profile.acpWorkspaceRootBase)
      expect(env.ACP_ALLOWED_SESSION_CWD_ROOTS).toBe(profile.acpWorkspaceRootBase)
      expect(env.ACP_RUNNER_CWD).toBe(path.join(repoRoot, "tools/tldw-agent"))
      expect(env.MCP_MODULES_CONFIG).toBe(profile.mcpModulesConfigPath)
      expect(env.WATCHLIST_TEMPLATE_DIR).toBe(profile.watchlistTemplateDir)
      expect(path.resolve(env.WATCHLIST_TEMPLATE_DIR).startsWith(
        `${path.resolve(profile.runDir)}${path.sep}`
      )).toBe(true)
      expect(env.ACP_AUDIT_DB_PATH).toBe(profile.databasePaths.acpAudit)
      expect(env.MONITORING_ALERTS_DB).toBe(profile.databasePaths.monitoringAlerts)
      expect(env.SYSTEM_LOG_FILE_PATH).toBe(profile.systemLogFilePath)
      for (const runtimePath of [
        env.ACP_AUDIT_DB_PATH,
        env.MONITORING_ALERTS_DB,
        env.SYSTEM_LOG_FILE_PATH,
      ]) {
        expect(path.resolve(runtimePath).startsWith(
          `${path.resolve(profile.runDir)}${path.sep}`
        )).toBe(true)
      }
    } finally {
      rmSync(baseTmpDir, { recursive: true, force: true })
    }
  })
})

describe("live Tier UAT API-interception inventory", () => {
  it("does not install the shared title-settings API stub during strict live Tier UAT", () => {
    expect(shouldInstallTitleSettingsStub({ TLDW_LIVE_TIER_UAT: "1" })).toBe(false)
    expect(shouldInstallTitleSettingsStub({})).toBe(true)
    expect(shouldInstallSmokeApiStubs({ TLDW_LIVE_TIER_UAT: "1" })).toBe(false)
    expect(shouldInstallSmokeApiStubs({})).toBe(true)
  })

  it("marks page.route API fulfillment as intercepted coverage", () => {
    const source = `
      test("uses intercepted notes", async ({ page }) => {
        await page.route("**/api/v1/notes", route => route.fulfill({ json: [] }))
      })
    `

    expect(inventorySource(source, { project: "tier-1", file: "notes.spec.ts" }))
      .toEqual([
        expect.objectContaining({
          project: "tier-1",
          file: "notes.spec.ts",
          matcher: "**/api/v1/notes",
          kind: "intercepted",
          test: "uses intercepted notes",
        }),
      ])
  })

  it("does not classify passive API request assertions as intercepted", () => {
    const source = `
      test("uses the backend", async ({ page }) => {
        const response = page.waitForResponse(/\\/api\\/v1\\/notes/)
        await page.getByRole("button", { name: "Save" }).click()
        await response
      })
    `

    expect(inventorySource(source, { project: "tier-1", file: "notes.spec.ts" }))
      .toEqual([])
  })

  it("detects route interception delegated to a fulfillment helper", () => {
    const source = `
      const fulfillJson = async (route, body) => route.fulfill({ json: body })
      test("uses delegated interception", async ({ page }) => {
        await page.route("**/api/v1/delegated", async route => {
          await fulfillJson(route, [])
        })
      })
    `

    expect(inventorySource(source, { project: "tier-2", file: "delegated.spec.ts" }))
      .toEqual([
        expect.objectContaining({
          matcher: "**/api/v1/delegated",
          test: "uses delegated interception",
        }),
      ])
  })

  it("attributes nested runnable tests without treating describe or skip as tests", () => {
    const source = `
      test.describe("outer suite", () => {
        test.skip("disabled route", async ({ page }) => {
          await page.route("**/api/v1/disabled", route => route.abort())
        })
        test("live route", async ({ page }) => {
          await page.route("**/api/v1/live", route => route.abort())
        })
      })
    `

    expect(inventorySource(source, { project: "tier-3", file: "nested.spec.ts" }))
      .toEqual([
        expect.objectContaining({
          matcher: "**/api/v1/live",
          test: "live route",
        }),
      ])
  })
})

describe("live Tier UAT reporting", () => {
  it("records exact per-project denominators from Playwright list output", () => {
    const listed = parseListOutput(`
      [tier-1] › one.spec.ts:1:1 › one
      [tier-1] › one.spec.ts:2:1 › two
      [tier-2] › two.spec.ts:1:1 › three
      Total: 3 tests in 2 files
    `)

    expect(listed).toEqual({ "tier-1": 2, "tier-2": 1 })
  })

  it("summarizes passed, failed, skipped, and interrupted Playwright results", () => {
    const summary = summarizePlaywrightReport({
      suites: [{
        title: "suite",
        specs: [
          { title: "passes", tests: [{ projectName: "tier-1", status: "expected", results: [{ status: "passed", duration: 11 }] }] },
          { title: "fails", tests: [{ projectName: "tier-1", status: "unexpected", results: [{ status: "failed", duration: 12 }] }] },
          { title: "skips", tests: [{ projectName: "tier-2", status: "skipped", results: [{ status: "skipped", duration: 0 }] }] },
          { title: "interrupts", tests: [{ projectName: "tier-2", status: "unexpected", results: [{ status: "interrupted", duration: 3 }] }] },
        ],
      }],
    })

    expect(summary).toEqual({
      "tier-1": { passed: 1, failed: 1, skipped: 0, interrupted: 0, elapsedMs: 23 },
      "tier-2": { passed: 0, failed: 0, skipped: 1, interrupted: 1, elapsedMs: 3 },
    })
  })

  it("collects exact skipped test titles and annotation reasons", () => {
    const skipped = collectSkippedTests({
      suites: [{
        title: "suite",
        specs: [{
          title: "does not run",
          tests: [{
            projectName: "tier-2",
            status: "skipped",
            annotations: [{ type: "skip", description: "missing fixture" }],
            results: [{ status: "skipped", duration: 0 }],
          }],
        }],
      }],
    })

    expect(skipped).toEqual([
      { project: "tier-2", title: "does not run", reason: "missing fixture" },
    ])
  })

  it("renders denominators, live/intercepted counts, health, and artifacts", () => {
    const markdown = renderMarkdownReport({
      runId: "run-1",
      commit: "abc123",
      listed: { "tier-1": 2 },
      results: { "tier-1": { passed: 2, failed: 0, skipped: 0, interrupted: 0, elapsedMs: 45 } },
      inventory: [{ project: "tier-1", kind: "intercepted", file: "one.spec.ts", line: 3, matcher: "**/api/v1/x", test: "mocked" }],
      health: { before: true, after: true },
      artifacts: { root: "/tmp/run-1" },
      skippedTests: [{ project: "tier-2", title: "skipped flow", reason: "no fixture" }],
    })

    expect(markdown).toContain("| tier-1 | 2 | 2 | 0 | 0 | 1 | 1 |")
    expect(markdown).toContain("Health before tests: healthy")
    expect(markdown).toContain("`/tmp/run-1`")
    expect(markdown).toContain("skipped flow")
    expect(markdown).toContain("no fixture")
  })
})

describe("live Tier UAT runner contract", () => {
  it("accepts only Tier 1-3 projects and defaults to a serial complete run", () => {
    expect(parseArgs([])).toMatchObject({
      projects: ["tier-1", "tier-2", "tier-3"],
      workers: 1,
      listOnly: false,
      grep: null,
      failOnSkip: true,
    })
    expect(parseArgs(["--allow-skips"]).failOnSkip).toBe(false)
    expect(() => parseArgs(["--projects=tier-4"])).toThrow(/tier-1, tier-2, or tier-3/)
  })

  it("builds retry-free Playwright commands with offline fallback disabled", () => {
    const profile = {
      repoRoot,
      runDir: "/tmp/live-tier",
      logsDir: "/tmp/live-tier/logs",
      reportsDir: "/tmp/live-tier/reports",
      configPath: "/tmp/live-tier/config.txt",
      envPath: "/tmp/live-tier/.env",
      usersDbPath: "/tmp/live-tier/users.db",
      databaseDir: "/tmp/live-tier/Databases",
      fixtureRoot: path.join(frontendRoot, "e2e/fixtures/media"),
      acpWorkspaceRootBase: "/tmp/live-tier/acp-workspaces",
      databasePaths: { evaluations: "/tmp/live-tier/evaluations.db" },
    }
    const commands = buildCommands({
      repoRoot,
      frontendRoot,
      ports: { backend: 18180, web: 18181, mock: 18182 },
      profile,
      projects: ["tier-1", "tier-2"],
      workers: 1,
      runId: "run-contract",
      baseEnv: { NODE_ENV: "test", PATH: process.env.PATH },
    })

    expect(commands.playwrightList.args).toEqual(expect.arrayContaining([
      "test", "--list", "--project=tier-1", "--project=tier-2",
    ]))
    expect(commands.playwrightRun.args).toEqual(expect.arrayContaining([
      "test", "--project=tier-1", "--project=tier-2", "--workers=1", "--retries=0",
    ]))
    expect(commands.playwrightRun.args.join(" ")).not.toContain("--grep")
    expect(commands.playwrightRun.env.TLDW_E2E_ALLOW_OFFLINE).toBe("0")
    expect(commands.playwrightRun.env.TLDW_E2E_ACP_WORKSPACE_ROOT_BASE).toBe(
      profile.acpWorkspaceRootBase
    )
    expect(commands.playwrightRun.env.TLDW_E2E_INGESTION_SOURCE_ROOT).toBe(
      profile.fixtureRoot
    )
    expect(commands.playwrightRun.env.TLDW_WEB_AUTOSTART).toBe("false")
    expect(commands.frontend.env.TLDW_NEXT_DIST_DIR).toBe(".next-live-tier-run-contract")
    expect(commands.nextDistPath).toBe(path.join(frontendRoot, ".next-live-tier-run-contract"))

    const filteredCommands = buildCommands({
      repoRoot,
      frontendRoot,
      ports: { backend: 18180, web: 18181, mock: 18182 },
      profile,
      projects: ["tier-2"],
      workers: 1,
      runId: "run-contract-filtered",
      grep: "Sources",
      baseEnv: { NODE_ENV: "test", PATH: process.env.PATH },
    })
    expect(filteredCommands.playwrightList.args).toEqual(expect.arrayContaining([
      "--grep", "Sources",
    ]))
    expect(filteredCommands.playwrightRun.args).toEqual(expect.arrayContaining([
      "--grep", "Sources",
    ]))
  })

  it("stops only registered process groups in reverse startup order", async () => {
    const stop = vi.fn(async (_record: { name: string }) => undefined)
    const records = [{ name: "mock" }, { name: "backend" }, { name: "web" }]

    await stopSpawnedProcesses(records, stop)

    expect(stop.mock.calls.map(([record]) => record.name)).toEqual(["web", "backend", "mock"])
  })

  it("attempts every registered process stop even when an earlier stop fails", async () => {
    const stop = vi.fn(async (record: { name: string }) => {
      if (record.name === "web") throw new Error("web stop failed")
    })
    const records = [{ name: "mock" }, { name: "backend" }, { name: "web" }]

    await expect(stopSpawnedProcesses(records, stop)).rejects.toThrow(/web stop failed/)

    expect(stop.mock.calls.map(([record]) => record.name)).toEqual(["web", "backend", "mock"])
  })

  it("keeps termination handlers installed while repeated signals share one abort", () => {
    const processLike = new EventEmitter()
    const controller = new AbortController()
    const termination = installTerminationHandlers({ processLike, controller })

    processLike.emit("SIGINT")
    processLike.emit("SIGTERM")

    expect(controller.signal.aborted).toBe(true)
    expect(controller.signal.reason).toMatchObject({ signalName: "SIGINT" })
    expect(processLike.listenerCount("SIGINT")).toBe(1)
    expect(processLike.listenerCount("SIGTERM")).toBe(1)
    termination.dispose()
    expect(processLike.listenerCount("SIGINT")).toBe(0)
    expect(processLike.listenerCount("SIGTERM")).toBe(0)
  })

  it("waits for an aborted detached command group to stop before resolving", async () => {
    const runDir = mkdtempSync(path.join(tmpdir(), "tldw-live-tier-abort-"))
    const pidPath = path.join(runDir, "pids.json")
    const logPath = path.join(runDir, "command.log")
    const childScript = [
      'const { spawn } = require("node:child_process")',
      'const { writeFileSync } = require("node:fs")',
      'const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: "ignore" })',
      `writeFileSync(${JSON.stringify(pidPath)}, JSON.stringify({ parent: process.pid, child: child.pid }))`,
      'setInterval(() => {}, 1000)',
    ].join(";")
    const controller = new AbortController()
    let stopFinished = false

    try {
      const resultPromise = runCommand({
        name: "abort-regression",
        command: process.execPath,
        args: ["-e", childScript],
        cwd: runDir,
        env: process.env,
      }, logPath, {
        signal: controller.signal,
        stop: async (record) => {
          await stopProcessTree(record)
          await new Promise((resolve) => setTimeout(resolve, 50))
          stopFinished = true
        },
      })

      for (let attempt = 0; attempt < 100 && !existsSync(pidPath); attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 10))
      }
      expect(existsSync(pidPath)).toBe(true)
      const pids = JSON.parse(readFileSync(pidPath, "utf8")) as { parent: number; child: number }

      controller.abort(new Error("test abort"))
      const result = await resultPromise

      expect(result).toMatchObject({ aborted: true })
      expect(stopFinished).toBe(true)
      for (const pid of [pids.parent, pids.child]) {
        expect(() => process.kill(pid, 0)).toThrow(expect.objectContaining({ code: "ESRCH" }))
      }
    } finally {
      rmSync(runDir, { recursive: true, force: true })
    }
  })

  it("rejects reused run targets before certification starts", () => {
    expect(() => assertFreshRunTargets(
      ["/tmp/artifacts", "/tmp/profile"],
      (target) => target === "/tmp/artifacts"
    )).toThrow(/already exists.*artifacts/i)
  })

  it("reports and cleans generated paths when profile setup fails", async () => {
    const testRoot = mkdtempSync(path.join(tmpdir(), "tldw-live-tier-preflight-"))
    const testRepoRoot = path.join(testRoot, "repo-without-config")
    const testFrontendRoot = path.join(testRoot, "frontend")
    const runId = `preflight-${process.pid}-${Date.now()}`
    const artifactRoot = path.join(
      testFrontendRoot,
      "test-results/live-tier-uat",
      runId
    )
    const profileRoot = path.join(tmpdir(), `tldw-onboarding-uat-${runId}`)
    const nextDistPath = path.join(testFrontendRoot, `.next-live-tier-${runId}`)
    mkdirSync(testRepoRoot, { recursive: true })
    mkdirSync(testFrontendRoot, { recursive: true })

    try {
      const result = await runLiveTierUat({
        options: { ...parseArgs([]), runId },
        repoRoot: testRepoRoot,
        frontendRoot: testFrontendRoot,
        baseEnv: { NODE_ENV: "test", PATH: process.env.PATH },
      })

      expect(result.status).toBe("failed")
      expect(result.error).toMatch(/config\.txt/)
      expect(readFileSync(result.reportPath, "utf8")).toContain("config.txt")
      expect(existsSync(result.summaryPath)).toBe(true)
      expect(existsSync(profileRoot)).toBe(false)
      expect(existsSync(nextDistPath)).toBe(false)
      expect(existsSync(artifactRoot)).toBe(true)
    } finally {
      rmSync(profileRoot, { recursive: true, force: true })
      rmSync(testRoot, { recursive: true, force: true })
    }
  })

  it("requires a Playwright JSON report instead of accepting missing evidence", () => {
    expect(() => readPlaywrightReport("/tmp/missing-playwright.json", {
      exists: () => false,
    })).toThrow(/Playwright JSON report was not produced/)
  })

  it("requires exact per-project accounting for certification", () => {
    expect(() => assertProjectAccounting({
      projects: ["tier-1"],
      listed: { "tier-1": 2 },
      results: {
        "tier-1": { passed: 1, failed: 0, skipped: 0, interrupted: 0, elapsedMs: 10 },
      },
    })).toThrow(/tier-1.*listed 2.*accounted 1/i)

    expect(() => assertProjectAccounting({
      projects: ["tier-1"],
      listed: { "tier-1": 1 },
      results: {
        "tier-1": { passed: 0, failed: 1, skipped: 0, interrupted: 0, elapsedMs: 10 },
      },
    })).toThrow(/tier-1.*failed 1/i)
  })

  it("labels only strict skip-failing complete runs as certification", () => {
    const complete = {
      listOnly: false,
      grep: null,
      failOnSkip: true,
      projects: ["tier-1", "tier-2", "tier-3"],
      workers: 1,
    }

    expect(isCertificationRun(complete)).toBe(true)
    expect(isCertificationRun({ ...complete, failOnSkip: false })).toBe(false)
    expect(isCertificationRun({ ...complete, grep: "focused" })).toBe(false)
    expect(isCertificationRun({ ...complete, projects: ["tier-1"] })).toBe(false)
    expect(isCertificationRun({ ...complete, workers: 2 })).toBe(false)
  })

  it("rejects certification when any spawned service remains reachable", () => {
    expect(() => assertServicesStopped(false)).toThrow(/did not stop/i)
  })

  it("retries bounded cleanup races only for the exact generated target", async () => {
    const busy = Object.assign(new Error("directory not empty"), { code: "ENOTEMPTY" })
    const remove = vi.fn()
      .mockImplementationOnce(() => { throw busy })
      .mockImplementationOnce(() => undefined)
    const wait = vi.fn(async () => undefined)

    await removeGeneratedPath("/tmp/.next-live-tier-run", {
      expectedParent: "/tmp",
      expectedPrefix: ".next-live-tier-",
      remove,
      wait,
      maxAttempts: 3,
      retryDelayMs: 1,
    })

    expect(remove).toHaveBeenCalledTimes(2)
    expect(remove).toHaveBeenLastCalledWith(
      "/tmp/.next-live-tier-run",
      { recursive: true, force: true }
    )
    expect(wait).toHaveBeenCalledWith(1)
    await expect(removeGeneratedPath("/tmp", {
      expectedParent: "/",
      expectedPrefix: ".next-live-tier-",
      remove,
      wait,
    })).rejects.toThrow(/cleanup target/)
  })

  it("cleans generated Go module caches whose directories are read-only", async () => {
    const cleanupParent = mkdtempSync(path.join(tmpdir(), "tldw-live-tier-cleanup-"))
    const target = path.join(cleanupParent, "profile-owned")
    const moduleDir = path.join(target, "go/pkg/mod/example.invalid/module@v1")
    mkdirSync(moduleDir, { recursive: true })
    writeFileSync(path.join(moduleDir, "module.go"), "package module\n", "utf8")
    chmodSync(moduleDir, 0o500)

    try {
      await removeGeneratedPath(target, {
        expectedParent: cleanupParent,
        expectedPrefix: "profile-",
      })
      expect(existsSync(target)).toBe(false)
    } finally {
      try { chmodSync(moduleDir, 0o700) } catch {}
      rmSync(cleanupParent, { recursive: true, force: true })
    }
  })

  it("rejects backend logs that reveal a mutable worktree database path", () => {
    expect(() => assertNoMutableRepoDatabasePaths(
      `Initialized /repo/Databases/user_databases/1/Media_DB_v2.db`,
      "/repo"
    )).toThrow(/mutable database path escaped/)
    expect(() => assertNoMutableRepoDatabasePaths(
      `Initialized /tmp/live-tier/Databases/user_databases/1/Media_DB_v2.db`,
      "/repo"
    )).not.toThrow()
  })

  it("rejects backend HTTP requests to non-loopback provider endpoints", () => {
    expect(() => assertOnlyLoopbackHttpRequests([
      '"url.full": "http://127.0.0.1:18195/v1/models"',
      'HTTP Request: POST http://localhost:18195/v1/chat/completions "HTTP/1.1 200 OK"',
    ].join("\n"))).not.toThrow()

    expect(() => assertOnlyLoopbackHttpRequests(
      '"url.full": "http://192.168.2.235:5000/v1/models"'
    )).toThrow(/non-loopback HTTP request.*192\.168\.2\.235/i)
  })

  it("exposes the package command documented by the certification plan", () => {
    const packageJson = JSON.parse(readFileSync(path.join(frontendRoot, "package.json"), "utf8"))

    expect(packageJson.scripts["uat:live-tiers"]).toBe("node scripts/live-tier-uat/run.mjs")
  })

  it("registers Chatbooks roundtrip coverage without a live-tier skip gate", () => {
    const source = readFileSync(
      path.join(
        frontendRoot,
        "e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts"
      ),
      "utf8"
    )

    expect(source).toContain("liveTierEnabled")
    expect(source).toContain("if (liveTierEnabled)")
    expect(source).not.toContain('test.skip(!enabled, "Run through chatbooks_full_account_browser_uat.py")')
  })

  it("lets the runner select a unique validated Next build directory", () => {
    const nextConfig = readFileSync(path.join(frontendRoot, "next.config.mjs"), "utf8")

    expect(nextConfig).toContain("TLDW_NEXT_DIST_DIR")
    expect(nextConfig).toContain("liveTierDistDir")
  })
})
