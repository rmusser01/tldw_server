import { describe, expect, it } from "vitest"
import { EventEmitter } from "node:events"
import { execFileSync } from "node:child_process"
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import { createServer } from "node:http"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { reservePort, reservePorts } from "../onboarding-uat/ports.mjs"
import {
  SYNTHETIC_SECRETS,
  assertNoSecretLeaks,
  cleanupRunArtifacts,
  createRunArtifacts,
  redactText,
} from "../onboarding-uat/artifacts.mjs"
import {
  buildBackendEnv,
  createRuntimeProfile,
} from "../onboarding-uat/profile.mjs"
import {
  spawnLoggedProcess,
  stopProcessTree,
  waitForHttpOk,
} from "../onboarding-uat/processes.mjs"
import {
  buildCommands,
  copyReviewedEvidence,
  formatUsage,
  parseArgs,
  resolveEffectiveMockConfig,
  runOnboardingUat,
} from "../onboarding-uat/run.mjs"
import { tierAScenarios } from "../../e2e/onboarding-uat/scenarios"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(testDir, "../..")
const repoRoot = path.resolve(frontendRoot, "../..")
const mockOpenAiRoot = path.join(frontendRoot, "e2e/onboarding-uat/mock-openai")

const configFiles = [
  "hosted-success.json",
  "local-success.json",
  "local-models-unavailable.json",
  "chat-fail-once.json",
  "model-unavailable.json",
  "local-model-unavailable.json",
]

const responseFiles = [
  "chat/analysis-one.json",
  "chat/analysis-two.json",
  "chat/default.json",
  "chat/source-summary.json",
  "embeddings/default.json",
]

const secretMarkers = [
  "sk-",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "ghp_",
  "xoxb-",
  "AKIA",
  "BEGIN " + "PRIVATE " + "KEY",
]

function readJson(relativePath: string) {
  return JSON.parse(readFileSync(path.join(mockOpenAiRoot, relativePath), "utf8"))
}

function readText(relativePath: string) {
  return readFileSync(path.join(frontendRoot, relativePath), "utf8")
}

function modelIds(config: { models?: Array<{ id?: string }> }) {
  return (config.models ?? []).map((model) => model.id)
}

describe("onboarding UAT static fixtures", () => {
  it("scrubs copied provider secrets and redirects provider endpoints to the mock", () => {
    const fixtureRoot = mkdtempSync(path.join(tmpdir(), "tldw-hostile-profile-"))
    const fixtureRepo = path.join(fixtureRoot, "repo")
    const fixtureFrontend = path.join(fixtureRepo, "apps/tldw-frontend")
    const configDir = path.join(fixtureRepo, "tldw_Server_API/Config_Files")
    mkdirSync(configDir, { recursive: true })
    mkdirSync(fixtureFrontend, { recursive: true })
    writeFileSync(
      path.join(configDir, "config.txt"),
      [
        "[Setup]",
        "enable_first_time_setup = false",
        "setup_completed = true",
        "",
        "[AuthNZ]",
        "auth_mode = multi_user",
        "single_user_api_key = copied-auth-key",
        "database_url = postgresql://host-user:host-password@db.example.test/app",
        "",
        "[Providers]",
        "OPENAI_API_KEY = host-provider-secret",
        "service token = host-service-token",
        "database.password = host-database-password",
        "byok_encryption_key = host-encryption-key",
        "tts_history_hash_key = host-hash-key",
        "llama_api_IP = http://192.168.2.235:5000/v1",
        "custom_openai_api_ip = https://api.example.test/v1",
        "web_scraper_api_url = https://scraper.example.test/api",
        "provider_name = retained-provider",
        "",
      ].join("\n"),
      "utf8"
    )

    try {
      const profile = createRuntimeProfile({
        repoRoot: fixtureRepo,
        frontendRoot: fixtureFrontend,
        runId: "hostile-source",
        mockPort: 18195,
        baseTmpDir: fixtureRoot,
      })
      const config = readFileSync(profile.configPath, "utf8")
      const mockBaseUrl = "http://127.0.0.1:18195/v1"

      expect(config).toContain(`llama_api_IP = ${mockBaseUrl}`)
      expect(config).toContain(`custom_openai_api_ip = ${mockBaseUrl}`)
      expect(config).toContain(`web_scraper_api_url = ${mockBaseUrl}`)
      expect(config).toContain(`database_url = sqlite:///${profile.usersDbPath}`)
      expect(config).toContain("provider_name = retained-provider")
      for (const forbidden of [
        "copied-auth-key",
        "host-provider-secret",
        "host-service-token",
        "host-database-password",
        "host-encryption-key",
        "host-hash-key",
        "192.168.2.235",
        "api.example.test",
        "scraper.example.test",
        "db.example.test",
      ]) {
        expect(config).not.toContain(forbidden)
      }
    } finally {
      rmSync(fixtureRoot, { recursive: true, force: true })
    }
  })

  it("provides mock OpenAI config files with the expected scenario shapes", () => {
    const configs = Object.fromEntries(
      configFiles.map((file) => [file, readJson(`configs/${file}`)])
    )

    expect(modelIds(configs["hosted-success.json"])).toEqual(
      expect.arrayContaining(["gpt-4.1-mini", "text-embedding-3-small"])
    )
    expect(configs["hosted-success.json"].server?.require_auth).toBe(true)

    expect(modelIds(configs["local-success.json"])).toEqual(
      expect.arrayContaining(["llama3.2:3b", "local-uat-chat"])
    )
    expect(configs["local-success.json"].server?.require_auth).toBe(false)

    const localModelsUnavailable = configs["local-models-unavailable.json"]
    expect(modelIds(localModelsUnavailable)).toContain("llama3.2:3b")
    expect(
      localModelsUnavailable.scenario_failures?.models?.[0]
    ).toMatchObject({
      status_code: 404,
      code: "models_unavailable",
      times: 999,
    })

    expect(
      configs["chat-fail-once.json"].scenario_failures?.chat_completions?.[0]
    ).toMatchObject({
      status_code: 503,
      code: "uat_fail_once",
      times: 1,
    })

    const unavailable = configs["model-unavailable.json"]
    const failsSelectedChatModel = (
      unavailable.scenario_failures?.chat_completions ?? []
    ).some(
      (failure: {
        match?: { model?: string }
        status_code?: number
        code?: string
      }) =>
        failure.match?.model === "gpt-4.1-mini" &&
        failure.status_code === 404 &&
        failure.code === "model_not_found"
    )

    expect(modelIds(unavailable)).toContain("gpt-4.1-mini")
    expect(failsSelectedChatModel).toBe(true)

    const localUnavailable = configs["local-model-unavailable.json"]
    const failsSelectedLocalChatModel = (
      localUnavailable.scenario_failures?.chat_completions ?? []
    ).some(
      (failure: {
        match?: { model?: string }
        status_code?: number
        code?: string
      }) =>
        failure.match?.model === "llama3.2:3b" &&
        failure.status_code === 404 &&
        failure.code === "model_not_found"
    )

    expect(modelIds(localUnavailable)).toContain("llama3.2:3b")
    expect(failsSelectedLocalChatModel).toBe(true)
  })

  it("registers explicit local-provider PR4 scenarios", () => {
    const scenarioIds = tierAScenarios.map((scenario) => scenario.id)

    expect(scenarioIds).toEqual(
      expect.arrayContaining([
        "local-openai-discovered-model-first-chat",
        "local-openai-manual-model-first-chat",
        "local-openai-model-unavailable-recovery",
        "local-to-hosted-switch-state-isolated",
      ])
    )
    expect(
      tierAScenarios.find(
        (scenario) =>
          scenario.id === "local-openai-discovered-model-first-chat"
      )?.viewports
    ).toEqual(expect.arrayContaining(["desktop", "mobile"]))
  })

  it("keeps Playwright output outside the runner artifact root", () => {
    const config = readText("e2e/onboarding-uat/playwright.config.ts")

    expect(config).toContain(
      'outputDir: "test-results/playwright-onboarding-uat"'
    )
  })

  it("keeps mock response paths resolvable from each config file", () => {
    for (const file of configFiles) {
      const configPath = path.join(mockOpenAiRoot, "configs", file)
      const config = readJson(`configs/${file}`)
      const responseBaseDir = path.resolve(
        path.dirname(configPath),
        config.response_base_dir
      )

      expect(responseBaseDir).toBe(path.join(mockOpenAiRoot, "responses"))

      for (const endpoint of Object.values(config.responses ?? {}) as Array<{
        default?: string
        patterns?: Array<{ response_file?: string }>
      }>) {
        if (endpoint.default) {
          expect(existsSync(path.join(responseBaseDir, endpoint.default))).toBe(true)
        }
        for (const pattern of endpoint.patterns ?? []) {
          if (pattern.response_file) {
            expect(existsSync(path.join(responseBaseDir, pattern.response_file))).toBe(true)
          }
        }
      }
    }
  })

  it("keeps every referenced response fixture eligible for source control", () => {
    for (const file of ["chat/analysis-one.json", "chat/analysis-two.json"]) {
      const fixturePath = path.join(mockOpenAiRoot, "responses", file)

      expect(() =>
        execFileSync(
          "git",
          ["check-ignore", "--no-index", "--quiet", fixturePath],
          { cwd: repoRoot, stdio: "ignore" }
        )
      ).toThrow()
    }
  })

  it("keeps JSON fixtures static, synthetic, and free of obvious real-secret markers", () => {
    for (const file of [
      ...configFiles.map((name) => `configs/${name}`),
      ...responseFiles.map((name) => `responses/${name}`),
    ]) {
      const content = readFileSync(path.join(mockOpenAiRoot, file), "utf8")

      expect(() => JSON.parse(content)).not.toThrow()
      for (const marker of secretMarkers) {
        expect(content).not.toContain(marker)
      }
    }
  })

  it("provides stable chat and embedding responses for future UAT assertions", () => {
    const analysisOne = readJson("responses/chat/analysis-one.json")
    const analysisTwo = readJson("responses/chat/analysis-two.json")
    const defaultChat = readJson("responses/chat/default.json")
    const sourceSummary = readJson("responses/chat/source-summary.json")
    const embeddings = readJson("responses/embeddings/default.json")

    expect(defaultChat.choices?.[0]?.message?.content).toContain(
      "onboarding UAT ready"
    )
    expect(analysisOne.choices?.[0]?.message?.content).toBe("LIVE_TIER_ANALYSIS_ONE")
    expect(analysisTwo.choices?.[0]?.message?.content).toBe("LIVE_TIER_ANALYSIS_TWO")
    expect(analysisOne.choices?.[0]?.message?.content).not.toBe(
      analysisTwo.choices?.[0]?.message?.content
    )
    expect(sourceSummary.choices?.[0]?.message?.content).toContain(
      "short first-run wizard"
    )
    expect(sourceSummary.choices?.[0]?.message?.content).toContain(
      "Deterministic evidence"
    )
    expect(defaultChat.model).toBeUndefined()
    expect(sourceSummary.model).toBeUndefined()

    expect(embeddings.object).toBe("list")
    expect(embeddings.data?.[0]).toMatchObject({
      object: "embedding",
      index: 0,
    })
    expect(embeddings.data?.[0]?.embedding.length).toBeGreaterThan(0)
    const embeddingDimension = embeddings.data[0].embedding.length
    expect(embeddings.data.length).toBeGreaterThanOrEqual(3)
    for (const item of embeddings.data) {
      expect(item.embedding.length).toBe(embeddingDimension)
    }
    expect(
      embeddings.data[0].embedding.every((value: unknown) => typeof value === "number")
    ).toBe(true)
  })

  it("provides structured markdown and HTML source fixtures", () => {
    const markdownPath = "e2e/fixtures/media/onboarding-uat-note.md"
    const htmlPath = "public/e2e/onboarding-uat-research-note.html"

    expect(existsSync(path.join(frontendRoot, markdownPath))).toBe(true)
    expect(existsSync(path.join(frontendRoot, htmlPath))).toBe(true)

    const markdown = readText(markdownPath)
    const html = readText(htmlPath)

    for (const fixture of [markdown, html]) {
      expect(fixture).toContain("Onboarding UAT Research Note")
      expect(fixture).toContain("2026-06-02")
      expect(fixture).toContain("Claims")
      expect(fixture).toContain("short first-run wizard")
      expect(fixture).toContain("Deterministic evidence")
      expect(fixture).toContain("Action Items")
      expect(fixture).toContain("Verify first chat")
      expect(fixture).toContain("Add one source")
      expect(fixture).toContain("Ask for a summary")
    }
  })
})

describe("onboarding UAT runner helpers", () => {
  it("redacts synthetic secrets and common API key carrier formats", () => {
    const redacted = redactText(
      [
        "plain sk-uat-mock-openai",
        "api THIS-IS-A-SECURE-KEY-123-UAT",
        "Authorization: Bearer sk-live-never-log-this",
        "x-api-key: sk-header-never-log-this",
      ].join("\n")
    )

    for (const secret of SYNTHETIC_SECRETS) {
      expect(redacted).not.toContain(secret)
    }
    expect(redacted).toContain("Bearer [REDACTED]")
    expect(redacted).toContain("x-api-key: [REDACTED]")
  })

  it("creates and cleans up the expected artifact layout", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-artifacts",
      preserve: false,
    })

    expect(artifacts.root).toBe(
      path.join(frontendRoot, "test-results/onboarding-uat/unit-artifacts")
    )
    expect(existsSync(artifacts.summaryPath)).toBe(true)
    expect(existsSync(artifacts.logs.backend)).toBe(true)
    expect(existsSync(artifacts.logs.frontend)).toBe(true)
    expect(existsSync(artifacts.logs.mockOpenai)).toBe(true)
    expect(existsSync(artifacts.logs.runner)).toBe(true)
    expect(existsSync(artifacts.browserDiagnosticsPath)).toBe(true)
    expect(existsSync(artifacts.screenshotsDir)).toBe(true)
    expect(existsSync(artifacts.runtimeProfileManifestPath)).toBe(true)

    cleanupRunArtifacts(artifacts)

    expect(existsSync(artifacts.root)).toBe(false)
  })

  it("refuses to clean artifact paths outside the onboarding UAT result root", () => {
    expect(() =>
      cleanupRunArtifacts({
        preserve: false,
        root: path.join(tmpdir(), "unsafe-onboarding-cleanup-target"),
      })
    ).toThrow(/refusing to remove/i)
  })

  it("detects unredacted synthetic secret leaks in artifact files", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-leak-check",
      preserve: false,
    })

    try {
      writeFileSync(artifacts.logs.runner, "leaked sk-uat-mock-openai", "utf8")

      expect(() => assertNoSecretLeaks(artifacts.root)).toThrow(
        /secret leak/i
      )
    } finally {
      cleanupRunArtifacts(artifacts)
    }
  })

  it("detects common unredacted secret-like artifact content", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-real-secret-leak-check",
      preserve: false,
    })

    try {
      writeFileSync(artifacts.logs.runner, "ANTHROPIC_API_KEY=real-secret", "utf8")

      expect(() => assertNoSecretLeaks(artifacts.root)).toThrow(
        /secret leak/i
      )
    } finally {
      cleanupRunArtifacts(artifacts)
    }
  })

  it("accepts explicit redaction markers in secret carrier formats", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-redacted-secret-check",
      preserve: false,
    })

    try {
      writeFileSync(
        artifacts.logs.runner,
        [
          "ANTHROPIC_API_KEY=[REDACTED]",
          "x-api-key: [REDACTED]",
          '"authorization": "[REDACTED]"',
        ].join("\n"),
        "utf8"
      )

      expect(() => assertNoSecretLeaks(artifacts.root)).not.toThrow()
    } finally {
      cleanupRunArtifacts(artifacts)
    }
  })

  it("detects raw and JSON-shaped secret-like artifact leaks", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-json-secret-leak-check",
      preserve: false,
    })
    const privateKeyMarker = ["-----BEGIN", "PRIVATE " + "KEY-----"].join(" ")

    try {
      writeFileSync(
        artifacts.browserDiagnosticsPath,
        JSON.stringify(
          {
            headers: { "x-api-key": "sk-live-json-leak" },
            env: { OPENAI_API_KEY: "sk-live-json-leak" },
            tokens: ["ghp_realgithubtoken", "xoxb-real-slack-token", "AKIAIOSFODNN7EXAMPLE"],
            privateKey: privateKeyMarker,
          },
          null,
          2
        ),
        "utf8"
      )

      expect(() => assertNoSecretLeaks(artifacts.root)).toThrow(
        /secret leak/i
      )
    } finally {
      cleanupRunArtifacts(artifacts)
    }
  })

  it("detects header-array secret leaks in JSON diagnostics", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-header-array-secret-leak-check",
      preserve: false,
    })

    try {
      writeFileSync(
        artifacts.browserDiagnosticsPath,
        JSON.stringify(
          {
            headers: [
              { name: "accept", value: "application/json" },
              { name: "x-api-key", value: "real-secret-value" },
            ],
          },
          null,
          2
        ),
        "utf8"
      )

      expect(() => assertNoSecretLeaks(artifacts.root)).toThrow(
        /secret leak/i
      )
    } finally {
      cleanupRunArtifacts(artifacts)
    }
  })

  it("creates an isolated runtime profile and backend env for the mock provider", () => {
    const profile = createRuntimeProfile({
      repoRoot,
      frontendRoot,
      runId: "unit-profile",
      mockPort: 43210,
      baseTmpDir: path.join(tmpdir(), "tldw-onboarding-uat-tests"),
    })

    try {
      const config = readFileSync(profile.configPath, "utf8")
      const envText = readFileSync(profile.envPath, "utf8")

      expect(config).toContain("enable_first_time_setup = true")
      expect(config).toContain("setup_completed = false")
      expect(config).toContain("openai_model = gpt-4.1-mini")
      expect(config).toContain("custom_openai_api_ip = http://127.0.0.1:43210/v1")
      expect(config).toContain("custom_openai_api_model = local-uat-chat")
      expect(config).toContain("ollama_api_IP = http://127.0.0.1:43210/v1")
      expect(config).toContain("ollama_model = llama3.2:3b")
      expect(config).toContain(
        `USER_DB_BASE_DIR = ${path.join(profile.root, "Databases/user_databases")}`
      )
      expect(config).toContain(
        `ingestion_source_allowed_roots = ${path.join(frontendRoot, "e2e/fixtures/media")}`
      )

      expect(envText).toContain("AUTH_MODE=single_user")
      expect(envText).toContain("SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT")
      expect(envText).toContain("DEFAULT_LLM_PROVIDER=openai")
      expect(envText).toContain("OPENAI_API_KEY=sk-uat-mock-openai")
      expect(envText).toContain("OPENAI_API_BASE_URL=http://127.0.0.1:43210/v1")
      expect(envText).toContain("CUSTOM_OPENAI_API_KEY=sk-uat-mock-openai")
      expect(envText).toContain(`DATABASE_URL=sqlite:///${profile.usersDbPath}`)
      expect(envText).toContain("WORKFLOWS_EGRESS_BLOCK_PRIVATE=false")
      expect(envText).toContain("WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443,43210")
      expect(envText).not.toContain(path.join(repoRoot, "tldw_Server_API/Config_Files/.env"))

      const backendEnv = buildBackendEnv({
        profile,
        mockPort: 43210,
        baseEnv: {
          PATH: "/test/bin",
          HOME: "/test/home",
          OPENAI_API_KEY: "real-key",
          ANTHROPIC_API_KEY: "anthropic-real-key",
          GITHUB_TOKEN: "github-real-token",
        },
      })

      expect(backendEnv.PATH).toBe("/test/bin")
      expect(backendEnv.HOME).toBe("/test/home")
      expect(backendEnv.TLDW_CONFIG_FILE).toBe(profile.configPath)
      expect(backendEnv.TLDW_ENV_FILE).toBe(profile.envPath)
      expect(backendEnv.DATABASE_URL).toBe(`sqlite:///${profile.usersDbPath}`)
      expect(backendEnv.AUTH_MODE).toBe("single_user")
      expect(backendEnv.SINGLE_USER_API_KEY).toBe("THIS-IS-A-SECURE-KEY-123-UAT")
      expect(backendEnv.DEFAULT_LLM_PROVIDER).toBe("openai")
      expect(backendEnv.OPENAI_API_KEY).toBe("sk-uat-mock-openai")
      expect(backendEnv.OPENAI_API_BASE_URL).toBe("http://127.0.0.1:43210/v1")
      expect(backendEnv.CUSTOM_OPENAI_API_KEY).toBe("sk-uat-mock-openai")
      expect(backendEnv.WORKFLOWS_EGRESS_BLOCK_PRIVATE).toBe("false")
      expect(backendEnv.WORKFLOWS_EGRESS_ALLOWED_PORTS).toBe("80,443,43210")
      expect(backendEnv.ANTHROPIC_API_KEY).toBeUndefined()
      expect(backendEnv.GITHUB_TOKEN).toBeUndefined()
      expect(backendEnv.INGESTION_SOURCE_ALLOWED_ROOTS).toBe(
        path.join(frontendRoot, "e2e/fixtures/media")
      )
    } finally {
      rmSync(profile.root, { recursive: true, force: true })
    }
  })

  it("reserves distinct loopback ports", async () => {
    const onePort = await reservePort()
    const ports = await reservePorts(["backend", "web", "mock"])

    expect(onePort).toBeGreaterThan(0)
    expect(ports).toEqual({
      backend: expect.any(Number),
      web: expect.any(Number),
      mock: expect.any(Number),
    })
    expect(new Set(Object.values(ports)).size).toBe(3)
  })

  it("waits for HTTP readiness against a local server", async () => {
    const port = await reservePort()
    const server = createServer((request, response) => {
      if (request.url === "/ready") {
        response.writeHead(204)
      } else {
        response.writeHead(404)
      }
      response.end()
    })

    await new Promise<void>((resolve) => {
      server.listen(port, "127.0.0.1", resolve)
    })

    try {
      await expect(
        waitForHttpOk(`http://127.0.0.1:${port}/ready`, {
          timeoutMs: 500,
          intervalMs: 10,
        })
      ).resolves.toMatchObject({ ok: true, status: 204 })
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()))
      })
    }
  })

  it("aborts HTTP readiness waits when the runner is interrupted", async () => {
    const controller = new AbortController()
    controller.abort(new Error("test interruption"))

    await expect(waitForHttpOk("http://127.0.0.1:1/health", {
      timeoutMs: 1_000,
      signal: controller.signal,
    })).rejects.toThrow(/test interruption/)
  })

  it("writes redacted child process output to logs", async () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-process",
      preserve: false,
    })

    const record = spawnLoggedProcess({
      name: "unit-child",
      command: process.execPath,
      args: ["-e", "console.log('token sk-uat-mock-openai')"],
      cwd: frontendRoot,
      env: process.env,
      logPath: artifacts.logs.runner,
    })

    try {
      const exitCode = await new Promise<number | null>((resolve) => {
        record.child.once("exit", (code) => resolve(code))
      })

      expect(exitCode).toBe(0)
      const log = readFileSync(artifacts.logs.runner, "utf8")
      expect(log).toContain("[REDACTED]")
      expect(log).not.toContain("sk-uat-mock-openai")
    } finally {
      await stopProcessTree(record, { timeoutMs: 50 })
      cleanupRunArtifacts(artifacts)
    }
  })

  it("returns an owned record when its log sink is invalid", async () => {
    const record = spawnLoggedProcess({
      name: "invalid-log-child",
      command: process.execPath,
      args: ["-e", "setTimeout(() => { console.log('one'); console.log('two') }, 20); setInterval(() => {}, 1000)"],
      cwd: frontendRoot,
      env: process.env,
      logPath: path.join(process.execPath, "skills-certification.log"),
    })
    try {
      expect(record.child.pid).toBeTruthy()
      await new Promise<void>((resolve) => record.child.stdout?.once("data", () => resolve()))
      expect(record.loggingErrors.length).toBe(1)
      await stopProcessTree(record, { timeoutMs: 50 })
      expect(record.child.exitCode ?? record.child.signalCode).not.toBeNull()
    } finally {
      await stopProcessTree(record, { timeoutMs: 50 })
    }
  })

  it("waits for one-shot child process stdio close before artifact cleanup", () => {
    const runnerSource = readText("scripts/onboarding-uat/run.mjs")

    expect(runnerSource).toContain('child.once("close"')
    expect(runnerSource).not.toContain('child.once("exit"')
  })

  it("waits for process exit after SIGKILL escalation", async () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-process-kill",
      preserve: false,
    })

    const record = spawnLoggedProcess({
      name: "unit-stubborn-child",
      command: process.execPath,
      args: [
        "-e",
        "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)",
      ],
      cwd: frontendRoot,
      env: process.env,
      logPath: artifacts.logs.runner,
    })

    try {
      await stopProcessTree(record, { timeoutMs: 0 })
      expect(record.child.exitCode ?? record.child.signalCode).not.toBeNull()
    } finally {
      await stopProcessTree(record, { timeoutMs: 50 })
      cleanupRunArtifacts(artifacts)
    }
  })

  it("does not resolve stopProcessTree until SIGKILL exit is observed", async () => {
    class FakeChild extends EventEmitter {
      pid = 123456
      exitCode: number | null = null
      signalCode: string | null = null
      killed = false
      signals: string[] = []

      kill(signal: string) {
        this.signals.push(signal)
        if (signal === "SIGKILL") {
          setTimeout(() => {
            this.signalCode = signal
            this.emit("exit", null, signal)
          }, 20)
        }
        return true
      }
    }

    const child = new FakeChild()
    await stopProcessTree(child, { timeoutMs: 0 })

    expect(child.signals).toEqual(["SIGTERM", "SIGKILL"])
    expect(child.signalCode).toBe("SIGKILL")
  })

  it("returns when a child exits before the stop listener settles", async () => {
    class AlreadyExitedChild extends EventEmitter {
      pid = 123457
      exitCode: number | null = null
      signalCode: string | null = null
      killed = false
      signals: string[] = []

      kill(signal: string) {
        this.signals.push(signal)
        this.exitCode = 0
        return true
      }
    }

    const child = new AlreadyExitedChild()
    await stopProcessTree(child, { timeoutMs: 0 })

    expect(child.signals).toEqual(["SIGTERM"])
    expect(child.exitCode).toBe(0)
  })

  it("terminates a surviving detached process group after its parent exits", async () => {
    class ExitedDetachedChild extends EventEmitter {
      pid = 123460
      exitCode = 0
      signalCode: string | null = null
      spawnfile = process.execPath
    }

    const child = new ExitedDetachedChild()
    const signals: Array<[number, string | number]> = []
    let groupAlive = true
    const killProcess = (target: number, signal: string | number) => {
      signals.push([target, signal])
      if (signal === 0) {
        if (groupAlive) return true
        throw Object.assign(new Error("process group is gone"), { code: "ESRCH" })
      }
      if (signal === "SIGTERM") groupAlive = false
      return true
    }

    await stopProcessTree(child, {
      platform: "darwin",
      timeoutMs: 0,
      killProcess,
      wait: async () => undefined,
    })

    expect(signals).toContainEqual([-child.pid, "SIGTERM"])
    expect(signals).toContainEqual([-child.pid, 0])
    expect(groupAlive).toBe(false)
  })

  it("uses taskkill tree termination and verifies parent exit on Windows", async () => {
    class WindowsChild extends EventEmitter {
      pid = 123458
      exitCode: number | null = null
      signalCode: string | null = null
    }
    const child = new WindowsChild()
    await stopProcessTree(child, {
      platform: "win32",
      taskkill: async (pid: number, timeoutMs: number) => {
        expect(pid).toBe(child.pid)
        expect(timeoutMs).toBe(50)
        child.exitCode = 0
        child.emit("exit", 0, null)
      },
      timeoutMs: 50,
    })
    expect(child.exitCode).toBe(0)
  })

  it("rejects when injected Windows taskkill fails", async () => {
    class WindowsChild extends EventEmitter {
      pid = 123459
      exitCode: number | null = null
      signalCode: string | null = null
    }
    await expect(stopProcessTree(new WindowsChild(), {
      platform: "win32",
      taskkill: async () => { throw new Error("taskkill failed") },
      timeoutMs: 50,
    })).rejects.toThrow("taskkill failed")
  })
})

describe("onboarding UAT runner command assembly", () => {
  it("builds service and Playwright commands from ports, profile, and mock config", () => {
    const profile = {
      root: "/tmp/tldw-onboarding-uat-unit",
      configPath: "/tmp/tldw-onboarding-uat-unit/Config_Files/config.txt",
      envPath: "/tmp/tldw-onboarding-uat-unit/Config_Files/.env",
      usersDbPath: "/tmp/tldw-onboarding-uat-unit/Databases/users.db",
      databaseDir: "/tmp/tldw-onboarding-uat-unit/Databases",
      fixtureRoot: path.join(frontendRoot, "e2e/fixtures/media"),
    }
    const commands = buildCommands({
      repoRoot,
      frontendRoot,
      ports: { backend: 18110, web: 18111, mock: 18112 },
      profile,
      mockConfig: "hosted-success.json",
      scenario: "hosted-openai-first-chat",
      viewport: "desktop",
      baseEnv: { PATH: "/usr/bin", PYTHON: "/opt/project-python" },
    })

    expect(commands.mockOpenai).toMatchObject({
      name: "mock-openai",
      command: "/opt/project-python",
      cwd: mockOpenAiRoot,
    })
    expect(commands.mockOpenai.args).toEqual([
      "-m",
      "mock_openai.server",
      "--config",
      path.join(
        frontendRoot,
        "e2e/onboarding-uat/mock-openai/configs/hosted-success.json"
      ),
      "--host",
      "127.0.0.1",
      "--port",
      "18112",
    ])
    expect(commands.mockOpenai.env.PYTHONPATH).toContain(
      path.join(repoRoot, "mock_openai_server")
    )
    expect(
      existsSync(path.join(commands.mockOpenai.cwd, "responses/chat/default.json"))
    ).toBe(true)

    expect(commands.backend).toMatchObject({
      name: "backend",
      command: "/opt/project-python",
      cwd: repoRoot,
    })
    expect(commands.backend.args).toEqual([
      "-m",
      "uvicorn",
      "tldw_Server_API.app.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      "18110",
    ])
    expect(commands.backend.env.TLDW_CONFIG_FILE).toBe(profile.configPath)
    expect(commands.backend.env.OPENAI_API_KEY).toBe("sk-uat-mock-openai")
    expect(commands.backend.env.WORKFLOWS_EGRESS_BLOCK_PRIVATE).toBe("false")
    expect(commands.backend.env.WORKFLOWS_EGRESS_ALLOWED_PORTS).toBe(
      "80,443,18112"
    )

    expect(commands.authInit).toMatchObject({
      name: "auth-init",
      command: "/opt/project-python",
      cwd: repoRoot,
    })
    expect(commands.authInit.args).toEqual([
      "-m",
      "tldw_Server_API.app.core.AuthNZ.initialize",
      "--non-interactive",
    ])
    expect(commands.authInit.env.TLDW_ENV_FILE).toBe(profile.envPath)

    expect(commands.frontend).toMatchObject({
      name: "frontend",
      command: "bun",
      cwd: frontendRoot,
    })
    expect(commands.frontend.args).toEqual(["run", "dev", "--", "-p", "18111"])
    expect(commands.frontend.env.NEXT_PUBLIC_API_URL).toBe("http://127.0.0.1:18110")
    expect(commands.frontend.env.NEXT_PUBLIC_X_API_KEY).toBe(
      "THIS-IS-A-SECURE-KEY-123-UAT"
    )

    expect(commands.playwright).toMatchObject({
      name: "playwright",
      command: "bunx",
      cwd: frontendRoot,
    })
    expect(commands.playwright.args).toEqual([
      "playwright",
      "test",
      "-c",
      "e2e/onboarding-uat/playwright.config.ts",
      "--reporter=line",
      "--grep",
      "hosted-openai-first-chat",
      "--project",
      "uat-desktop",
    ])
    expect(commands.playwright.env.TLDW_ONBOARDING_UAT).toBe("1")
    expect(commands.playwright.env.TLDW_WEB_URL).toBe("http://localhost:18111")
    expect(commands.playwright.env.TLDW_SERVER_URL).toBe("http://127.0.0.1:18110")
    expect(commands.playwright.env.TLDW_MOCK_OPENAI_URL).toBe(
      "http://127.0.0.1:18112/v1"
    )
  })

  it("parses help and debug flags without requiring service startup", () => {
    expect(parseArgs(["--help"])).toMatchObject({ help: true })
    expect(
      parseArgs([
        "--scenario",
        "hosted-openai-first-chat",
        "--viewport",
        "mobile",
        "--mock-config",
        "chat-fail-once.json",
        "--preserve-runtime",
        "--reviewed-evidence",
      ])
    ).toMatchObject({
      scenario: "hosted-openai-first-chat",
      viewport: "mobile",
      mockConfig: "chat-fail-once.json",
      preserveRuntime: true,
      reviewedEvidence: true,
    })
    expect(formatUsage()).toContain("e2e:onboarding:uat")
    expect(formatUsage()).toContain("--scenario")
  })

  it("uses the registered scenario mock config when no override is provided", () => {
    for (const scenario of tierAScenarios) {
      expect(
        resolveEffectiveMockConfig({
          scenario: scenario.id,
          mockConfig: null,
        })
      ).toBe(scenario.mockConfig)
    }

    expect(
      resolveEffectiveMockConfig({
        scenario: "local-openai-manual-model-first-chat",
        mockConfig: "hosted-success.json",
      })
    ).toBe("hosted-success.json")
    expect(resolveEffectiveMockConfig({ scenario: null, mockConfig: null })).toBe(
      "hosted-success.json"
    )
  })

  it("returns help without allocating ports or starting services", async () => {
    const writes: string[] = []
    const originalWrite = process.stdout.write
    process.stdout.write = ((chunk: string | Uint8Array) => {
      writes.push(String(chunk))
      return true
    }) as typeof process.stdout.write

    try {
      await expect(
        runOnboardingUat({ options: parseArgs(["--help"]) })
      ).resolves.toEqual({ status: "help" })
    } finally {
      process.stdout.write = originalWrite
    }

    expect(writes.join("")).toContain("Usage: bun run e2e:onboarding:uat")
  })

  it("copies preserved artifacts into the reviewed evidence tree when requested", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-reviewed-evidence",
      preserve: false,
    })
    const evidenceRoot = path.join(
      repoRoot,
      "Docs/Product/WebUI/evidence/onboarding_uat/unit-reviewed-evidence"
    )

    try {
      writeFileSync(artifacts.summaryPath, "{\"status\":\"passed\"}\n", "utf8")

      expect(copyReviewedEvidence({ artifacts, repoRoot })).toBe(evidenceRoot)
      expect(existsSync(path.join(evidenceRoot, "summary.json"))).toBe(true)
      expect(readFileSync(path.join(evidenceRoot, "summary.json"), "utf8")).toContain(
        "\"status\":\"passed\""
      )
    } finally {
      cleanupRunArtifacts(artifacts)
      rmSync(evidenceRoot, { recursive: true, force: true })
    }
  })
})
