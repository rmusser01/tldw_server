import { describe, expect, it } from "vitest"
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs"
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

const testDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(testDir, "../..")
const repoRoot = path.resolve(frontendRoot, "../..")
const mockOpenAiRoot = path.join(frontendRoot, "e2e/onboarding-uat/mock-openai")

const configFiles = [
  "hosted-success.json",
  "local-success.json",
  "chat-fail-once.json",
  "model-unavailable.json",
]

const responseFiles = [
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
  "BEGIN PRIVATE KEY",
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
  it("provides mock OpenAI config files with the expected scenario shapes", () => {
    const configs = Object.fromEntries(
      configFiles.map((file) => [file, readJson(`configs/${file}`)])
    )

    expect(modelIds(configs["hosted-success.json"])).toEqual(
      expect.arrayContaining(["gpt-4.1-mini", "text-embedding-3-small"])
    )

    expect(modelIds(configs["local-success.json"])).toEqual(
      expect.arrayContaining(["llama3.2:3b", "local-uat-chat"])
    )

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
        failure.match?.model === "missing-uat-model" &&
        failure.status_code === 404 &&
        failure.code === "model_not_found"
    )

    expect(modelIds(unavailable)).not.toContain("missing-uat-model")
    expect(failsSelectedChatModel).toBe(true)
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
    const defaultChat = readJson("responses/chat/default.json")
    const sourceSummary = readJson("responses/chat/source-summary.json")
    const embeddings = readJson("responses/embeddings/default.json")

    expect(defaultChat.choices?.[0]?.message?.content).toContain(
      "onboarding UAT ready"
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

  it("detects unredacted synthetic secret leaks in artifact files", () => {
    const artifacts = createRunArtifacts({
      frontendRoot,
      runId: "unit-leak-check",
      preserve: false,
    })

    try {
      writeFileSync(artifacts.logs.runner, "leaked sk-uat-mock-openai", "utf8")

      expect(() => assertNoSecretLeaks(artifacts.root)).toThrow(
        /synthetic secret leak/i
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
      expect(config).not.toContain("ingestion_source_allowed_roots =")

      expect(envText).toContain("AUTH_MODE=single_user")
      expect(envText).toContain("SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT")
      expect(envText).toContain("DEFAULT_LLM_PROVIDER=openai")
      expect(envText).toContain("OPENAI_API_KEY=sk-uat-mock-openai")
      expect(envText).toContain("OPENAI_API_BASE_URL=http://127.0.0.1:43210/v1")
      expect(envText).toContain(`DATABASE_URL=sqlite:///${profile.usersDbPath}`)
      expect(envText).not.toContain(path.join(repoRoot, "tldw_Server_API/Config_Files/.env"))

      const backendEnv = buildBackendEnv({
        profile,
        mockPort: 43210,
        baseEnv: { PATH: "/test/bin", OPENAI_API_KEY: "real-key" },
      })

      expect(backendEnv.PATH).toBe("/test/bin")
      expect(backendEnv.TLDW_CONFIG_FILE).toBe(profile.configPath)
      expect(backendEnv.TLDW_ENV_FILE).toBe(profile.envPath)
      expect(backendEnv.DATABASE_URL).toBe(`sqlite:///${profile.usersDbPath}`)
      expect(backendEnv.AUTH_MODE).toBe("single_user")
      expect(backendEnv.SINGLE_USER_API_KEY).toBe("THIS-IS-A-SECURE-KEY-123-UAT")
      expect(backendEnv.DEFAULT_LLM_PROVIDER).toBe("openai")
      expect(backendEnv.OPENAI_API_KEY).toBe("sk-uat-mock-openai")
      expect(backendEnv.OPENAI_API_BASE_URL).toBe("http://127.0.0.1:43210/v1")
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
})
