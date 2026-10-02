import { describe, expect, it } from "vitest"
import { buildCollectionCommand } from "../list-release-tests.mjs"

describe("release collection profile", () => {
  it("always lists with no app autostart and all import requirements supplied", () => {
    const command = buildCollectionCommand({ env: {} })
    expect(command.args).toEqual(expect.arrayContaining(["test", "--list", "--reporter=json", "--retries=0", "--workers=1"]))
    expect(command.env).toMatchObject({
      TLDW_WEB_AUTOSTART: "false",
      TLDW_SERVER_URL: "http://127.0.0.1:1",
      TLDW_MOCK_OPENAI_URL: "http://127.0.0.1:1/v1",
      TLDW_E2E_API_KEY: "collection-only-placeholder",
      TLDW_SKILLS_CERT_SKILL_NAME: "collection-only-skill",
    })
    expect(command.env.TLDW_SKILLS_CERT_WEB_RESULT).toContain("collection-only-unused.json")
  })
  it("does not inherit credentials, targets, auth mode, reporter output or Node preload options", () => {
    const command = buildCollectionCommand({ env: {
      PATH: "/local/bin", OPENAI_API_KEY: "private", TLDW_E2E_API_KEY: "private",
      TLDW_SERVER_URL: "https://real.example", TLDW_WEB_AUTOSTART: "true", AUTH_MODE: "multi_user",
      PLAYWRIGHT_JSON_OUTPUT_NAME: "/private/report.json", NODE_OPTIONS: "--require side-effects.cjs",
    } })
    expect(command.env.PATH).toBe("/local/bin")
    expect(JSON.stringify(command.env)).not.toMatch(/private|real\.example|side-effects/)
    expect(command.env).not.toHaveProperty("AUTH_MODE")
  })
  it("selects the workflow directory across projects without dropping root or browser variants", () => {
    const command = buildCollectionCommand({ workflows: true, env: {} })
    expect(command.args).toContain("e2e/workflows")
    expect(command.args.some(arg => arg.startsWith("--project"))).toBe(false)
  })
})

describe("collection command line", () => {
  it("rejects arguments that could change selection into execution", async () => {
    const { spawnSync } = await import("node:child_process")
    const command = buildCollectionCommand({ env: {} })
    const result = spawnSync(process.execPath, ["scripts/list-release-tests.mjs", "--workers=2"], {
      cwd: command.cwd, encoding: "utf8",
    })
    expect(result.status).toBe(2)
    expect(result.stderr).toContain("Usage:")
  })
})
