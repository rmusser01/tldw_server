import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")
const repoRoot = path.resolve(appDir, "..", "..")
const workflowsDir = path.join(repoRoot, ".github", "workflows")
const packageJsonPath = path.join(appDir, "package.json")

const readWorkflow = (fileName: string) =>
  readFileSync(path.join(workflowsDir, fileName), "utf8")

const getJobBlock = (workflow: string, jobId: string) => {
  const lines = workflow.split("\n")
  const startIndex = lines.findIndex((line) => line === `  ${jobId}:`)

  if (startIndex === -1) {
    throw new Error(`Unable to locate job "${jobId}" in workflow`)
  }

  const bodyLines: string[] = []

  for (let index = startIndex + 1; index < lines.length; index += 1) {
    const line = lines[index]

    if (/^ {2}[a-z0-9-]+:$/.test(line)) {
      break
    }

    bodyLines.push(line)
  }

  return bodyLines.join("\n")
}

describe("frontend CI workflow networking", () => {
  it("pins advanced-mode browser API settings for the frontend UX gates", () => {
    const workflow = readWorkflow("frontend-ux-gates.yml")

    for (const jobId of ["onboarding-gate", "smoke-gate"]) {
      const jobBlock = getJobBlock(workflow, jobId)

      expect(jobBlock).toContain("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: advanced")
      expect(jobBlock).toContain("NEXT_PUBLIC_API_URL: http://127.0.0.1:8000")
    }
  })

  it("forces the smoke gate to build the production frontend artifact explicitly", () => {
    const workflow = readWorkflow("frontend-ux-gates.yml")
    const jobBlock = getJobBlock(workflow, "smoke-gate")

    expect(jobBlock).toContain("run: bun run build:prod")
  })

  it("pins advanced-mode browser API settings for the frontend E2E tiers", () => {
    const workflow = readWorkflow("frontend-e2e-tiers.yml")

    for (const jobId of ["critical", "features", "admin"]) {
      const jobBlock = getJobBlock(workflow, jobId)

      expect(jobBlock).toContain("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: advanced")
      expect(jobBlock).toContain("NEXT_PUBLIC_API_URL: http://127.0.0.1:8000")
    }
  })

  it("keeps the focused /chat real-server cockpit proof in the UX gate rotation", () => {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
      scripts?: Record<string, string>
    }
    const script = packageJson.scripts?.["e2e:chat-cockpit:real:focused"] ?? ""

    expect(script).toContain("chat-cockpit.real-server.spec.ts")
    expect(script).toContain("uses the running server and keeps cockpit/focus controls working")
    expect(script).toContain("keeps mobile cockpit tabs and focus composer usable")
    expect(script).toContain("sends a real mobile focus conversation")
    expect(script).toContain("proves model provider confidence")
    expect(script).toContain("captures streaming stop and regenerate controls")
    expect(script).toContain("assert-playwright-no-skips.mjs")

    const workflow = readWorkflow("frontend-ux-gates.yml")
    const smokeGate = getJobBlock(workflow, "smoke-gate")

    expect(smokeGate).toContain("OPENAI_API_KEY: sk-mock-key-12345")
    expect(smokeGate).toContain("OPENAI_API_BASE_URL: http://127.0.0.1:18080/v1")
    expect(smokeGate).toContain("Start mock OpenAI server for /chat cockpit gate")
    expect(smokeGate).toContain("Run /chat real-server cockpit regression gate")
    expect(smokeGate).toContain("bun run e2e:chat-cockpit:real:focused")
    expect(smokeGate).toContain("Stop mock OpenAI server")
  })
})
