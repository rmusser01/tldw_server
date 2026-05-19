import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")
const repoRoot = path.resolve(appDir, "..", "..")
const workflowsDir = path.join(repoRoot, ".github", "workflows")

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
})
