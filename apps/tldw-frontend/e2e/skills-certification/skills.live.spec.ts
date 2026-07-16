import { expect, test } from "@playwright/test"

import {
  runSkillsLiveCertification,
  SKILLS_CERT_ARGUMENTS,
  SKILLS_CERT_RENDERED,
} from "../utils/skills-live-certification"
import { seedAuth } from "../utils/helpers"

const requireEnv = (name: string): string => {
  const value = process.env[name]?.trim()
  if (!value) throw new Error(`${name} is required for Skills WebUI certification`)
  return value
}

const serverUrl = (process.env.TLDW_SERVER_URL || process.env.TLDW_E2E_SERVER_URL)?.trim()
if (!serverUrl) {
  throw new Error("TLDW_SERVER_URL or TLDW_E2E_SERVER_URL is required for Skills WebUI certification")
}

const apiKey = requireEnv("TLDW_E2E_API_KEY")
const name = requireEnv("TLDW_SKILLS_CERT_SKILL_NAME")
const resultPath = requireEnv("TLDW_SKILLS_CERT_WEB_RESULT")

const boundedDiagnostics = (pageErrorCount: number, failedSkillsRequestCount: number): string =>
  `page_errors=${pageErrorCount}; failed_skills_requests=${failedSkillsRequestCount}`.slice(0, 200)

test("certifies the complete live Skills WebUI lifecycle", async ({ page }) => {
  const { writeSanitizedJson } = await import("../../scripts/skills-certification/evidence.mjs")
  writeSanitizedJson(resultPath, { status: "running" })

  let pageErrorCount = 0
  let failedSkillsRequestCount = 0
  let failed = false
  let resultWriteError: unknown

  page.on("pageerror", () => {
    pageErrorCount += 1
  })
  page.on("requestfailed", (request) => {
    try {
      if (new URL(request.url()).pathname.startsWith("/api/v1/skills")) {
        failedSkillsRequestCount += 1
      }
    } catch {
      // Request diagnostics intentionally retain no request URL or payload.
    }
  })

  try {
    await seedAuth(page, { serverUrl, apiKey, allowOffline: false })
    await page.goto("/skills", { waitUntil: "domcontentloaded" })
    await runSkillsLiveCertification({
      page,
      expect,
      initialExpectation: "empty-library-and-trash",
      name,
      arguments: SKILLS_CERT_ARGUMENTS,
      expectedRenderedPrompt: SKILLS_CERT_RENDERED,
      step: test.step,
    })

    if (pageErrorCount > 0 || failedSkillsRequestCount > 0) {
      throw new Error(boundedDiagnostics(pageErrorCount, failedSkillsRequestCount))
    }
  } catch (error) {
    failed = true
    throw error
  } finally {
    const status = failed ? "failed" : "passed"
    const result = {
      status,
      categories: status === "failed" ? ["webui_workflow"] : [],
      ...(status === "failed"
        ? { detail: boundedDiagnostics(pageErrorCount, failedSkillsRequestCount) }
        : {}),
    }

    try {
      writeSanitizedJson(resultPath, result)
    } catch (resultError) {
      if (!failed) resultWriteError = resultError
    }
  }

  if (resultWriteError) throw resultWriteError
})
