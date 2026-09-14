import { expect, test, type Request, type Response } from "@playwright/test"

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

type SanitizedJsonWriter = (filePath: string, value: unknown) => unknown

const boundedDiagnostics = (
  pageErrorCount: number,
  failedSkillsRequestCount: number,
  skillsHttpErrorCount: number
): string =>
  [
    `page_errors=${pageErrorCount}`,
    `failed_skills_requests=${failedSkillsRequestCount}`,
    `skills_http_errors=${skillsHttpErrorCount}`,
  ].join("; ").slice(0, 200)

const isSkillsRequestUrl = (url: string): boolean => {
  try {
    return new URL(url).pathname.startsWith("/api/v1/skills")
  } catch {
    return false
  }
}

test("certifies the complete live Skills WebUI lifecycle", async ({ page }) => {
  let pageErrorCount = 0
  let failedSkillsRequestCount = 0
  let skillsHttpErrorCount = 0
  let writeSanitizedJson: SanitizedJsonWriter | undefined
  let hasOriginalError = false
  let originalError: unknown
  let resultWriteError: unknown

  const captureOriginalError = (error: unknown): void => {
    originalError = hasOriginalError
      ? new AggregateError([originalError, error], "Skills WebUI workflow failed")
      : error
    hasOriginalError = true
  }
  const onPageError = (): void => {
    pageErrorCount += 1
  }
  const onRequestFailed = (request: Request): void => {
    if (isSkillsRequestUrl(request.url())) {
      failedSkillsRequestCount += 1
    }
  }
  const onResponse = (response: Response): void => {
    if (response.status() >= 400 && isSkillsRequestUrl(response.url())) {
      skillsHttpErrorCount += 1
    }
  }

  try {
    const evidence = await import("../../scripts/skills-certification/evidence.mjs")
    writeSanitizedJson = evidence.writeSanitizedJson
    try {
      writeSanitizedJson(resultPath, { status: "running" })
    } catch (error) {
      captureOriginalError(error)
    }

    page.on("pageerror", onPageError)
    page.on("requestfailed", onRequestFailed)
    page.on("response", onResponse)

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

      if (pageErrorCount > 0 || failedSkillsRequestCount > 0 || skillsHttpErrorCount > 0) {
        throw new Error(
          boundedDiagnostics(pageErrorCount, failedSkillsRequestCount, skillsHttpErrorCount)
        )
      }
    } catch (error) {
      captureOriginalError(error)
    }
  } catch (error) {
    captureOriginalError(error)
  } finally {
    page.off("pageerror", onPageError)
    page.off("requestfailed", onRequestFailed)
    page.off("response", onResponse)

    if (writeSanitizedJson) {
      const status = hasOriginalError ? "failed" : "passed"
      const result = {
        status,
        categories: status === "failed" ? ["webui_workflow"] : [],
        ...(status === "failed"
          ? {
              detail: boundedDiagnostics(
                pageErrorCount,
                failedSkillsRequestCount,
                skillsHttpErrorCount
              ),
            }
          : {}),
      }

      try {
        writeSanitizedJson(resultPath, result)
      } catch (error) {
        resultWriteError = error
      }
    }
  }

  if (hasOriginalError && resultWriteError) {
    throw new AggregateError([originalError, resultWriteError], "Skills WebUI certification failed")
  }
  if (hasOriginalError) throw originalError
  if (resultWriteError) throw resultWriteError
})
