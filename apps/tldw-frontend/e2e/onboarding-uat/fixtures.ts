import { test as base, expect, type Page, type TestInfo } from "@playwright/test"
import { mkdirSync, writeFileSync } from "node:fs"
import path from "node:path"

export interface DiagnosticsData {
  console: Array<{
    type: string
    text: string
    location?: { url: string; lineNumber: number }
  }>
  pageErrors: Array<{ message: string; stack: string }>
  requestFailures: Array<{ url: string; errorText: string }>
}

export type StepArtifact = {
  scenarioId: string
  stepName: string
  screenshotPath: string
  jsonPath: string
}

export interface OnboardingArtifact {
  root: string
  screenshotsDir: string
  stepsDir: string
  diagnosticsPath: string
  writeJson: (relativePath: string, payload: unknown) => string
}

export interface OnboardingUatFixtures {
  diagnostics: DiagnosticsData
  artifact: OnboardingArtifact
  firstRunPage: Page
}

type OptionalStorageArea = {
  clear?: () => unknown
}

type BrowserStorageShim = {
  storage?: {
    sync?: OptionalStorageArea
    local?: OptionalStorageArea
  }
}

const safeSegment = (value: string): string =>
  value.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "step"

const artifactRootFor = (testInfo: TestInfo): string => {
  const root = process.env.TLDW_ONBOARDING_UAT_ARTIFACT_ROOT
  if (root && root.trim()) {
    return root
  }
  return path.join(testInfo.outputDir, "onboarding-uat-artifacts")
}

const createArtifact = (testInfo: TestInfo): OnboardingArtifact => {
  const root = artifactRootFor(testInfo)
  const screenshotsDir = path.join(root, "screenshots")
  const stepsDir = path.join(root, "steps")
  const diagnosticsPath = path.join(root, "browser", "console-and-network.json")
  mkdirSync(screenshotsDir, { recursive: true })
  mkdirSync(stepsDir, { recursive: true })
  mkdirSync(path.dirname(diagnosticsPath), { recursive: true })

  return {
    root,
    screenshotsDir,
    stepsDir,
    diagnosticsPath,
    writeJson(relativePath, payload) {
      const outPath = path.join(root, relativePath)
      mkdirSync(path.dirname(outPath), { recursive: true })
      writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8")
      return outPath
    },
  }
}

const installCleanFirstRunState = async (page: Page): Promise<void> => {
  await page.context().clearCookies()
  await page.addInitScript(() => {
    try {
      localStorage.clear()
    } catch {}
    try {
      sessionStorage.clear()
    } catch {}
    try {
      const chromeLike = (window as unknown as { chrome?: BrowserStorageShim }).chrome
      chromeLike?.storage?.sync?.clear?.()
      chromeLike?.storage?.local?.clear?.()
    } catch {}
  })
}

export const test = base.extend<OnboardingUatFixtures>({
  diagnostics: async ({ page, artifact }, use, testInfo) => {
    const data: DiagnosticsData = {
      console: [],
      pageErrors: [],
      requestFailures: [],
    }

    page.on("console", (msg) => {
      const location = msg.location()
      data.console.push({
        type: msg.type(),
        text: msg.text(),
        location: location.url
          ? { url: location.url, lineNumber: location.lineNumber }
          : undefined,
      })
    })

    page.on("pageerror", (error) => {
      data.pageErrors.push({
        message: error.message,
        stack: error.stack || "",
      })
    })

    page.on("requestfailed", (request) => {
      data.requestFailures.push({
        url: request.url(),
        errorText: request.failure()?.errorText || "",
      })
    })

    await use(data)

    const diagnosticsJson = JSON.stringify(data, null, 2)
    writeFileSync(artifact.diagnosticsPath, `${diagnosticsJson}\n`, "utf8")

    if (data.console.length || data.pageErrors.length || data.requestFailures.length) {
      await testInfo.attach("onboarding-uat-diagnostics.json", {
        body: diagnosticsJson,
        contentType: "application/json",
      })
    }
  },

  artifact: async ({}, use, testInfo) => {
    await use(createArtifact(testInfo))
  },

  firstRunPage: async ({ page }, use) => {
    const webUrl = process.env.TLDW_WEB_URL || "http://localhost:18111"
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], {
      origin: new URL(webUrl).origin,
    })
    await installCleanFirstRunState(page)
    await use(page)
  },
})

export { expect, safeSegment }
