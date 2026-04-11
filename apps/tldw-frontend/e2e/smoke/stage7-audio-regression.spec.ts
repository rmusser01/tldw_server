import {
  test,
  expect,
  seedAuth,
  getCriticalIssues,
  classifySmokeIssues,
  SMOKE_LOAD_TIMEOUT
} from "./smoke.setup"
import {
  dismissConnectionModals,
  getVisibleAntdSelectDropdown,
  getAntdSelectTrigger,
  waitForAppShell
} from "../utils/helpers"
import type { Page, Route } from "@playwright/test"

const LOAD_TIMEOUT = SMOKE_LOAD_TIMEOUT
const UNRESOLVED_TEMPLATE_PATTERN = /\{\{[^{}\n]{1,120}\}\}/g

type AudioRoute = {
  path: string
  name: string
  expectedPath?: string
}

const AUDIO_ROUTES: AudioRoute[] = [
  { path: "/tts", name: "TTS" },
  { path: "/stt", name: "STT" },
  { path: "/speech", name: "Speech" },
  { path: "/audio", name: "Audio Alias", expectedPath: "/speech" }
]

const ELEVENLABS_CORS_HEADERS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, OPTIONS",
  "access-control-allow-headers": "*"
}

const fulfillJson = async (route: Route, status: number, data: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(data)
  })
}

const stubDocsInfo = async (page: Page) => {
  await page.route("**/api/v1/config/docs-info", async (route) => {
    await fulfillJson(route, 200, {
      info: { version: "e2e" },
      capabilities: {}
    })
  })
}

async function openSpeechInputSourcePicker(page: Page) {
  const inputSourcePicker = getAntdSelectTrigger(page, {
    ariaLabel: "Speech playground input source"
  })
  const inputSourcePickerVisible = await inputSourcePicker.isVisible().catch(() => false)
  if (!inputSourcePickerVisible) {
    const roundTripMode = page.getByRole("radio", { name: /^Round-trip$/i }).first()
    if (await roundTripMode.isVisible().catch(() => false)) {
      await roundTripMode.check({ force: true })
    }
  }
  await expect(inputSourcePicker).toBeVisible({ timeout: LOAD_TIMEOUT })
  await inputSourcePicker.click({ force: true })
  const dropdown = getVisibleAntdSelectDropdown(page)
  await dropdown.waitFor({ state: "visible", timeout: LOAD_TIMEOUT })
  return dropdown
}

test.describe("Stage 7 audio regression gate", () => {
  test("audio routes enforce console/error/template budgets", async ({
    page,
    diagnostics
  }) => {
    await seedAuth(page)

    for (const route of AUDIO_ROUTES) {
      diagnostics.console.length = 0
      diagnostics.pageErrors.length = 0
      diagnostics.requestFailures.length = 0

      const response = await page.goto(route.path, {
        waitUntil: "domcontentloaded",
        timeout: LOAD_TIMEOUT
      })

      const status = response?.status() ?? 0
      expect(status, `Expected ${route.path} to return HTTP 2xx/3xx`).toBeGreaterThanOrEqual(
        200
      )
      expect(status, `Expected ${route.path} to return HTTP 2xx/3xx`).toBeLessThan(400)

      const expectedPath = route.expectedPath || route.path
      await page.waitForURL((url) => url.pathname === expectedPath, {
        timeout: LOAD_TIMEOUT
      })
      await waitForAppShell(page, LOAD_TIMEOUT)

      const issues = getCriticalIssues(diagnostics)
      const classified = classifySmokeIssues(route.path, issues)

      expect(
        issues.pageErrors,
        `Uncaught page errors on ${route.path}: ${issues.pageErrors
          .map((entry) => entry.message)
          .join(" | ")}`
      ).toHaveLength(0)
      expect(
        classified.unexpectedConsoleErrors,
        `Unexpected console errors on ${route.path}: ${classified.unexpectedConsoleErrors
          .map((entry) => entry.text)
          .join(" | ")}`
      ).toHaveLength(0)
      expect(
        classified.unexpectedRequestFailures,
        `Unexpected request failures on ${route.path}: ${classified.unexpectedRequestFailures
          .map((entry) => `${entry.url} (${entry.errorText})`)
          .join(" | ")}`
      ).toHaveLength(0)

      const bodyText = await page.evaluate(() => document.body?.innerText || "")
      const unresolvedTemplates = Array.from(bodyText.matchAll(UNRESOLVED_TEMPLATE_PATTERN)).map(
        (match) => match[0]
      )
      const uniqueUnresolvedTemplates = Array.from(new Set(unresolvedTemplates))

      expect(
        uniqueUnresolvedTemplates,
        `Unresolved template placeholders on ${route.path}: ${uniqueUnresolvedTemplates.join(
          " | "
        )}`
      ).toHaveLength(0)
    }
  })

  test("tts ElevenLabs timeout state shows retry and recovers", async ({ page }) => {
    await seedAuth(page)
    await stubDocsInfo(page)
    await page.addInitScript(() => {
      try {
        localStorage.setItem("ttsProvider", "elevenlabs")
        localStorage.setItem("elevenLabsApiKey", "elevenlabs-e2e-key")
      } catch {}
    })

    let shouldFailMetadata = true
    let voicesGetHits = 0
    let modelsGetHits = 0

    await page.route("https://api.elevenlabs.io/v1/voices**", async (route) => {
      const method = route.request().method()
      if (method !== "GET") {
        await route.fulfill({ status: 204, headers: ELEVENLABS_CORS_HEADERS })
        return
      }
      voicesGetHits += 1
      if (shouldFailMetadata) {
        await route.abort("timedout")
        return
      }
      await route.fulfill({
        status: 200,
        headers: ELEVENLABS_CORS_HEADERS,
        contentType: "application/json",
        body: JSON.stringify({
          voices: [{ voice_id: "voice-1", name: "Voice One" }]
        })
      })
    })

    await page.route("https://api.elevenlabs.io/v1/models**", async (route) => {
      const method = route.request().method()
      if (method !== "GET") {
        await route.fulfill({ status: 204, headers: ELEVENLABS_CORS_HEADERS })
        return
      }
      modelsGetHits += 1
      if (shouldFailMetadata) {
        await route.abort("timedout")
        return
      }
      await route.fulfill({
        status: 200,
        headers: ELEVENLABS_CORS_HEADERS,
        contentType: "application/json",
        body: JSON.stringify([{ model_id: "model-1", name: "Model One" }])
      })
    })

    await page.goto("/tts", { waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })

    const timeoutAlert = page.locator(".ant-alert").filter({
      hasText: /ElevenLabs voices unavailable/i
    })
    await expect(timeoutAlert).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(
      timeoutAlert.getByText(/Loading voices\/models took longer than 10 seconds/i)
    ).toBeVisible({ timeout: LOAD_TIMEOUT })

    shouldFailMetadata = false
    await dismissConnectionModals(page)
    await timeoutAlert.getByRole("button", { name: /^Retry$/i }).click()

    await expect.poll(() => voicesGetHits).toBeGreaterThanOrEqual(2)
    await expect.poll(() => modelsGetHits).toBeGreaterThanOrEqual(2)
    await expect(timeoutAlert).toHaveCount(0)
  })

  test("speech ElevenLabs timeout state shows retry and recovers", async ({ page }) => {
    await seedAuth(page)
    await stubDocsInfo(page)
    await page.addInitScript(() => {
      try {
        localStorage.setItem("ttsProvider", "elevenlabs")
        localStorage.setItem("elevenLabsApiKey", "elevenlabs-e2e-key")
        localStorage.setItem("speechPlaygroundMode", "listen")
      } catch {}
    })

    let shouldFailMetadata = true
    let voicesGetHits = 0
    let modelsGetHits = 0

    await page.route("https://api.elevenlabs.io/v1/voices**", async (route) => {
      const method = route.request().method()
      if (method !== "GET") {
        await route.fulfill({ status: 204, headers: ELEVENLABS_CORS_HEADERS })
        return
      }
      voicesGetHits += 1
      if (shouldFailMetadata) {
        await route.abort("timedout")
        return
      }
      await route.fulfill({
        status: 200,
        headers: ELEVENLABS_CORS_HEADERS,
        contentType: "application/json",
        body: JSON.stringify({
          voices: [{ voice_id: "voice-1", name: "Voice One" }]
        })
      })
    })

    await page.route("https://api.elevenlabs.io/v1/models**", async (route) => {
      const method = route.request().method()
      if (method !== "GET") {
        await route.fulfill({ status: 204, headers: ELEVENLABS_CORS_HEADERS })
        return
      }
      modelsGetHits += 1
      if (shouldFailMetadata) {
        await route.abort("timedout")
        return
      }
      await route.fulfill({
        status: 200,
        headers: ELEVENLABS_CORS_HEADERS,
        contentType: "application/json",
        body: JSON.stringify([{ model_id: "model-1", name: "Model One" }])
      })
    })

    await page.goto("/speech", { waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })

    const timeoutAlert = page.locator(".ant-alert").filter({
      hasText: /ElevenLabs voices unavailable/i
    })
    await expect(timeoutAlert).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(
      timeoutAlert.getByText(/Loading voices\/models took longer than 10 seconds/i)
    ).toBeVisible({ timeout: LOAD_TIMEOUT })

    shouldFailMetadata = false
    await dismissConnectionModals(page)
    await timeoutAlert.getByRole("button", { name: /^Retry$/i }).click()

    await expect.poll(() => voicesGetHits).toBeGreaterThanOrEqual(2)
    await expect.poll(() => modelsGetHits).toBeGreaterThanOrEqual(2)
    await expect(timeoutAlert).toHaveCount(0)
  })

  test("speech exposes the dedicated mic input source picker", async ({ page }) => {
    await seedAuth(page)
    await stubDocsInfo(page)
    await page.goto("/speech", { waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })
    await waitForAppShell(page, LOAD_TIMEOUT)
    await dismissConnectionModals(page)

    const dropdown = await openSpeechInputSourcePicker(page)
    await expect(
      dropdown.locator(".ant-select-item-option-content").filter({ hasText: /Default microphone/i })
    ).toBeVisible()
    await expect(
      dropdown.locator(".ant-select-item-option-content").filter({ hasText: /Tab audio/i })
    ).toHaveCount(0)
    await expect(
      dropdown.locator(".ant-select-item-option-content").filter({ hasText: /System audio/i })
    ).toHaveCount(0)
  })

  test("stt transcription-model timeout state shows retry and recovers", async ({
    page
  }) => {
    await seedAuth(page)

    let shouldFailModels = true
    let modelCalls = 0
    await page.route("**/api/v1/media/transcription-models**", async (route) => {
      modelCalls += 1
      if (shouldFailModels) {
        await fulfillJson(route, 504, {
          detail: "timeout while loading transcription models"
        })
        return
      }
      await fulfillJson(route, 200, {
        all_models: ["whisper-1", "parakeet-tdt", "canary"]
      })
    })

    await page.goto("/stt", { waitUntil: "domcontentloaded", timeout: LOAD_TIMEOUT })

    await expect
      .poll(() => modelCalls, {
        message:
          "Expected /api/v1/media/transcription-models interceptor to be hit at least once for /stt"
      })
      .toBeGreaterThan(0)

    const retryButton = page.getByRole("button", { name: /retry/i }).first()
    await expect(retryButton).toBeVisible({ timeout: LOAD_TIMEOUT })

    const callsBeforeRetry = modelCalls
    shouldFailModels = false
    await retryButton.click()

    await expect.poll(() => modelCalls).toBeGreaterThan(callsBeforeRetry)
    await expect(retryButton).toHaveCount(0)
  })
})
