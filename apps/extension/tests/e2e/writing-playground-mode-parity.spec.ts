import { test, expect, type Page } from "@playwright/test"
import path from "path"
import { grantHostPermission } from "./utils/permissions"
import { requireRealServerConfig, launchWithExtensionOrSkip } from "./utils/real-server"
import { waitForConnectionStore } from "./utils/connection"

const WRITING_VIEWPORT = { width: 1440, height: 900 }
const TIMED_FLOW_P95_TARGET_MS = 60000
const TIMED_FLOW_SAMPLE_PROMPTS = [
  "Write one concise sentence about morning rain.",
  "Write one concise sentence about city traffic at dusk.",
  "Write one concise sentence about a quiet library.",
  "Write one concise sentence about a late train platform.",
  "Write one concise sentence about a windy coastline."
]
const CONTROL_BASELINE_ENV = "TLDW_WRITING_CONTROL_BASELINE_COUNT"
const CONTROL_REDUCTION_TARGET_PERCENT = 35

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value.replace(/\/$/, "") : `http://${value}`

const fetchWritingCapabilities = async (serverUrl: string, apiKey: string) => {
  const res = await fetch(
    `${serverUrl}/api/v1/writing/capabilities?include_providers=false`,
    {
      headers: { "x-api-key": apiKey }
    }
  ).catch(() => null)
  if (!res || !res.ok) return null
  return await res.json().catch(() => null)
}

const waitForConnected = async (page: Page, label: string) => {
  await waitForConnectionStore(page, label)
  await page.evaluate(() => {
    const store = (window as any).__tldw_useConnectionStore
    try {
      store?.getState?.().markFirstRunComplete?.()
      store?.getState?.().checkOnce?.()
    } catch {
      // ignore connection triggers
    }
    window.dispatchEvent(new CustomEvent("tldw:check-connection"))
  })
  await page.waitForFunction(
    () => {
      const store = (window as any).__tldw_useConnectionStore
      const state = store?.getState?.().state
      return state?.isConnected === true && state?.phase === "connected"
    },
    undefined,
    { timeout: 20000 }
  )
}

const ensurePageVisible = async (page: Page) => {
  try {
    await page.bringToFront()
  } catch {
    // ignore bringToFront failures in headless contexts
  }
  try {
    await page.waitForFunction(
      () => document.visibilityState === "visible",
      undefined,
      { timeout: 5000 }
    )
  } catch {
    // ignore visibility polling failures
  }
}

const openWritingPlayground = async (page: Page, optionsUrl: string) => {
  await ensurePageVisible(page)
  await page.goto(optionsUrl + "#/writing-playground", {
    waitUntil: "domcontentloaded"
  })
  await page.waitForFunction(() => !!document.querySelector("#root"), undefined, {
    timeout: 10000
  })
  await page.evaluate(() => {
    const navigate = (window as any).__tldwNavigate
    if (typeof navigate === "function") {
      navigate("/writing-playground")
    }
  })
}

const createSession = async (page: Page, name: string) => {
  await page.getByRole("button", { name: /New session/i }).click()
  const modal = page.getByRole("dialog", { name: /New session/i })
  await expect(modal).toBeVisible()
  const input = modal.getByRole("textbox")
  await input.fill(name)
  const okButton = modal.getByRole("button", { name: /OK|Create/i })
  await okButton.click()
  await expect(modal).toBeHidden()
  const row = page.locator(".ant-list-item").filter({ hasText: name }).first()
  await expect(row).toBeVisible({ timeout: 15000 })
  await row.click()
}

const percentile = (values: number[], p: number): number => {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const rank = Math.ceil((p / 100) * sorted.length) - 1
  const index = Math.max(0, Math.min(sorted.length - 1, rank))
  return sorted[index]
}

const waitForGenerationCompletion = async (
  page: Page,
  expectedInitialLength: number,
  timeoutMs: number
) => {
  await page.waitForFunction(
    ({ expectedLength }) => {
      const root = document.querySelector(".writing-playground")
      if (!root) return false
      const generate = Array.from(root.querySelectorAll("button")).find((btn) =>
        /^Generate$/i.test((btn.textContent || "").trim())
      )
      const stopVisible = Array.from(root.querySelectorAll("button")).some((btn) =>
        /^Stop$/i.test((btn.textContent || "").trim())
      )
      const loading =
        Boolean(generate?.querySelector(".ant-btn-loading-icon")) ||
        Boolean(generate && /\bant-btn-loading\b/.test(generate.className))
      const textarea = root.querySelector("textarea")
      const currentLength = textarea?.value.length ?? 0
      const seenKey = "__writing_generation_seen__"
      const seenBefore = Boolean((window as any)[seenKey])
      const seenNow = seenBefore || loading || stopVisible || currentLength > expectedLength
      ;(window as any)[seenKey] = seenNow
      return seenNow && !loading && !stopVisible
    },
    { expectedLength: expectedInitialLength },
    { timeout: timeoutMs }
  )
}

const getConfiguredControlBaseline = (
  testContext: Parameters<typeof requireRealServerConfig>[0]
): number => {
  const raw = process.env[CONTROL_BASELINE_ENV]
  const parsed = raw ? Number.parseInt(raw, 10) : NaN
  if (!Number.isFinite(parsed) || parsed <= 0) {
    testContext.skip(
      true,
      `Set ${CONTROL_BASELINE_ENV} to a recorded pre-redesign above-fold control baseline.`
    )
  }
  return parsed
}

const setupModeParityPage = async (
  test: Parameters<typeof requireRealServerConfig>[0]
) => {
  const { serverUrl, apiKey } = requireRealServerConfig(test)
  const normalizedServerUrl = normalizeServerUrl(serverUrl)
  const caps = await fetchWritingCapabilities(normalizedServerUrl, apiKey)
  if (!caps?.server?.sessions) {
    test.skip(true, "Writing sessions not available on the configured server.")
  }

  const extPath = path.resolve("build/chrome-mv3")
  const { context, page, extensionId, optionsUrl } = await launchWithExtensionOrSkip(
    test,
    extPath,
    {
      seedConfig: {
        __tldw_first_run_complete: true,
        __tldw_allow_offline: true,
        tldwConfig: {
          serverUrl: normalizedServerUrl,
          authMode: "single-user",
          apiKey
        }
      }
    }
  )

  const origin = new URL(normalizedServerUrl).origin + "/*"
  const granted = await grantHostPermission(context, extensionId, origin)
  if (!granted) {
    test.skip(
      true,
      "Host permission not granted for tldw_server origin; allow it in chrome://extensions > tldw Assistant > Site access, then re-run"
    )
  }

  await page.setViewportSize(WRITING_VIEWPORT)
  await openWritingPlayground(page, optionsUrl)
  await waitForConnected(page, "writing-playground-mode-parity")
  await expect(
    page.getByRole("heading", { name: /Writing Playground/i })
  ).toBeVisible()

  return { context, page }
}

const createUniqueSessionForTest = async (page: Page, testNamePrefix: string) => {
  const unique = `${Date.now()}-${test.info().workerIndex}-${Math.random()
    .toString(36)
    .slice(2, 6)}`
  const sessionName = `${testNamePrefix} ${unique}`
  await createSession(page, sessionName)
}

test.describe("Writing Playground mode parity", () => {
  test("switches modes with keyboard support and preserves draft input", async () => {
    const { context, page } = await setupModeParityPage(test)

    try {
      await createUniqueSessionForTest(page, "E2E Writing Mode")

      await expect(page.getByTestId("writing-revision-action-bar")).toBeVisible()
      await expect(page.getByTestId("writing-revision-queue")).toBeVisible()
      await expect(page.getByTestId("writing-status-word-count")).toBeVisible()

      const modeSwitch = page.getByTestId("writing-workspace-mode-switch")
      const draftRadio = modeSwitch.getByRole("radio", { name: /^Draft$/i })
      const manageRadio = modeSwitch.getByRole("radio", { name: /^Manage$/i })

      await expect(modeSwitch).toBeVisible()
      await expect(draftRadio).toHaveAttribute("aria-checked", "true")
      await expect(manageRadio).toHaveAttribute("aria-checked", "false")
      await expect(page.getByTestId("writing-section-draft-editor")).toBeVisible()
      await expect(page.getByTestId("writing-section-manage-generation")).toBeHidden()

      const editor = page.getByPlaceholder(/Start writing your prompt/i)
      const draftText = "Draft text that must survive mode toggle"
      await editor.fill(draftText)

      await draftRadio.focus()
      await page.keyboard.press("ArrowRight")
      await expect(manageRadio).toBeFocused()
      await expect(manageRadio).toHaveAttribute("aria-checked", "true")
      await expect(page.getByTestId("writing-section-manage-generation")).toBeVisible()
      await expect(page.getByTestId("writing-mode-live-region")).toContainText(/Manage/i)
      await expect(page.getByTestId("writing-section-draft-editor")).toBeHidden()

      await page.keyboard.press("ArrowLeft")
      await expect(draftRadio).toBeFocused()
      await expect(draftRadio).toHaveAttribute("aria-checked", "true")
      await expect(page.getByTestId("writing-section-draft-editor")).toBeVisible()
      await expect(page.getByTestId("writing-mode-live-region")).toContainText(/Draft/i)
      await expect(editor).toHaveValue(draftText)
    } finally {
      await context.close()
    }
  })

  test("timed first generation path <= 60s (timed first generation)", async () => {
    const { context, page } = await setupModeParityPage(test)
    try {
      await createUniqueSessionForTest(page, "E2E Writing Timed Flow")
      const editor = page.getByPlaceholder(/Start writing your prompt/i)
      const generateButton = page.getByRole("button", { name: /^Generate$/i })
      await expect(generateButton).toBeVisible()
      if (await generateButton.isDisabled()) {
        test.skip(
          true,
          "Generate is disabled in this environment (likely missing model/provider configuration)."
        )
      }

      const elapsedSamplesMs: number[] = []
      for (const prompt of TIMED_FLOW_SAMPLE_PROMPTS) {
        await editor.fill(prompt)
        await page.evaluate(() => {
          delete (window as any).__writing_generation_seen__
        })
        const startedAt = Date.now()
        await generateButton.click()
        await waitForGenerationCompletion(page, prompt.length, TIMED_FLOW_P95_TARGET_MS)
        elapsedSamplesMs.push(Date.now() - startedAt)
      }

      expect(elapsedSamplesMs).toHaveLength(TIMED_FLOW_SAMPLE_PROMPTS.length)
      const p95Ms = percentile(elapsedSamplesMs, 95)
      expect(p95Ms).toBeLessThanOrEqual(TIMED_FLOW_P95_TARGET_MS)
    } finally {
      await context.close()
    }
  })

  test("control density delta >= 35% in draft vs manage (control density delta)", async () => {
    const { context, page } = await setupModeParityPage(test)
    try {
      await createUniqueSessionForTest(page, "E2E Writing Control Density")
      const preRedesignBaseline = getConfiguredControlBaseline(test)

      const countVisibleControlsAboveFold = async () =>
        page.evaluate(() => {
          const root = document.querySelector(".writing-playground")
          if (!root) return 0
          const viewportHeight = window.innerHeight
          const selectors = [
            "button",
            "input",
            "textarea",
            "select",
            "[role='button']",
            "[role='switch']",
            "[role='combobox']",
            "[role='checkbox']",
            "[role='radio']",
            "[role='slider']",
            "[role='spinbutton']"
          ].join(", ")
          const canonicalSelector = [
            ".ant-select",
            ".ant-input-number",
            ".ant-switch",
            ".ant-btn",
            "[role='combobox']",
            "[role='switch']",
            "[role='checkbox']",
            "[role='radio']",
            "[role='slider']",
            "[role='spinbutton']",
            "button",
            "input",
            "textarea",
            "select"
          ].join(", ")
          const elements = Array.from(root.querySelectorAll<HTMLElement>(selectors))
          const seen = new Set<HTMLElement>()
          let count = 0
          for (const element of elements) {
            const canonical =
              (element.closest(canonicalSelector) as HTMLElement | null) ?? element
            if (seen.has(canonical)) continue
            seen.add(canonical)
            if (canonical.closest("[aria-hidden='true'], [hidden], [inert]")) continue
            if (canonical.getAttribute("aria-disabled") === "true") continue
            const style = window.getComputedStyle(canonical)
            if (style.visibility === "hidden" || style.display === "none") continue
            if ((element as HTMLInputElement).disabled) continue
            const rect = canonical.getBoundingClientRect()
            const visible =
              rect.width > 0 &&
              rect.height > 0 &&
              rect.top < viewportHeight &&
              rect.bottom > 0
            if (visible) count += 1
          }
          return count
        })

      await page.getByTestId("writing-mode-draft").click()
      const draftCount = await countVisibleControlsAboveFold()
      await page.getByTestId("writing-mode-manage").click()
      await expect(page.getByTestId("writing-section-manage-generation")).toBeVisible()
      const manageCount = await countVisibleControlsAboveFold()

      expect(manageCount).toBeGreaterThan(0)
      expect(manageCount).toBeGreaterThan(draftCount)
      const reductionPercent =
        ((preRedesignBaseline - draftCount) / preRedesignBaseline) * 100
      expect(reductionPercent).toBeGreaterThanOrEqual(CONTROL_REDUCTION_TARGET_PERCENT)
    } finally {
      await context.close()
    }
  })
})
