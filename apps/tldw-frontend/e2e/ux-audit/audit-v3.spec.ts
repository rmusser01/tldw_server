/**
 * UX Audit v3 - Data Collection Script
 *
 * Captures 4 screenshot variants (desktop/mobile x light/dark) and diagnostics
 * for all 92 routes in the tldw-frontend application.
 */
import * as fs from "fs"
import * as path from "path"
import { PAGES, type PageEntry } from "../smoke/page-inventory"
import {
  test,
  seedAuth,
  getCriticalIssues,
  classifySmokeIssues,
  type DiagnosticsData,
} from "../smoke/smoke.setup"
import { dismissModals, waitForVisualSettle } from "../utils/helpers"

// ─── Extra routes not in the page-inventory ────────────────────────────────
const EXTRA_ROUTES: PageEntry[] = [
  { path: "/settings/guardian", name: "Guardian Settings", category: "settings" },
  { path: "/settings/splash", name: "Splash Settings", category: "settings" },
  { path: "/research-workspace", name: "Research Workspace", category: "workspace" },
  { path: "/document-workspace", name: "Document Workspace", category: "workspace" },
  { path: "/workflow-editor", name: "Workflow Editor", category: "workspace" },
  { path: "/audiobook-studio", name: "Audiobook Studio", category: "audio" },
  { path: "/model-playground", name: "Model Playground", category: "workspace" },
  { path: "/writing-playground", name: "Writing Playground", category: "workspace" },
  { path: "/acp-playground", name: "ACP Playground", category: "workspace" },
  { path: "/skills", name: "Skills", category: "workspace" },
  { path: "/chatbooks-playground", name: "Chatbooks Playground", category: "workspace" },
]

// Merge and deduplicate by path
const existingPaths = new Set(PAGES.map((p) => p.path))
const ALL_ROUTES: PageEntry[] = [
  ...PAGES,
  ...EXTRA_ROUTES.filter((r) => !existingPaths.has(r.path)),
]

// ─── Output directories ────────────────────────────────────────────────────
const AUDIT_ROOT = path.resolve(__dirname, "../../ux-audit-v3")
const SCREENSHOTS_DIR = path.join(AUDIT_ROOT, "screenshots")
const DATA_DIR = path.join(AUDIT_ROOT, "data")

// Ensure directories exist
fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true })
fs.mkdirSync(DATA_DIR, { recursive: true })

// ─── Helpers ────────────────────────────────────────────────────────────────

function slugify(routePath: string): string {
  return routePath
    .replace(/^\//, "")
    .replace(/\//g, "-")
    .replace(/[^a-zA-Z0-9_-]/g, "_") || "root"
}

async function waitForAuditRenderableSurface(
  page: import("@playwright/test").Page,
  timeoutMs = 10_000
) {
  await page
    .waitForFunction(() => {
      const redirectPanel = document.querySelector('[data-testid="route-redirect-panel"]')
      const placeholderPanel = document.querySelector('[data-testid="route-placeholder-panel"]')
      const mainLandmark = document.querySelector("main, [role='main']")
      const heading = document.querySelector("h1, h2, h3, h4, h5, h6")
      const focusableCount = document.querySelectorAll(
        "a[href], button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
      ).length
      const visibleText = (document.body?.innerText || "").trim().length

      return Boolean(
        redirectPanel ||
          placeholderPanel ||
          heading ||
          mainLandmark ||
          focusableCount > 0 ||
          visibleText > 32
      )
    }, { timeout: timeoutMs })
    .catch(() => {})
}

// ─── Test suite ─────────────────────────────────────────────────────────────

test.describe("UX Audit v3 - Data Collection", () => {
  test.describe.configure({ mode: "serial" })

  for (const route of ALL_ROUTES) {
    const slug = slugify(route.path)

    test(`[${route.category}] ${route.name} (${route.path})`, async ({ page, diagnostics }) => {
      test.setTimeout(120_000)

      const routeData: Record<string, unknown> = {
        path: route.path,
        name: route.name,
        category: route.category,
        slug,
        timestamp: new Date().toISOString(),
        navigationError: null,
        finalUrl: null,
        redirected: false,
        httpStatus: null,
        screenshots: {} as Record<string, string>,
        diagnostics: {} as Record<string, unknown>,
        performance: null,
        accessibility: null,
      }

      // 1. Seed auth
      await seedAuth(page)

      // 2. Set desktop viewport
      await page.setViewportSize({ width: 1440, height: 900 })

      // 3. Navigate with error catching
      let navigationOk = true
      try {
        const response = await page.goto(route.path, {
          waitUntil: "domcontentloaded",
          timeout: 30_000,
        })
        routeData.httpStatus = response?.status() ?? null
        routeData.finalUrl = page.url()
        routeData.redirected = page.url() !== (page.context().pages()[0]
          ? new URL(route.path, page.url()).href
          : route.path)
      } catch (err: unknown) {
        navigationOk = false
        routeData.navigationError = err instanceof Error ? err.message : String(err)
      }

      // 4. Wait for the route to paint enough UI for stable diagnostics/screenshots.
      if (navigationOk) {
        await waitForVisualSettle(page, 15_000)
        await waitForAuditRenderableSurface(page, 10_000)
        routeData.finalUrl = page.url()
        routeData.redirected = page.url() !== (page.context().pages()[0]
          ? new URL(route.path, page.url()).href
          : route.path)
      }

      // 5. Dismiss modals
      try {
        await dismissModals(page)
        // Also dismiss antd notifications
        const notifications = page.locator(".ant-notification-notice-close")
        const notifCount = await notifications.count()
        for (let i = 0; i < notifCount; i++) {
          const btn = notifications.nth(i)
          if (await btn.isVisible()) {
            await btn.click().catch(() => {})
          }
        }
      } catch {
        // Non-critical
      }

      // 6. Collect diagnostics
      const criticalIssues = getCriticalIssues(diagnostics)
      const classified = classifySmokeIssues(route.path, criticalIssues)
      routeData.diagnostics = {
        pageErrors: classified.pageErrors,
        unexpectedConsoleErrors: classified.unexpectedConsoleErrors,
        allowlistedConsoleErrors: classified.allowlistedConsoleErrors.map((e) => ({
          text: e.entry.text,
          ruleId: e.rule.id,
        })),
        unexpectedRequestFailures: classified.unexpectedRequestFailures,
        allowlistedRequestFailures: classified.allowlistedRequestFailures.map((e) => ({
          url: e.entry.url,
          ruleId: e.rule.id,
        })),
        totalConsoleMessages: diagnostics.console.length,
        consoleWarnings: diagnostics.console.filter((c) => c.type === "warning").length,
      }

      // 7. Collect performance metrics
      try {
        routeData.performance = await page.evaluate(() => {
          const nav = performance.getEntriesByType("navigation")[0] as PerformanceNavigationTiming | undefined
          const paint = performance.getEntriesByType("paint")
          const fcp = paint.find((e) => e.name === "first-contentful-paint")
          return {
            domContentLoaded: nav?.domContentLoadedEventEnd ?? null,
            loadComplete: nav?.loadEventEnd ?? null,
            firstContentfulPaint: fcp?.startTime ?? null,
            resourceCount: performance.getEntriesByType("resource").length,
          }
        })
      } catch {
        routeData.performance = null
      }

      // 8. Collect accessibility quick-check
      try {
        routeData.accessibility = await page.evaluate(() => {
          const landmarks = document.querySelectorAll(
            "main, nav, header, footer, aside, [role='main'], [role='navigation'], [role='banner'], [role='contentinfo'], [role='complementary']"
          )
          const headings = document.querySelectorAll("h1, h2, h3, h4, h5, h6")
          const headingLevels = Array.from(headings).map((h) => parseInt(h.tagName[1]))
          const imagesWithoutAlt = document.querySelectorAll("img:not([alt])")
          const focusable = document.querySelectorAll(
            "a[href], button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
          )
          // Check heading hierarchy
          let headingOrderValid = true
          for (let i = 1; i < headingLevels.length; i++) {
            if (headingLevels[i] > headingLevels[i - 1] + 1) {
              headingOrderValid = false
              break
            }
          }
          return {
            landmarkCount: landmarks.length,
            landmarkTypes: Array.from(new Set(
              Array.from(landmarks).map((l) => l.getAttribute("role") || l.tagName.toLowerCase())
            )),
            headingCount: headings.length,
            headingLevels,
            headingOrderValid,
            imagesWithoutAlt: imagesWithoutAlt.length,
            focusableElements: focusable.length,
          }
        })
      } catch {
        routeData.accessibility = null
      }

      // 9. Take 4 screenshots
      const screenshotPaths: Record<string, string> = {}

      // Desktop Light
      try {
        await page.evaluate(() => {
          document.documentElement.classList.remove("dark")
          document.documentElement.classList.add("light")
        })
        await waitForVisualSettle(page, 5_000)
        const desktopLightPath = path.join(SCREENSHOTS_DIR, `${slug}_desktop_light.png`)
        await page.screenshot({ path: desktopLightPath, fullPage: true })
        screenshotPaths.desktop_light = `${slug}_desktop_light.png`
      } catch (err) {
        screenshotPaths.desktop_light = `ERROR: ${err instanceof Error ? err.message : String(err)}`
      }

      // Desktop Dark
      try {
        await page.evaluate(() => {
          document.documentElement.classList.remove("light")
          document.documentElement.classList.add("dark")
        })
        await waitForVisualSettle(page, 5_000)
        const desktopDarkPath = path.join(SCREENSHOTS_DIR, `${slug}_desktop_dark.png`)
        await page.screenshot({ path: desktopDarkPath, fullPage: true })
        screenshotPaths.desktop_dark = `${slug}_desktop_dark.png`
      } catch (err) {
        screenshotPaths.desktop_dark = `ERROR: ${err instanceof Error ? err.message : String(err)}`
      }

      // Mobile Dark (375x812)
      try {
        await page.setViewportSize({ width: 375, height: 812 })
        await waitForVisualSettle(page, 5_000)
        const mobileDarkPath = path.join(SCREENSHOTS_DIR, `${slug}_mobile_dark.png`)
        await page.screenshot({ path: mobileDarkPath, fullPage: true })
        screenshotPaths.mobile_dark = `${slug}_mobile_dark.png`
      } catch (err) {
        screenshotPaths.mobile_dark = `ERROR: ${err instanceof Error ? err.message : String(err)}`
      }

      // Mobile Light
      try {
        await page.evaluate(() => {
          document.documentElement.classList.remove("dark")
          document.documentElement.classList.add("light")
        })
        await waitForVisualSettle(page, 5_000)
        const mobileLightPath = path.join(SCREENSHOTS_DIR, `${slug}_mobile_light.png`)
        await page.screenshot({ path: mobileLightPath, fullPage: true })
        screenshotPaths.mobile_light = `${slug}_mobile_light.png`
      } catch (err) {
        screenshotPaths.mobile_light = `ERROR: ${err instanceof Error ? err.message : String(err)}`
      }

      routeData.screenshots = screenshotPaths

      // 10. Write JSON data file
      const jsonPath = path.join(DATA_DIR, `${slug}.json`)
      fs.writeFileSync(jsonPath, JSON.stringify(routeData, null, 2))
    })
  }
})
