import AxeBuilder from "@axe-core/playwright"
import { expect, test, type Page } from "@playwright/test"

/**
 * a11y smoke for the chat-composer variants. Runs axe-core scoped to
 * the composer wrapper for each of V1, V3, V5 on the Sidepanel
 * surface (deterministic narrow viewport, no chat history noise).
 *
 * We only fail on critical violations — serious/moderate issues are
 * surfaced as warnings in the test output but don't block the build,
 * since some come from app-wide patterns outside the composer's
 * control. Critical violations within the composer wrapper would
 * indicate a real regression in the variant or its primitives.
 */

const bypassOnboarding = async (page: Page) => {
  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("assistant_setup_dismissed", "true")
    } catch {
      /* ignore */
    }
  })
}

const setVariant = (variant: "v1" | "v3" | "v5") => async (page: Page) => {
  await page.addInitScript((v: string) => {
    try {
      window.localStorage.setItem("tldw:composerVariant", v)
    } catch {
      /* ignore */
    }
  }, variant)
}

const mockComposerProfile = async (page: Page) => {
  await page.route(/\/api\/v1\/users\/me\/profile.*/, async (route) => {
    const method = route.request().method()
    if (method === "GET") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          profile_version: "2026-04-20T00:00:00Z",
          preferences: {},
        }),
      })
      return
    }
    if (method === "PATCH") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ applied: [], skipped: [] }),
      })
      return
    }
    await route.continue()
  })
}

for (const variant of ["v1", "v3", "v5"] as const) {
  test(`composer a11y · sidepanel ${variant} has no critical axe violations`, async ({
    page,
  }) => {
    test.setTimeout(90_000)
    await bypassOnboarding(page)
    await setVariant(variant)(page)
    await mockComposerProfile(page)
    await page.setViewportSize({ width: 480, height: 1000 })

    await page.goto("/__debug__/sidepanel-chat?nextgenComposer=1")
    await page
      .waitForLoadState("networkidle", { timeout: 30_000 })
      .catch(() => {})

    const wrapper = page.locator('[data-testid="nextgen-composer-wrapper"]')
    await expect(wrapper).toBeVisible({ timeout: 30_000 })
    await expect(wrapper.locator(`[data-variant='${variant}']`)).toBeVisible()

    const results = await new AxeBuilder({ page })
      .include('[data-testid="nextgen-composer-wrapper"]')
      .analyze()

    const critical = results.violations.filter((v) => v.impact === "critical")
    if (critical.length > 0) {
      const summary = critical
        .map(
          (v) =>
            `  • [${v.id}] ${v.help} — ${v.nodes.length} node(s)\n      ${v.helpUrl}`
        )
        .join("\n")
      throw new Error(
        `Critical axe violations in ${variant} composer:\n${summary}`
      )
    }

    // Surface non-critical issues as test annotations for visibility.
    const nonCritical = results.violations.filter(
      (v) => v.impact !== "critical"
    )
    if (nonCritical.length > 0) {
      test.info().annotations.push({
        type: "axe-non-critical",
        description: `${variant}: ${nonCritical.length} ${nonCritical.length === 1 ? "issue" : "issues"} (${nonCritical.map((v) => v.id).join(", ")})`,
      })
    }
  })
}
