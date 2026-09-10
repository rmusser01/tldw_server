/* Live probe for #2922: open /admin/watchlists-items as an admin, pick alice
 * in the user selector, and assert her seeded feeds render in the read-only
 * oversight tables. Uses the API-key context (round-4 Part B pattern) since
 * hard navigation with a fresh JWT races the auth redirect. */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8001"
const ADMIN_KEY = process.env.ADMIN_KEY || ""
const OUT = process.env.OUT_DIR || "/tmp/oversight-probe"
if (!ADMIN_KEY) {
  console.error("Set ADMIN_KEY to an admin user's API key")
  process.exit(2)
}
await fs.mkdir(OUT, { recursive: true })

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
await ctx.addInitScript(({ serverUrl, apiKey }) => {
  localStorage.setItem(
    "tldwConfig",
    JSON.stringify({ serverUrl, authMode: "single-user", apiKey })
  )
  localStorage.setItem("isMigrated", "true")
}, { serverUrl: SERVER, apiKey: ADMIN_KEY })
const page = await ctx.newPage()
const fails = []
const check = (name, ok, detail) => {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`)
  if (!ok) fails.push(name)
}

await page.goto(WEB + "/admin/watchlists-items", { waitUntil: "domcontentloaded" })
// networkidle regularly times out on this SPA (long-lived connections keep the
// network busy); the timeout is a best-effort settle, not a pass/fail signal -
// the explicit checks below are what decide the probe's outcome.
try { await page.waitForLoadState("networkidle", { timeout: 12000 }) } catch {}
await page.waitForTimeout(2500)
await page.screenshot({ path: `${OUT}/1-initial.png` })
const initial = await page.evaluate(() => document.body.innerText)
check("oversight-title", /Watchlists Oversight/.test(initial))
check("user-selector", /Select User:/.test(initial))
check("hint-before-selection", /Select a user above to inspect/.test(initial))
check("no-personal-triage", !/Mark selected as reviewed|Create a watchlist/.test(initial))

await page.locator(".ant-select").first().click()
await page.waitForTimeout(600)
await page.locator(".ant-select-item-option", { hasText: "alice" }).first().click()
await page.waitForTimeout(3000)
await page.screenshot({ path: `${OUT}/2-alice.png` })
const after = await page.evaluate(() => document.body.innerText)
check("alice-feed-battery", /Battery Tech News/.test(after))
check("alice-feed-grid", /Grid Storage Policy/.test(after))
check("summary-stats", /Feeds[\s\S]*Collected items[\s\S]*Unread[\s\S]*Recent runs/.test(after))
const summaryVisible = await page.locator('[data-testid="oversight-summary"]').count()
check("summary-testid", summaryVisible === 1, `count=${summaryVisible}`)

await browser.close()
if (fails.length) {
  console.log(`RESULT: ${fails.length} failure(s): ${fails.join(", ")}`)
  process.exit(1)
}
console.log("RESULT: all checks passed")
