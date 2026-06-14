#!/usr/bin/env node
/*
 * /media-multi UAT driver — live browser walkthrough of MediaReviewPage (bulk review).
 * Seeds single-user auth, exercises selection + batch-toolbar flows from first-time
 * and power-user perspectives, screenshots, emits observations JSON.
 * Config is env-overridable; the fake default matches the repo's e2e smoke fixture.
 */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const SHOTS = process.env.SHOTS_DIR || "/tmp/media-multi-uat-shots"

const obs = { steps: [], consoleErrors: [], pageErrors: [], apiCalls: {}, presence: {} }
const note = (s, st, d) => { obs.steps.push({ step: s, status: st, detail: d }); console.log(`[${st}] ${s}${d ? " :: " + d : ""}`) }
const TESTIDS = [
  "media-review-results-list", "media-review-results-header", "media-review-status-bar",
  "media-bulk-toolbar", "media-bulk-select-all", "media-bulk-clear", "media-bulk-tag",
  "media-bulk-delete", "media-bulk-export", "media-bulk-export-format",
  "media-bulk-add-collection", "media-bulk-open-multi", "media-multi-selection-status",
  "view-selected-items-button", "selected-items-drawer", "first-ingest-tutorial",
]
async function vis(page, t) { const l = page.getByTestId(t).first(); try { if (!(await l.count())) return "absent"; return (await l.isVisible()) ? "visible" : "present-hidden" } catch { return "error" } }
async function shot(page, name) { const p = `${SHOTS}/${name}.png`; try { await page.screenshot({ path: p }); note(`screenshot ${name}`, "ok", p) } catch (e) { note(`screenshot ${name}`, "warn", e.message) } }

async function main() {
  try { await fs.mkdir(SHOTS, { recursive: true }) } catch (e) { console.warn(`mkdir ${SHOTS}: ${e.message}`) }
  const browser = await chromium.launch()
  try {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
    await ctx.addInitScript(({ s, k }) => {
      localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: s, authMode: "single-user", apiKey: k, accessToken: "" }))
      for (const [a, v] of Object.entries({ isMigrated: "true", serverUrl: s, tldwServerUrl: s, authMode: "single-user", apiKey: k, accessToken: "", __tldw_first_run_complete: "true", assistant_setup_dismissed: "true", __tldw_test_bypass: "true", __tldw_allow_offline: "true" })) localStorage.setItem(a, v)
    }, { s: SERVER, k: API_KEY })
    const page = await ctx.newPage()
    page.on("console", (m) => { if (m.type() === "error") obs.consoleErrors.push(m.text().slice(0, 200)) })
    page.on("pageerror", (e) => obs.pageErrors.push(String(e).slice(0, 200)))
    page.on("request", (r) => { if (r.method() !== "GET") return; const u = r.url(); if (!u.includes("/api/v1/")) return; const p = u.split("/api/v1")[1].split("?")[0]; obs.apiCalls[p] = (obs.apiCalls[p] || 0) + 1 })

    // ---- Step 1: load ----
    await page.goto(`${WEB}/media-multi`, { waitUntil: "domcontentloaded", timeout: 60000 })
    try { await page.getByTestId("media-review-results-list").first().waitFor({ state: "visible", timeout: 30000 }); note("load /media-multi", "ok") }
    catch (e) { note("load /media-multi", "FAIL", e.message) }
    await page.waitForTimeout(3000)
    await shot(page, "01-initial")
    for (const t of TESTIDS) obs.presence[t] = await vis(page, t)
    note("inventory", "ok", `${Object.values(obs.presence).filter(v => v === "visible").length}/${TESTIDS.length} visible`)
    obs.passiveLoadDupes = Object.fromEntries(Object.entries(obs.apiCalls).filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]))
    note("passive-load duplicate GETs", "ok", JSON.stringify(obs.passiveLoadDupes))

    // ---- Step 2: select items (rows) ----
    try {
      const rows = page.locator('[data-testid^="results-select-"]')
      const n = await rows.count()
      if (n > 0) {
        await rows.nth(0).click({ timeout: 4000 }).catch(() => {})
        if (n > 1) await rows.nth(1).click({ timeout: 4000 }).catch(() => {})
        if (n > 2) await rows.nth(2).click({ timeout: 4000 }).catch(() => {})
        await page.waitForTimeout(800)
        await shot(page, "02-items-selected")
        const status = await page.getByTestId("media-multi-selection-status").first().innerText().catch(() => "")
        note("select items", "ok", `rows=${n} status="${status.slice(0, 60)}"`)
      } else {
        // fall back to bulk select-all
        const sa = page.getByTestId("media-bulk-select-all").first()
        if (await sa.count()) { await sa.click().catch(() => {}); await page.waitForTimeout(800); await shot(page, "02-select-all"); note("select-all", "ok") }
        else note("select items", "warn", "no result rows or select-all")
      }
    } catch (e) { note("select items", "warn", e.message) }

    // ---- Step 3: bulk toolbar presence after selection ----
    for (const t of ["media-bulk-toolbar", "media-bulk-tag", "media-bulk-delete", "media-bulk-export", "media-bulk-add-collection"]) obs.presence[t + "@selected"] = await vis(page, t)
    note("bulk toolbar after selection", "ok", JSON.stringify(Object.fromEntries(["media-bulk-toolbar", "media-bulk-tag", "media-bulk-delete", "media-bulk-export"].map(t => [t, obs.presence[t + "@selected"]]))))
    await shot(page, "03-bulk-toolbar")

    // ---- Step 4: selected-items drawer + Escape ----
    try {
      const vsb = page.getByTestId("view-selected-items-button").first()
      if (await vsb.count() && await vsb.isVisible()) {
        await vsb.click({ timeout: 4000 }); await page.waitForTimeout(700)
        const drawerOpen = await vis(page, "selected-items-drawer")
        await shot(page, "04-selected-drawer")
        // Escape should close it (CommandPalette fix merged)
        await page.keyboard.press("Escape"); await page.waitForTimeout(600)
        const afterEsc = await vis(page, "selected-items-drawer")
        note("selected-items drawer + Escape", "ok", `open=${drawerOpen} afterEsc=${afterEsc}`)
      } else note("selected-items drawer", "warn", "view-selected-items-button absent/hidden")
    } catch (e) { note("selected drawer", "warn", e.message) }

    // ---- Step 5: export format dropdown ----
    try {
      const ef = page.getByTestId("media-bulk-export-format").first()
      if (await ef.count() && await ef.isVisible()) { await ef.click({ timeout: 4000 }).catch(() => {}); await page.waitForTimeout(500); await shot(page, "05-export-format") ; note("export format control", "ok") ; await page.keyboard.press("Escape").catch(() => {}) }
      else note("export format", "warn", "absent/hidden")
    } catch (e) { note("export format", "warn", e.message) }

    // ---- Step 6: keyboard tab order from results ----
    try {
      const rl = page.getByTestId("media-review-results-list").first()
      await rl.click({ position: { x: 10, y: 10 } }).catch(() => {})
      const order = []
      for (let i = 0; i < 6; i++) { await page.keyboard.press("Tab"); order.push(await page.evaluate(() => { const e = document.activeElement; return e ? `${e.tagName.toLowerCase()}${e.getAttribute("data-testid") ? "#" + e.getAttribute("data-testid") : ""}${e.getAttribute("aria-label") ? "[" + (e.getAttribute("aria-label") || "").slice(0, 22) + "]" : ""}` : "none" })) }
      obs.tabOrder = order; note("tab order", "ok", order.join(" → "))
    } catch (e) { note("tab order", "warn", e.message) }

    // ---- Step 7: mobile ----
    try {
      await page.setViewportSize({ width: 390, height: 844 }); await page.waitForTimeout(1500); await shot(page, "06-mobile")
      obs.mobile = { resultsList: await vis(page, "media-review-results-list"), mobileBadge: await vis(page, "mobile-view-mode-badge"), statusBar: await vis(page, "media-review-status-bar") }
      note("mobile", "ok", JSON.stringify(obs.mobile))
    } catch (e) { note("mobile", "warn", e.message) }

    obs.allLoadDupes = Object.fromEntries(Object.entries(obs.apiCalls).filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]))
  } finally {
    await browser.close().catch(() => {})
  }
  await fs.writeFile(`${SHOTS}/observations.json`, JSON.stringify(obs, null, 2)).catch((e) => console.warn(`observations.json: ${e.message}`))
  console.log("\n==== SUMMARY ====")
  console.log("console errors:", obs.consoleErrors.length, "| page errors:", obs.pageErrors.length)
  console.log("passive-load dupes:", JSON.stringify(obs.passiveLoadDupes))
}
main().catch((e) => { console.error("DRIVER CRASH:", e); process.exit(1) })
