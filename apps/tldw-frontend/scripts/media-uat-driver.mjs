#!/usr/bin/env node
/*
 * /media UAT driver — live browser walkthrough of ViewMediaPage.
 * Seeds single-user auth, exercises search/filter/list/detail flows from
 * first-time + power-user perspectives, screenshots, emits observations JSON.
 * Config is env-overridable; the fake default matches the repo's e2e smoke fixture.
 */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const SHOTS = process.env.SHOTS_DIR || "/tmp/media-uat-shots"

const obs = { steps: [], consoleErrors: [], pageErrors: [], apiCalls: {}, mediaDetailCalls: [], presence: {} }
const note = (step, status, detail) => { obs.steps.push({ step, status, detail }); console.log(`[${status}] ${step}${detail ? " :: " + detail : ""}`) }

const TESTIDS = [
  "media-search-input", "media-search-submit", "media-search-clear",
  "filter-panel", "filter-panel-media-types", "media-results-list",
  "media-review-results-list", "pagination", "media-library-stats-panel",
  "media-ingest-jobs-panel", "first-ingest-tutorial", "content-viewer-empty",
  "media-metadata-bar", "media-detail-fetch-error", "media-intelligence-section",
]
async function visState(page, t) { const l = page.getByTestId(t).first(); try { if (!(await l.count())) return "absent"; return (await l.isVisible()) ? "visible" : "present-hidden" } catch { return "error" } }
async function shot(page, name) { const p = `${SHOTS}/${name}.png`; try { await page.screenshot({ path: p }); note(`screenshot ${name}`, "ok", p) } catch (e) { note(`screenshot ${name}`, "warn", e.message) } }

async function main() {
  await fs.mkdir(SHOTS, { recursive: true })
  const browser = await chromium.launch()
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  await ctx.addInitScript(({ s, k }) => {
    localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: s, authMode: "single-user", apiKey: k, accessToken: "" }))
    for (const [a, v] of Object.entries({ isMigrated: "true", serverUrl: s, tldwServerUrl: s, authMode: "single-user", apiKey: k, accessToken: "", __tldw_first_run_complete: "true", assistant_setup_dismissed: "true", __tldw_test_bypass: "true", __tldw_allow_offline: "true" })) localStorage.setItem(a, v)
  }, { s: SERVER, k: API_KEY })

  const page = await ctx.newPage()
  page.on("console", (m) => { if (m.type() === "error") obs.consoleErrors.push(m.text().slice(0, 300)) })
  page.on("pageerror", (e) => obs.pageErrors.push(String(e).slice(0, 300)))
  page.on("request", (r) => {
    if (r.method() !== "GET") return
    const u = r.url(); if (!u.includes("/api/v1/")) return
    const p = u.split("/api/v1")[1].split("?")[0]
    obs.apiCalls[p] = (obs.apiCalls[p] || 0) + 1
  })

  // ---- Step 1: load ----
  await page.goto(`${WEB}/media`, { waitUntil: "domcontentloaded", timeout: 60000 })
  await page.waitForTimeout(4000)
  await shot(page, "01-desktop-initial")
  for (const t of TESTIDS) obs.presence[t] = await visState(page, t)
  note("load + inventory", "ok", `${Object.values(obs.presence).filter(v => v === "visible").length}/${TESTIDS.length} visible`)
  obs.initialBody = (await page.locator("body").innerText().catch(() => "")).replace(/\n{2,}/g, "\n").slice(0, 1400)
  obs.passiveLoadDupes = Object.fromEntries(Object.entries(obs.apiCalls).filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]))
  note("passive-load dupes", "ok", JSON.stringify(obs.passiveLoadDupes))

  // ---- Step 2: search ----
  try {
    const si = page.getByTestId("media-search-input").first()
    if (await si.count()) {
      await si.click(); await si.fill("the"); await page.waitForTimeout(400)
      const sub = page.getByTestId("media-search-submit").first()
      if (await sub.count()) await sub.click(); else await page.keyboard.press("Enter")
      await page.waitForTimeout(2000); await shot(page, "02-search-results")
      note("search", "ok")
    } else note("search", "warn", "no media-search-input")
  } catch (e) { note("search", "warn", e.message) }

  // ---- Step 3: open an item -> detail/ContentViewer (count detail fetches) ----
  try {
    const before = { ...obs.apiCalls }
    // click the first result row/link
    const row = page.locator('[data-testid^="results-select-"], [data-testid="media-results-list"] a, [data-testid="media-review-results-list"] [role="button"]').first()
    let clicked = false
    if (await row.count()) { await row.click({ timeout: 5000 }).catch(() => {}); clicked = true }
    else {
      const anyItem = page.getByText(/research-workspace-uat|quick-ingest-sample|\.mp4|document/i).first()
      if (await anyItem.count()) { await anyItem.click({ timeout: 5000 }).catch(() => {}); clicked = true }
    }
    await page.waitForTimeout(2500)
    await shot(page, "03-item-detail")
    // how many media-detail fetches happened on this single open?
    const detailDelta = Object.entries(obs.apiCalls).filter(([p]) => /\/media\/\d+|\/media\/.*\/(metadata|analysis|intelligence)/.test(p)).map(([p, n]) => `${p}:${n - (before[p] || 0)}`)
    obs.mediaDetailCalls = detailDelta
    note("open item -> detail", clicked ? "ok" : "warn", "detailFetches=" + JSON.stringify(detailDelta))
    obs.detailBody = (await page.locator("body").innerText().catch(() => "")).replace(/\n{2,}/g, "\n").slice(0, 800)
  } catch (e) { note("open item", "warn", e.message) }

  // ---- Step 4: no-match search (empty-state ambiguity) ----
  try {
    const si = page.getByTestId("media-search-input").first()
    if (await si.count()) {
      await si.click(); await si.fill("zzzqqq-no-such-media-xyz-123"); await page.keyboard.press("Enter")
      await page.waitForTimeout(2000); await shot(page, "04-no-match")
      const body = await page.locator("body").innerText().catch(() => "")
      obs.noMatchText = (body.match(/no .{0,40}(found|results|match|items)[^\n]*/i) || ["?"])[0]
      note("no-match search", "ok", obs.noMatchText)
    }
  } catch (e) { note("no-match", "warn", e.message) }

  // ---- Step 5: keyboard tab order from search ----
  try {
    const si = page.getByTestId("media-search-input").first()
    if (await si.count()) {
      await si.focus(); const order = []
      for (let i = 0; i < 6; i++) { await page.keyboard.press("Tab"); order.push(await page.evaluate(() => { const e = document.activeElement; return e ? `${e.tagName.toLowerCase()}${e.getAttribute("data-testid") ? "#" + e.getAttribute("data-testid") : ""}${e.getAttribute("aria-label") ? "[" + e.getAttribute("aria-label").slice(0, 20) + "]" : ""}` : "none" })) }
      obs.tabOrder = order; note("tab order", "ok", order.join(" → "))
    }
  } catch (e) { note("tab order", "warn", e.message) }

  // ---- Step 6: mobile ----
  try { await page.setViewportSize({ width: 390, height: 844 }); await page.waitForTimeout(1500); await shot(page, "05-mobile"); note("mobile", "ok") } catch (e) { note("mobile", "warn", e.message) }

  await browser.close()
  obs.allLoadDupes = Object.fromEntries(Object.entries(obs.apiCalls).filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]))
  await fs.writeFile(`${SHOTS}/observations.json`, JSON.stringify(obs, null, 2))
  console.log("\n==== SUMMARY ====")
  console.log("console errors:", obs.consoleErrors.length, "| page errors:", obs.pageErrors.length)
  console.log("passive-load dupes:", JSON.stringify(obs.passiveLoadDupes))
  console.log("media detail fetches on one open:", JSON.stringify(obs.mediaDetailCalls))
}
main().catch((e) => { console.error("DRIVER CRASH:", e); process.exit(1) })
