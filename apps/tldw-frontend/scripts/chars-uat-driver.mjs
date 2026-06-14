#!/usr/bin/env node
/*
 * /characters UAT driver — live browser walkthrough.
 * Seeds single-user auth (mirrors e2e/smoke/smoke.setup.ts), exercises core flows
 * from first-time + power-user perspectives, screenshots, emits observations JSON.
 * Config is env-overridable; the fake default matches the repo's e2e smoke fixture.
 */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const SHOTS = process.env.SHOTS_DIR || "/tmp/chars-uat-shots"

const obs = { steps: [], consoleErrors: [], pageErrors: [], apiCalls: {}, presence: {} }
const note = (step, status, detail) => {
  obs.steps.push({ step, status, detail })
  console.log(`[${status}] ${step}${detail ? " :: " + detail : ""}`)
}

const TESTIDS = [
  "characters-page", "characters-new-button", "characters-search-input",
  "characters-view-mode-segmented", "characters-scope-segmented",
  "characters-table-view", "characters-gallery-view",
  "active-filter-chips", "generate-character-panel", "character-import-dropzone",
]

async function visState(page, testid) {
  const loc = page.getByTestId(testid).first()
  try {
    if (!(await loc.count())) return "absent"
    return (await loc.isVisible()) ? "visible" : "present-hidden"
  } catch { return "error" }
}
async function shot(page, name) {
  const p = `${SHOTS}/${name}.png`
  try { await page.screenshot({ path: p }); note(`screenshot ${name}`, "ok", p) }
  catch (e) { note(`screenshot ${name}`, "warn", e.message) }
}

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

  // ---- Step 1: load + initial gallery ----
  await page.goto(`${WEB}/characters`, { waitUntil: "domcontentloaded", timeout: 60000 })
  try {
    await page.getByTestId("characters-page").first().waitFor({ state: "visible", timeout: 30000 })
    note("load /characters", "ok")
  } catch (e) { note("load /characters", "FAIL", e.message) }
  await page.waitForTimeout(3000)
  await shot(page, "01-desktop-initial")

  for (const t of TESTIDS) obs.presence[t] = await visState(page, t)
  note("inventory", "ok", `${Object.values(obs.presence).filter(v => v === "visible").length}/${TESTIDS.length} visible`)
  obs.initialBody = (await page.locator("body").innerText().catch(() => "")).replace(/\n{2,}/g, "\n").slice(0, 1200)

  // ---- Step 2: search ----
  try {
    const s = page.getByTestId("characters-search-input").first()
    if (await s.count()) {
      await s.click(); await s.fill("uat-test-char-03"); await page.waitForTimeout(1200)
      await shot(page, "02-search")
      const cnt = await page.locator("body").innerText()
      note("search filters list", "ok", /uat-test-char-03/.test(cnt) ? "match shown" : "no visible match")
      await s.fill(""); await page.waitForTimeout(600)
    } else note("search", "warn", "no search input")
  } catch (e) { note("search", "warn", e.message) }

  // ---- Step 3: view-mode toggle (table/gallery) ----
  try {
    const seg = page.getByTestId("characters-view-mode-segmented").first()
    if (await seg.count()) {
      const opts = seg.locator("label, button, [role=radio]")
      const n = await opts.count()
      if (n > 1) { await opts.nth(1).click({ timeout: 4000 }); await page.waitForTimeout(800); await shot(page, "03-view-toggled") }
      note("view-mode toggle", "ok", `${n} options`)
    } else note("view-mode toggle", "warn", "absent")
  } catch (e) { note("view-mode toggle", "warn", e.message) }

  // ---- Step 4: scope (active/deleted) ----
  try {
    const sc = page.getByTestId("characters-scope-segmented").first()
    if (await sc.count()) {
      const opts = sc.locator("label, button, [role=radio]")
      if (await opts.count() > 1) { await opts.nth(1).click({ timeout: 4000 }); await page.waitForTimeout(1000); await shot(page, "04-scope-deleted") ; await opts.nth(0).click().catch(()=>{}) }
      note("scope toggle", "ok")
    } else note("scope toggle", "warn", "absent")
  } catch (e) { note("scope toggle", "warn", e.message) }

  // ---- Step 5: new character (open editor) ----
  try {
    const nb = page.getByTestId("characters-new-button").first()
    if (await nb.count()) {
      await nb.click({ timeout: 5000 }); await page.waitForTimeout(1200)
      await shot(page, "05-new-character-editor")
      const body = await page.locator("body").innerText()
      note("new character opens editor", "ok", /Character name|name|greeting|system prompt/i.test(body) ? "form fields visible" : "?")
      // try Escape to close (the /chat Escape-swallow lens)
      const before = body.length
      await page.keyboard.press("Escape"); await page.waitForTimeout(600)
      await shot(page, "06-after-escape")
    } else note("new character", "warn", "absent")
  } catch (e) { note("new character", "warn", e.message) }

  // ---- Step 6: keyboard focus order from search ----
  try {
    const s = page.getByTestId("characters-search-input").first()
    if (await s.count()) {
      await s.focus()
      const order = []
      for (let i = 0; i < 6; i++) {
        await page.keyboard.press("Tab")
        order.push(await page.evaluate(() => { const el = document.activeElement; return el ? `${el.tagName.toLowerCase()}${el.getAttribute("data-testid") ? "#" + el.getAttribute("data-testid") : ""}${el.getAttribute("aria-label") ? "[" + el.getAttribute("aria-label").slice(0, 24) + "]" : ""}` : "none" }))
      }
      obs.tabOrder = order
      note("tab order from search", "ok", order.join(" → "))
    }
  } catch (e) { note("tab order", "warn", e.message) }

  // ---- Step 7: mobile ----
  try {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.waitForTimeout(1500)
    await shot(page, "07-mobile")
    note("mobile snapshot", "ok")
  } catch (e) { note("mobile", "warn", e.message) }

  await browser.close()
  obs.apiDuplicates = Object.fromEntries(Object.entries(obs.apiCalls).filter(([, n]) => n > 1).sort((a, b) => b[1] - a[1]))
  await fs.writeFile(`${SHOTS}/observations.json`, JSON.stringify(obs, null, 2))
  console.log("\n==== SUMMARY ====")
  console.log("console errors:", obs.consoleErrors.length, "| page errors:", obs.pageErrors.length)
  console.log("duplicate GET endpoints:", JSON.stringify(obs.apiDuplicates))
}
main().catch((e) => { console.error("DRIVER CRASH:", e); process.exit(1) })
