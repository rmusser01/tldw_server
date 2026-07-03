#!/usr/bin/env node
/*
 * Chat page UAT driver — live browser walkthrough of /chat.
 * Seeds single-user auth (mirrors e2e/smoke/smoke.setup.ts), drives core flows
 * from first-time + power-user perspectives, screenshots, and emits observations JSON.
 */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const SHOTS = process.env.SHOTS_DIR || "/tmp/chat-uat-shots"
const SEND_LLM = process.env.SEND_LLM !== "0" // actually send 1 real chat message

const obs = { steps: [], consoleErrors: [], pageErrors: [], requestFailures: [], presence: {} }
const note = (step, status, detail) => {
  obs.steps.push({ step, status, detail })
  console.log(`[${status}] ${step}${detail ? " :: " + detail : ""}`)
}

const TESTIDS = [
  "playground-cockpit-shell","playground-cockpit-left-rail","playground-cockpit-right-rail",
  "playground-cockpit-main","playground-cockpit-status-strip","playground-context-rail",
  "playground-empty-shell","playground-empty-mode-deck","playground-chat-shell",
  "chat-input","nextgen-composer-wrapper","composer-inline-send-control","composer-context-strip",
  "attachment-button","dictation-button","tools-button","voice-chat-button",
  "composer-advanced-toggle","composer-options-toggle","composer-formatting-guide-toggle",
  "model-selector","model-list-scope-toggle","model-recommendations-panel",
  "mcp-tools-toggle","knowledge-search-toggle","web-search-toggle","open-quick-ingest",
  "playground-shortcuts-help-trigger","playground-artifacts-trigger",
  "playground-active-chat-mode","playground-cockpit-mode-summary",
  "role-play-setup-drawer","character-chat-readiness-panel","startup-template-controls",
  "session-insights-panel","playground-runtime-inspector","playground-chat-error-banner",
  "playground-composer-disconnected-notice","playground-composer-degraded-notice",
]

async function visState(page, testid) {
  const loc = page.getByTestId(testid).first()
  try {
    const count = await loc.count()
    if (!count) return "absent"
    return (await loc.isVisible()) ? "visible" : "present-hidden"
  } catch { return "error" }
}

async function shot(page, name) {
  const p = `${SHOTS}/${name}.png`
  try { await page.screenshot({ path: p, fullPage: false }); note(`screenshot ${name}`, "ok", p) }
  catch (e) { note(`screenshot ${name}`, "warn", e.message) }
}

async function main() {
  await fs.mkdir(SHOTS, { recursive: true })
  const browser = await chromium.launch()
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })

  await context.addInitScript(({ serverUrl, apiKey }) => {
    const cfg = { serverUrl, authMode: "single-user", apiKey, accessToken: "" }
    // lgtm[js/clear-text-storage-of-sensitive-data]: synthetic UAT browser auth seed only.
    localStorage.setItem("tldwConfig", JSON.stringify(cfg))
    localStorage.setItem("isMigrated", "true")
    localStorage.setItem("serverUrl", serverUrl)
    localStorage.setItem("tldwServerUrl", serverUrl)
    localStorage.setItem("authMode", "single-user")
    // lgtm[js/clear-text-storage-of-sensitive-data]: test-only legacy auth compatibility key.
    localStorage.setItem("apiKey", apiKey)
    localStorage.setItem("accessToken", "")
    localStorage.setItem("__tldw_first_run_complete", "true")
    localStorage.setItem("assistant_setup_dismissed", "true")
    localStorage.setItem("__tldw_test_bypass", "true")
    localStorage.setItem("__tldw_allow_offline", "true")
  }, { serverUrl: SERVER, apiKey: API_KEY })

  const page = await context.newPage()
  page.on("console", (m) => { if (m.type() === "error") obs.consoleErrors.push(m.text().slice(0, 300)) })
  page.on("pageerror", (e) => obs.pageErrors.push(String(e).slice(0, 300)))
  page.on("requestfailed", (r) => {
    const u = r.url()
    if (u.includes("/api/")) obs.requestFailures.push(`${r.failure()?.errorText || "fail"} ${u.slice(0, 120)}`)
  })

  // ---- Step 1: first load / empty state ----
  await page.goto(`${WEB}/chat`, { waitUntil: "domcontentloaded", timeout: 60000 })
  try {
    await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30000 })
    note("load /chat — composer visible", "ok")
  } catch (e) {
    note("load /chat — composer visible", "FAIL", e.message)
  }
  await page.waitForTimeout(2500) // let async panels settle
  await shot(page, "01-desktop-empty")

  // ---- Step 2: inventory presence map ----
  for (const t of TESTIDS) obs.presence[t] = await visState(page, t)
  note("inventory presence map", "ok", `${Object.values(obs.presence).filter(v=>v==="visible").length} visible / ${TESTIDS.length}`)

  // ---- Step 3: empty-state heading / first-time affordances ----
  try {
    const bodyText = (await page.locator("body").innerText()).slice(0, 4000)
    obs.emptyStateText = bodyText.replace(/\n{2,}/g, "\n").slice(0, 1500)
    note("captured empty-state copy", "ok")
  } catch (e) { note("empty-state copy", "warn", e.message) }

  // ---- Step 4: model selector ----
  try {
    const ms = page.getByTestId("model-selector").first()
    if (await ms.count()) {
      await ms.click({ timeout: 5000 })
      await page.waitForTimeout(800)
      await shot(page, "02-model-selector-open")
      note("model selector opens", "ok")
      await page.keyboard.press("Escape")
    } else note("model selector", "warn", "no model-selector testid visible")
  } catch (e) { note("model selector", "warn", e.message) }

  // ---- Step 5: tools button / popover ----
  try {
    const tb = page.getByTestId("tools-button").first()
    if (await tb.count()) {
      await tb.click({ timeout: 5000 }); await page.waitForTimeout(600)
      await shot(page, "03-tools-popover")
      note("tools popover opens", "ok")
      await page.keyboard.press("Escape")
    } else note("tools button", "warn", "absent")
  } catch (e) { note("tools button", "warn", e.message) }

  // ---- Step 6: composer advanced/options toggle (pro user) ----
  try {
    const adv = page.getByTestId("composer-advanced-toggle").first()
    const opt = page.getByTestId("composer-options-toggle").first()
    const target = (await adv.count()) ? adv : (await opt.count()) ? opt : null
    if (target) {
      await target.click({ timeout: 5000 }); await page.waitForTimeout(600)
      await shot(page, "04-composer-advanced")
      note("composer advanced/options toggle", "ok")
    } else note("composer advanced toggle", "warn", "absent")
  } catch (e) { note("composer advanced toggle", "warn", e.message) }

  // ---- Step 7: shortcuts help (power user) ----
  try {
    const sc = page.getByTestId("playground-shortcuts-help-trigger").first()
    if (await sc.count()) {
      await sc.click({ timeout: 5000 }); await page.waitForTimeout(500)
      await shot(page, "05-shortcuts-help")
      const panel = await visState(page, "playground-shortcuts-help-panel")
      note("shortcuts help panel", panel === "visible" ? "ok" : "warn", panel)
      await page.keyboard.press("Escape")
    } else note("shortcuts help", "warn", "absent")
  } catch (e) { note("shortcuts help", "warn", e.message) }

  // ---- Step 8: type a message + send-control enablement ----
  try {
    const input = page.getByTestId("chat-input").first()
    await input.click()
    await input.fill("Reply with exactly the word: pong")
    await page.waitForTimeout(400)
    await shot(page, "06-typed-message")
    note("type into composer", "ok")

    if (SEND_LLM) {
      await page.keyboard.press("Enter")
      note("sent message (Enter)", "ok")
      // wait for an assistant response token / streaming to appear
      let got = false
      for (let i = 0; i < 30; i++) {
        await page.waitForTimeout(1000)
        const txt = await page.locator("body").innerText().catch(() => "")
        if (/pong/i.test(txt) && !/(Reply with exactly)/.test(txt.split("pong")[0].slice(-40))) { got = true; break }
        if (await page.getByTestId("playground-chat-error-banner").first().count()) {
          if (await page.getByTestId("playground-chat-error-banner").first().isVisible()) { break }
        }
      }
      await shot(page, "07-after-send")
      note("assistant response received", got ? "ok" : "warn", got ? "got 'pong'" : "no clear response in 30s")
    }
  } catch (e) { note("send message flow", "warn", e.message) }

  // ---- Step 9: keyboard focus order from composer (a11y) ----
  try {
    await page.getByTestId("chat-input").first().focus()
    const order = []
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press("Tab")
      const info = await page.evaluate(() => {
        const el = document.activeElement
        if (!el) return "none"
        return `${el.tagName.toLowerCase()}${el.getAttribute("data-testid") ? "#" + el.getAttribute("data-testid") : ""}${el.getAttribute("aria-label") ? "[" + el.getAttribute("aria-label").slice(0,30) + "]" : ""}`
      })
      order.push(info)
    }
    obs.tabOrderFromComposer = order
    note("tab order from composer", "ok", order.join(" → "))
  } catch (e) { note("tab order", "warn", e.message) }

  // ---- Step 10: mobile viewport ----
  try {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.waitForTimeout(1500)
    await shot(page, "08-mobile")
    obs.presenceMobile = {
      composer: await visState(page, "chat-input"),
      mobileRails: await visState(page, "playground-cockpit-mobile-rails"),
      parityNotice: await visState(page, "playground-mobile-parity-notice"),
      sendControl: await visState(page, "composer-inline-send-control"),
    }
    note("mobile viewport snapshot", "ok", JSON.stringify(obs.presenceMobile))
  } catch (e) { note("mobile viewport", "warn", e.message) }

  await browser.close()
  await fs.writeFile(`${SHOTS}/observations.json`, JSON.stringify(obs, null, 2))
  note("wrote observations.json", "ok", `${SHOTS}/observations.json`)
  // summary
  console.log("\n==== SUMMARY ====")
  console.log("console errors:", obs.consoleErrors.length, "| page errors:", obs.pageErrors.length, "| api request failures:", obs.requestFailures.length)
}

main().catch((e) => { console.error("DRIVER CRASH:", e); process.exit(1) })
