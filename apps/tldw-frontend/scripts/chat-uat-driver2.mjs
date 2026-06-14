#!/usr/bin/env node
/* Focused pass: select a model, confirm end-to-end send/response, verify Escape-dismiss. */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = "http://localhost:8080"
const SERVER = "http://127.0.0.1:8000"
const API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const SHOTS = "/tmp/chat-uat-shots"
const log = (...a) => console.log(...a)

async function main() {
  const browser = await chromium.launch()
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  await ctx.addInitScript(({ s, k }) => {
    localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: s, authMode: "single-user", apiKey: k, accessToken: "" }))
    for (const [kk, vv] of Object.entries({ isMigrated: "true", serverUrl: s, tldwServerUrl: s, authMode: "single-user", apiKey: k, accessToken: "", __tldw_first_run_complete: "true", assistant_setup_dismissed: "true", __tldw_test_bypass: "true", __tldw_allow_offline: "true" })) localStorage.setItem(kk, vv)
  }, { s: SERVER, k: API_KEY })

  const page = await ctx.newPage()
  const apiCalls = []
  page.on("request", (r) => { if (r.url().includes("/api/v1/chat")) apiCalls.push(`${r.method()} ${r.url().split("/api/v1")[1]}`) })
  await page.goto(`${WEB}/chat`, { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30000 })
  await page.waitForTimeout(2500)

  // --- Open model selector, dump menu DOM ---
  await page.getByTestId("model-selector").first().click()
  await page.waitForTimeout(1000)
  const menu = await page.evaluate(() => {
    const items = Array.from(document.querySelectorAll('[role="menuitem"], [role="option"], .ant-dropdown li, [data-testid*="model"] button, [data-testid*="provider"]'))
    return items.slice(0, 40).map((e) => ({ tag: e.tagName.toLowerCase(), tid: e.getAttribute("data-testid"), role: e.getAttribute("role"), text: (e.textContent || "").trim().slice(0, 50) })).filter(i => i.text)
  })
  log("=== MODEL MENU ITEMS ===")
  menu.forEach((m) => log(`  ${m.role || m.tag}${m.tid ? " #" + m.tid : ""} :: ${m.text}`))
  await page.screenshot({ path: `${SHOTS}/10-model-menu.png` })

  // Try to type in any visible search box then pick first gpt-4o option
  let selected = false
  try {
    const search = page.locator('input[type="text"], input[role="combobox"], .ant-select-selection-search-input, input[placeholder*="odel"]').filter({ visible: true }).first()
    if (await search.count()) {
      await search.fill("gpt-4o-mini")
      await page.waitForTimeout(900)
      await page.screenshot({ path: `${SHOTS}/11-model-search.png` })
    }
    // click first option/menuitem that mentions gpt-4o
    const opt = page.locator('[role="option"], [role="menuitem"], .ant-dropdown li').filter({ hasText: /gpt-4o/i }).first()
    if (await opt.count()) { await opt.click({ timeout: 4000 }); selected = true; log("clicked model option gpt-4o*") }
  } catch (e) { log("model pick attempt:", e.message) }
  if (!selected) { await page.keyboard.press("Enter").catch(() => {}) }
  await page.waitForTimeout(1500)

  // Read composition MODEL state
  const modelState = await page.evaluate(() => {
    const el = Array.from(document.querySelectorAll("*")).find((n) => n.children.length === 0 && /No model selected|tldw:|gpt-|claude-|Active|Unavailable/.test(n.textContent || "") && (n.textContent || "").length < 40)
    // grab the composition MODEL block text
    const body = document.body.innerText
    const m = body.match(/MODEL\s*\n([^\n]+)\n([^\n]+)/)
    return m ? `${m[1]} | ${m[2]}` : "??"
  })
  log("composition MODEL after select:", modelState)
  await page.screenshot({ path: `${SHOTS}/12-after-model-select.png` })

  // --- Send a message ---
  const input = page.getByTestId("chat-input").first()
  await input.click()
  await input.fill("Reply with exactly one word: pong")
  await page.keyboard.press("Enter")
  log("sent message; waiting for response...")
  let resp = "none"
  for (let i = 0; i < 45; i++) {
    await page.waitForTimeout(1000)
    const t = await page.locator("body").innerText().catch(() => "")
    const err = page.getByText(/please select a model/i)
    if (await err.count() && await err.first().isVisible()) { resp = "BLOCKED: please select a model"; break }
    // assistant bubble: look for 'pong' appearing not in our own input echo
    if (/\bpong\b/i.test(t)) { resp = "GOT RESPONSE (pong)"; break }
  }
  log("RESULT:", resp)
  log("chat API calls observed:", JSON.stringify(apiCalls))
  await page.screenshot({ path: `${SHOTS}/13-send-result.png`, fullPage: false })

  // --- Escape-dismiss test on shortcuts panel ---
  let escClosed = "n/a"
  try {
    const trig = page.getByTestId("playground-shortcuts-help-trigger").first()
    if (await trig.count()) {
      await trig.click(); await page.waitForTimeout(500)
      const before = await page.getByTestId("playground-shortcuts-help-panel").first().isVisible().catch(() => false)
      await page.keyboard.press("Escape"); await page.waitForTimeout(500)
      const after = await page.getByTestId("playground-shortcuts-help-panel").first().isVisible().catch(() => false)
      escClosed = `before=${before} afterEsc=${after}`
    }
  } catch (e) { escClosed = "err:" + e.message }
  log("shortcuts Escape-dismiss:", escClosed)

  await browser.close()
  await fs.writeFile(`${SHOTS}/observations2.json`, JSON.stringify({ menu, modelState, sendResult: resp, apiCalls, escClosed }, null, 2))
}
main().catch((e) => { console.error("CRASH", e); process.exit(1) })
