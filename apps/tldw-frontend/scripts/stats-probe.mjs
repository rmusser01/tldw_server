#!/usr/bin/env node
/* Probe: why does /admin/server never request /api/v1/admin/stats? */
import { chromium } from "@playwright/test"

const WEB = "http://localhost:8080"
const SERVER = "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || ""

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
await ctx.addInitScript(({ serverUrl, apiKey }) => {
  const cfg = { serverUrl, authMode: "single-user", apiKey }
  localStorage.setItem("tldwConfig", JSON.stringify(cfg))
  localStorage.setItem("isMigrated", "true")
  localStorage.setItem("__tldw_first_run_complete", "true")
  localStorage.setItem("assistant_setup_dismissed", "true")
  localStorage.setItem("serverUrl", serverUrl)
  localStorage.setItem("tldwServerUrl", serverUrl)
  localStorage.setItem("tldw-api-host", serverUrl)
  localStorage.setItem("authMode", "single-user")
  localStorage.setItem("apiKey", apiKey)
}, { serverUrl: SERVER, apiKey: API_KEY })

const page = await ctx.newPage()
const consoleLines = []
page.on("console", (m) => {
  const t = m.text()
  if (/stats|error|Error|failed|Failed|timeout/i.test(t) && consoleLines.length < 60)
    consoleLines.push(`${m.type()}: ${t.slice(0, 240)}`)
})
page.on("pageerror", (e) => consoleLines.push(`PAGEERROR: ${String(e.message).slice(0, 240)}`))
const requests = []
page.on("request", (r) => {
  if (r.url().includes("stats")) requests.push(`REQ ${r.method()} ${r.url()}`)
})
page.on("response", (r) => {
  if (r.url().includes("stats")) requests.push(`RESP ${r.status()} ${r.url()}`)
})

await page.goto(WEB + "/admin/server", { waitUntil: "domcontentloaded" })
await page.waitForTimeout(12000)

const bodyHas = await page.evaluate(() => ({
  emptyPanel: document.body.innerText.includes("No system statistics"),
  errorPanel: document.body.innerText.includes("Unable to load system statistics"),
  timeoutCopy: document.body.innerText.includes("longer than 10 seconds"),
  usersLoaded: document.body.innerText.includes("single_user")
}))

console.log("STATS REQUESTS:", JSON.stringify(requests))
console.log("PANELS:", JSON.stringify(bodyHas))
console.log("CONSOLE:")
for (const line of consoleLines) console.log(" ", line)

await browser.close()
