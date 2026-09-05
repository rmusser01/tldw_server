#!/usr/bin/env node
/* Admin UAT pass D — targeted interaction probes. */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || ""
const OUT = process.env.OUT_DIR || "/tmp/admin-uat"

const out = { steps: [] }
const note = (s, detail) => { out.steps.push({ s, detail }); console.log(`[step] ${s} :: ${detail || ""}`) }

const seed = ({ serverUrl, apiKey }) => {
  const cfg = { serverUrl, authMode: "single-user", apiKey }
  try { localStorage.setItem("tldwConfig", JSON.stringify(cfg)) } catch {}
  try { localStorage.setItem("isMigrated", "true") } catch {}
  try { localStorage.setItem("__tldw_first_run_complete", "true") } catch {}
  try { localStorage.setItem("assistant_setup_dismissed", "true") } catch {}
  try {
    localStorage.setItem("serverUrl", serverUrl)
    localStorage.setItem("tldwServerUrl", serverUrl)
    localStorage.setItem("tldw-api-host", serverUrl)
    localStorage.setItem("authMode", "single-user")
    localStorage.setItem("apiKey", apiKey)
  } catch {}
}

async function shot(page, name) {
  try { await page.screenshot({ path: `${OUT}/${name}.png` }) } catch {}
}

async function main() {
  await fs.mkdir(OUT, { recursive: true })
  const browser = await chromium.launch()

  // D1: first-time user clicks "Skip for now" on /admin interstitial
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
    const page = await ctx.newPage()
    await page.goto(WEB + "/admin", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(2500)
    const skip = page.getByText("Skip for now").first()
    if (await skip.count()) {
      await skip.click()
      await page.waitForTimeout(2500)
      await shot(page, "D1-admin-after-skip")
      note("D1 skip-for-now", `url=${page.url()} :: body starts: ${(await page.evaluate(() => document.body.innerText.slice(0, 400))).replace(/\n/g, " | ")}`)
    } else {
      note("D1 skip-for-now", "no interstitial found")
      await shot(page, "D1-admin-no-interstitial")
    }
    await ctx.close()
  }

  // Authed context for the rest
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  await ctx.addInitScript(seed, { serverUrl: SERVER, apiKey: API_KEY })
  const page = await ctx.newPage()

  // D2: server admin — Create role with empty name (validation?) then real create (feedback?)
  {
    await page.goto(WEB + "/admin/server", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(3500)
    const createBtn = page.getByRole("button", { name: /create role/i }).first()
    if (await createBtn.count()) {
      await createBtn.scrollIntoViewIfNeeded()
      await createBtn.click()
      await page.waitForTimeout(1200)
      await shot(page, "D2a-create-role-empty")
      note("D2a create-role empty submit", (await page.evaluate(() => document.body.innerText)).match(/required|enter|name|invalid|error/i)?.[0] || "no visible validation keyword")
      const nameInput = page.getByPlaceholder(/role name/i).first()
      if (await nameInput.count()) {
        await nameInput.fill("uat-analyst")
        await createBtn.click()
        await page.waitForTimeout(2000)
        await shot(page, "D2b-create-role-filled")
        const body = await page.evaluate(() => document.body.innerText)
        note("D2b create-role uat-analyst", body.includes("uat-analyst") ? "role appears in list" : "role NOT visible after create")
      }
    } else note("D2 create-role", "button not found")
  }

  // D3: api-keys — open user select, type, see whether any options / error appear
  {
    await page.goto(WEB + "/admin/api-keys", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(2500)
    const sel = page.locator(".ant-select-selector").first()
    if (await sel.count()) {
      await sel.click()
      await page.keyboard.type("admin")
      await page.waitForTimeout(2500)
      await shot(page, "D3-api-keys-user-search")
      const dropdown = await page.evaluate(() => document.querySelector(".ant-select-dropdown")?.innerText?.slice(0, 200) || "NO DROPDOWN")
      note("D3 api-keys user search", dropdown.replace(/\n/g, " | "))
    } else note("D3 api-keys", "no select found")
  }

  // D4: data-ops — Create Backup with no dataset selected (validation?)
  {
    await page.goto(WEB + "/admin/data-ops", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(2500)
    const btn = page.getByRole("button", { name: /create backup/i }).first()
    if (await btn.count()) {
      await btn.click()
      await page.waitForTimeout(1500)
      await shot(page, "D4-create-backup-novalue")
      const toasts = await page.evaluate(() => [...document.querySelectorAll(".ant-message, .ant-notification, [role='alert']")].map(n => n.innerText).join(" || ").slice(0, 300))
      note("D4 create-backup no dataset", toasts || "no toast/alert seen")
    } else note("D4 data-ops", "create backup button not found")
  }

  // D5: maintenance — Create Incident empty (validation?)
  {
    await page.goto(WEB + "/admin/maintenance", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(2500)
    const btn = page.getByRole("button", { name: /create incident/i }).first()
    if (await btn.count()) {
      await btn.click()
      await page.waitForTimeout(1500)
      await shot(page, "D5-create-incident-empty")
      const toasts = await page.evaluate(() => [...document.querySelectorAll(".ant-message, .ant-notification, [role='alert']")].map(n => n.innerText).join(" || ").slice(0, 300))
      note("D5 create-incident empty", toasts || "no toast/alert seen")
    } else note("D5 maintenance", "create incident button not found")
  }

  // D6: keyboard focus probe on /admin/server
  {
    await page.goto(WEB + "/admin/server", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(3000)
    const seq = []
    for (let i = 0; i < 18; i++) {
      await page.keyboard.press("Tab")
      const info = await page.evaluate(() => {
        const el = document.activeElement
        if (!el) return "none"
        const cs = getComputedStyle(el)
        const outline = cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0
        const ring = (cs.boxShadow || "").includes("px") && cs.boxShadow !== "none"
        const label = (el.getAttribute("aria-label") || el.textContent || el.tagName).trim().slice(0, 28)
        return `${el.tagName.toLowerCase()}:${label}:${outline || ring ? "RING" : "no-ring"}`
      })
      seq.push(info)
    }
    note("D6 tab sequence /admin/server", seq.join(" -> "))
    await shot(page, "D6-focus-state")
  }

  // D7: monitoring — check tabs/sections + alert rule form discoverability
  {
    await page.goto(WEB + "/admin/monitoring", { waitUntil: "domcontentloaded" })
    await page.waitForTimeout(3000)
    const btns = await page.evaluate(() => [...document.querySelectorAll("button")].map(b => b.innerText.trim()).filter(Boolean).slice(0, 40))
    note("D7 monitoring buttons", btns.join(" | "))
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
    await page.waitForTimeout(800)
    await shot(page, "D7-monitoring-bottom")
  }

  await ctx.close()
  await browser.close()
  await fs.writeFile(`${OUT}/interactions.json`, JSON.stringify(out, null, 2))
  console.log("interactions written")
}

main().catch((e) => { console.error(e); process.exit(1) })
