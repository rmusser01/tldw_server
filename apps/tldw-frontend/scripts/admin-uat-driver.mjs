#!/usr/bin/env node
/*
 * Admin WebUI UAT driver — live browser walkthrough of all /admin routes.
 * Pass A: first-time user, no seeded auth (real first-run experience).
 * Pass B: configured power user (seeded single-user auth), sweep all admin pages.
 * Emits screenshots + observations JSON for heuristic evaluation.
 */
import { chromium } from "@playwright/test"
import fs from "node:fs/promises"

const WEB = process.env.WEB_URL || "http://localhost:8080"
const SERVER = process.env.SERVER_URL || "http://127.0.0.1:8000"
const API_KEY = process.env.TLDW_API_KEY || ""
const OUT = process.env.OUT_DIR || "/tmp/admin-uat"

const ADMIN_ROUTES = [
  "/admin", "/admin/server", "/admin/api-keys", "/admin/billing",
  "/admin/data-ops", "/admin/integrations", "/admin/llamacpp",
  "/admin/maintenance", "/admin/mlx", "/admin/monitoring", "/admin/orgs",
  "/admin/rate-limiting", "/admin/rbac", "/admin/sources", "/admin/usage",
  "/admin/watchlists-items", "/admin/watchlists-runs"
]

const MOBILE_ROUTES = new Set(["/admin", "/admin/server", "/admin/monitoring"])

const report = { passA: [], passB: [], meta: { web: WEB, server: SERVER, when: new Date().toISOString() } }

function attachCollectors(page, bucket) {
  bucket.consoleErrors = []
  bucket.pageErrors = []
  bucket.failedRequests = []
  bucket.apiCalls = []
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") {
      const t = m.text()
      if (bucket.consoleErrors.length < 40) bucket.consoleErrors.push(`${m.type()}: ${t.slice(0, 300)}`)
    }
  })
  page.on("pageerror", (e) => bucket.pageErrors.push(String(e.message).slice(0, 300)))
  page.on("response", async (r) => {
    const url = r.url()
    if (url.includes("/api/")) {
      const rec = { url: url.replace(SERVER, "").replace(WEB, ""), status: r.status() }
      if (bucket.apiCalls.length < 200) bucket.apiCalls.push(rec)
      if (r.status() >= 400 && bucket.failedRequests.length < 60) bucket.failedRequests.push(rec)
    }
  })
  page.on("requestfailed", (req) => {
    if (bucket.failedRequests.length < 60)
      bucket.failedRequests.push({ url: req.url().replace(SERVER, "").replace(WEB, "").slice(0, 200), status: "net:" + (req.failure()?.errorText || "?") })
  })
}

async function probePage(page) {
  return await page.evaluate(() => {
    const txt = (el) => (el?.innerText || "").trim()
    const h1s = [...document.querySelectorAll("h1")].map(txt).filter(Boolean)
    const h2s = [...document.querySelectorAll("h2")].map(txt).filter(Boolean).slice(0, 20)
    const buttons = [...document.querySelectorAll("button")]
    const unnamedButtons = buttons.filter((b) => {
      const name = (b.getAttribute("aria-label") || b.title || b.innerText || "").trim()
      return !name
    }).length
    const landmarks = ["main", "nav", "header", "aside"].map((t) => `${t}:${document.querySelectorAll(t).length}`).join(" ")
    const tables = document.querySelectorAll("table").length
    const forms = document.querySelectorAll("form").length
    const inputs = document.querySelectorAll("input,select,textarea").length
    const bodyText = (document.body.innerText || "").replace(/\n{3,}/g, "\n\n").slice(0, 3000)
    return {
      title: document.title, h1s, h2s, buttons: buttons.length, unnamedButtons,
      landmarks, tables, forms, inputs, bodyText
    }
  })
}

async function visit(page, route, bucket, shotName) {
  const url = WEB + route
  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout
: 30000 })
  } catch (e) {
    bucket.gotoError = e.message.slice(0, 200)
  }
  try { await page.waitForLoadState("networkidle", { timeout: 12000 }) } catch {}
  await page.waitForTimeout(1500)
  bucket.finalUrl = page.url()
  try {
    bucket.probe = await probePage(page)
  } catch (e) { bucket.probeError = e.message.slice(0, 200) }
  try {
    await page.screenshot({ path: `${OUT}/${shotName}.png`, fullPage: false })
    await page.screenshot({ path: `${OUT}/${shotName}-full.png`, fullPage: true })
  } catch (e) { bucket.shotError = e.message.slice(0, 200) }
  console.log(`[done] ${route} -> ${bucket.finalUrl} (h1: ${bucket.probe?.h1s?.[0] || "-"})`)
}

function seedScript({ serverUrl, apiKey }) {
  return ({ serverUrl, apiKey }) => {
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
}

async function main() {
  await fs.mkdir(OUT, { recursive: true })
  const browser = await chromium.launch()

  // ---------- Pass A: first-time user, nothing configured ----------
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
    const page = await ctx.newPage()
    for (const route of ["/", "/admin", "/admin/server"]) {
      const bucket = { route }
      attachCollectors(page, bucket)
      await visit(page, route, bucket, "A" + (route === "/" ? "-root" : route.replaceAll("/", "-")))
      report.passA.push(bucket)
      page.removeAllListeners("console"); page.removeAllListeners("pageerror")
      page.removeAllListeners("response"); page.removeAllListeners("requestfailed")
    }
    await ctx.close()
  }

  // ---------- Pass B: configured power user ----------
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
    await ctx.addInitScript(seedScript({}), { serverUrl: SERVER, apiKey: API_KEY })
    const page = await ctx.newPage()

    // Discoverability probe: from home, how many links point at /admin?
    {
      const bucket = { route: "home-discoverability" }
      attachCollectors(page, bucket)
      await visit(page, "/", bucket, "B-home")
      try {
        bucket.adminLinks = await page.evaluate(() =>
          [...document.querySelectorAll("a[href*='/admin']")].map((a) => ({
            href: a.getAttribute("href"), text: (a.innerText || a.getAttribute("aria-label") || "").trim().slice(0, 60)
          })).slice(0, 30)
        )
      } catch {}
      report.passB.push(bucket)
      page.removeAllListeners("console"); page.removeAllListeners("pageerror")
      page.removeAllListeners("response"); page.removeAllListeners("requestfailed")
    }

    for (const route of ADMIN_ROUTES) {
      const bucket = { route }
      attachCollectors(page, bucket)
      await visit(page, route, bucket, "B" + route.replaceAll("/", "-"))
      report.passB.push(bucket)
      page.removeAllListeners("console"); page.removeAllListeners("pageerror")
      page.removeAllListeners("response"); page.removeAllListeners("requestfailed")
    }
    await ctx.close()
  }

  // ---------- Pass C: mobile viewport spot-check ----------
  {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 } })
    await ctx.addInitScript(seedScript({}), { serverUrl: SERVER, apiKey: API_KEY })
    const page = await ctx.newPage()
    for (const route of MOBILE_ROUTES) {
      const bucket = { route: "mobile:" + route }
      attachCollectors(page, bucket)
      await visit(page, route, bucket, "M" + route.replaceAll("/", "-"))
      report.passB.push(bucket)
      page.removeAllListeners("console"); page.removeAllListeners("pageerror")
      page.removeAllListeners("response"); page.removeAllListeners("requestfailed")
    }
    await ctx.close()
  }

  await browser.close()
  await fs.writeFile(`${OUT}/report.json`, JSON.stringify(report, null, 2))
  console.log(`report written to ${OUT}/report.json`)
}

main().catch((e) => { console.error(e); process.exit(1) })
