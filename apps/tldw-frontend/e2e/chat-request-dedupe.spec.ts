import { test, expect, seedAuth } from "./smoke/smoke.setup"

/**
 * UAT Finding #6: identical GET requests fire many times concurrently on /chat load.
 * In-flight coalescing in the bgRequest layer collapses concurrent identical GETs to
 * a single network call. (Endpoints fetched via fetchWithAuth — e.g. /persona/profiles
 * — return a single-read Response and are intentionally out of scope: a Response body
 * cannot be safely shared across callers.)
 *
 * Baseline before fix: /users/me/profile x5, /config/providers x3, /characters x2,
 * /persona/catalog x2.
 */
const WEB = "http://localhost:8080"
const SERVER = "http://127.0.0.1:8000"
const KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"

const TARGETS = [
  "/users/me/profile",
  "/config/providers",
  "/characters/",
  "/persona/catalog",
]

test("identical concurrent GETs are coalesced on /chat load", async ({ page }) => {
  test.setTimeout(90_000)
  const counts = new Map<string, number>()
  page.on("request", (r) => {
    if (r.method() !== "GET") return
    const u = r.url()
    if (!u.includes("/api/v1/")) return
    const path = u.split("/api/v1")[1].split("?")[0]
    for (const target of TARGETS) {
      if (path === target) counts.set(target, (counts.get(target) ?? 0) + 1)
    }
  })

  await seedAuth(page, { serverUrl: SERVER, apiKey: KEY })
  await page.goto(`${WEB}/chat`, { waitUntil: "domcontentloaded" })
  await page.getByTestId("chat-input").first().waitFor({ state: "visible", timeout: 30_000 })
  await page.waitForTimeout(6000)

  const summary = TARGETS.map((t) => `${t}=${counts.get(t) ?? 0}`).join(" ")
  console.log(`[dedupe] ${summary}`)

  for (const target of TARGETS) {
    expect(counts.get(target) ?? 0, `${target} should be coalesced to one concurrent request`).toBeLessThanOrEqual(1)
  }
})
