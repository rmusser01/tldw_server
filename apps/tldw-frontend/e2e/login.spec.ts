import { test, expect } from "@playwright/test"

const serverUrl = process.env.TLDW_SERVER_URL || "http://127.0.0.1:8000"
const apiKey =
  process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const hostedMode =
  String(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || "").trim().toLowerCase() ===
  "hosted"

const normalizeUrl = (value: string) => value.replace(/\/$/, "")

if (hostedMode) {
  test("hosted login uses the auth form instead of redirecting to settings", async ({
    page,
  }) => {
    let loginFormBody = ""

    await page.route("**/api/auth/login", async (route) => {
      loginFormBody = route.request().postData() || ""
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          access_token: "server-managed",
          refresh_token: "server-managed-refresh",
          token_type: "bearer",
          expires_in: 1800,
        }),
      })
    })

    await page.route("**/api/auth/session", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          authenticated: true,
          user: {
            id: 7,
            username: "hosted-user",
            email: "user@example.com",
            role: "user",
            is_active: true,
          },
        }),
      })
    })

    await page.route("**/api/proxy/orgs", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          items: [{ id: 23, name: "Personal workspace" }],
        }),
      })
    })

    await page.goto("/login", { waitUntil: "domcontentloaded" })

    await expect(page).toHaveURL(/\/login$/)
    await expect(
      page.getByRole("heading", { name: /^sign in$/i })
    ).toBeVisible()

    await page.getByLabel(/account identifier/i).fill("user@example.com")
    await page.getByLabel(/^password$/i).fill("HostedPass123!")
    await page.getByRole("button", { name: /^sign in$/i }).click()

    await page.waitForURL(/\/chat/)

    const stored = await page.evaluate(() => {
      const raw = localStorage.getItem("tldwConfig")
      return raw ? JSON.parse(raw) : null
    })

    expect(Object.fromEntries(new URLSearchParams(loginFormBody))).toEqual({
      username: "user@example.com",
      password: "HostedPass123!",
    })
    expect(stored).toMatchObject({
      authMode: "multi-user",
      orgId: 23,
    })
    expect(stored?.accessToken ?? "").toBe("")
    expect(stored?.refreshToken ?? "").toBe("")
  })
} else {
  test("login via settings saves config", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.clear()
    })

    await page.goto("/login", { waitUntil: "domcontentloaded" })
    await page.waitForURL(/\/settings\/tldw/)

    const serverInput = page.getByLabel(/server url/i)
    await serverInput.waitFor({ state: "visible" })
    await serverInput.fill(serverUrl)

    const apiKeyInput = page.locator("#apiKey")
    await apiKeyInput.fill(apiKey)

    await page.getByRole("button", { name: /save/i }).click()
    await page.waitForFunction(() => Boolean(localStorage.getItem("tldwConfig")))

    const stored = await page.evaluate(() => localStorage.getItem("tldwConfig"))
    expect(stored).toBeTruthy()
    const parsed = JSON.parse(stored || "{}")
    expect(normalizeUrl(parsed.serverUrl || "")).toBe(normalizeUrl(serverUrl))
    expect(parsed.apiKey).toBe(apiKey)
  })

  test("chat renders with saved config", async ({ page }) => {
    await page.addInitScript(
      (cfg) => {
        localStorage.setItem("tldwConfig", JSON.stringify(cfg))
        localStorage.setItem("__tldw_first_run_complete", "true")
      },
      {
        serverUrl,
        apiKey,
        authMode: "single-user",
      }
    )

    await page.goto("/chat", { waitUntil: "domcontentloaded" })
    await expect(page.getByPlaceholder(/type a message/i).first()).toBeVisible()
  })
}
