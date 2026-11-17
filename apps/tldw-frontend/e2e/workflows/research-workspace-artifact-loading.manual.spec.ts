import { expect, test, type Response } from "@playwright/test"

const workspaceId = process.env.TLDW_RESEARCH_WORKSPACE_ID?.trim()

test("measures cold and warm Studio entry with unopened section outputs", async ({ page, baseURL }, testInfo) => {
  test.skip(!workspaceId, "Set TLDW_RESEARCH_WORKSPACE_ID to a prepared workspace with saved section outputs.")
  test.setTimeout(240_000)
  const workspacePath = `/api/v1/workspaces/${encodeURIComponent(workspaceId!)}`
  const catalogPath = `${workspacePath}/artifact-revisions`
  const fullPath = `${workspacePath}/artifacts`
  const url = new URL("/research-workspace", process.env.TLDW_WEB_URL || baseURL)
  url.searchParams.set("source_workspace_id", workspaceId!)
  url.searchParams.set("tab", "studio")

  const cardSelector = '[data-testid^="studio-artifact-card-"]'
  await page.addInitScript((selector) => {
    performance.setResourceTimingBufferSize(4000)
    const observer = new MutationObserver(() => {
      const card = document.querySelector(selector)
      if (card?.getBoundingClientRect().width) {
        performance.mark("first-saved-studio-card")
        observer.disconnect()
      }
    })
    observer.observe(document, { childList: true, subtree: true, attributes: true })
  }, cardSelector)

  const eagerReads: string[] = []
  const catalogs: Response[] = []
  page.on("response", (response) => {
    if (new URL(response.url()).pathname === catalogPath) catalogs.push(response)
  })
  page.on("request", (request) => {
    const path = new URL(request.url()).pathname
    if (request.method() === "GET" && (path === fullPath || path.startsWith(`${fullPath}/`))) {
      eagerReads.push(path)
    }
  })
  for (const mode of ["cold", "warm"]) {
    catalogs.length = 0
    if (mode === "cold") await page.goto(url.href, { waitUntil: "commit" })
    else await page.reload({ waitUntil: "commit" })
    await expect(page.locator(cardSelector).first()).toBeVisible({ timeout: 110_000 })
    const sample = await page.evaluate(({ catalogPath, fullPath }) => ({
      firstCardMs: performance.getEntriesByName("first-saved-studio-card")[0]?.startTime,
      requests: (performance.getEntriesByType("resource") as PerformanceResourceTiming[])
        .filter(({ name }) => {
          const path = new URL(name).pathname
          return path === catalogPath || path === fullPath || path.startsWith(`${fullPath}/`)
        })
        .map(({ name, duration, transferSize, encodedBodySize, decodedBodySize }) => ({
          path: new URL(name).pathname, duration, transferSize, encodedBodySize, decodedBodySize
        }))
    }), { catalogPath, fullPath })
    await testInfo.attach(`studio-${mode}.json`, {
      body: JSON.stringify(sample, null, 2), contentType: "application/json"
    })
    console.info(mode, JSON.stringify(sample))
    expect(sample.firstCardMs).toBeGreaterThan(0)
    expect(eagerReads, "Opening Studio fetched the full artifact list or an unopened output").toEqual([])
    expect(catalogs.length).toBeGreaterThan(0)
    for (const response of catalogs) {
      expect(response.ok()).toBe(true)
      const rows: { defer_detail: boolean }[] = await response.json()
      expect(rows.length).toBeGreaterThan(0)
      expect(rows.every(row => row.defer_detail), "Prepare a workspace with only deferred section outputs").toBe(true)
    }
    expect(sample.requests.some(request => request.path === catalogPath && request.decodedBodySize > 0)).toBe(true)
  }
})
