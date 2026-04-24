/**
 * World Books Workflow E2E Tests
 *
 * Tests the complete world book lifecycle with the two-panel layout:
 * - Create world book (modal, unchanged)
 * - Select row → detail panel opens with Entries tab
 * - Edit via Settings tab in detail panel (no more edit modal)
 * - Delete via overflow menu on the row
 * - Manage entries via detail panel Entries tab
 * - Character attachment via detail panel Attachments tab
 * - Import / Export via toolbar Tools dropdown
 * - Statistics via detail panel Stats tab
 *
 * Run: npx playwright test e2e/workflows/world-books.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors
} from "../utils/fixtures"
import { WorldBooksPage } from "../utils/page-objects/WorldBooksPage"
import { seedAuth, generateTestId } from "../utils/helpers"

test.describe("World Books Workflow", () => {
  let wbPage: WorldBooksPage
  const testPrefix = generateTestId("wb")

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    wbPage = new WorldBooksPage(page)
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.1  Create World Book
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Create World Book", () => {
    test("should navigate to world books page and render layout", async ({
      authedPage,
      diagnostics
    }) => {
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should create a new world book and show it in the list", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-create`

      const [apiResult] = await Promise.all([
        wbPage.waitForApiCall(/\/api\/v1\/characters\/world-books\/?$/, "POST"),
        wbPage.createWorldBook(name, "E2E test world book")
      ])

      expect(apiResult.status).toBeLessThan(300)

      // Verify row appears in the list panel
      const row = await wbPage.findWorldBookRow(name)
      await expect(row).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should select a created world book and show detail panel", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-select`
      await wbPage.createWorldBook(name, "E2E test world book for selection")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({ timeout: 10_000 })

      // Click the row to select it
      await wbPage.selectWorldBookRow(name)

      // Verify detail panel shows the world book name and Entries tab
      await wbPage.expectDetailPanelTitle(name)
      await wbPage.expectDetailTabActive("Entries")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.2  Edit World Book (via Settings tab)
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Edit World Book", () => {
    test("should edit a world book description via Settings tab", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-edit`
      await wbPage.createWorldBook(name, "Original")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      // Open the Settings tab in the detail panel
      await wbPage.openSettingsTab(name)

      // Change description
      await wbPage.fillSettingsDescription("Updated description via E2E")

      // Submit
      const [apiResult] = await Promise.all([
        wbPage.waitForApiCall(/\/api\/v1\/characters\/world-books\/\d+/, "PUT"),
        wbPage.submitSettingsForm()
      ])

      expect(apiResult.status).toBeLessThan(300)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.3  Manage Entries (via detail panel Entries tab)
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Manage Entries", () => {
    test("should add a single entry with keywords", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-entries`
      await wbPage.createWorldBook(name, "For entry tests")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      // Open the Entries tab in the detail panel
      await wbPage.openEntriesTab(name)

      // Add entry
      await wbPage.fillEntryForm("dragon, fire", "Dragons breathe fire", 50)
      const [entryResult] = await Promise.all([
        wbPage.waitForApiCall(/\/api\/v1\/characters\/world-books\/\d+\/entries$/, "POST"),
        wbPage.submitEntry()
      ])
      expect(entryResult.status).toBeLessThan(300)

      // Verify entry count
      await expect
        .poll(async () => wbPage.getEntryCount(), { timeout: 30_000 })
        .toBeGreaterThanOrEqual(1)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should surface bulk add failures in bulk mode", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      await authedPage.route(/\/api\/v1\/characters\/world-books\/\d+\/entries$/, async (route) => {
        await route.fulfill({
          status: 429,
          contentType: "application/json",
          body: JSON.stringify({ detail: "rate_limited" })
        })
      })

      const name = `${testPrefix}-bulk`
      await wbPage.createWorldBook(name, "For bulk entry tests")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      // Open the Entries tab in the detail panel
      await wbPage.openEntriesTab(name)

      // Toggle bulk mode
      await wbPage.toggleBulkAddMode()

      // Fill bulk text using -> separator
      await wbPage.fillBulkText("elf, forest -> Elves live in ancient forests")

      // Submit
      await wbPage.submitEntry()

      const panel = wbPage.detailPanel()
      await expect(panel.getByText("Parsed entries: 1")).toBeVisible({ timeout: 10_000 })
      await expect(panel.getByText(/failed entries \(1\)/i)).toBeVisible({ timeout: 10_000 })
      await expect(panel.getByText(/line 1/i)).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.4  Attach to Characters (via detail panel Attachments tab)
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Attach to Characters", () => {
    test("should open Attachments tab for a world book", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-attach`
      await wbPage.createWorldBook(name, "For attachment tests")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      await wbPage.openAttachmentsTab(name)
      await wbPage.expectDetailTabActive("Attachments")

      const panel = wbPage.detailPanel()
      await expect(panel.getByRole("heading", { name: /attached characters/i })).toBeVisible({
        timeout: 10_000
      })
      await expect(panel.getByText(/no characters attached\./i)).toBeVisible({
        timeout: 10_000
      })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should open relationship matrix via Tools dropdown", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-matrix`
      await wbPage.createWorldBook(name, "For matrix tests")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      await wbPage.clickRelationshipMatrix()
      await expect(
        authedPage.getByText(/(?:list|matrix) view active \(\d+ characters\)\./i)
      ).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.5  Delete World Book (via overflow menu)
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Delete World Book", () => {
    test("should delete a world book via overflow menu with confirmation", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-del`
      await wbPage.createWorldBook(name, "To be deleted")
      const row = await wbPage.findWorldBookRow(name)
      await expect(row).toBeVisible({ timeout: 10_000 })

      // Delete via the overflow menu
      await wbPage.clickDeleteOnRow(name)
      await wbPage.confirmDeletion()

      await expect(wbPage.pendingDeleteBanner()).toBeVisible({ timeout: 10_000 })
      await expect(row.getByText(/pending delete/i)).toBeVisible({ timeout: 10_000 })
      await expect(row).toBeHidden({ timeout: 20_000 })
      await expect(wbPage.pendingDeleteBanner()).toBeHidden({ timeout: 20_000 })

      await assertNoCriticalErrors(diagnostics)
    })

    test("should support undo after delete", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-undo`
      await wbPage.createWorldBook(name, "Will undo delete")
      const row = await wbPage.findWorldBookRow(name)
      await expect(row).toBeVisible({ timeout: 10_000 })

      // Delete via the overflow menu
      await wbPage.clickDeleteOnRow(name)
      await wbPage.confirmDeletion()

      await expect(wbPage.pendingDeleteBanner()).toBeVisible({ timeout: 10_000 })
      await expect(row.getByText(/pending delete/i)).toBeVisible({ timeout: 10_000 })

      await wbPage.cancelPendingDelete()

      await expect(wbPage.pendingDeleteBanner()).toBeHidden({ timeout: 10_000 })
      await expect(row).toBeVisible({ timeout: 10_000 })
      await expect(row.getByText(/pending delete/i)).toBeHidden({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.6  Import / Export (via Tools dropdown and overflow menu)
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Import / Export", () => {
    test("should export a world book via overflow menu", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-export`
      await wbPage.createWorldBook(name, "For export tests")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      const [apiResult] = await Promise.all([
        wbPage.waitForApiCall(/\/api\/v1\/characters\/world-books\/\d+\/export/, "GET"),
        wbPage.clickExportOnRow(name)
      ])

      expect(apiResult.status).toBe(200)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should open import modal via Tools dropdown", async ({
      authedPage,
      diagnostics
    }) => {
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      await wbPage.clickImport()

      const modal = authedPage.locator(".ant-modal")
      await expect(modal).toBeVisible({ timeout: 10_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.7  Statistics (via detail panel Stats tab)
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Statistics", () => {
    test("should display world book statistics in Stats tab", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-stats`
      await wbPage.createWorldBook(name, "For statistics tests")
      await expect(await wbPage.findWorldBookRow(name)).toBeVisible({
        timeout: 10_000
      })

      await wbPage.openStatsTab(name)
      await wbPage.expectDetailTabActive("Stats")

      const content = await wbPage.getStatsTabContent()
      expect(content).toContain("Entries")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // ═════════════════════════════════════════════════════════════════════
  // 2.8  Full Create → View → Edit → Delete Flow
  // ═════════════════════════════════════════════════════════════════════

  test.describe("Full Lifecycle", () => {
    test("should create, select, edit, and delete a world book", async ({
      authedPage,
      serverInfo,
      diagnostics
    }) => {
      skipIfServerUnavailable(serverInfo)
      wbPage = new WorldBooksPage(authedPage)
      await wbPage.goto()
      await wbPage.waitForReady()

      const name = `${testPrefix}-lifecycle`

      // 1. Create via modal (unchanged)
      await wbPage.createWorldBook(name, "Lifecycle test")
      const row = await wbPage.findWorldBookRow(name)
      await expect(row).toBeVisible({ timeout: 10_000 })

      // 2. Click the row to select it in the list panel
      await wbPage.selectWorldBookRow(name)

      // 3. Verify detail panel shows name and default Entries tab
      await wbPage.expectDetailPanelTitle(name)
      await wbPage.expectDetailTabActive("Entries")

      // 4. Click Settings tab to edit
      await wbPage.clickDetailTab("Settings")

      // 5. Update description and save
      let putResult: { status: number; body: any } | null = null
      for (let attempt = 1; attempt <= 4; attempt += 1) {
        await wbPage.fillSettingsDescription("Updated lifecycle description")
        ;[putResult] = await Promise.all([
          wbPage.waitForApiCall(/\/api\/v1\/characters\/world-books\/\d+/, "PUT"),
          wbPage.submitSettingsForm()
        ])

        if (putResult.status < 300) {
          break
        }

        if (putResult.status !== 429 || attempt === 4) {
          expect(putResult.status).toBeLessThan(300)
        }

        await authedPage.waitForTimeout(1_000 * attempt)
      }

      await authedPage.reload({ waitUntil: "domcontentloaded" })
      await wbPage.waitForReady()
      const refreshedRow = await wbPage.findWorldBookRow(name)
      await expect(refreshedRow).toBeVisible({ timeout: 30_000 })

      // 6. Delete via overflow menu on the row
      await wbPage.clickDeleteOnRow(name)
      await wbPage.confirmDeletion()
      await expect(wbPage.pendingDeleteBanner()).toBeVisible({ timeout: 10_000 })
      await expect(refreshedRow).toBeHidden({ timeout: 20_000 })

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
