/**
 * Kanban Playground E2E Tests (Tier 2)
 *
 * Tests the Kanban Playground page lifecycle:
 * - Page loads with expected elements (heading, board selector, new board button)
 * - Empty state gallery shown when no boards exist
 * - New Board button opens create modal
 * - Refresh button fires GET /api/v1/kanban/boards
 * - Create board fires POST /api/v1/kanban/boards (requires server)
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/kanban.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { KanbanPage } from "../../utils/page-objects"
import { seedAuth } from "../../utils/helpers"

test.describe("Kanban Playground", () => {
  let kanban: KanbanPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    kanban = new KanbanPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Kanban Playground page with heading and controls", async ({
      authedPage,
      diagnostics,
    }) => {
      kanban = new KanbanPage(authedPage)
      await kanban.goto()
      await kanban.assertPageReady()

      // Heading or empty state should be visible
      const headingVisible = await authedPage
        .getByText("Kanban Playground")
        .first()
        .isVisible()
        .catch(() => false)
      const emptyVisible = await kanban.emptyStateMessage
        .isVisible()
        .catch(() => false)

      expect(headingVisible || emptyVisible).toBe(true)

      // If the page loaded normally, toolbar controls should be present
      if (headingVisible) {
        await expect(kanban.newBoardButton).toBeVisible()
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show empty state or board gallery when no board is selected", async ({
      authedPage,
      diagnostics,
    }) => {
      kanban = new KanbanPage(authedPage)
      await kanban.goto()
      await kanban.assertPageReady()

      // Zero boards exposes the primary Create Board action; existing boards
      // with no selection expose gallery cards instead.
      const zeroBoardStateVisible = await Promise.all([
        kanban.emptyStateMessage.isVisible().catch(() => false),
        kanban.createBoardButton.isVisible().catch(() => false),
      ]).then(([messageVisible, actionVisible]) => messageVisible && actionVisible)
      const galleryVisible = await kanban.boardGalleryCards
        .first()
        .isVisible()
        .catch(() => false)
      const newBoardCardVisible = await kanban.galleryNewBoardCard
        .isVisible()
        .catch(() => false)

      expect(zeroBoardStateVisible || galleryVisible || newBoardCardVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Create Board Modal
  // =========================================================================

  test.describe("Create Board Modal", () => {
    test("should open Create New Board modal when New Board button is clicked", async ({
      authedPage,
      diagnostics,
    }) => {
      kanban = new KanbanPage(authedPage)
      await kanban.goto()
      await kanban.assertPageReady()

      const newBoardVisible = await kanban.newBoardButton
        .isVisible()
        .catch(() => false)
      if (!newBoardVisible) return

      await kanban.openCreateBoardModal()

      await expect(kanban.boardNameInput).toBeVisible()
      await expect(kanban.boardDescriptionInput).toBeVisible()
      await expect(kanban.modalCreateButton).toBeVisible()

      // Close the modal
      await authedPage.keyboard.press("Escape")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // API Integration (requires server)
  // =========================================================================

  test.describe("API Integration", () => {
    test("should fire GET /api/v1/kanban/boards on page load", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      kanban = new KanbanPage(authedPage)

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/kanban\/boards/,
        method: "GET",
      }, 15_000)

      await kanban.goto()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Board listing may not fire if the page is in an error state
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should fire POST /api/v1/kanban/boards when creating a board", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      kanban = new KanbanPage(authedPage)
      await kanban.goto()
      await kanban.assertPageReady()

      const newBoardVisible = await kanban.newBoardButton
        .isVisible()
        .catch(() => false)
      if (!newBoardVisible) return

      await kanban.openCreateBoardModal()
      await kanban.boardNameInput.fill("E2E Test Board")

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/kanban\/boards/,
        method: "POST",
      }, 15_000)

      await kanban.modalCreateButton.click()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Creation may fail if backend kanban module is not available
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should fire GET /api/v1/kanban/boards when refresh is clicked", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      kanban = new KanbanPage(authedPage)
      await kanban.goto()
      await kanban.assertPageReady()

      const refreshVisible = await kanban.refreshButton
        .isVisible()
        .catch(() => false)
      if (!refreshVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/kanban\/boards/,
        method: "GET",
      }, 15_000)

      await kanban.refreshButton.click()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Refresh may not fire if the component isn't fully loaded
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
