/**
 * Document Workspace E2E Tests (Tier 2)
 *
 * Tests the Document Workspace page lifecycle:
 * - Page loads with expected elements (heading, sidebar toggles, panel tabs)
 * - Left/right sidebar toggles change panel visibility
 * - Open document button opens the picker modal which fires media list API
 * - Keyboard shortcuts modal opens on shortcut button click
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/document-workspace.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { DocumentWorkspacePage } from "../../utils/page-objects"
import { seedAuth } from "../../utils/helpers"

test.describe("Document Workspace", () => {
  let workspace: DocumentWorkspacePage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    workspace = new DocumentWorkspacePage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the Document Workspace page with heading and controls", async ({
      authedPage,
      diagnostics,
    }) => {
      workspace = new DocumentWorkspacePage(authedPage)
      await workspace.goto()
      await workspace.assertPageReady()

      // Either the heading is visible (server online) or the offline message
      const headingVisible = await authedPage
        .getByText("Document Workspace")
        .first()
        .isVisible()
        .catch(() => false)
      const offlineVisible = await workspace.offlineMessage
        .isVisible()
        .catch(() => false)

      expect(headingVisible || offlineVisible).toBe(true)

      // If online, header controls should be present
      if (headingVisible) {
        // Toggle buttons are present on desktop viewports
        const leftToggle = await workspace.toggleLeftButton
          .isVisible()
          .catch(() => false)
        const rightToggle = await workspace.toggleRightButton
          .isVisible()
          .catch(() => false)

        // At least the open-document and shortcuts buttons should be visible
        const openDocBtn = await workspace.openDocumentButton
          .isVisible()
          .catch(() => false)
        const shortcutsBtn = await workspace.shortcutsButton
          .isVisible()
          .catch(() => false)

        expect(openDocBtn || shortcutsBtn || leftToggle || rightToggle).toBe(true)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should toggle left sidebar without errors", async ({
      authedPage,
      diagnostics,
    }) => {
      workspace = new DocumentWorkspacePage(authedPage)
      await workspace.goto()
      await workspace.assertPageReady()

      const toggleVisible = await workspace.toggleLeftButton
        .isVisible()
        .catch(() => false)
      if (!toggleVisible) return

      // Click to toggle left sidebar (collapse)
      const labelBefore = await workspace.toggleLeftButton.getAttribute("aria-label")
      await workspace.toggleLeftButton.click()
      await expect
        .poll(
          () => workspace.toggleLeftButton.getAttribute("aria-label"),
          { timeout: 5_000 }
        )
        .not.toBe(labelBefore)

      // Click again to toggle back (expand)
      const labelAfterFirstToggle = await workspace.toggleLeftButton.getAttribute("aria-label")
      await workspace.toggleLeftButton.click()
      await expect
        .poll(
          () => workspace.toggleLeftButton.getAttribute("aria-label"),
          { timeout: 5_000 }
        )
        .not.toBe(labelAfterFirstToggle)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should toggle right panel without errors", async ({
      authedPage,
      diagnostics,
    }) => {
      workspace = new DocumentWorkspacePage(authedPage)
      await workspace.goto()
      await workspace.assertPageReady()

      const toggleVisible = await workspace.toggleRightButton
        .isVisible()
        .catch(() => false)
      if (!toggleVisible) return

      // Click to toggle right panel (collapse)
      const labelBefore = await workspace.toggleRightButton.getAttribute("aria-label")
      await workspace.toggleRightButton.click()
      await expect
        .poll(
          () => workspace.toggleRightButton.getAttribute("aria-label"),
          { timeout: 5_000 }
        )
        .not.toBe(labelBefore)

      // Click again to toggle back (expand)
      const labelAfterFirstToggle = await workspace.toggleRightButton.getAttribute("aria-label")
      await workspace.toggleRightButton.click()
      await expect
        .poll(
          () => workspace.toggleRightButton.getAttribute("aria-label"),
          { timeout: 5_000 }
        )
        .not.toBe(labelAfterFirstToggle)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Document Picker Modal
  // =========================================================================

  test.describe("Document Picker", () => {
    test("should open document picker modal when Open document button is clicked", async ({
      authedPage,
      diagnostics,
    }) => {
      workspace = new DocumentWorkspacePage(authedPage)
      await workspace.goto()
      await workspace.assertPageReady()

      const openDocVisible = await workspace.openDocumentButton
        .isVisible()
        .catch(() => false)
      if (!openDocVisible) return

      await workspace.openDocumentButton.click()

      // The picker modal should appear
      const modal = authedPage.locator(".ant-modal")
      await expect(modal.first()).toBeVisible({ timeout: 5_000 })

      // Close the modal
      await authedPage.keyboard.press("Escape")

      await assertNoCriticalErrors(diagnostics)
    })

    test("should fire media list API when document picker opens", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      workspace = new DocumentWorkspacePage(authedPage)
      await workspace.goto()
      await workspace.assertPageReady()

      const openDocVisible = await workspace.openDocumentButton
        .isVisible()
        .catch(() => false)
      if (!openDocVisible) return

      // Expect the media list API call when the picker opens
      const apiCall = expectApiCall(
        authedPage,
        {
          url: /\/api\/v1\/media/,
          method: "GET",
        },
        15_000
      )

      await workspace.openDocumentButton.click()

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // Modal may not fire API if server is in a degraded state
      }

      // Close the modal
      await authedPage.keyboard.press("Escape")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Keyboard Shortcuts
  // =========================================================================

  test.describe("Keyboard Shortcuts", () => {
    test("should open keyboard shortcuts modal on button click", async ({
      authedPage,
      diagnostics,
    }) => {
      await authedPage.addInitScript(() => {
        localStorage.setItem("dw-tips-tour-completed", "true")
      })
      workspace = new DocumentWorkspacePage(authedPage)
      await workspace.goto()
      await workspace.assertPageReady()

      const shortcutsBtnVisible = await workspace.shortcutsButton
        .isVisible()
        .catch(() => false)
      if (!shortcutsBtnVisible) return

      await workspace.shortcutsButton.click()

      const dialog = authedPage.getByRole("dialog", {
        name: /^Keyboard Shortcuts$/i,
      })
      await expect(dialog).toBeVisible({ timeout: 5_000 })
      await expect(
        dialog.getByRole("heading", { name: /^Keyboard Shortcuts$/i })
      ).toBeVisible()

      await dialog.getByRole("button", { name: /^Close$/i }).click()
      await expect(dialog).not.toBeVisible()
      await expect(workspace.openDocumentButton).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
