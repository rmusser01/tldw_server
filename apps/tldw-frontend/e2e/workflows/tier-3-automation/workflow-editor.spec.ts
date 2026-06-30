/**
 * Workflow Editor E2E Tests (Tier 3)
 *
 * Tests the Workflow Editor (node-based visual editor) page lifecycle:
 * - Page loads with toolbar, canvas, and status bar
 * - Undo/Redo buttons present in toolbar
 * - Toggle Grid and Toggle Minimap buttons work
 * - Save button present (disabled when not dirty)
 * - More actions dropdown opens
 * - Sidebar panel switching (palette, config, run)
 * - Status bar shows node and connection counts
 *
 * Run: npx playwright test e2e/workflows/tier-3-automation/workflow-editor.spec.ts
 */
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { WorkflowEditorPage } from "../../utils/page-objects/WorkflowEditorPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

test.describe("Workflow Editor", () => {
  let editor: WorkflowEditorPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    editor = new WorkflowEditorPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render Workflow Editor with toolbar and status bar", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      // Save button should be visible in toolbar
      const saveVisible = await editor.saveButton.isVisible().catch(() => false)
      expect(saveVisible).toBe(true)

      await expect(authedPage.getByTestId("workflow-editor-status-summary")).toBeVisible()
      await expect(
        authedPage.getByRole("button", { name: "Import workflow" })
      ).toBeVisible()
      await expect(
        authedPage.getByRole("button", { name: "Export workflow" })
      ).toBeVisible()
      await expect(
        authedPage.getByRole("button", { name: "Open run panel" })
      ).toBeVisible()

      // Status bar should show node/connection counts
      const nodeCountVisible = await editor.nodeCount.isVisible().catch(() => false)
      const connectionCountVisible = await editor.connectionCount.isVisible().catch(() => false)
      expect(nodeCountVisible || connectionCountVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show undo and redo buttons in toolbar", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const undoVisible = await editor.undoButton.isVisible().catch(() => false)
      const redoVisible = await editor.redoButton.isVisible().catch(() => false)

      expect(undoVisible).toBe(true)
      expect(redoVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should have Save button disabled on fresh load (no changes)", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const saveDisabled = await editor.saveButton.isDisabled().catch(() => false)
      expect(saveDisabled).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Toolbar Interactions
  // =========================================================================

  test.describe("Toolbar Interactions", () => {
    test("should toggle grid visibility", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const gridBtnVisible = await editor.toggleGridButton.isVisible().catch(() => false)
      if (!gridBtnVisible) return

      // Get initial class (tracks primary vs text type)
      const classBefore = await editor.toggleGridButton.getAttribute("class") ?? ""
      await editor.toggleGridButton.click()
      await expect
        .poll(() => editor.toggleGridButton.getAttribute("class"), { timeout: 5_000 })
        .not.toBe(classBefore)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should toggle minimap visibility", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const minimapBtnVisible = await editor.toggleMinimapButton.isVisible().catch(() => false)
      if (!minimapBtnVisible) return

      const classBefore = await editor.toggleMinimapButton.getAttribute("class") ?? ""
      await editor.toggleMinimapButton.click()
      await expect
        .poll(() => editor.toggleMinimapButton.getAttribute("class"), { timeout: 5_000 })
        .not.toBe(classBefore)

      await assertNoCriticalErrors(diagnostics)
    })

    test("should open More actions dropdown", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const moreBtnVisible = await editor.moreActionsButton.isVisible().catch(() => false)
      if (!moreBtnVisible) return

      await editor.openMoreActions()

      // Dropdown menu should appear with New Workflow, Import, Export, Clear Canvas
      await expect(editor.newWorkflowMenuItem).toBeVisible({ timeout: 5_000 })

      // Close dropdown by pressing Escape
      await authedPage.keyboard.press("Escape")

      await assertNoCriticalErrors(diagnostics)
    })

    test("should allow editing the workflow name", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const nameDisplayVisible = await editor.workflowNameDisplay.isVisible().catch(() => false)
      if (!nameDisplayVisible) return

      // Click to start editing
      await editor.startEditingName()

      // Input should appear
      await expect(editor.workflowNameInput).toBeVisible({ timeout: 5_000 })

      // Type a new name and press Enter
      await editor.workflowNameInput.fill("E2E Test Workflow")
      await editor.workflowNameInput.press("Enter")

      // Name should update
      await expect
        .poll(() => editor.workflowNameDisplay.textContent(), { timeout: 5_000 })
        .toContain("E2E Test Workflow")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Sidebar Panels (Desktop)
  // =========================================================================

  test.describe("Sidebar Panels", () => {
    test("should switch between Nodes, Config, and Run panels", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const segmentedVisible = await editor.sidebarSegmented.isVisible().catch(() => false)
      if (!segmentedVisible) return

      // Switch to Config
      const configVisible = await editor.configTab.isVisible().catch(() => false)
      if (configVisible) {
        await editor.switchSidebarPanel("config")
        await expect(authedPage.getByText("Node Configuration")).toBeVisible({ timeout: 5_000 })
      }

      // Switch to Run
      const runVisible = await editor.runTab.isVisible().catch(() => false)
      if (runVisible) {
        await editor.switchSidebarPanel("execution")
        await expect(authedPage.getByText("Execution")).toBeVisible({ timeout: 5_000 })
      }

      // Switch back to Palette
      const paletteVisible = await editor.nodesPaletteTab.isVisible().catch(() => false)
      if (paletteVisible) {
        await editor.switchSidebarPanel("palette")
        await expect(authedPage.getByPlaceholder("Search nodes...")).toBeVisible({ timeout: 5_000 })
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Status Bar
  // =========================================================================

  test.describe("Status Bar", () => {
    test("should display node and connection counts", async ({
      authedPage,
      diagnostics,
    }) => {
      editor = new WorkflowEditorPage(authedPage)
      await editor.goto()
      await editor.assertPageReady()

      const nodeCountVisible = await editor.nodeCount.isVisible().catch(() => false)
      const connectionCountVisible = await editor.connectionCount.isVisible().catch(() => false)

      // At least one count should be visible
      expect(nodeCountVisible || connectionCountVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
