/**
 * Audio Studio E2E Tests (Tier 2)
 *
 * Run: bunx playwright test e2e/workflows/tier-2-features/audio-studio.spec.ts
 */
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { AudioStudioPage, type AudioStudioWorkflow } from "../../utils/page-objects/AudioStudioPage"
import { seedAuth } from "../../utils/helpers"

test.describe("Audio Studio", () => {
  let studio: AudioStudioPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    studio = new AudioStudioPage(page)
  })

  test("renders the canonical route with first-class workflows", async ({
    authedPage,
    diagnostics,
  }) => {
    studio = new AudioStudioPage(authedPage)
    await studio.goto()
    await studio.assertPageReady()

    await expect(studio.heading).toBeVisible()
    await expect(studio.projectTitleInput).toBeVisible()
    await expect(studio.newProjectButton).toBeVisible()
    await expect(studio.generationPanel).toBeVisible()
    await expect(studio.renderExportPanel).toBeVisible()
    await expect(studio.timelinePanel).toBeVisible()

    for (const workflow of ["narration", "podcast", "briefing", "music"] as AudioStudioWorkflow[]) {
      await expect(studio.workflowTab(workflow)).toBeVisible()
    }

    await studio.expectWorkflowControls("narration")
    await assertNoCriticalErrors(diagnostics)
  })

  test("switches between Narration, Podcast, Briefing, and Music controls", async ({
    authedPage,
    diagnostics,
  }) => {
    studio = new AudioStudioPage(authedPage)
    await studio.goto()
    await studio.assertPageReady()

    for (const workflow of ["podcast", "briefing", "music", "narration"] as AudioStudioWorkflow[]) {
      await studio.switchWorkflow(workflow)
      await studio.expectWorkflowControls(workflow)
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("opens workflow-specific URLs", async ({ authedPage, diagnostics }) => {
    studio = new AudioStudioPage(authedPage)

    for (const workflow of ["narration", "podcast", "briefing", "music"] as AudioStudioWorkflow[]) {
      await studio.goto(workflow)
      await studio.assertPageReady()
      await expect(studio.workflowTab(workflow)).toHaveAttribute("aria-selected", "true")
      await studio.expectWorkflowControls(workflow)
    }

    await assertNoCriticalErrors(diagnostics)
  })

  test("keeps the legacy Audiobook route compatible with Narration", async ({
    authedPage,
    diagnostics,
  }) => {
    studio = new AudioStudioPage(authedPage)
    await studio.gotoCompatibilityRoute()
    await studio.assertPageReady()

    await expect(authedPage).toHaveURL(/\/audio-studio\?workflow=narration/)
    await expect(studio.workflowTab("narration")).toHaveAttribute("aria-selected", "true")
    await studio.expectWorkflowControls("narration")

    await assertNoCriticalErrors(diagnostics)
  })
})
