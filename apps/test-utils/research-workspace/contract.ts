import { ResearchWorkspaceParityPage } from "./page"
import type { ResearchWorkspaceParityContext } from "./types"

export async function runResearchWorkspaceParityContract(
  ctx: ResearchWorkspaceParityContext
): Promise<void> {
  const workspacePage = new ResearchWorkspaceParityPage(ctx.page)

  await workspacePage.goto(ctx.platform, ctx.optionsUrl)
  await workspacePage.waitForReady()

  await workspacePage.assertBaselinePanesVisible()
  await workspacePage.expectComposerVisibleWithoutPageScroll()
  await workspacePage.hideSourcesPane()
  await workspacePage.restoreSourcesPane()
  await workspacePage.hideStudioPane()
  await workspacePage.restoreStudioPane()

  await workspacePage.openOutputTypesSection()
  await workspacePage.openGeneratedOutputsSection()
  await workspacePage.seedDeterministicArtifact()

  await workspacePage.expectParityArtifactVisible()
  await workspacePage.expectArtifactActionButtons()

  await workspacePage.collapseGeneratedOutputsSection()
  await workspacePage.expectGeneratedOutputsSectionHidden()
  await workspacePage.openGeneratedOutputsSection()
  await workspacePage.expectParityArtifactVisible()
}
