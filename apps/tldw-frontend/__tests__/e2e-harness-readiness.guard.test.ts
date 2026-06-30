import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8")

describe("e2e harness readiness contracts", () => {
  it("keeps smoke and review harnesses off direct networkidle waits", () => {
    const smokeSource = readSource("e2e/smoke/all-pages.spec.ts")
    const reviewSource = readSource("e2e/review/parallel-review.spec.ts")

    expect(smokeSource).toContain("waitForAppShell")
    expect(reviewSource).toContain("waitForAppShell")
    expect(smokeSource).not.toContain("waitForLoadState('networkidle'")
    expect(reviewSource).not.toContain('waitForLoadState("networkidle"')
  })

  it("keeps BasePage state change checks on polling instead of fixed sleeps", () => {
    const basePageSource = readSource("e2e/utils/page-objects/BasePage.ts")

    expect(basePageSource).toContain("expect\n                .poll")
    expect(basePageSource).not.toContain("waitForTimeout(500)")
  })

  it("keeps journey helpers on explicit quick-ingest and stream readiness markers", () => {
    const helperSource = readSource("e2e/utils/journey-helpers.ts")

    expect(helperSource).toContain("wizard-results-step")
    expect(helperSource).toContain("quick-ingest-complete")
    expect(helperSource).toContain("/api/v1/media/ingest/jobs/")
    expect(helperSource).toContain("article[aria-label*='Assistant message']")
    expect(helperSource).toContain("Generating response")
    expect(helperSource).not.toContain("waitForTimeout(1_000)")
  })

  it("keeps the media review workflow off fixed sleeps", () => {
    const mediaReviewSource = readSource("e2e/workflows/media-review.spec.ts")

    expect(mediaReviewSource).not.toContain("waitForTimeout(")
    expect(mediaReviewSource).toContain("expect\n        .poll")
    expect(mediaReviewSource).not.toContain("const sleep =")
    expect(mediaReviewSource).not.toContain("setTimeout(resolve, 150)")
  })

  it("keeps refreshed journey workflows off blind timeout waits", () => {
    const characterJourneySource = readSource("e2e/workflows/journeys/character-chat.spec.ts")
    const ingestSearchChatSource = readSource("e2e/workflows/journeys/ingest-search-chat.spec.ts")
    const ingestEvaluateReviewSource = readSource("e2e/workflows/journeys/ingest-evaluate-review.spec.ts")

    expect(characterJourneySource).not.toContain("waitForTimeout(")
    expect(ingestSearchChatSource).not.toContain("waitForTimeout(")
    expect(ingestEvaluateReviewSource).not.toContain("waitForTimeout(")
  })

  it("keeps the media ingest workflow off blind timeout waits", () => {
    const mediaIngestSource = readSource("e2e/workflows/media-ingest.spec.ts")

    expect(mediaIngestSource).not.toContain("waitForTimeout(")
    expect(mediaIngestSource).not.toContain("waitForLoadState('networkidle'")
    expect(mediaIngestSource).not.toContain('waitForLoadState("networkidle"')
    expect(mediaIngestSource).not.toContain("setTimeout(resolve, 1400)")
  })

  it("keeps the collections stage 3 workflow on polling instead of local sleep wrappers", () => {
    const collectionsStage3Source = readSource("e2e/workflows/collections-stage3.spec.ts")

    expect(collectionsStage3Source).toContain("expect.poll(")
    expect(collectionsStage3Source).not.toContain("const sleep =")
  })

  it("keeps the media navigation UX verification on polling instead of timeout sleeps", () => {
    const mediaNavigationSource = readSource("e2e/workflows/media-navigation-ux-verification.spec.ts")

    expect(mediaNavigationSource).not.toContain("waitForTimeout(")
    expect(mediaNavigationSource).toContain(".poll(")
  })

  it("keeps the UX audit harness on visual-settle helpers instead of networkidle or sleeps", () => {
    const uxAuditSource = readSource("e2e/ux-audit/audit-v3.spec.ts")

    expect(uxAuditSource).toContain("waitForVisualSettle")
    expect(uxAuditSource).toContain("waitForAuditRenderableSurface")
    expect(uxAuditSource).not.toContain("waitForTimeout(")
    expect(uxAuditSource).not.toContain("waitForLoadState('networkidle'")
    expect(uxAuditSource).not.toContain('waitForLoadState("networkidle"')
  })

  it("keeps the Knowledge QA workflow on stage-driven waits instead of blind timeout sleeps", () => {
    const knowledgeQaSource = readSource("e2e/workflows/knowledge-qa.spec.ts")

    expect(knowledgeQaSource).not.toContain("waitForTimeout(")
    expect(knowledgeQaSource).toContain("Reranking results")
    expect(knowledgeQaSource).toContain("setTimeout(resolve, 6_500)")
  })

  it("keeps the next smoke harness slice off direct networkidle waits", () => {
    const aliasRollupSource = readSource("e2e/smoke/alias-rollup-capture.spec.ts")
    const stage4AxeSource = readSource("e2e/smoke/stage4-axe-high-risk-routes.spec.ts")
    const stage5GateSource = readSource("e2e/smoke/stage5-release-gate.spec.ts")

    expect(aliasRollupSource).toContain("waitForVisualSettle")
    expect(aliasRollupSource).not.toContain("waitForLoadState('networkidle'")
    expect(aliasRollupSource).not.toContain("waitForTimeout(")

    expect(stage4AxeSource).toContain("waitForAppShell")
    expect(stage4AxeSource).not.toContain('waitForLoadState("networkidle"')

    expect(stage5GateSource).toContain("waitForAppShell")
    expect(stage5GateSource).toContain("setTimeout(resolve, NAVIGATION_RETRY_WAIT_MS)")
    expect(stage5GateSource).not.toContain('waitForLoadState("networkidle"')
    expect(stage5GateSource).not.toContain("waitForTimeout(")
  })

  it("keeps the simple smoke evidence batch on app-shell or visual-settle helpers", () => {
    const invalidApiKeySource = readSource("e2e/smoke/invalid-api-key.spec.ts")
    const labelEvidenceSource = readSource("e2e/smoke/m1-2-label-evidence.spec.ts")
    const focusEvidenceSource = readSource("e2e/smoke/m3-2-a11y-focus-evidence.spec.ts")
    const stage1MatrixSource = readSource("e2e/smoke/stage1-route-matrix-capture.spec.ts")
    const mobileSidebarSource = readSource("e2e/smoke/stage4-mobile-sidebar.spec.ts")
    const aliasNaturalSource = readSource("e2e/smoke/alias-rollup-natural-capture.spec.ts")
    const stage6Stage1Source = readSource("e2e/smoke/stage6-interaction-stage1.spec.ts")
    const stage4AccessibilitySource = readSource("e2e/smoke/stage4-accessibility-controls.spec.ts")
    const stage7AudioSource = readSource("e2e/smoke/stage7-audio-regression.spec.ts")
    const stage3ResilienceSource = readSource("e2e/smoke/stage3-rendering-resilience.spec.ts")
    const routeContractSource = readSource("e2e/smoke/route-contract-stage2.spec.ts")

    expect(invalidApiKeySource).toContain("waitForAppShell")
    expect(invalidApiKeySource).not.toContain('waitForLoadState("networkidle"')

    expect(labelEvidenceSource).toContain("waitForVisualSettle")
    expect(labelEvidenceSource).not.toContain("waitForTimeout(")
    expect(labelEvidenceSource).not.toContain("waitForLoadState('networkidle'")

    expect(focusEvidenceSource).toContain("waitForVisualSettle")
    expect(focusEvidenceSource).not.toContain('waitForLoadState("networkidle"')

    expect(stage1MatrixSource).toContain("waitForAppShell")
    expect(stage1MatrixSource).not.toContain("waitForLoadState('networkidle'")

    expect(mobileSidebarSource).toContain("waitForAppShell")
    expect(mobileSidebarSource).not.toContain("waitForLoadState('networkidle'")

    expect(aliasNaturalSource).toContain("waitForVisualSettle")
    expect(aliasNaturalSource).not.toContain("waitForLoadState('networkidle'")

    expect(stage6Stage1Source).toContain("waitForAppShell")
    expect(stage6Stage1Source).not.toContain('waitForLoadState("networkidle"')

    expect(stage4AccessibilitySource).toContain("waitForAppShell")
    expect(stage4AccessibilitySource).not.toContain('waitForLoadState("networkidle"')

    expect(stage7AudioSource).toContain("waitForAppShell")
    expect(stage7AudioSource).not.toContain('waitForLoadState("networkidle"')

    expect(stage3ResilienceSource).toContain("waitForAppShell")
    expect(stage3ResilienceSource).not.toContain("waitForLoadState('networkidle'")

    expect(routeContractSource).toContain("waitForAppShell")
    expect(routeContractSource).not.toContain("waitForLoadState('networkidle'")
  })

  it("keeps the stage 6 interaction stage 2 smoke spec off fixed sleeps and direct networkidle waits", () => {
    const stage6Stage2Source = readSource("e2e/smoke/stage6-interaction-stage2.spec.ts")

    expect(stage6Stage2Source).toContain("waitForAppShell")
    expect(stage6Stage2Source).not.toContain("waitForTimeout(")
    expect(stage6Stage2Source).not.toContain("waitForLoadState('networkidle'")
    expect(stage6Stage2Source).not.toContain('waitForLoadState("networkidle"')
  })

  it("keeps writing playground in Stage 5 instead of the all-pages traversal", () => {
    const pageInventorySource = readSource("e2e/smoke/page-inventory.ts")
    const stage5GateSource = readSource("e2e/smoke/stage5-release-gate.spec.ts")

    expect(stage5GateSource).toContain('{ path: "/writing-playground", name: "Writing Playground" }')
    expect(pageInventorySource).toContain('path: "/writing-playground"')
    expect(pageInventorySource).toContain(
      'skip: "Covered in Stage 5 release gate; intermittently trips the global error boundary during full all-pages traversal in CI."'
    )
  })

  it("keeps the interactive review harness on app-shell readiness instead of direct networkidle waits", () => {
    const interactiveReviewSource = readSource("e2e/interactive-review.ts")

    expect(interactiveReviewSource).toContain("waitForAppShell")
    expect(interactiveReviewSource).not.toContain("waitForLoadState('networkidle'")
    expect(interactiveReviewSource).not.toContain('waitForLoadState("networkidle"')
  })
})
