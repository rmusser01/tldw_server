import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

import {
  REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED,
  createRealServerWorkflowTldwConfig,
  createRealServerWorkflowStorageSeed,
  resolveRunnableChatModel,
  toSelectedModelId
} from "../../test-utils/real-server-workflows"

const readSource = (relativePath: string) =>
  readFileSync(path.join(process.cwd(), relativePath), "utf8")

describe("e2e harness readiness contracts", () => {
  it("keeps the Kanban forced-error fixture aligned with its canonical boundary", () => {
    const smokeSource = readSource("e2e/smoke/all-pages.spec.ts")

    expect(smokeSource).toMatch(
      /name: 'Kanban Playground',[\s\S]*?path: '\/kanban',[\s\S]*?routeId: 'kanban',[\s\S]*?routeLabel: 'Kanban'/
    )
    expect(smokeSource).not.toContain("routeId: 'kanban-playground'")
  })

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
    expect(helperSource).toContain("completed items")
    expect(helperSource).toContain("error items")
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
    expect(interactiveReviewSource).not.toContain(
      "waitForLoadState('networkidle'"
    )
    expect(interactiveReviewSource).not.toContain(
      'waitForLoadState("networkidle"'
    )
  })

  it("coverage maps every legacy real-server workflow and keeps only honest live cases", () => {
    const workflowSource = readSource("../test-utils/real-server-workflows.ts")
    const coverageMap = readSource(
      "../../Docs/superpowers/reviews/2026-08-25-real-server-workflow-coverage-map.md"
    )
    const rows = coverageMap
      .split("\n")
      .filter((line) => /^\|\s*\d+\s*\|/.test(line))
      .map((line) => line.split("|").map((cell) => cell.trim()))

    expect(rows).toHaveLength(17)
    const mappedTitles = rows.map((cells) => cells[2])
    expect(new Set(mappedTitles).size).toBe(17)
    const decisions = rows.map((cells) => cells[6])
    expect(
      decisions.every((decision) =>
        ["delete-redundant", "move-to-tier", "retain-live-gate"].includes(
          decision
        )
      )
    ).toBe(true)

    const inventoryMatch = workflowSource.match(
      /export const LEGACY_REAL_SERVER_WORKFLOW_TITLES = \[([\s\S]*?)\] as const/
    )
    expect(inventoryMatch).not.toBeNull()
    const inventoryTitles = Array.from(
      (inventoryMatch?.[1] ?? "").matchAll(/"([^"]+)"/g),
      (match) => match[1]
    )
    expect(inventoryTitles).toEqual(mappedTitles)

    const registrations = Array.from(
      workflowSource.matchAll(/^\s*test\(\s*\n?\s*"([^"]+)"/gm)
    )
    const registeredTitles = registrations.map((match) => match[1])
    const retainedTitles = rows
      .filter((cells) => cells[6] === "retain-live-gate")
      .map((cells) => cells[2])
    expect(registeredTitles).toEqual(retainedTitles)

    const mutationMarkers: Record<string, string> = {
      "chat -> save to notes -> open linked conversation":
        "createCharacterByName(",
      "chat -> save to flashcards -> review card": "createCharacterByName(",
      "media trash -> delete -> restore": ".setInputFiles(",
      "media ingestion -> analysis -> review -> re-analyze": ".setInputFiles("
    }
    for (const title of retainedTitles) {
      const registrationIndex = registrations.findIndex(
        (match) => match[1] === title
      )
      expect(registrationIndex).toBeGreaterThanOrEqual(0)
      const start = registrations[registrationIndex]?.index ?? -1
      expect(start).toBeGreaterThanOrEqual(0)
      const nextTest = registrations[registrationIndex + 1]?.index ?? -1
      const block = workflowSource.slice(
        start,
        nextTest >= 0 ? nextTest : workflowSource.length
      )
      const marker = mutationMarkers[title]
      const mutation = block.indexOf(marker)
      expect(mutation).toBeGreaterThanOrEqual(0)
      const afterMutation = block.slice(mutation)
      expect(afterMutation).not.toContain("skipOrThrow(")
      expect(afterMutation).not.toContain("test.skip(")
      expect(block).not.toMatch(/catch\s*\{\s*\}/)
      if (title.startsWith("chat ->")) {
        expect(block).toContain("selectTrackedCharacterFromRuntimeRail(")
        expect(block).not.toContain("setSelectedCharacterInStorage(")
      }
    }

    expect(workflowSource).toContain("resolveRunnableChatModel")
    expect(workflowSource).toContain('getByTestId("chat-character-controls-trigger")')
    expect(workflowSource).toContain('getByTestId("chat-character-controls-sheet")')
    expect(workflowSource).toContain('name: "Start tracked character chat"')
    expect(workflowSource).toContain('getByTestId("assistant-select-panel")')
    expect(workflowSource).toContain('getByTestId("media-search-input")')
    expect(workflowSource).toContain('expected_version=')
    expect(workflowSource).toContain(
      '`/media?id=${encodeURIComponent(String(mediaId))}`'
    )
    expect(workflowSource).not.toContain("getFirstModelId")
    expect(workflowSource).not.toContain("void waitForQuickIngestCompletion")
    expect(workflowSource).toContain('getByTestId("wizard-results-step")')
    expect(workflowSource).toContain(
      'getByRole("button", { name: "Next", exact: true })'
    )
    expect(workflowSource).toContain(
      'getByRole("heading", { name: /^Processing$/i })'
    )
    expect(workflowSource).not.toContain("const getRunState")
    expect(workflowSource).not.toContain(
      "if (/use defaults|start processing"
    )
    expect(workflowSource).toContain('[aria-current="step"]')
    expect(workflowSource).not.toContain(".test(clickedLabel)")
    expect(workflowSource).toContain('name: "Quick preset"')
    expect(workflowSource).toContain("selectQuickIngestQuickPreset(modal)")
    expect(workflowSource).toContain("await configureButton.click()")
    expect(workflowSource).toContain("closeQuickIngestModal(modal)")
    expect(workflowSource).toContain(
      'getByText("Moved to trash", { exact: true })'
    )
    expect(workflowSource).toContain(
      "getByText(expectedTitle, { exact: true })"
    )
    expect(workflowSource).toContain(
      'getByRole("dialog", { name: /^Generate$/i })'
    )
    expect(workflowSource).not.toContain("name: /Generate Analysis/i")
    expect(workflowSource).toContain("pollForPersistedMediaAnalysis(")
    expect(workflowSource).toContain(
      'getByText("Pending save", { exact: true })'
    )
    expect(workflowSource).not.toContain("getByText(token)")
    expect(workflowSource).not.toContain(
      'name: /Use defaults & process/i'
    )
    expect(workflowSource).not.toContain(
      "Use defaults & process|Start Processing|Run quick ingest|Configure|Review|Process|Ingest"
    )
    expect(workflowSource).toContain("Checking chat model readiness")
    expect(workflowSource).toContain("Composer did not dispatch the chat request")
    expect(workflowSource).toContain("first_message")
  })

  it("selects an explicitly configured chat provider using its callable provider key", () => {
    const selected = resolveRunnableChatModel({
      providers: [
        {
          name: "openai",
          chat_provider: "openai",
          is_configured: false,
          models: ["gpt-4.1-mini"]
        },
        {
          name: "custom_openai_api",
          chat_provider: "custom-openai-api",
          is_configured: true,
          models: ["local-uat-chat"]
        }
      ]
    })

    expect(selected).toEqual({
      id: "local-uat-chat",
      provider: "custom-openai-api"
    })
    expect(selected && toSelectedModelId(selected)).toBe(
      "tldw:custom-openai-api:local-uat-chat"
    )
  })

  it("opens linked notes conversations through the current overflow action", () => {
    const workflowSource = readSource("../test-utils/real-server-workflows.ts")

    expect(workflowSource).toContain('getByTestId("notes-overflow-menu-button")')
    expect(workflowSource).toMatch(
      /getByRole\("menuitem",\s*\{\s*name:\s*\/Open linked conversation\|Open conversation\/i/
    )
    expect(workflowSource).not.toContain(
      'getByRole("button", {\n          name: /Open conversation/i'
    )
  })

  it("runs the extension background liveness probe only on the extension surface", () => {
    const workflowSource = readSource("../test-utils/real-server-workflows.ts")

    expect(workflowSource).toMatch(
      /const waitForConnected = async \(\s*page: Page,\s*label: string,\s*surface: WorkflowDriver\["kind"\]\s*\)/
    )
    expect(workflowSource).toMatch(
      /if \(surface === "extension"\) \{[\s\S]*?pingBackgroundScript\(page\)/
    )
    expect(workflowSource).toContain(
      'waitForConnected(chatPage, "workflow-chat-notes", driver.kind)'
    )
    expect(workflowSource).toContain(
      'waitForConnected(page, "workflow-media-trash-view", driver.kind)'
    )
    expect(workflowSource).not.toMatch(
      /waitForConnected\([^\n]*"workflow-[^"]+"\)/
    )
  })

  it("opens retained media workflows from the media surface without an onboarding fallback", () => {
    const workflowSource = readSource("../test-utils/real-server-workflows.ts")

    expect(
      workflowSource.match(/await driver\.goto\(page, "\/media", \{/g)
    ).toHaveLength(2)
    expect(workflowSource).toContain("__tldwPendingQuickIngestOpen")
    expect(workflowSource).not.toContain("workflow-media-trash-ingest-fallback")
    expect(workflowSource).not.toContain("workflow-analysis-ingest-fallback")
  })

  it("saves chat flashcards through the visible message overflow action", () => {
    const workflowSource = readSource("../test-utils/real-server-workflows.ts")

    expect(workflowSource).toContain("clickMessageOverflowAction(")
    expect(workflowSource).toContain('getByRole("tab", { name: /^Manage$/i })')
    expect(workflowSource).toContain('getByRole("tab", { name: /^Study$/i })')
    expect(workflowSource).toContain('getByTestId("flashcards-review-all-due")')
    expect(workflowSource).toContain("cleanupFlashcard(")
    expect(workflowSource).not.toMatch(
      /assistantMessage\.getByRole\("button",\s*\{\s*name:\s*\/Save to Flashcards\/i/
    )
    expect(workflowSource).not.toContain(
      'getByRole("tab", { name: /Cards/i })'
    )
    expect(workflowSource).not.toContain(
      'getByRole("tab", { name: /Review/i })'
    )
  })

  it("shares intentional first-run and tour dismissal state across both live wrappers", () => {
    const storageSeed = createRealServerWorkflowStorageSeed(123)

    expect(storageSeed).toMatchObject({
      __tldw_first_run_complete: true,
      assistant_setup_dismissed: true,
      tldw_skip_landing_hub: true,
      quickIngestInspectorIntroDismissed: true,
      quickIngestOnboardingDismissed: true,
      "tldw:workflow:landing-config": {
        dismissedAt: 123
      }
    })
    expect(REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED).toHaveProperty(
      "playground-tour-completed",
      "true"
    )
    expect(REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED).toHaveProperty(
      "notes-tutorial-shown",
      "1"
    )
    expect(REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED).toHaveProperty(
      "tldw-tutorials"
    )

    const webWrapper = readSource("e2e/real-server-workflows.spec.ts")
    const extensionWrapper = readSource(
      "../extension/tests/e2e/real-server-workflows.spec.ts"
    )
    for (const wrapper of [webWrapper, extensionWrapper]) {
      expect(wrapper).toContain("createRealServerWorkflowStorageSeed")
      expect(wrapper).toContain("REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED")
    }
  })

  it("seeds a complete manual credential record on every live browser document", () => {
    expect(
      createRealServerWorkflowTldwConfig(
        "http://127.0.0.1:8000/",
        "test-api-key"
      )
    ).toEqual({
      serverUrl: "http://127.0.0.1:8000",
      apiKey: "test-api-key",
      authMode: "single-user",
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "http://127.0.0.1:8000"
    })

    const webWrapper = readSource("e2e/real-server-workflows.spec.ts")
    const extensionWrapper = readSource(
      "../extension/tests/e2e/real-server-workflows.spec.ts"
    )
    for (const wrapper of [webWrapper, extensionWrapper]) {
      expect(wrapper).toContain("createRealServerWorkflowTldwConfig")
    }
  })

  it("quiesces the WebUI page before disposable workflow fixtures are deleted", () => {
    const webWrapper = readSource("e2e/real-server-workflows.spec.ts")
    const extensionWrapper = readSource(
      "../extension/tests/e2e/real-server-workflows.spec.ts"
    )

    expect(webWrapper).toContain("await page.close()")
    expect(webWrapper).not.toContain("about:blank")
    expect(extensionWrapper).not.toContain("about:blank")
  })
})
