import type { Route } from "@playwright/test"
import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from "./smoke.setup"
import { waitForAppShell } from "../utils/helpers"
import { EvaluationsPage } from "../utils/page-objects/EvaluationsPage"

const LOAD_TIMEOUT = SMOKE_LOAD_TIMEOUT

const fulfillJson = async (route: Route, status: number, data: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(data)
  })
}

test.describe("RAG answer quality recipe smoke", () => {
  test("validates and launches from guided controls", async ({ page }) => {
    await seedAuth(page)

    let lastValidateBody: Record<string, any> | null = null
    let lastRunBody: Record<string, any> | null = null

    await page.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
      await fulfillJson(route, 200, {
        status: "ok",
        auth_mode: "single_user",
        test_api_key: "THIS-IS-A-SECURE-KEY-123-LOCAL-TEST"
      })
    })

    await page.route(/\/api\/v1\/evaluations(?:\/.*)?(?:\?.*)?$/, async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const method = request.method().toUpperCase()
      const { pathname } = url

      if (method === "GET" && pathname === "/api/v1/evaluations/recipes") {
        await fulfillJson(route, 200, [
          {
            recipe_id: "rag_answer_quality",
            recipe_version: "1",
            name: "RAG Answer Quality",
            description: "Compare answer-generation candidates against fixed or live RAG anchors.",
            supported_modes: ["labeled", "unlabeled"],
            tags: ["rag", "generation"],
            launchable: true,
            capabilities: {
              evaluation_modes: ["fixed_context", "live_end_to_end"],
              supervision_modes: ["rubric", "reference_answer", "pairwise", "mixed"],
              candidate_dimensions: [
                "generation_model",
                "prompt_variant",
                "formatting_citation_mode"
              ]
            },
            default_run_config: {
              evaluation_mode: "fixed_context",
              supervision_mode: "rubric"
            }
          }
        ])
        return
      }

      if (
        method === "GET" &&
        /\/api\/v1\/evaluations\/recipes\/[^/]+\/launch-readiness$/.test(pathname)
      ) {
        await fulfillJson(route, 200, {
          recipe_id: "rag_answer_quality",
          ready: true,
          can_enqueue_runs: true,
          can_reuse_completed_runs: true,
          runtime_checks: {
            recipe_run_worker_enabled: true
          },
          message: null
        })
        return
      }

      if (
        method === "POST" &&
        /\/api\/v1\/evaluations\/recipes\/[^/]+\/validate-dataset$/.test(pathname)
      ) {
        lastValidateBody = (request.postDataJSON() as Record<string, any>) || null
        await fulfillJson(route, 200, {
          valid: true,
          errors: [],
          dataset_mode: "labeled",
          sample_count: Array.isArray(lastValidateBody?.dataset) ? lastValidateBody?.dataset.length : 0,
          review_sample: {
            required: false,
            sample_size: 0,
            sample_ids: []
          }
        })
        return
      }

      if (
        method === "POST" &&
        /\/api\/v1\/evaluations\/recipes\/[^/]+\/runs$/.test(pathname)
      ) {
        lastRunBody = (request.postDataJSON() as Record<string, any>) || null
        await fulfillJson(route, 202, {
          run_id: "rag-answer-quality-run-1",
          recipe_id: "rag_answer_quality",
          recipe_version: "1",
          status: "completed",
          created_at: "2026-03-29T18:00:00Z",
          metadata: {}
        })
        return
      }

      if (
        method === "GET" &&
        pathname === "/api/v1/evaluations/recipe-runs/rag-answer-quality-run-1/report"
      ) {
        await fulfillJson(route, 200, {
          run: {
            run_id: "rag-answer-quality-run-1",
            recipe_id: "rag_answer_quality",
            recipe_version: "1",
            status: "completed",
            created_at: "2026-03-29T18:00:00Z",
            metadata: {
              recipe_report: {
                candidates: [
                  {
                    candidate_id: "fixed-best",
                    candidate_run_id: "fixed-best",
                    metrics: {
                      quality_score: 0.9,
                      grounding: 0.93,
                      answer_relevance: 0.87,
                      format_style_compliance: 0.84,
                      abstention_behavior: 0.89
                    }
                  }
                ]
              }
            }
          },
          confidence_summary: {
            kind: "aggregate",
            confidence: 0.84,
            sample_count: 1
          },
          recommendation_slots: {
            best_overall: {
              candidate_run_id: "fixed-best",
              reason_code: "highest_quality_score",
              explanation: "Selected 'fixed-best' for strongest grounded answer quality."
            },
            best_cheap: {
              candidate_run_id: null,
              reason_code: "not_available",
              explanation: "No recommendation is available for 'best_cheap'."
            },
            best_local: {
              candidate_run_id: null,
              reason_code: "not_available",
              explanation: "No recommendation is available for 'best_local'."
            }
          }
        })
        return
      }

      if (method === "GET" && pathname === "/api/v1/evaluations/datasets") {
        await fulfillJson(route, 200, {
          object: "list",
          data: [],
          total: 0
        })
        return
      }

      if (method === "GET" && pathname === "/api/v1/evaluations/") {
        await fulfillJson(route, 200, {
          object: "list",
          data: []
        })
        return
      }

      if (method === "GET" && pathname === "/api/v1/evaluations/history") {
        await fulfillJson(route, 200, {
          data: []
        })
        return
      }

      if (method === "GET" && pathname === "/api/v1/evaluations/webhooks") {
        await fulfillJson(route, 200, [])
        return
      }

      await fulfillJson(route, 200, {})
    })

    await page.route(/\/api\/v1\/notifications(?:\/.*)?(?:\?.*)?$/, async (route) => {
      const method = route.request().method().toUpperCase()
      if (method === "GET") {
        await fulfillJson(route, 200, {
          notifications: [],
          unread_count: 0
        })
        return
      }
      await fulfillJson(route, 200, { success: true })
    })

    const evaluations = new EvaluationsPage(page)
    await evaluations.goto()
    await waitForAppShell(page, LOAD_TIMEOUT)
    await evaluations.assertPageReady()

    await evaluations.selectRecipe("RAG Answer Quality")

    await page.getByLabel("Context snapshot reference").fill("ctx-release-2026-03-29")
    await page.getByLabel("Supervision mode").selectOption("mixed")
    await page.getByLabel("Candidate ID 1").fill("fixed-best")
    await page.getByLabel("Generation model 1").fill("openai:gpt-4.1-mini")
    await page.getByLabel("Prompt variant 1").fill("cite-v2")
    await page.getByLabel("Formatting/citation mode 1").selectOption("markdown_citations")
    await page.getByLabel("Sample ID 1").fill("sample-42")
    await page.getByLabel("Query text 1").fill("What changed in the rollout?")
    await page.getByLabel("Expected behavior 1").selectOption("hedge")
    await page
      .getByLabel("Retrieved contexts 1")
      .fill("Doc A :: Rollout completed on Friday.\nDoc B :: Beta remained invite-only.")
    await page
      .getByLabel("Reference answer 1 (optional)")
      .fill("The rollout completed on Friday, while beta access stayed limited.")

    await evaluations.validateCurrentRecipe()
    await expect(page.getByText("Dataset format is valid.")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })

    await expect
      .poll(() => lastValidateBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected validation request body to be captured"
      })
      .not.toBeNull()

    expect(lastValidateBody).toMatchObject({
      run_config: {
        evaluation_mode: "fixed_context",
        supervision_mode: "mixed",
        context_snapshot_ref: "ctx-release-2026-03-29",
        candidates: [
          {
            candidate_id: "fixed-best",
            generation_model: "openai:gpt-4.1-mini",
            prompt_variant: "cite-v2",
            formatting_citation_mode: "markdown_citations"
          }
        ]
      },
      dataset: [
        {
          sample_id: "sample-42",
          query: "What changed in the rollout?",
          expected_behavior: "hedge",
          retrieved_contexts: [
            { content: "Doc A :: Rollout completed on Friday." },
            { content: "Doc B :: Beta remained invite-only." }
          ],
          reference_answer:
            "The rollout completed on Friday, while beta access stayed limited."
        }
      ]
    })

    await evaluations.runEvaluation()

    await expect
      .poll(() => lastRunBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected run request body to be captured"
      })
      .not.toBeNull()

    expect(lastRunBody).toMatchObject(lastValidateBody as Record<string, any>)
    await evaluations.assertRecipeRunOutcome()
    await expect(
      page.getByText("Selected 'fixed-best' for strongest grounded answer quality.")
    ).toBeVisible({ timeout: LOAD_TIMEOUT })
  })
})
