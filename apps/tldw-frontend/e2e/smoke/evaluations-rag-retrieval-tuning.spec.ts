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

test.describe("RAG retrieval tuning recipe smoke", () => {
  test("validates and launches from guided controls", async ({ page }) => {
    await seedAuth(page)

    let lastValidateBody: Record<string, any> | null = null
    let lastRunBody: Record<string, any> | null = null

    await page.route(/\/api\/v1\/health(?:\/.*)?$/, async (route) => {
      await fulfillJson(route, 200, {
        status: "ok",
        auth_mode: "single_user",
        test_api_key: "local-smoke-test-auth-token"
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
            recipe_id: "rag_retrieval_tuning",
            recipe_version: "1",
            name: "RAG Retrieval Tuning",
            description: "Tune retrieval candidates.",
            supported_modes: ["labeled", "unlabeled"],
            tags: ["rag", "retrieval"],
            launchable: true,
            capabilities: {
              corpus_sources: ["media_db", "notes"],
              candidate_creation_modes: ["auto_sweep", "manual"],
              graded_relevance_scale: { min: 0, max: 3 }
            },
            default_run_config: {
              candidate_creation_mode: "auto_sweep",
              corpus_scope: { sources: ["media_db", "notes"] },
              weak_supervision_budget: {
                review_sample_fraction: 0.2,
                max_review_samples: 25,
                min_review_samples: 3,
                synthetic_query_limit: 20
              }
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
          recipe_id: "rag_retrieval_tuning",
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
          run_id: "rag-recipe-run-1",
          recipe_id: "rag_retrieval_tuning",
          recipe_version: "1",
          status: "completed",
          created_at: "2026-03-29T18:00:00Z",
          metadata: {}
        })
        return
      }

      if (
        method === "GET" &&
        pathname === "/api/v1/evaluations/recipe-runs/rag-recipe-run-1/report"
      ) {
        await fulfillJson(route, 200, {
          run: {
            run_id: "rag-recipe-run-1",
            recipe_id: "rag_retrieval_tuning",
            recipe_version: "1",
            status: "completed",
            created_at: "2026-03-29T18:00:00Z",
            metadata: {
              recipe_report: {
                candidates: [
                  {
                    candidate_id: "manual-hybrid",
                    candidate_run_id: "manual-hybrid",
                    metrics: {
                      retrieval_quality_score: 0.88,
                      pre_rerank_recall_at_k: 0.91,
                      post_rerank_ndcg_at_k: 0.84
                    },
                    latency_ms: 123,
                    sample_count: 1,
                    is_local: true
                  }
                ]
              }
            }
          },
          confidence_summary: {
            kind: "aggregate",
            confidence: 0.82,
            sample_count: 1
          },
          recommendation_slots: {
            best_overall: {
              candidate_run_id: "manual-hybrid",
              reason_code: "highest_retrieval_quality_score",
              explanation: "Selected 'manual-hybrid' for best_overall with retrieval quality 0.880."
            },
            best_cheap: {
              candidate_run_id: null,
              reason_code: "not_available",
              explanation: "No recommendation is available for 'best_cheap'."
            },
            best_local: {
              candidate_run_id: "manual-hybrid",
              reason_code: "best_local_candidate",
              explanation: "Selected 'manual-hybrid' for best_local with retrieval quality 0.880."
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

      if (method === "POST" && pathname === "/api/v1/evaluations/history") {
        await fulfillJson(route, 200, {
          total_count: 0,
          items: [],
          aggregations: {}
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

    await evaluations.selectRecipe("RAG Retrieval Tuning")

    await page.getByLabel("Media IDs").fill("10, 12")
    await page.getByLabel("Note IDs").fill("note-7")
    await page.getByLabel("Indexing fixed").check()
    await page.getByLabel("Candidate creation mode").selectOption("manual")
    await page.getByLabel("Candidate ID 1").fill("manual-hybrid")
    await page.getByLabel("Search mode 1").selectOption("hybrid")
    await page.getByLabel("Top K 1").fill("12")
    await page.getByLabel("Hybrid alpha 1").fill("0.6")
    await page.getByLabel("Reranking strategy 1").selectOption("cross_encoder")
    await page.getByLabel("Chunking preset 1").selectOption("compact")
    await page.getByLabel("Sample ID 1").fill("q-42")
    await page.getByLabel("Query text 1").fill("find the beta architecture note")
    await page.getByLabel("Relevant media targets 1 (optional)").fill("10:3\n12:1")
    await page.getByLabel("Relevant note targets 1 (optional)").fill("note-7:2")
    await page.getByLabel("Relevant spans 1 (optional)").fill("media_db,10,0,42,3")

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
        candidate_creation_mode: "manual",
        corpus_scope: {
          sources: ["media_db", "notes"],
          media_ids: [10, 12],
          note_ids: ["note-7"],
          indexing_fixed: true
        }
      },
      dataset: [
        {
          sample_id: "q-42",
          query: "find the beta architecture note"
        }
      ]
    })

    await evaluations.runEvaluation()
    await evaluations.assertRecipeRunOutcome()

    await expect
      .poll(() => lastRunBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected run request body to be captured"
      })
      .not.toBeNull()

    expect(lastRunBody).toMatchObject({
      run_config: {
        candidate_creation_mode: "manual",
        corpus_scope: {
          sources: ["media_db", "notes"],
          media_ids: [10, 12],
          note_ids: ["note-7"],
          indexing_fixed: true
        },
        candidates: [
          {
            candidate_id: "manual-hybrid",
            retrieval_config: {
              search_mode: "hybrid",
              top_k: 12,
              hybrid_alpha: 0.6,
              reranking_strategy: "cross_encoder"
            },
            indexing_config: {
              chunking_preset: "compact"
            }
          }
        ]
      }
    })
  })
})
