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

test.describe("Synthetic generation handoff smoke", () => {
  test("retrieval generation hands off into the shared review queue", async ({
    page
  }) => {
    await seedAuth(page)

    let lastGenerationBody: Record<string, any> | null = null
    let queueItems = [
      {
        sample_id: "draft-1",
        recipe_kind: "rag_retrieval_tuning",
        provenance: "synthetic_from_corpus",
        review_state: "draft",
        sample_payload: {
          query: "What changed in the rollout?",
          relevant_media_ids: [{ id: "10", grade: 3 }]
        },
        sample_metadata: {
          generation_batch_id: "batch-123"
        },
        source_kind: "media_db"
      }
    ]

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
      const { pathname, searchParams } = url

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

      if (method === "POST" && pathname === "/api/v1/evaluations/synthetic/drafts/generate") {
        lastGenerationBody = (request.postDataJSON() as Record<string, any>) || null
        queueItems = [
          {
            sample_id: "draft-1",
            recipe_kind: "rag_retrieval_tuning",
            provenance: "synthetic_from_corpus",
            review_state: "draft",
            sample_payload: {
              query: "What changed in the rollout?",
              relevant_media_ids: [{ id: "10", grade: 3 }]
            },
            sample_metadata: {
              generation_batch_id: "batch-123"
            },
            source_kind: "media_db"
          }
        ]
        await fulfillJson(route, 200, {
          generation_batch_id: "batch-123",
          samples: queueItems,
          source_breakdown: {
            media_db: 1
          }
        })
        return
      }

      if (method === "GET" && pathname === "/api/v1/evaluations/synthetic/queue") {
        const recipeKind = searchParams.get("recipe_kind")
        const generationBatchId = searchParams.get("generation_batch_id")
        const filtered = queueItems.filter((sample) => {
          if (recipeKind && sample.recipe_kind !== recipeKind) return false
          if (
            generationBatchId &&
            sample.sample_metadata?.generation_batch_id !== generationBatchId
          ) {
            return false
          }
          return true
        })
        await fulfillJson(route, 200, {
          data: filtered,
          total: filtered.length
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

    await evaluations.selectRecipe("RAG Retrieval Tuning")
    await page.getByLabel("Media IDs").fill("10")
    await evaluations.generateSyntheticDrafts()

    await expect
      .poll(() => lastGenerationBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected synthetic generation request body to be captured"
      })
      .not.toBeNull()

    expect(lastGenerationBody).toMatchObject({
      recipe_kind: "rag_retrieval_tuning",
      corpus_scope: {
        media_ids: [10]
      },
      target_sample_count: 20
    })

    await expect(page.getByText("Synthetic review queue")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })
    await evaluations.assertSyntheticReviewBatch("batch-123")
    await expect(page.getByText("draft-1")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })
  })
})
