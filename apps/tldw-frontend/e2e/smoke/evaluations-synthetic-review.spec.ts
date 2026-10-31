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

test.describe("Synthetic review queue smoke", () => {
  test("reviews and promotes synthetic drafts from the shared queue", async ({
    page
  }) => {
    await seedAuth(page)

    let lastReviewBody: Record<string, any> | null = null
    let lastPromotionBody: Record<string, any> | null = null
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
          difficulty: "medium"
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
              corpus_scope: { sources: ["media_db", "notes"] }
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

      if (method === "GET" && pathname === "/api/v1/evaluations/synthetic/queue") {
        const recipeKind = searchParams.get("recipe_kind")
        const filtered = queueItems.filter((sample) => {
          if (recipeKind && sample.recipe_kind !== recipeKind) return false
          return true
        })
        await fulfillJson(route, 200, {
          data: filtered,
          total: filtered.length
        })
        return
      }

      if (
        method === "POST" &&
        /\/api\/v1\/evaluations\/synthetic\/queue\/[^/]+\/review$/.test(pathname)
      ) {
        lastReviewBody = (request.postDataJSON() as Record<string, any>) || null
        queueItems = queueItems.map((sample) =>
          sample.sample_id === "draft-1"
            ? {
                ...sample,
                review_state:
                  lastReviewBody?.action === "reject"
                    ? "rejected"
                    : "approved"
              }
            : sample
        )
        await fulfillJson(route, 200, {
          action_id: "action-1",
          sample_id: "draft-1",
          action: lastReviewBody?.action || "approve",
          reviewer_id: "user_123",
          notes: lastReviewBody?.notes || null,
          action_payload: lastReviewBody?.action_payload || {},
          resulting_review_state:
            lastReviewBody?.action === "reject" ? "rejected" : "approved",
          created_at: "2026-03-30T12:00:00Z"
        })
        return
      }

      if (method === "POST" && pathname === "/api/v1/evaluations/synthetic/promotions") {
        lastPromotionBody = (request.postDataJSON() as Record<string, any>) || null
        await fulfillJson(route, 200, {
          dataset_id: "dataset_synthetic_1",
          dataset_snapshot_ref: "dataset-snapshot-1",
          promotion_ids: ["promotion-1"],
          sample_count: Array.isArray(lastPromotionBody?.sample_ids)
            ? lastPromotionBody.sample_ids.length
            : 0
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
    await page
      .getByRole("button", { name: "Review synthetic retrieval drafts" })
      .click()

    await expect(page.getByText("Synthetic review queue")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })

    await page.getByLabel("Review notes draft-1").fill("Relevant and realistic retrieval sample.")
    await page.getByRole("button", { name: "Edit & approve" }).click()

    await expect
      .poll(() => lastReviewBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected review request body to be captured"
      })
      .not.toBeNull()

    expect(lastReviewBody).toMatchObject({
      action: "edit_and_approve",
      notes: "Relevant and realistic retrieval sample."
    })

    await page.getByLabel("Select draft-1").click()
    await page
      .getByLabel("Synthetic promoted dataset name")
      .fill("reviewed synthetic retrieval")
    await page.getByRole("button", { name: "Promote selected" }).click()

    await expect
      .poll(() => lastPromotionBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected promotion request body to be captured"
      })
      .not.toBeNull()

    expect(lastPromotionBody).toMatchObject({
      sample_ids: ["draft-1"],
      dataset_name: "reviewed synthetic retrieval"
    })

    await expect(page.getByText("dataset_synthetic_1 (1)")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })
  })
})
