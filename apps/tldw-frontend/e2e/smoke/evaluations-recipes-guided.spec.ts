import type { Route } from "@playwright/test"
import { test, expect, seedAuth, SMOKE_LOAD_TIMEOUT } from "./smoke.setup"
import { waitForAppShell } from "../utils/helpers"

const LOAD_TIMEOUT = SMOKE_LOAD_TIMEOUT

const fulfillJson = async (route: Route, status: number, data: unknown) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(data)
  })
}

test.describe("Evaluations recipes guided smoke", () => {
  test("guided recipe flow validates, launches, and renders a report from mocked APIs", async ({
    page,
    diagnostics
  }) => {
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
            recipe_id: "summarization_quality",
            recipe_version: "1",
            name: "Summarization Quality",
            description: "Compare summarization candidates.",
            supported_modes: ["labeled", "unlabeled"],
            tags: ["summarization"]
          },
          {
            recipe_id: "embeddings_model_selection",
            recipe_version: "1",
            name: "Embeddings Model Selection",
            description: "Compare embedding candidates.",
            supported_modes: ["labeled", "unlabeled"],
            tags: ["embeddings"]
          }
        ])
        return
      }

      if (
        method === "GET" &&
        /\/api\/v1\/evaluations\/recipes\/[^/]+\/launch-readiness$/.test(pathname)
      ) {
        const recipeId = pathname.split("/").at(-2) || "summarization_quality"
        await fulfillJson(route, 200, {
          recipe_id: recipeId,
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
        const dataset = Array.isArray(lastValidateBody?.dataset) ? lastValidateBody?.dataset : []
        await fulfillJson(route, 200, {
          valid: true,
          errors: [],
          dataset_mode: dataset.some((sample) => sample.expected) ? "labeled" : "unlabeled",
          sample_count: dataset.length,
          review_sample: {
            required: false,
            sample_size: 0,
            sample_ids: dataset.map((_: unknown, index: number) => `sample-${index}`)
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
          run_id: "recipe-run-smoke-1",
          recipe_id: "summarization_quality",
          recipe_version: "1",
          status: "completed",
          created_at: "2026-03-29T18:00:00Z",
          metadata: {}
        })
        return
      }

      if (
        method === "GET" &&
        pathname === "/api/v1/evaluations/recipe-runs/recipe-run-smoke-1/report"
      ) {
        await fulfillJson(route, 200, {
          run: {
            run_id: "recipe-run-smoke-1",
            recipe_id: "summarization_quality",
            recipe_version: "1",
            status: "completed",
            created_at: "2026-03-29T18:00:00Z",
            metadata: {
              recipe_report: {
                candidates: [
                  {
                    candidate_id: "openai:gpt-4.1-mini",
                    candidate_run_id: "openai:gpt-4.1-mini",
                    provider: "openai",
                    model: "gpt-4.1-mini",
                    metrics: {
                      grounding: 0.94,
                      coverage: 0.9,
                      usefulness: 0.86,
                      quality_score: 0.91
                    }
                  }
                ]
              }
            }
          },
          confidence_summary: {
            kind: "aggregate",
            confidence: 0.88,
            sample_count: 1
          },
          recommendation_slots: {
            best_overall: {
              candidate_run_id: "openai:gpt-4.1-mini",
              reason_code: "highest_quality_score",
              explanation: "Best grounding and coverage balance."
            },
            best_cheap: {
              candidate_run_id: null,
              reason_code: "not_available",
              explanation: "No cheaper candidate in this run."
            },
            best_local: {
              candidate_run_id: null,
              reason_code: "not_available",
              explanation: "No local candidate in this run."
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

    await page.goto("/evaluations", {
      waitUntil: "domcontentloaded",
      timeout: LOAD_TIMEOUT
    })
    await waitForAppShell(page, LOAD_TIMEOUT)

    await expect(page.getByTestId("evaluations-page-title")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })
    await expect(page.getByText("Guided setup")).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(page.getByText("Inline dataset JSON")).toHaveCount(0)

    await page.getByLabel("Source text 1").fill("Guided smoke transcript for alpha rollout.")
    await page.getByLabel("Reference summary 1 (optional)").fill("Alpha rollout completed.")
    await page
      .getByLabel("Candidate models")
      .fill("openai:gpt-4.1-mini\nlocal:mistral-small")
    await page.getByLabel("Judge provider").fill("openai")
    await page.getByLabel("Judge model").fill("gpt-4.1-mini")

    await page.getByRole("button", { name: "Validate dataset" }).click()
    await expect(page.getByText("Dataset format is valid.")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })

    await expect
      .poll(() => lastValidateBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected validation request body to be captured"
      })
      .not.toBeNull()
    expect(lastValidateBody?.dataset).toEqual([
      {
        input: "Guided smoke transcript for alpha rollout.",
        expected: "Alpha rollout completed."
      }
    ])

    await page.getByRole("button", { name: "Run recipe" }).click()
    await expect(page.getByText("Current run")).toBeVisible({ timeout: LOAD_TIMEOUT })
    await expect(page.getByText("Best grounding and coverage balance.")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })
    await expect(
      page.locator(".font-medium", { hasText: "gpt-4.1-mini" }).first()
    ).toBeVisible({ timeout: LOAD_TIMEOUT })

    await expect
      .poll(() => lastRunBody, {
        timeout: LOAD_TIMEOUT,
        message: "Expected run request body to be captured"
      })
      .not.toBeNull()
    expect(lastRunBody?.run_config?.candidate_model_ids).toEqual([
      "openai:gpt-4.1-mini",
      "local:mistral-small"
    ])

    await page.getByRole("button", { name: /Advanced JSON/ }).click()
    await expect(page.getByText("Inline dataset JSON")).toBeVisible({
      timeout: LOAD_TIMEOUT
    })

    expect(diagnostics.pageErrors).toEqual([])
  })
})
