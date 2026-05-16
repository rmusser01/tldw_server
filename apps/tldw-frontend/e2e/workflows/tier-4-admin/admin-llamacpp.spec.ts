/**
 * Admin Llama.cpp E2E Tests (Tier 4)
 *
 * Tests the /admin/llamacpp page:
 * - Page loads without critical errors
 * - Llama.cpp Admin heading or admin guard visible
 * - Guided readiness, inventory, and launch panels
 * - Start Server calls start-by-model with stable model_id
 * - Use this in Chat is explicit and calls provider wiring API
 *
 * Run: npx playwright test e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts
 */
import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { AdminPage } from "../../utils/page-objects"
import { seedAuth } from "../../utils/helpers"

const fulfillJson = async (
  route: Route,
  data: unknown,
  status = 200
): Promise<void> => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(data),
  })
}

const stoppedStatus = {
  state: "stopped",
  model: null,
  port: 8080,
  backend: "llamacpp",
}

const runningStatus = {
  state: "running",
  model: "toy-7b-q4_k_m.gguf",
  port: 8080,
  backend: "llamacpp",
}

const mockConfig = {
  saved_config: {
    enabled: true,
    executable_path: "/opt/llama-server",
    models_dir: "/srv/models/gguf",
    default_host: "127.0.0.1",
    default_port: 8080,
    default_threads: 8,
    default_n_gpu_layers: 0,
    default_ctx_size: 4096,
    allow_unvalidated_args: false,
    allow_cli_secrets: false,
    port_autoselect: true,
    port_probe_max: 10,
    allowed_paths: ["/srv/models"],
    registered_model_paths: [],
    log_output_file: null,
  },
  active_config: {
    handler_configured: false,
    enabled: null,
    executable_path: null,
    models_dir: null,
    default_host: null,
    default_port: null,
    active_model: null,
    active_host: null,
    active_port: null,
    active_pid: null,
  },
  restart_required: true,
  restart_reasons: ["handler_not_configured"],
  env_overrides: {
    models_dir: true,
  },
  warnings: ["Saved config is loaded on API server restart."],
}

const mockInventory = {
  models: [
    {
      model_id: "gguf:toy-model-id",
      display_name: "Toy 7B Q4_K_M",
      basename: "toy-7b-q4_k_m.gguf",
      source: "models_dir",
      path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
      size_bytes: 4_200_000_000,
      modified_at: "2026-05-15T10:00:00Z",
      metadata: {
        quantization: "Q4_K_M",
        parameter_hint: "7B",
        context_hint: 4096,
      },
      warnings: ["Metadata is filename-derived."],
    },
  ],
  warnings: [],
  scan_limited: false,
}

const mockHardware = {
  ram_total_bytes: 16_000_000_000,
  ram_available_bytes: 8_000_000_000,
  cpu_count: 8,
  gpus: [],
  warnings: ["GPU probe unavailable."],
}

async function mockLlamacppAdminFlow(page: Page) {
  let currentStatus = stoppedStatus
  let startRequestBody: unknown = null
  let useInChatCalled = false

  await page.route("**/api/v1/llamacpp/config", async (route) => {
    await fulfillJson(route, mockConfig)
  })

  await page.route("**/api/v1/llamacpp/status", async (route) => {
    await fulfillJson(route, currentStatus)
  })

  await page.route("**/api/v1/llamacpp/inventory", async (route) => {
    await fulfillJson(route, mockInventory)
  })

  await page.route("**/api/v1/llamacpp/hardware", async (route) => {
    await fulfillJson(route, mockHardware)
  })

  await page.route("**/api/v1/llamacpp/start-by-model", async (route) => {
    startRequestBody = route.request().postDataJSON()
    currentStatus = runningStatus
    await fulfillJson(route, {
      status: "started",
      backend: "llamacpp",
      model_id: "gguf:toy-model-id",
    })
  })

  await page.route("**/api/v1/llamacpp/use-in-chat", async (route) => {
    useInChatCalled = true
    await fulfillJson(route, {
      provider: "llamacpp",
      endpoint: "http://127.0.0.1:8080",
      updated: true,
      effective: true,
      warnings: [],
    })
  })

  return {
    getStartRequestBody: () => startRequestBody,
    wasUseInChatCalled: () => useInChatCalled,
  }
}

test.describe("Admin Llama.cpp", () => {
  let admin: AdminPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    admin = new AdminPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should load admin/llamacpp page without critical errors", async ({
      authedPage,
      diagnostics,
    }) => {
      admin = new AdminPage(authedPage)
      await admin.gotoSection("llamacpp")
      await admin.assertSectionReady("llamacpp")

      await assertNoCriticalErrors(diagnostics)
    })

    test("should display Llama.cpp Admin heading or admin guard alert", async ({
      authedPage,
      diagnostics,
    }) => {
      admin = new AdminPage(authedPage)
      await admin.gotoSection("llamacpp")
      await admin.assertSectionReady("llamacpp")

      const hasHeading = await admin.llamacppHeading.isVisible().catch(() => false)
      const hasGuard = await admin.adminGuardAlert.isVisible().catch(() => false)

      expect(hasHeading || hasGuard).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // Key Controls
  // =========================================================================

  test.describe("Key Controls", () => {
    test("should show guided readiness inventory and launch panels", async ({
      authedPage,
      diagnostics,
    }) => {
      await mockLlamacppAdminFlow(authedPage)
      admin = new AdminPage(authedPage)
      await admin.gotoSection("llamacpp")
      await admin.assertSectionReady("llamacpp")

      await expect(admin.llamacppReadinessCard).toBeVisible()
      await expect(admin.llamacppInventoryCard).toBeVisible()
      await expect(admin.llamacppLaunchCard).toBeVisible()
      await expect(authedPage.getByText("API server restart required")).toBeVisible()
      await expect(admin.llamacppInventoryCard).toContainText("Toy 7B Q4_K_M")
      await expect(admin.llamacppStartButton).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should show Export and Import preset buttons", async ({
      authedPage,
      diagnostics,
    }) => {
      await mockLlamacppAdminFlow(authedPage)
      admin = new AdminPage(authedPage)
      await admin.gotoSection("llamacpp")
      await admin.assertSectionReady("llamacpp")

      const exportVisible = await admin.llamacppExportPresetButton
        .isVisible()
        .catch(() => false)
      const importVisible = await admin.llamacppImportPresetButton
        .isVisible()
        .catch(() => false)

      expect(exportVisible).toBe(true)
      expect(importVisible).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // API Interactions
  // =========================================================================

  test.describe("API Interactions", () => {
    test("should start selected inventory model and explicitly wire chat", async ({
      authedPage,
      diagnostics,
    }) => {
      const api = await mockLlamacppAdminFlow(authedPage)

      admin = new AdminPage(authedPage)
      await admin.gotoSection("llamacpp")
      await admin.assertSectionReady("llamacpp")

      await expect(admin.llamacppInventoryCard).toContainText("Toy 7B Q4_K_M")
      await expect(admin.llamacppUseInChatButton).toHaveCount(0)

      await admin.llamacppStartButton.click()
      await expect(admin.llamacppUseInChatButton).toBeVisible()

      expect(api.getStartRequestBody()).toEqual(
        expect.objectContaining({
          model_id: "gguf:toy-model-id",
        })
      )
      expect(api.wasUseInChatCalled()).toBe(false)

      await admin.llamacppUseInChatButton.click()
      await expect(authedPage.getByText("Chat provider updated.")).toBeVisible()
      expect(api.wasUseInChatCalled()).toBe(true)

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
