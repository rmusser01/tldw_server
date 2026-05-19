/**
 * Admin Llama.cpp E2E Tests (Tier 4)
 *
 * Tests the /admin/llamacpp page:
 * - Page loads without critical errors
 * - Llama.cpp Admin heading or admin guard visible
 * - Guided readiness, inventory, and launch panels
 * - Start Server calls start-by-model with stable model_id
 * - Use this in Chat is explicit and calls provider wiring API
 * - Acquisition import/download flows stay mocked and do not require remote downloads
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
    imported_asset_folders: [],
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

const mockGgufAsset = {
  asset_id: "gguf:toy-model-id",
  kind: "gguf",
  identity_basis: "resolved_path",
  path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
  resolved_path: "/srv/models/gguf/toy-7b-q4_k_m.gguf",
  display_name: "Toy 7B Q4_K_M",
  source: "models_dir",
  size_bytes: 4_200_000_000,
  modified_at: "2026-05-15T10:00:00Z",
  metadata: {
    quantization: "Q4_K_M",
    parameter_hint: "7B",
    context_hint: 4096,
    family_hint: "toy",
  },
  capabilities: ["chat"],
  mmproj_asset_ids: ["mmproj:toy-vision"],
  base_model_asset_ids: [],
  warnings: [],
}

const mockMmprojAsset = {
  asset_id: "mmproj:toy-vision",
  kind: "mmproj",
  identity_basis: "resolved_path",
  path: "/srv/models/gguf/toy-mmproj.gguf",
  resolved_path: "/srv/models/gguf/toy-mmproj.gguf",
  display_name: "Toy Vision Projector",
  source: "models_dir",
  size_bytes: 320_000_000,
  modified_at: "2026-05-15T10:05:00Z",
  metadata: {
    family_hint: "toy",
  },
  capabilities: ["vision"],
  mmproj_asset_ids: [],
  base_model_asset_ids: ["gguf:toy-model-id"],
  warnings: [],
}

const mockImportedFolderAsset = {
  asset_id: "folder:imported",
  kind: "folder",
  identity_basis: "resolved_path",
  path: "/srv/models/imported",
  resolved_path: "/srv/models/imported",
  display_name: "imported",
  source: "imported_folder",
  size_bytes: null,
  modified_at: "2026-05-15T11:00:00Z",
  metadata: {},
  capabilities: [],
  mmproj_asset_ids: [],
  base_model_asset_ids: [],
  warnings: [],
}

const mockImportedGgufAsset = {
  asset_id: "gguf:imported-toy",
  kind: "gguf",
  identity_basis: "resolved_path",
  path: "/srv/models/imported/imported-toy.gguf",
  resolved_path: "/srv/models/imported/imported-toy.gguf",
  display_name: "Imported Toy",
  source: "imported_folder",
  size_bytes: 2_100_000_000,
  modified_at: "2026-05-15T11:01:00Z",
  metadata: {
    quantization: "Q4_K_M",
    parameter_hint: "3B",
  },
  capabilities: ["chat"],
  mmproj_asset_ids: [],
  base_model_asset_ids: [],
  warnings: [],
}

const mockDownloadedAsset = {
  asset_id: "gguf:downloaded-toy",
  kind: "gguf",
  identity_basis: "resolved_path",
  path: "/srv/models/downloaded-toy.gguf",
  resolved_path: "/srv/models/downloaded-toy.gguf",
  display_name: "Downloaded Toy",
  source: "models_dir",
  size_bytes: 2_200_000_000,
  modified_at: "2026-05-15T12:00:00Z",
  metadata: {
    quantization: "Q4_K_M",
    parameter_hint: "3B",
  },
  capabilities: ["chat"],
  mmproj_asset_ids: [],
  base_model_asset_ids: [],
  warnings: [],
}

const mockQueuedDownloadJob = {
  job_id: "42",
  status: "queued",
  operation: "download",
  queue: "acquisition",
  source_label: "Toy download",
  destination_path: "/srv/models/downloaded-toy.gguf",
  asset_id: null,
  progress: {},
  warnings: [],
  error_message: null,
}

const mockCompletedDownloadJob = {
  ...mockQueuedDownloadJob,
  status: "completed",
  asset_id: "gguf:downloaded-toy",
  progress: {
    progress_percent: 100,
  },
}

async function mockLlamacppAdminFlow(page: Page) {
  let currentStatus = stoppedStatus
  let currentAssets = {
    assets: [mockGgufAsset, mockMmprojAsset],
    warnings: [],
    scan_limited: false,
  }
  let downloadJobs: unknown[] = []
  let assetsRequestCount = 0
  let startRequestBody: unknown = null
  let importRequestBody: unknown = null
  let downloadRequestBody: unknown = null
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

  await page.route("**/api/v1/llamacpp/assets", async (route) => {
    assetsRequestCount += 1
    await fulfillJson(route, currentAssets)
  })

  await page.route("**/api/v1/llamacpp/assets/import-folder/preview", async (route) => {
    await fulfillJson(route, {
      folder: mockImportedFolderAsset,
      assets: [mockImportedGgufAsset, mockMmprojAsset],
      asset_counts: {
        gguf: 1,
        mmproj: 1,
      },
      warnings: ["Preview skipped unreadable sidecar file."],
      scan_limited: false,
      will_persist: false,
    })
  })

  await page.route("**/api/v1/llamacpp/assets/import-folder", async (route) => {
    importRequestBody = route.request().postDataJSON()
    currentAssets = {
      assets: [mockGgufAsset, mockMmprojAsset, mockImportedFolderAsset, mockImportedGgufAsset],
      warnings: [],
      scan_limited: false,
    }
    await fulfillJson(route, mockImportedFolderAsset)
  })

  await page.route("**/api/v1/llamacpp/assets/downloads", async (route) => {
    if (route.request().method() === "POST") {
      downloadRequestBody = route.request().postDataJSON()
      downloadJobs = [mockQueuedDownloadJob]
      await fulfillJson(route, mockQueuedDownloadJob)
      return
    }

    await fulfillJson(route, {
      jobs: downloadJobs,
    })
  })

  await page.route("**/api/v1/llamacpp/profiles", async (route) => {
    await fulfillJson(route, {
      profiles: [],
    })
  })

  await page.route("**/api/v1/llamacpp/instances", async (route) => {
    await fulfillJson(route, {
      runtimes: [],
      warnings: [],
    })
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
    getImportRequestBody: () => importRequestBody,
    getDownloadRequestBody: () => downloadRequestBody,
    getAssetsRequestCount: () => assetsRequestCount,
    markDownloadCompleted: () => {
      downloadJobs = [mockCompletedDownloadJob]
      currentAssets = {
        assets: [
          mockGgufAsset,
          mockMmprojAsset,
          mockImportedFolderAsset,
          mockImportedGgufAsset,
          mockDownloadedAsset,
        ],
        warnings: [],
        scan_limited: false,
      }
    },
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

    test("should import folders and refresh completed downloads without live downloads", async ({
      authedPage,
      diagnostics,
    }) => {
      const api = await mockLlamacppAdminFlow(authedPage)

      admin = new AdminPage(authedPage)
      await admin.gotoSection("llamacpp")
      await admin.assertSectionReady("llamacpp")

      await expect(authedPage.getByText("Assets")).toBeVisible()
      await expect(authedPage.getByText("Toy 7B Q4_K_M").first()).toBeVisible()

      await authedPage
        .getByLabel("Import local asset folder")
        .fill("/srv/models/imported")
      await authedPage.getByRole("button", { name: "Preview folder" }).click()

      await expect(authedPage.getByText("Import preview")).toBeVisible()
      await expect(authedPage.getByText("GGUF: 1")).toBeVisible()
      await expect(authedPage.getByText("mmproj: 1")).toBeVisible()
      expect(api.getImportRequestBody()).toBeNull()

      await authedPage.getByRole("button", { name: "Confirm import" }).click()
      await expect(authedPage.getByText("Imported Toy")).toBeVisible()
      expect(api.getImportRequestBody()).toEqual({
        path: "/srv/models/imported",
      })

      await authedPage
        .getByLabel("Download source URL")
        .fill("https://example.com/downloaded-toy.gguf")
      await authedPage
        .getByLabel("Download destination directory")
        .fill("/srv/models")
      await authedPage
        .getByLabel("Download filename")
        .fill("downloaded-toy.gguf")
      await authedPage.getByRole("button", { name: "Queue download" }).click()

      await expect(authedPage.getByText("Toy download")).toBeVisible()
      await expect(authedPage.getByText("queued")).toBeVisible()
      expect(api.getDownloadRequestBody()).toEqual({
        url: "https://example.com/downloaded-toy.gguf",
        destination_dir: "/srv/models",
        filename: "downloaded-toy.gguf",
      })

      const assetsRequestsBeforeCompletion = api.getAssetsRequestCount()
      api.markDownloadCompleted()
      await authedPage.getByRole("button", { name: /Refresh downloads/i }).click()

      await expect(authedPage.getByText("completed")).toBeVisible()
      await expect(authedPage.getByText("Downloaded Toy")).toBeVisible()
      expect(api.getAssetsRequestCount()).toBeGreaterThan(
        assetsRequestsBeforeCompletion
      )
      expect(api.wasUseInChatCalled()).toBe(false)

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
