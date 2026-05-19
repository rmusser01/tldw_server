import { test } from "@playwright/test"
import { registerRealServerWorkflows, type CreateWorkflowDriver, withFeatures, ALL_FEATURE_FLAGS_ENABLED } from "../../../test-utils/real-server-workflows"
import { launchWithBuiltExtensionOrSkip } from "./utils/real-server"
import { grantHostPermission } from "./utils/permissions"

const shouldSkipHostPermission =
  process.env.TLDW_E2E_SKIP_HOST_PERMISSION !== "0" &&
  process.env.TLDW_E2E_SKIP_HOST_PERMISSION !== "false"

const normalizeRoute = (route: string) => {
  const trimmed = String(route || "").trim()
  if (!trimmed) return "/"
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`
}

const createExtensionDriver: CreateWorkflowDriver = async ({
  serverUrl,
  apiKey,
  featureFlags,
  testRef
}) => {
  const baseSeed = {
    __tldw_first_run_complete: true,
    tldwConfig: {
      serverUrl,
      authMode: "single-user",
      apiKey
    },
    quickIngestInspectorIntroDismissed: true,
    quickIngestOnboardingDismissed: true,
    // Skip the "What would you like to do?" landing hub modal
    // Note: chrome.storage handles serialization automatically - don't JSON.stringify
    tldw_skip_landing_hub: true,
    // Dismiss the workflow landing modal (shown on first run)
    // Note: loadFromStorage in workflows.ts reads directly without JSON.parse
    "tldw:workflow:landing-config": {
      showOnFirstRun: true,
      dismissedAt: Date.now(),
      completedWorkflows: []
    }
  }
  const enabledFlags = Object.entries(featureFlags || {})
    .filter(([, value]) => value)
    .map(([key]) => key as keyof typeof ALL_FEATURE_FLAGS_ENABLED)
  const seedConfig = enabledFlags.length
    ? withFeatures(enabledFlags, baseSeed)
    : baseSeed

  // Seed localStorage for tutorials and tours that don't use chrome.storage
  const seedLocalStorage = {
    // Skip the playground tour ("Choose a Model", etc.)
    "playground-tour-completed": "true",
    // Skip all tutorial prompts via zustand persisted state
    "tldw-tutorials": JSON.stringify({
      state: {
        completedTutorials: ["playground", "chat", "notes", "media", "settings"],
        seenPromptPages: ["/", "/chat", "/notes", "/media", "/settings", "/playground", "/research-studio"]
      },
      version: 0
    })
  }

  const launchResult = await launchWithBuiltExtensionOrSkip(testRef ?? test, {
    seedConfig,
    seedLocalStorage
  })
  const { context, page, extensionId, optionsUrl, sidepanelUrl, openSidepanel } =
    launchResult

  return {
    kind: "extension",
    serverUrl,
    apiKey,
    context,
    page,
    optionsUrl,
    sidepanelUrl,
    openSidepanel,
    goto: async (targetPage, route, options) => {
      const normalized = normalizeRoute(route)
      await targetPage.goto(`${optionsUrl}#${normalized}`, options)
    },
    ensureHostPermission: async () => {
      if (shouldSkipHostPermission) {
        return true
      }
      const origin = new URL(serverUrl).origin + "/*"
      return grantHostPermission(context, extensionId, origin)
    },
    close: async () => {
      await context.close()
    }
  }
}

registerRealServerWorkflows(createExtensionDriver)
