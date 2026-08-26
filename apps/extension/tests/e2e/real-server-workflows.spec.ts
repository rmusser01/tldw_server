import { test } from "@playwright/test"

import {
  ALL_FEATURE_FLAGS_ENABLED,
  type CreateWorkflowDriver,
  REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED,
  createRealServerWorkflowTldwConfig,
  createRealServerWorkflowStorageSeed,
  registerRealServerWorkflows,
  withFeatures
} from "../../../test-utils/real-server-workflows"
import { grantHostPermission } from "./utils/permissions"
import { launchWithBuiltExtensionOrSkip } from "./utils/real-server"

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
    ...createRealServerWorkflowStorageSeed(),
    tldwConfig: createRealServerWorkflowTldwConfig(serverUrl, apiKey)
  }
  const enabledFlags = Object.entries(featureFlags || {})
    .filter(([, value]) => value)
    .map(([key]) => key as keyof typeof ALL_FEATURE_FLAGS_ENABLED)
  const seedConfig = enabledFlags.length
    ? withFeatures(enabledFlags, baseSeed)
    : baseSeed

  const launchResult = await launchWithBuiltExtensionOrSkip(testRef ?? test, {
    seedConfig,
    seedLocalStorage: REAL_SERVER_WORKFLOW_LOCAL_STORAGE_SEED
  })
  const {
    context,
    page,
    extensionId,
    optionsUrl,
    sidepanelUrl,
    openSidepanel
  } = launchResult

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
