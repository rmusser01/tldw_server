import path from "path"

import { expect, test, type BrowserContext, type Page, type Route } from "@playwright/test"

import { waitForConnectionStore, forceConnected } from "./utils/connection"
import { launchWithExtensionOrSkip } from "./utils/real-server"
import { grantHostPermission } from "./utils/permissions"

const EXT_PATH = path.resolve("build/chrome-mv3")
const MOCK_SERVER_URL = "http://127.0.0.1:8000"
const PERSONA_ID = "garden-helper"
const PERSONA_NAME = "Garden Helper"

type PersonaSetupState = {
  status: "not_started" | "in_progress" | "completed"
  version: number
  current_step: "persona" | "voice" | "commands" | "safety" | "test"
  completed_steps: string[]
  completed_at: string | null
  last_test_type: "dry_run" | "live_session" | null
}

type PersonaProfileMock = {
  id: string
  version: number
  use_persona_state_context_default: boolean
  voice_defaults: Record<string, unknown>
  setup: PersonaSetupState
}

type PersonaMockState = {
  profile: PersonaProfileMock
  commandCount: number
  connections: Array<{ id: string; name: string; base_url: string; auth_type: string }>
  failStarterCommandOnce: boolean
  failedStarterCommand: boolean
}

const buildVoiceDefaults = () => ({
  stt_language: "en-US",
  stt_model: "whisper-1",
  tts_provider: "tldw",
  tts_voice: "af_heart",
  confirmation_mode: "destructive_only",
  voice_chat_trigger_phrases: ["hey helper"],
  auto_resume: true,
  barge_in: false,
  auto_commit_enabled: true,
  vad_threshold: 0.5,
  min_silence_ms: 250,
  turn_stop_secs: 0.2,
  min_utterance_secs: 0.4
})

const buildInitialProfile = (): PersonaProfileMock => ({
  id: PERSONA_ID,
  version: 7,
  use_persona_state_context_default: true,
  voice_defaults: buildVoiceDefaults(),
  setup: {
    status: "in_progress",
    version: 1,
    current_step: "persona",
    completed_steps: [],
    completed_at: null,
    last_test_type: null
  }
})

const buildVoiceAnalytics = () => ({
  persona_id: PERSONA_ID,
  summary: {
    total_events: 0,
    direct_command_count: 0,
    planner_fallback_count: 0,
    success_rate: 0,
    fallback_rate: 0,
    avg_response_time_ms: 0
  },
  live_voice: {
    total_committed_turns: 0,
    vad_auto_commit_count: 0,
    manual_commit_count: 0,
    vad_auto_rate: 0,
    manual_commit_rate: 0,
    degraded_session_count: 0
  },
  commands: [],
  fallbacks: {
    total_invocations: 0,
    success_count: 0,
    error_count: 0,
    avg_response_time_ms: 0,
    last_used: null
  },
  recent_live_sessions: []
})

const buildOpenApiSpec = () => ({
  openapi: "3.1.0",
  info: {
    title: "tldw_server mock",
    version: "e2e"
  },
  paths: {
    "/api/v1/persona/catalog": {
      get: {}
    },
    "/api/v1/persona/session": {
      post: {}
    },
    "/api/v1/persona/stream": {
      get: {}
    }
  }
})

const fulfillJson = async (
  route: Route,
  status: number,
  body: unknown
) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body)
  })
}

const handleMockApiRequest = async (route: Route, state: PersonaMockState) => {
  const request = route.request()
  const url = new URL(request.url())
  const method = request.method().toUpperCase()
  const pathname = url.pathname

  if (method === "OPTIONS") {
    await route.fulfill({ status: 204, body: "" })
    return
  }

  if (pathname === "/openapi.json") {
    await fulfillJson(route, 200, buildOpenApiSpec())
    return
  }

  if (pathname === "/api/v1/config/docs-info") {
    await fulfillJson(route, 200, {
      capabilities: {
        persona: true,
        personalization: true
      },
      supported_features: {
        persona: true,
        personalization: true
      }
    })
    return
  }

  if (pathname === "/api/v1/health" || pathname === "/api/v1/health/live") {
    await fulfillJson(route, 200, { ok: true, status: "ok" })
    return
  }

  if (pathname === "/api/v1/persona/catalog") {
    await fulfillJson(route, 200, [
      { id: "research_assistant", name: "Research Assistant" },
      { id: PERSONA_ID, name: PERSONA_NAME }
    ])
    return
  }

  if (pathname === `/api/v1/persona/profiles/${PERSONA_ID}/voice-analytics`) {
    await fulfillJson(route, 200, buildVoiceAnalytics())
    return
  }

  if (pathname === `/api/v1/persona/profiles/${PERSONA_ID}` && method === "GET") {
    await fulfillJson(route, 200, state.profile)
    return
  }

  if (pathname === `/api/v1/persona/profiles/${PERSONA_ID}` && method === "PATCH") {
    const body = request.postDataJSON() as {
      setup?: PersonaSetupState
      voice_defaults?: Record<string, unknown>
    }
    if (body?.setup) {
      state.profile.setup = body.setup
    }
    if (body?.voice_defaults) {
      state.profile.voice_defaults = body.voice_defaults
    }
    state.profile.version += 1
    await fulfillJson(route, 200, state.profile)
    return
  }

  if (pathname === `/api/v1/persona/profiles/${PERSONA_ID}/voice-commands` && method === "POST") {
    if (state.failStarterCommandOnce && !state.failedStarterCommand) {
      state.failedStarterCommand = true
      await fulfillJson(route, 500, {
        error: "Failed to create starter command"
      })
      return
    }
    state.commandCount += 1
    await fulfillJson(route, 201, {
      id: `cmd-${state.commandCount}`
    })
    return
  }

  if (pathname === `/api/v1/persona/profiles/${PERSONA_ID}/connections` && method === "GET") {
    await fulfillJson(route, 200, state.connections)
    return
  }

  if (pathname === `/api/v1/persona/profiles/${PERSONA_ID}/connections` && method === "POST") {
    const body = request.postDataJSON() as {
      name?: string
      base_url?: string
      auth_type?: string
    }
    const nextConnection = {
      id: `conn-setup-${state.connections.length + 1}`,
      name: String(body?.name || "Setup Connection"),
      base_url: String(body?.base_url || "https://api.example.com"),
      auth_type: String(body?.auth_type || "none")
    }
    state.connections = [nextConnection, ...state.connections]
    await fulfillJson(route, 201, nextConnection)
    return
  }

  if (
    pathname === `/api/v1/persona/profiles/${PERSONA_ID}/voice-commands/test` &&
    method === "POST"
  ) {
    const body = request.postDataJSON() as {
      heard_text?: string
    }
    const heardText = String(body?.heard_text || "").trim()
    await fulfillJson(route, 200, {
      heard_text: heardText,
      matched: true,
      command_name: "Search Notes"
    })
    return
  }

  if (pathname === "/api/v1/persona/sessions" && method === "GET") {
    await fulfillJson(route, 200, [])
    return
  }

  await fulfillJson(route, 404, {
    error: `Unhandled mock route: ${method} ${pathname}`
  })
}

const installPersonaSetupMocks = async (
  context: BrowserContext,
  { failStarterCommandOnce = false }: { failStarterCommandOnce?: boolean } = {}
) => {
  const state: PersonaMockState = {
    profile: buildInitialProfile(),
    commandCount: 0,
    connections: [],
    failStarterCommandOnce,
    failedStarterCommand: false
  }

  await context.route(`${MOCK_SERVER_URL}/**/*`, async (route) => {
    await handleMockApiRequest(route, state)
  })

  return state
}

const enableDirectRequestFallback = async (page: Page) => {
  await page.evaluate(() => {
    const patchRuntimeSendMessage = (runtime: any): boolean => {
      if (!runtime || typeof runtime.sendMessage !== "function") return false
      if (runtime.__tldwE2EDirectFallbackEnabled) return true

      const originalSendMessage = runtime.sendMessage.bind(runtime)
      const wrappedSendMessage = (...args: any[]) => {
        const message = args[0]
        if (message && typeof message === "object" && message.type === "tldw:request") {
          throw new Error("Could not establish connection. Receiving end does not exist.")
        }
        return originalSendMessage(...args)
      }

      try {
        runtime.sendMessage = wrappedSendMessage
      } catch {
        try {
          Object.defineProperty(runtime, "sendMessage", {
            value: wrappedSendMessage,
            configurable: true
          })
        } catch {
          return false
        }
      }

      runtime.__tldwE2EDirectFallbackEnabled = true
      return true
    }

    const browserRuntime = (globalThis as { browser?: { runtime?: unknown } }).browser?.runtime
    const chromeRuntime = (globalThis as { chrome?: { runtime?: unknown } }).chrome?.runtime
    patchRuntimeSendMessage(browserRuntime) || patchRuntimeSendMessage(chromeRuntime)
  })
}

const launchPersonaSetupPage = async (
  page: Page,
  sidepanelUrl: string,
  targetTab: "profiles" | "commands" = "profiles"
) => {
  await page.goto(`${sidepanelUrl}#/persona?persona_id=${PERSONA_ID}&tab=${targetTab}`, {
    waitUntil: "domcontentloaded"
  })
  await waitForConnectionStore(page, "persona-assistant-setup")
  await forceConnected(
    page,
    {
      serverUrl: MOCK_SERVER_URL,
      apiKey: "test-api-key"
    },
    "persona-assistant-setup:forceConnected"
  )
  await enableDirectRequestFallback(page)
  await expect(page.getByTestId("assistant-setup-overlay")).toBeVisible()
}

const completeSetupThroughDryRun = async (page: Page) => {
  await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Choose persona")
  await expect(page.getByTestId("assistant-setup-progress-step-persona")).toHaveAttribute(
    "data-status",
    "current"
  )

  await page.getByRole("button", { name: `Use ${PERSONA_NAME} persona` }).click()

  await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Configure voice")
  await expect(page.getByTestId("assistant-setup-progress-step-persona")).toHaveAttribute(
    "data-status",
    "completed"
  )
  await expect(page.getByTestId("assistant-setup-progress-step-voice")).toHaveAttribute(
    "data-status",
    "current"
  )
  await expect(page.getByRole("button", { name: "Save assistant defaults" })).toBeEnabled()
  await page.getByRole("button", { name: "Save assistant defaults" }).click()

  await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Add commands")
  await expect(page.getByTestId("assistant-setup-progress-step-voice")).toHaveAttribute(
    "data-status",
    "completed"
  )
  await page.getByRole("button", { name: "Search Notes", exact: true }).click()

  await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Review safety")
  await page.getByRole("button", { name: "Ask for destructive actions" }).click()
  await page.getByRole("button", { name: "No external connections for now" }).click()
  await page.getByRole("button", { name: "Save safety choices" }).click()

  await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Run a test")
  await page.getByPlaceholder("Try a spoken phrase").fill("search notes for project alpha")
  await page.getByRole("button", { name: "Run dry-run test" }).click()
  await expect(page.getByText(/Matched Search Notes/i)).toBeVisible()
  await page.getByRole("button", { name: "Finish with dry-run test" }).click()
}

test.describe("Persona assistant setup", () => {
  test("persona assistant setup can resume and finish with dry run", async () => {
    const { context, page, extensionId, sidepanelUrl } = (await launchWithExtensionOrSkip(
      test,
      EXT_PATH,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: true,
          tldw_skip_landing_hub: true,
          tldwConfig: {
            serverUrl: MOCK_SERVER_URL,
            authMode: "single-user",
            apiKey: "test-api-key"
          }
        }
      }
    )) as any

    try {
      const granted = await grantHostPermission(
        context,
        extensionId,
        `${new URL(MOCK_SERVER_URL).origin}/*`
      )
      if (!granted) {
        test.skip(
          true,
          "Host permission not granted for persona setup E2E origin."
        )
      }

      await installPersonaSetupMocks(context)
      await launchPersonaSetupPage(page, sidepanelUrl, "profiles")
      await completeSetupThroughDryRun(page)

      await expect(page.getByTestId("assistant-setup-overlay")).toHaveCount(0)
      await expect(page.getByRole("tab", { name: "Profiles" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
      await expect(page.getByTestId("persona-setup-handoff-card")).toBeVisible()
      await expect(page.getByText("Assistant setup complete")).toBeVisible()
      await expect(page.getByText("Recommended next step")).toBeVisible()
      await expect(page.getByText("Add a connection")).toBeVisible()

      await page.getByRole("button", { name: "Open connections", exact: true }).click()
      await expect(page.getByRole("tab", { name: "Connections" })).toHaveAttribute(
        "aria-selected",
        "true"
      )
      await expect(page.getByTestId("persona-setup-handoff-card")).toBeVisible()

      await page.getByTestId("persona-connections-name-input").fill("Slack Alerts")
      await page
        .getByTestId("persona-connections-base-url-input")
        .fill("https://hooks.example.com/incoming")
      await page.getByTestId("persona-connections-save").click()

      await expect(page.getByText("Setup complete")).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("persona assistant setup can recover from a starter-command failure", async () => {
    const { context, page, extensionId, sidepanelUrl } = (await launchWithExtensionOrSkip(
      test,
      EXT_PATH,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: true,
          tldw_skip_landing_hub: true,
          tldwConfig: {
            serverUrl: MOCK_SERVER_URL,
            authMode: "single-user",
            apiKey: "test-api-key"
          }
        }
      }
    )) as any

    try {
      const granted = await grantHostPermission(
        context,
        extensionId,
        `${new URL(MOCK_SERVER_URL).origin}/*`
      )
      if (!granted) {
        test.skip(
          true,
          "Host permission not granted for persona setup E2E origin."
        )
      }

      await installPersonaSetupMocks(context, { failStarterCommandOnce: true })
      await launchPersonaSetupPage(page, sidepanelUrl, "profiles")

      await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Choose persona")
      await page.getByRole("button", { name: `Use ${PERSONA_NAME} persona` }).click()
      await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Configure voice")
      await page.getByRole("button", { name: "Save assistant defaults" }).click()
      await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Add commands")

      await page.getByRole("button", { name: "Search Notes", exact: true }).click()
      await expect(page.getByText("Failed to create starter command")).toBeVisible()
      await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Add commands")

      await page.getByRole("button", { name: "Retry Search Notes", exact: true }).click()
      await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Review safety")
      await page.getByRole("button", { name: "Ask for destructive actions" }).click()
      await page.getByRole("button", { name: "No external connections for now" }).click()
      await page.getByRole("button", { name: "Save safety choices" }).click()
      await expect(page.getByTestId("assistant-setup-current-step")).toHaveText("Run a test")

      await page.getByPlaceholder("Try a spoken phrase").fill("search notes for retry path")
      await page.getByRole("button", { name: "Run dry-run test" }).click()
      await expect(page.getByText(/Matched Search Notes/i)).toBeVisible()
      await page.getByRole("button", { name: "Finish with dry-run test" }).click()

      await expect(page.getByTestId("assistant-setup-overlay")).toHaveCount(0)
      await expect(page.getByTestId("persona-setup-handoff-card")).toBeVisible()
      await expect(page.getByText("Assistant setup complete")).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
