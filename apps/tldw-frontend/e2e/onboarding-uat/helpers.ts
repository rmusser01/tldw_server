import { expect, type Locator, type Page } from "@playwright/test"
import { ChatPage } from "../utils/page-objects"
import {
  isBenign,
  waitForAppShell,
  waitForConnection,
} from "../utils/helpers"
import { waitForStreamComplete } from "../utils/journey-helpers"
import { safeSegment, type DiagnosticsData, type OnboardingArtifact } from "./fixtures"

export const DEFAULT_HOSTED_PROVIDER_API_KEY =
  process.env.TLDW_ONBOARDING_UAT_OPENAI_API_KEY || "sk-uat-mock-openai"
export const DEFAULT_HOSTED_PROVIDER_MODEL = "gpt-4.1-mini"
export const DEFAULT_LOCAL_PROVIDER_MODEL = "llama3.2:3b"
export const UNREACHABLE_LOCAL_PROVIDER_ENDPOINT = "http://127.0.0.1:65535/v1"

type SetupPathChoice = "docker" | "local"

type WizardProviderOptions = {
  label: string
  model: string
  apiKey?: string | null
  baseUrl?: string | null
  expectedDiscoveredModel?: string
}

type FirstChatResponsePayload = {
  status?: string
  response_text?: string | null
  failure_category?: string | null
  message?: string | null
}

type FirstSourceResultSummary = {
  status?: string
  firstMediaId?: string | null
  primarySourceLabel?: string | null
  errorMessage?: string | null
}

export type FirstSourceSessionSummary = {
  lifecycle?: string
  firstSourceAddMode?: string | null
  resultSummary?: FirstSourceResultSummary | null
}

export type FirstSourceStarterHandoff = {
  mediaId?: string
  title?: string
  mode?: string
  content?: string
}

const FIRST_SOURCE_PASTE_FIXTURE = [
  "Onboarding UAT research note",
  "",
  "Date: 2026-06-02",
  "",
  "- Claim: The harness verifies first-value onboarding.",
  "- Action item: Ask a starter question after ingest.",
  "- Detail: Starter questions must be shown only after source readiness.",
].join("\n")

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

export async function openFirstRunSetup(page: Page): Promise<void> {
  await page.goto("/", { waitUntil: "domcontentloaded" })
  await waitForAppShell(page)

  const gateOverlay = page.getByTestId("first-run-gate-overlay").first()
  if (await gateOverlay.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await page.getByTestId("first-run-get-started").click()
    await waitForAppShell(page)
    return
  }

  if (!page.url().includes("/setup")) {
    await page.goto("/setup", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)
  }
}

async function clickOnlyContinue(page: Page): Promise<void> {
  const continueButton = page.getByRole("button", { name: /^continue$/i })
  await expect(continueButton).toBeVisible({ timeout: 30_000 })
  await expect(continueButton).toBeEnabled({ timeout: 30_000 })
  await continueButton.click()
}

export async function chooseWizardSetupPath(
  page: Page,
  setupPath: SetupPathChoice = "local"
): Promise<void> {
  await expect(page.getByTestId("unified-setup-shell")).toBeVisible({
    timeout: 60_000,
  })
  await expect(
    page.getByRole("heading", { name: /choose your setup path/i })
  ).toBeVisible({ timeout: 60_000 })

  const buttonName =
    setupPath === "docker" ? /solo,\s*docker/i : /solo,\s*local install/i
  await page.getByRole("button", { name: buttonName }).click()
}

export async function acknowledgeWizardPrivacy(page: Page): Promise<void> {
  await expect(
    page.getByRole("heading", { name: /privacy and security/i })
  ).toBeVisible({ timeout: 30_000 })
  await page
    .getByLabel(/i understand local or remote setup access/i)
    .check()
  await clickOnlyContinue(page)
}

export async function openWizardProviderStep(
  page: Page,
  setupPath: SetupPathChoice = "local"
): Promise<void> {
  await chooseWizardSetupPath(page, setupPath)
  await acknowledgeWizardPrivacy(page)
  await expect(page.getByRole("heading", { name: /chat provider/i })).toBeVisible({
    timeout: 30_000,
  })
}

export async function validateWizardProvider(
  page: Page,
  label: string
): Promise<Locator> {
  const validateButton = page.getByRole("button", {
    name: new RegExp(`^validate ${escapeRegExp(label)}$`, "i"),
  })
  await expect(validateButton).toBeVisible({ timeout: 30_000 })
  await expect(validateButton).toBeEnabled({ timeout: 30_000 })
  await validateButton.click()

  const validationCopy = page
    .getByText(/first chat verifies this provider|provider validation is ready|local_provider_unreachable|provider api key is required/i)
    .last()
  await expect(validationCopy).toBeVisible({ timeout: 30_000 })
  return validationCopy
}

export async function configureWizardProvider(
  page: Page,
  options: WizardProviderOptions
): Promise<void> {
  await expect(page.getByRole("heading", { name: /chat provider/i })).toBeVisible({
    timeout: 30_000,
  })

  await page
    .getByLabel(new RegExp(`^select ${escapeRegExp(options.label)}$`, "i"))
    .check()

  const apiKeyInput = page.getByLabel(
    new RegExp(`^${escapeRegExp(options.label)} api key$`, "i")
  )
  if (options.apiKey !== undefined && (await apiKeyInput.isVisible().catch(() => false))) {
    await apiKeyInput.fill(options.apiKey ?? "")
  }

  const baseUrlInput = page.getByLabel(
    new RegExp(`^${escapeRegExp(options.label)} base url$`, "i")
  )
  if (options.baseUrl !== undefined && (await baseUrlInput.isVisible().catch(() => false))) {
    await baseUrlInput.fill(options.baseUrl ?? "")
  }

  await page.getByLabel(/^default model$/i).fill(options.model)
  await validateWizardProvider(page, options.label)

  if (options.expectedDiscoveredModel) {
    await expect(
      page.getByRole("button", { name: options.expectedDiscoveredModel })
    ).toBeVisible({ timeout: 30_000 })
  }
}

export async function saveWizardProviderAndContinue(page: Page): Promise<void> {
  const saveButton = page.getByRole("button", { name: /save providers/i })
  await expect(saveButton).toBeVisible({ timeout: 30_000 })
  await expect(saveButton).toBeEnabled({ timeout: 30_000 })
  await saveButton.click()
  await expect(page.getByText(/^saved/i).last()).toBeVisible({ timeout: 30_000 })
  await clickOnlyContinue(page)
}

export async function advanceWizardDefaultsToFirstChat(page: Page): Promise<void> {
  await expect(page.getByRole("heading", { name: /ingest defaults/i })).toBeVisible({
    timeout: 30_000,
  })
  await clickOnlyContinue(page)

  await expect(
    page.getByRole("heading", { name: /audio,\s*stt,\s*and tts/i })
  ).toBeVisible({ timeout: 30_000 })
  await clickOnlyContinue(page)

  await expect(
    page.getByRole("heading", { name: /optional advanced setup/i })
  ).toBeVisible({ timeout: 30_000 })
  await clickOnlyContinue(page)

  await expect(page.getByRole("heading", { name: /first chat/i })).toBeVisible({
    timeout: 30_000,
  })
}

export async function prepareHostedOpenAiFirstChat(
  page: Page,
  options: {
    setupPath?: SetupPathChoice
    apiKey?: string
    model?: string
  } = {}
): Promise<void> {
  await openWizardProviderStep(page, options.setupPath ?? "local")
  await configureWizardProvider(page, {
    label: "OpenAI",
    apiKey: options.apiKey ?? DEFAULT_HOSTED_PROVIDER_API_KEY,
    model: options.model ?? DEFAULT_HOSTED_PROVIDER_MODEL,
  })
  await saveWizardProviderAndContinue(page)
  await advanceWizardDefaultsToFirstChat(page)
}

export async function prepareLocalOllamaFirstChat(
  page: Page,
  options: {
    baseUrl: string
    model?: string
    setupPath?: SetupPathChoice
  }
): Promise<void> {
  await openWizardProviderStep(page, options.setupPath ?? "local")
  await configureWizardProvider(page, {
    label: "Ollama",
    baseUrl: options.baseUrl,
    model: options.model ?? DEFAULT_LOCAL_PROVIDER_MODEL,
    expectedDiscoveredModel: options.model ?? DEFAULT_LOCAL_PROVIDER_MODEL,
  })
  await saveWizardProviderAndContinue(page)
  await advanceWizardDefaultsToFirstChat(page)
}

export async function sendWizardFirstChat(
  page: Page,
  prompt: string
): Promise<FirstChatResponsePayload> {
  await expect(page.getByRole("heading", { name: /first chat/i })).toBeVisible({
    timeout: 30_000,
  })
  await page.getByLabel(/first prompt/i).fill(prompt)

  const firstChatResponse = page.waitForResponse(
    (response) =>
      response.url().includes("/api/v1/setup/first-run/first-chat") &&
      response.request().method().toUpperCase() === "POST",
    { timeout: 60_000 }
  )
  await page.getByRole("button", { name: /send test chat/i }).click()
  const response = await firstChatResponse
  return (await response.json().catch(() => ({}))) as FirstChatResponsePayload
}

export async function sendWizardFirstChatAndWaitForMilestone(
  page: Page,
  prompt: string
): Promise<FirstChatResponsePayload> {
  const payload = await sendWizardFirstChat(page, prompt)
  expect(payload.status).toBe("ready")
  await expect(
    page.getByRole("heading", { name: /add your first source/i })
  ).toBeVisible({ timeout: 60_000 })
  return payload
}

export async function completeFirstSourcePasteIngest(
  page: Page
): Promise<FirstSourceSessionSummary> {
  await expect(
    page.getByRole("textbox", { name: /pasted text input/i })
  ).toBeVisible({ timeout: 20_000 })
  await page
    .getByRole("textbox", { name: /pasted text input/i })
    .fill(FIRST_SOURCE_PASTE_FIXTURE)
  await page
    .getByRole("button", { name: /add pasted text to queue/i })
    .click()
  await expect(page.getByText(/queued/i)).toBeVisible({ timeout: 20_000 })
  await page.getByRole("button", { name: /use defaults & process/i }).click()
  await expect(page.getByTestId("wizard-results-step")).toBeVisible({
    timeout: 120_000,
  })

  const sessionHandle = await page.waitForFunction(
    () => {
      type QuickIngestSessionStoreWindow = Window & {
        __tldw_useQuickIngestSessionStore?: {
          getState?: () => {
            session?: FirstSourceSessionSummary | null
          }
        }
      }
      const store = (window as QuickIngestSessionStoreWindow)
        .__tldw_useQuickIngestSessionStore
      const session = store?.getState?.().session
      if (
        session?.resultSummary?.status === "success" &&
        session.resultSummary.firstMediaId
      ) {
        return {
          lifecycle: session.lifecycle,
          firstSourceAddMode: session.firstSourceAddMode,
          resultSummary: session.resultSummary,
        }
      }
      return null
    },
    undefined,
    { timeout: 120_000 }
  )
  const summary =
    (await sessionHandle.jsonValue()) as FirstSourceSessionSummary

  const resultsStep = page.getByTestId("wizard-results-step")
  await resultsStep
    .getByRole("button", { name: /close the ingest wizard/i })
    .click()
  await expect(resultsStep).toBeHidden({ timeout: 10_000 })
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          type QuickIngestSessionStoreWindow = Window & {
            __tldw_useQuickIngestSessionStore?: {
              getState?: () => {
                session?: { visibility?: string } | null
              }
            }
          }
          const store = (window as QuickIngestSessionStoreWindow)
            .__tldw_useQuickIngestSessionStore
          return store?.getState?.().session?.visibility ?? null
        }),
      { timeout: 10_000 }
    )
    .not.toBe("visible")
  await expect(page.getByText(/starter questions/i)).toBeVisible({
    timeout: 30_000,
  })
  return summary
}

export async function clickFirstSourceStarterQuestion(
  page: Page,
  question = "Summarize this source."
): Promise<FirstSourceStarterHandoff> {
  await page.evaluate(() => {
    type DiscussMediaWindow = Window & {
      __tldwLastDiscussMediaDetail?: FirstSourceStarterHandoff | null
    }
    const target = window as DiscussMediaWindow
    target.__tldwLastDiscussMediaDetail = null
    window.addEventListener(
      "tldw:discuss-media",
      ((event: CustomEvent<FirstSourceStarterHandoff>) => {
        target.__tldwLastDiscussMediaDetail = event.detail
      }) as EventListener,
      { once: true }
    )
  })
  await page.getByRole("button", { name: question }).click()
  const handoffHandle = await page.waitForFunction(
    () =>
      (
        window as Window & {
          __tldwLastDiscussMediaDetail?: FirstSourceStarterHandoff | null
        }
      ).__tldwLastDiscussMediaDetail,
    undefined,
    { timeout: 10_000 }
  )
  return (await handoffHandle.jsonValue()) as FirstSourceStarterHandoff
}

export async function waitForWizardFirstChatRecovery(
  page: Page
): Promise<Locator> {
  const alert = page
    .getByRole("alert")
    .filter({
      hasText:
        /credentials need attention|provider returned an error|model is unavailable|endpoint could not be reached|first chat did not complete/i,
    })
    .last()
  await expect(alert).toBeVisible({ timeout: 60_000 })
  await expectNoUnsafePrimaryDetails(alert)
  return alert
}

export async function attemptSingleUserConnection(
  page: Page,
  options: { serverUrl: string; apiKey: string }
): Promise<void> {
  const serverUrlInput = page.getByTestId("onboarding-server-url")
  const apiKeyInput = page.getByTestId("onboarding-api-key")

  await expect(serverUrlInput).toBeVisible({ timeout: 30_000 })
  await serverUrlInput.fill(options.serverUrl)
  await expect(apiKeyInput).toBeVisible({ timeout: 30_000 })
  await apiKeyInput.fill(options.apiKey)

  const connectButton = page.getByTestId("onboarding-connect")
  await expect(connectButton).toBeVisible({ timeout: 30_000 })
  await connectButton.click()
}

export async function connectSingleUser(
  page: Page,
  options: { serverUrl: string; apiKey: string }
): Promise<void> {
  await attemptSingleUserConnection(page, options)

  await waitForSetupConnectionReady(page)
}

async function waitForSetupConnectionReady(page: Page): Promise<void> {
  const successScreen = page.getByTestId("onboarding-success-screen")
  const chatInput = page
    .locator("#textarea-message")
    .or(page.getByTestId("chat-input"))
    .or(page.getByPlaceholder(/type a message/i))
    .first()
  const startChatButton = page.getByRole("button", { name: /start chatting/i }).first()

  await expect
    .poll(
      async () => {
        if (await successScreen.isVisible().catch(() => false)) {
          return "onboarding-success"
        }
        if (
          (await chatInput.isVisible().catch(() => false)) ||
          (await startChatButton.isVisible().catch(() => false))
        ) {
          return "chat-ready"
        }
        return "waiting"
      },
      { timeout: 60_000 }
    )
    .not.toBe("waiting")
}

export async function sendFirstChat(page: Page, prompt: string): Promise<string> {
  const chatPage = new ChatPage(page)
  const currentPath = new URL(page.url()).pathname
  if (currentPath !== "/chat") {
    await chatPage.goto()
  } else {
    await waitForBackendConnection(page)
  }
  await chatPage.waitForReady()
  await chatPage.sendMessage(prompt)
  await waitForStreamComplete(page)
  await chatPage.waitForResponse()

  const messages = await chatPage.getMessages()
  const assistantMessage = messages
    .filter((message) => message.role === "assistant")
    .at(-1)
  const content = assistantMessage?.content?.trim() ?? ""
  expect(content.length).toBeGreaterThan(0)
  return content
}

export async function captureStep(
  page: Page,
  artifact: OnboardingArtifact,
  scenarioId: string,
  stepName: string,
  extra: Record<string, unknown> = {}
): Promise<{ screenshotPath: string; jsonPath: string }> {
  const stem = `${safeSegment(scenarioId)}-${safeSegment(stepName)}`
  const screenshotPath = `${artifact.screenshotsDir}/${stem}.png`
  await page.screenshot({ path: screenshotPath, fullPage: true })
  const jsonPath = artifact.writeJson(`steps/${stem}.json`, {
    scenario_id: scenarioId,
    step_name: stepName,
    url: page.url(),
    captured_at: new Date().toISOString(),
    screenshot_path: screenshotPath,
    ...extra,
  })
  return { screenshotPath, jsonPath }
}

const UNSAFE_PRIMARY_DETAIL_PATTERN =
  /traceback|stack trace|authorization|x-api-key|request headers|\/Users\/|[A-Za-z]:\\|sk-[A-Za-z0-9_-]+/i

export async function expectNoUnsafePrimaryDetails(
  locator: Locator
): Promise<void> {
  await expect(locator).not.toContainText(UNSAFE_PRIMARY_DETAIL_PATTERN)
}

type ConsoleDiagnostic = DiagnosticsData["console"][number]
type DiagnosticsAllowance = {
  expectedEndpointOrigins?: string[]
  expectedConsoleText?: RegExp[]
}

const MODEL_METADATA_ENDPOINT = "/api/v1/llm/models/metadata"
const CHAT_SETTINGS_ENDPOINT_PREFIX = "/api/v1/chats/"
const CHAT_SETTINGS_ENDPOINT_SUFFIX = "/settings?scope_type=global"

function isBenignOnboardingConsoleEntry(entry: ConsoleDiagnostic): boolean {
  if (isBenign(entry.text)) {
    return true
  }

  if (
    /Failed to fetch (?:models from tldw|chat models): Error: Failed to fetch/.test(entry.text) &&
    entry.text.includes(`(GET ${MODEL_METADATA_ENDPOINT})`)
  ) {
    return true
  }

  const locationUrl = entry.location?.url ?? ""
  return (
    /404 \(Not Found\)/.test(entry.text) &&
    locationUrl.includes(CHAT_SETTINGS_ENDPOINT_PREFIX) &&
    locationUrl.includes(CHAT_SETTINGS_ENDPOINT_SUFFIX)
  )
}

const matchesExpectedEndpoint = (
  url: string | undefined,
  expectedEndpointOrigins: string[] = []
): boolean => {
  if (!url) return false
  return expectedEndpointOrigins.some((origin) =>
    url.startsWith(origin.replace(/\/$/, ""))
  )
}

export function assertNoCriticalDiagnostics(
  diagnostics: DiagnosticsData,
  allowance: DiagnosticsAllowance = {}
): void {
  const expectedEndpointOrigins = allowance.expectedEndpointOrigins ?? []
  const expectedConsoleText = allowance.expectedConsoleText ?? []
  const pageErrors = diagnostics.pageErrors.filter((error) => !isBenign(error.message))
  const consoleErrors = diagnostics.console.filter(
    (entry) =>
      entry.type === "error" &&
      !isBenignOnboardingConsoleEntry(entry) &&
      !matchesExpectedEndpoint(entry.location?.url, expectedEndpointOrigins) &&
      !expectedConsoleText.some((pattern) => pattern.test(entry.text))
  )
  const requestFailures = diagnostics.requestFailures.filter(
    (request) =>
      !isBenign(request.url) &&
      !isBenign(request.errorText) &&
      !matchesExpectedEndpoint(request.url, expectedEndpointOrigins)
  )

  if (pageErrors.length || consoleErrors.length || requestFailures.length) {
    throw new Error(
      [
        "Critical onboarding UAT diagnostics detected.",
        `pageErrors=${pageErrors.length}`,
        `consoleErrors=${consoleErrors.length}`,
        `requestFailures=${requestFailures.length}`,
      ].join(" ")
    )
  }
}

export async function waitForBackendConnection(page: Page): Promise<void> {
  await waitForConnection(page, 30_000)
}
