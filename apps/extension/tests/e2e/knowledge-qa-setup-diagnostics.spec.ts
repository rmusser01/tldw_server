import { expect, test } from "@playwright/test"

import { launchWithBuiltExtension } from "./utils/extension-build"
import {
  forceConnected,
  forceConnectionState,
  forceErrorUnreachable,
  forceUnconfigured,
  waitForConnectionStore,
} from "./utils/connection"

test.describe("Knowledge QA extension setup diagnostics", () => {
  test("shows concrete recovery checks when no server is configured", async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {},
      allowOffline: false,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-setup-diagnostics:missing-url")
      await forceUnconfigured(page, "knowledge-setup-diagnostics:missing-url")

      await expect(page.getByTestId("knowledge-setup-diagnostics")).toBeVisible()
      await expect(page.getByText("Server URL")).toBeVisible()
      await expect(
        page.getByText("Add a tldw server URL before Knowledge QA can search your library.")
      ).toBeVisible()
      await expect(
        page.getByText("Waiting for a server URL before checking credentials.")
      ).toBeVisible()
      await expect(page.getByRole("button", { name: "Finish setup" })).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("shows credential recovery when a server URL is saved without auth", async () => {
    const serverUrl = "http://127.0.0.1:8000"
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        tldwConfig: {
          serverUrl,
          authMode: "single-user",
        },
      },
      allowOffline: false,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-setup-diagnostics:missing-auth")
      await forceConnectionState(
        page,
        {
          phase: "unconfigured",
          serverUrl,
          isConnected: false,
          isChecking: false,
          offlineBypass: false,
          configStep: "auth",
          errorKind: "none",
          lastError: null,
          lastStatusCode: null,
          knowledgeStatus: "unknown",
          knowledgeLastCheckedAt: null,
          knowledgeError: null,
          hasCompletedFirstRun: true,
        },
        "knowledge-setup-diagnostics:missing-auth"
      )

      await expect(
        page.getByText("Add your credentials to use Knowledge QA")
      ).toBeVisible()
      await expect(page.getByText(`Configured server: ${serverUrl}`)).toBeVisible()
      await expect(
        page.getByText("Add the API key or login token for this tldw server.")
      ).toBeVisible()
      await expect(page.getByRole("button", { name: "Update credentials" })).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("shows host access and allowlist recovery when backend requests are blocked", async () => {
    const serverUrl = "http://127.0.0.1:8000"
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        tldwConfig: {
          serverUrl,
          authMode: "single-user",
          apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
        },
      },
      allowOffline: false,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-setup-diagnostics:blocked")
      await forceErrorUnreachable(
        page,
        {
          serverUrl,
          lastError:
            "Absolute URL requests are blocked unless the request origin is explicitly allowlisted.",
          lastStatusCode: 400,
        },
        "knowledge-setup-diagnostics:blocked"
      )

      await expect(
        page.getByText("Can't reach your tldw server right now")
      ).toBeVisible()
      await expect(page.getByText("Browser access")).toBeVisible()
      await expect(
        page.getByText(/Allowlist this server origin or grant extension host access/i)
      ).toBeVisible()
      await expect(page.getByRole("button", { name: "Retry connection" })).toBeVisible()
      await expect(
        page.getByRole("button", { name: "Health & diagnostics" })
      ).toBeVisible()
    } finally {
      await context.close()
    }
  })

  test("keeps the ready search workspace available after setup is healthy", async () => {
    const serverUrl = "http://dummy-tldw"
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        tldwConfig: {
          serverUrl,
          authMode: "single-user",
          apiKey: "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
        },
      },
      allowOffline: true,
    })

    try {
      await page.goto(`${optionsUrl}#/knowledge`, { waitUntil: "domcontentloaded" })
      await waitForConnectionStore(page, "knowledge-setup-diagnostics:ready")
      await forceConnected(page, { serverUrl }, "knowledge-setup-diagnostics:ready")

      await expect(page.getByTestId("knowledge-setup-diagnostics")).toBeHidden()
      await expect(page.getByRole("heading", { name: /Ask Your Library/i })).toBeVisible()
      await expect(page.getByLabel(/Search your knowledge base/i)).toBeVisible()
    } finally {
      await context.close()
    }
  })
})
