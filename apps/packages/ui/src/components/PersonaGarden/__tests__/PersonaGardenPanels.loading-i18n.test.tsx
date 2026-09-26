import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import commonEn from "@/assets/locale/en/common.json"
import sidepanelEn from "@/assets/locale/en/sidepanel.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { CommandsPanel } from "../CommandsPanel"
import { ConnectionsPanel } from "../ConnectionsPanel"
import { PoliciesPanel } from "../PoliciesPanel"
import { ScopesPanel } from "../ScopesPanel"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { fetchWithAuth: vi.fn() }
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasMcp: false },
    loading: false
  })
}))

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const apiResponse = (body: unknown) => ({
  ok: true,
  status: 200,
  json: async () => body
})

const renderPanel = async (panel: React.ReactNode) => {
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    ns: ["sidepanel", "common"],
    defaultNS: "sidepanel",
    resources: { en: { common: commonEn, sidepanel: sidepanelEn } },
    interpolation: { escapeValue: false }
  })
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } }
  })
  return render(
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>{panel}</QueryClientProvider>
    </I18nextProvider>
  )
}

const connection = {
  id: "conn-1",
  persona_id: "persona-1",
  name: "Example API",
  base_url: "https://example.test",
  auth_type: "none",
  headers_template: {},
  timeout_ms: 15000,
  allowed_hosts: ["example.test"],
  secret_configured: false,
  key_hint: null,
  created_at: null,
  last_modified: null
}

describe("Persona Garden loading labels with English resources and ICU", () => {
  beforeEach(() => {
    vi.mocked(tldwClient.fetchWithAuth).mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it.each([
    {
      name: "Scopes",
      panel: (
        <ScopesPanel
          selectedPersonaId="persona-1"
          selectedPersonaName="Persona One"
        />
      ),
      body: { rules: [] },
      empty: "No scope rules yet."
    },
    {
      name: "Policies",
      panel: (
        <PoliciesPanel selectedPersonaId="persona-1" hasPendingPlan={false} />
      ),
      body: { rules: [] },
      empty: "No policy rules yet."
    },
    {
      name: "Commands",
      panel: (
        <CommandsPanel
          selectedPersonaId="persona-1"
          selectedPersonaName="Persona One"
          isActive
        />
      ),
      body: { commands: [] },
      empty: /No direct voice commands yet/
    },
    {
      name: "Connections",
      panel: (
        <ConnectionsPanel
          selectedPersonaId="persona-1"
          selectedPersonaName="Persona One"
          isActive
        />
      ),
      body: [],
      empty: /No reusable connections yet/
    }
  ])(
    "renders $name pending and settled labels",
    async ({ panel, body, empty }) => {
      const pending = createDeferred<void>()
      vi.mocked(tldwClient.fetchWithAuth).mockImplementation(async (path) => {
        await pending.promise
        return apiResponse(String(path).endsWith("/connections") ? [] : body)
      })

      await renderPanel(panel)
      expect(screen.getByText("Loading…")).toBeInTheDocument()

      await act(async () => {
        pending.resolve()
      })
      expect(screen.queryByText("Loading…")).not.toBeInTheDocument()
      expect(screen.getByText(empty)).toBeInTheDocument()
    }
  )

  it("renders a scalar pending test label and then the connection result", async () => {
    const pending = createDeferred<ReturnType<typeof apiResponse>>()
    vi.mocked(tldwClient.fetchWithAuth)
      .mockResolvedValueOnce(apiResponse([connection]))
      .mockReturnValueOnce(pending.promise)

    await renderPanel(
      <ConnectionsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Persona One"
        isActive
      />
    )
    const test = await screen.findByRole("button", { name: "Test" })
    fireEvent.click(test)
    expect(test).toHaveTextContent("Loading…")
    expect(test).toBeDisabled()

    await act(async () => {
      pending.resolve(
        apiResponse({
          ok: true,
          status_code: 200,
          body_preview: "OK",
          latency_ms: 12,
          error: null
        })
      )
    })
    expect(screen.getByText("Test passed (200)")).toBeInTheDocument()
    expect(test).toHaveTextContent("Test")
    expect(test).toBeEnabled()
  })

  it("renders a scalar pending delete label and removes the completed connection", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true)
    const pending = createDeferred<ReturnType<typeof apiResponse>>()
    vi.mocked(tldwClient.fetchWithAuth)
      .mockResolvedValueOnce(apiResponse([connection]))
      .mockReturnValueOnce(pending.promise)

    await renderPanel(
      <ConnectionsPanel
        selectedPersonaId="persona-1"
        selectedPersonaName="Persona One"
        isActive
      />
    )
    const remove = await screen.findByRole("button", { name: "Delete" })
    fireEvent.click(remove)
    expect(remove).toHaveTextContent("Loading…")
    expect(remove).toBeDisabled()

    await act(async () => {
      pending.resolve(apiResponse(null))
    })
    await waitFor(() =>
      expect(screen.queryByText("Example API")).not.toBeInTheDocument()
    )
    expect(screen.getByText(/No reusable connections yet/)).toBeInTheDocument()
  })
})
