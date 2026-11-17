import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider, notification } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createInstance } from "i18next"
import { I18nextProvider, initReactI18next } from "react-i18next"
import common from "@/assets/locale/en/common.json"
import settings from "@/assets/locale/en/settings.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { PromptPageErrorBoundary } from "../../../PromptPageErrorBoundary"
import { TestCaseRunModal } from "../TestCaseRunModal"

const mocks = vi.hoisted(() => ({ listPrompts: vi.fn(), getLlmProviders: vi.fn(), runTestCases: vi.fn() }))
vi.mock("@/services/prompt-studio", () => ({
  listPrompts: (...args: unknown[]) => mocks.listPrompts(...args),
  getLlmProviders: (...args: unknown[]) => mocks.getLlmProviders(...args),
  runTestCases: (...args: unknown[]) => mocks.runTestCases(...args)
}))

beforeEach(() => {
  mocks.listPrompts.mockReset().mockResolvedValue({ ok: true, status: 200, data: { data: [{ id: 2, name: "Required template", version_number: 2 }] } })
  mocks.getLlmProviders.mockReset().mockResolvedValue({
    ok: true, status: 200, data: { default_provider: "ollama", providers: [{ name: "ollama", models: ["qwen2.5:0.5b"] }] }
  })
  mocks.runTestCases.mockReset()
  vi.spyOn(notification, "error").mockImplementation(() => undefined)
})
afterEach(() => vi.restoreAllMocks())

describe("Test case refusal remains visible without a transient toast", () => {
  it.each(["input envelope", "transport exception"])("retains %s guidance in the actual modal and clears it on Close", async (kind) => {
    const message = kind === "input envelope" ? "Required prompt inputs are missing or empty." : "Connection unavailable."
    if (kind === "input envelope") mocks.runTestCases.mockResolvedValue({ ok: false, status: 400, error: message })
    else mocks.runTestCases.mockRejectedValue(new Error(message))
    const i18n = createInstance()
    await i18n.use(ICUWithInterpolation).use(initReactI18next).init({
      lng: "en", fallbackLng: false, ns: ["settings", "common"], defaultNS: "settings",
      resources: { en: { common, settings } }, interpolation: { escapeValue: false },
      react: { useSuspense: false }
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const onClose = vi.fn()
    const view = (open: boolean) => (
      <I18nextProvider i18n={i18n}>
        <ConfigProvider theme={{ token: { motion: false } }}>
          <QueryClientProvider client={client}>
            <PromptPageErrorBoundary>
              <TestCaseRunModal open={open} projectId={1} testCaseIds={[2]} onClose={onClose} />
            </PromptPageErrorBoundary>
          </QueryClientProvider>
        </ConfigProvider>
      </I18nextProvider>
    )
    const mounted = render(view(true))
    const user = userEvent.setup()
    try {
      await screen.findByText("qwen2.5:0.5b", { exact: true })
      await user.click(screen.getByRole("combobox", { name: "Select Prompt to Run Against" }))
      await user.click(await screen.findByText("Required template (v2)", { exact: true }))
      await user.click(screen.getByRole("button", { name: "Run Tests" }))
      await waitFor(() => {
        expect(screen.queryByTestId("prompts-error-boundary")).not.toBeInTheDocument()
        const error = within(screen.getByRole("dialog")).getByRole("alert")
        expect(error).toBeVisible()
        expect(error).toHaveTextContent(common.error.label)
        expect(error).toHaveTextContent(message)
      })
      expect(notification.error).toHaveBeenCalledWith({ message: common.error.label, description: message })
      expect(mocks.runTestCases).toHaveBeenCalledExactlyOnceWith({
        prompt_id: 2, test_case_ids: [2], provider: "ollama", model: "qwen2.5:0.5b"
      })
      expect(screen.queryByText("Results", { exact: true })).not.toBeInTheDocument()
      expect(screen.queryByText(/ungraded/)).not.toBeInTheDocument()
      expect(within(screen.getByRole("dialog")).getByText("Required template (v2)", { exact: true })).toBeVisible()
      await user.click(within(screen.getByRole("dialog")).getAllByRole("button", { name: "Close" }).at(-1)!)
      expect(onClose).toHaveBeenCalledOnce()
      mounted.rerender(view(false))
      mounted.rerender(view(true))
      await waitFor(() => expect(within(screen.getByRole("dialog")).queryByRole("alert")).not.toBeInTheDocument())
    } finally {
      mounted.unmount()
      client.clear()
    }
  })
})
