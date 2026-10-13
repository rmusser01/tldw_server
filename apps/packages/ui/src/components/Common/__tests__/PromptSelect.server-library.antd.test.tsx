/**
 * CC-05 (#3112) with the real antd Dropdown: antd closes the menu after any
 * menu click, so this guards that picking a server prompt keeps the menu open
 * while the prompt loads (and when it fails), then closes once it applies.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  buildFakeServerPrompt,
  createFakePromptsApi
} from "@/services/__tests__/server-prompts-api-fixture"
import { PromptSelect } from "../PromptSelect"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getAllPrompts: vi.fn(async () => [] as unknown[]),
  getPromptById: vi.fn(async () => undefined)
}))

vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const i18n = createInstance()
  await i18n.init({ lng: "en", resources: {}, interpolation: { escapeValue: false } })
  return {
    useTranslation: () => ({
      t: (...args: Parameters<typeof i18n.t>) => i18n.t(...args)
    })
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

vi.mock("@/db/dexie/helpers", () => ({
  updatePrompt: vi.fn(),
  markPromptSyncError: vi.fn(),
  getAllPrompts: mocks.getAllPrompts,
  getPromptById: mocks.getPromptById,
  upsertServerLibraryPromptCopy: vi.fn()
}))

const createDeferred = () => {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((done, fail) => {
    resolve = done
    reject = fail
  })
  return { promise, resolve, reject }
}

/** Open = rendered and neither hidden nor leaving (jsdom never ends motion). */
const isMenuOpen = () => {
  const popup = screen.queryByRole("menu")?.closest(".ant-dropdown")
  if (!popup) return false
  return !Array.from(popup.classList).some(
    (name) => name === "ant-dropdown-hidden" || name.endsWith("-leave")
  )
}

describe("PromptSelect server prompts with the real antd Dropdown", { timeout: 60_000 }, () => {
  let detailGate = createDeferred()

  beforeEach(() => {
    vi.clearAllMocks()
    detailGate = createDeferred()
    const api = createFakePromptsApi([
      buildFakeServerPrompt(5, { name: "Rewrite plainly", user_prompt: "Rewrite this plainly" })
    ])
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (/\/api\/v1\/prompts\/5$/.test(String(request.path))) {
        await detailGate.promise
      }
      return api.handle(request) ?? {}
    })
  })

  const renderPicker = () => {
    const props = {
      selectedSystemPrompt: undefined,
      systemPrompt: "",
      setSystemPrompt: vi.fn(),
      setSelectedSystemPrompt: vi.fn(),
      setSelectedQuickPrompt: vi.fn()
    }
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <PromptSelect {...props} />
      </QueryClientProvider>
    )
    return props
  }

  it("keeps the menu open while a server prompt loads, then closes it once applied", async () => {
    const user = userEvent.setup()
    const props = renderPicker()

    await user.click(await screen.findByRole("button", { name: "selectAPrompt" }))
    await user.click(await screen.findByRole("menuitem", { name: /Rewrite plainly/ }, { timeout: 5_000 }))

    expect(await screen.findByText("Loading prompt…")).toBeInTheDocument()
    expect(isMenuOpen()).toBe(true)

    detailGate.resolve()
    await waitFor(() =>
      expect(props.setSelectedQuickPrompt).toHaveBeenCalledWith("Rewrite this plainly")
    )
    await waitFor(() => expect(isMenuOpen()).toBe(false))
  })

  it("keeps the menu open with a retry hint when a server prompt fails to load", async () => {
    const user = userEvent.setup()
    const props = renderPicker()

    await user.click(await screen.findByRole("button", { name: "selectAPrompt" }))
    await user.click(await screen.findByRole("menuitem", { name: /Rewrite plainly/ }, { timeout: 5_000 }))
    detailGate.reject(new Error("Request failed: 500"))

    expect(
      await screen.findByText("Couldn't load this prompt from the server. Select it to retry.")
    ).toBeInTheDocument()
    expect(isMenuOpen()).toBe(true)
    expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled()
  })
})
