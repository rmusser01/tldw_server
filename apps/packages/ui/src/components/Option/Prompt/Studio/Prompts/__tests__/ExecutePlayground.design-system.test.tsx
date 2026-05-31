import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ExecutePlayground } from "../ExecutePlayground"

const mocks = vi.hoisted(() => ({
  getPrompt: vi.fn(),
  executePrompt: vi.fn(),
  getLlmProviders: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        if (fallbackOrOptions.defaultValue) {
          return Object.entries(fallbackOrOptions).reduce(
            (acc, [name, value]) =>
              name === "defaultValue"
                ? acc
                : acc.replace(new RegExp(`{{${name}}}`, "g"), String(value)),
            fallbackOrOptions.defaultValue
          )
        }
        return key
      }
      return key
    }
  })
}))

vi.mock("@/services/prompt-studio", () => ({
  getPrompt: (...args: unknown[]) =>
    (mocks.getPrompt as (...args: unknown[]) => unknown)(...args),
  executePrompt: (...args: unknown[]) =>
    (mocks.executePrompt as (...args: unknown[]) => unknown)(...args),
  getLlmProviders: (...args: unknown[]) =>
    (mocks.getLlmProviders as (...args: unknown[]) => unknown)(...args)
}))

const renderPlayground = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <ExecutePlayground open promptId={42} onClose={vi.fn()} />
    </QueryClientProvider>
  )
}

describe("ExecutePlayground design-system feedback", () => {
  beforeEach(() => {
    mocks.getPrompt.mockReset()
    mocks.executePrompt.mockReset()
    mocks.getLlmProviders.mockReset()

    mocks.getPrompt.mockResolvedValue({
      data: {
        data: {
          id: 42,
          name: "Variable prompt",
          version_number: 3,
          user_prompt: "Summarize {{topic}} for {{audience}}"
        }
      }
    })
    mocks.getLlmProviders.mockResolvedValue({
      data: {
        providers: [
          {
            id: "openai",
            name: "OpenAI",
            models: [{ id: "gpt-4.1", name: "GPT-4.1" }]
          }
        ]
      }
    })
  })

  it("renders variable guidance through the design-system Alert", async () => {
    renderPlayground()

    const title = await screen.findByText("Variables detected in prompt")
    const alert = title.closest('[data-ds-component="Alert"]')
    expect(alert).not.toBeNull()
    expect(screen.getByText("Found variables: topic, audience")).toBeInTheDocument()
  })
})
