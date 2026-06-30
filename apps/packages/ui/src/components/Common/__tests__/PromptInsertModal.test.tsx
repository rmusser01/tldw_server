import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  useQuery: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: mocks.useQuery
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    getPrompts: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      return _key
    }
  })
}))

import { PromptInsertModal } from "../PromptInsertModal"

describe("PromptInsertModal", () => {
  beforeEach(() => {
    mocks.useQuery.mockReset()
  })

  it("renders prompt load errors with the design-system Alert", () => {
    mocks.useQuery.mockReturnValue({
      data: [],
      isLoading: false,
      isError: true,
      error: new Error("Prompt service unavailable")
    })

    render(
      <PromptInsertModal
        open
        onClose={vi.fn()}
        onInsertPrompt={vi.fn()}
      />
    )

    expect(
      screen
        .getByText("Prompt service unavailable")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByText("Error")).toBeInTheDocument()
  })

  it("allows retrying prompt loading from the error alert", async () => {
    const user = userEvent.setup()
    const refetch = vi.fn()
    mocks.useQuery.mockReturnValue({
      data: [],
      isLoading: false,
      isError: true,
      error: new Error("Prompt service unavailable"),
      refetch
    })

    render(
      <PromptInsertModal
        open
        onClose={vi.fn()}
        onInsertPrompt={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(refetch).toHaveBeenCalledTimes(1)
  })
})
