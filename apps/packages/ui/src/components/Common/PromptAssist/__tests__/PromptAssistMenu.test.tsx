import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { PromptAssistMenu } from "../PromptAssistMenu"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValue?: string,
      options?: Record<string, string>
    ) =>
      (defaultValue ?? _key).replace(
        /{{(\w+)}}/g,
        (_, key) => options?.[key] ?? ""
      )
  })
}))

describe("PromptAssistMenu", () => {
  it("exposes exactly the two Track A actions and discloses an Auto route", async () => {
    const user = userEvent.setup()
    const onImproveNow = vi.fn()
    const onReviewChanges = vi.fn()
    render(
      <PromptAssistMenu
        draft="Summarize this report."
        capability="supported"
        modelSelection={{ selected_model: "auto" }}
        onImproveNow={onImproveNow}
        onReviewChanges={onReviewChanges}
      />
    )

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))

    expect(screen.getByText("Active model: Auto")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Improve now/ })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Review changes/ })
    ).toBeInTheDocument()
    expect(screen.queryByRole("menu")).not.toBeInTheDocument()
    expect(screen.queryByText(/Build from recipe/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /Review changes/ }))
    expect(onReviewChanges).toHaveBeenCalledTimes(1)
    expect(onImproveNow).not.toHaveBeenCalled()
  })

  it("discloses the concrete route and invokes Improve now", async () => {
    const user = userEvent.setup()
    const onImproveNow = vi.fn()
    render(
      <PromptAssistMenu
        draft="Draft"
        capability="supported"
        modelSelection={{
          selected_model: "openai/gpt-5-mini",
          provider_hint: "openai"
        }}
        modelDisplayName="GPT-5 mini"
        onImproveNow={onImproveNow}
        onReviewChanges={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(
      screen.getByText("Active model: GPT-5 mini (openai)")
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /Improve now/ }))
    expect(onImproveNow).toHaveBeenCalledTimes(1)
  })

  it("offers model recovery and keeps model actions disabled without a route", async () => {
    const user = userEvent.setup()
    const onSelectModel = vi.fn()
    render(
      <PromptAssistMenu
        draft="Draft"
        capability="supported"
        modelSelection={null}
        onImproveNow={vi.fn()}
        onReviewChanges={vi.fn()}
        onSelectModel={onSelectModel}
      />
    )

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(
      screen.getByText("Select a chat model to improve this draft.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Select model" }))
    expect(onSelectModel).toHaveBeenCalledTimes(1)
  })

  it.each([
    [
      "unsupported" as const,
      "Prompt improvement requires a newer server version."
    ],
    ["unknown" as const, "Reconnect to check prompt improvement availability."]
  ])(
    "fails closed with %s capability recovery",
    async (capability, recovery) => {
      const user = userEvent.setup()
      render(
        <PromptAssistMenu
          draft="Draft"
          capability={capability}
          modelSelection={{ selected_model: "auto" }}
          onImproveNow={vi.fn()}
          onReviewChanges={vi.fn()}
        />
      )

      await user.click(screen.getByRole("button", { name: "Improve prompt" }))
      expect(screen.getByText(recovery)).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: /Review changes/ })
      ).toBeDisabled()
    }
  )

  it("explains the empty-draft recovery and returns focus on Escape", async () => {
    const user = userEvent.setup()
    render(
      <PromptAssistMenu
        draft="   "
        capability="supported"
        modelSelection={{ selected_model: "auto" }}
        onImproveNow={vi.fn()}
        onReviewChanges={vi.fn()}
      />
    )

    const trigger = screen.getByRole("button", { name: "Improve prompt" })
    trigger.focus()
    await user.click(trigger)
    expect(
      screen.getByText("Write a draft to enable prompt improvement.")
    ).toBeInTheDocument()
    await user.keyboard("{Escape}")
    expect(trigger).toHaveFocus()
  })

  it("treats a whitespace-only selected model as missing", async () => {
    const user = userEvent.setup()
    render(
      <PromptAssistMenu
        draft="Draft"
        capability="supported"
        modelSelection={{ selected_model: "   " }}
        onImproveNow={vi.fn()}
        onReviewChanges={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: "Improve prompt" }))
    expect(
      screen.getByText("Select a chat model to improve this draft.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Improve now/ })).toBeDisabled()
  })

  it("uses a native disclosure flow, closes on focus leave, and clamps to the viewport", async () => {
    const user = userEvent.setup()
    render(
      <>
        <PromptAssistMenu
          draft="Draft"
          capability="supported"
          modelSelection={{ selected_model: "auto" }}
          onImproveNow={vi.fn()}
          onReviewChanges={vi.fn()}
        />
        <button>After prompt actions</button>
      </>
    )

    const trigger = screen.getByRole("button", { name: "Improve prompt" })
    trigger.focus()
    await user.keyboard("{Enter}")
    const disclosure = screen.getByRole("group", {
      name: "Prompt improvement actions"
    })
    expect(disclosure).toHaveClass("max-w-[calc(100vw-1rem)]")
    expect(screen.queryByRole("menu")).not.toBeInTheDocument()

    await user.tab()
    expect(screen.getByRole("button", { name: /Improve now/ })).toHaveFocus()
    await user.tab()
    await user.tab()
    expect(
      screen.getByRole("button", { name: "After prompt actions" })
    ).toHaveFocus()
    expect(
      screen.queryByRole("group", { name: "Prompt improvement actions" })
    ).not.toBeInTheDocument()
  })
})
