import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import { SystemPromptTemplatesModal } from "../SystemPromptTemplates"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("antd", async () => {
  const React = await import("react")

  const Input = (props: any) => (
    <input
      aria-label={props["aria-label"] ?? props.placeholder}
      placeholder={props.placeholder}
      value={props.value}
      onChange={props.onChange}
    />
  )

  Input.TextArea = (props: any) => (
    <textarea
      aria-label={props["aria-label"] ?? props.placeholder}
      placeholder={props.placeholder}
      value={props.value}
      onChange={props.onChange}
      onBlur={props.onBlur}
    />
  )

  return {
    Modal: ({
      open,
      title,
      children
    }: {
      open?: boolean
      title?: React.ReactNode
      children: React.ReactNode
    }) =>
      open ? (
        <section role="dialog" aria-label={String(title || "Dialog")}>
          {children}
        </section>
      ) : null,
    Input,
    Tabs: ({
      items
    }: {
      items: Array<{ key: string; label: React.ReactNode }>
    }) => (
      <div>
        {items.map((item) => (
          <button key={item.key} type="button">
            {item.label}
          </button>
        ))}
      </div>
    ),
    Empty: ({ description }: { description?: React.ReactNode }) => (
      <div>{description}</div>
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
  }
})

describe("SystemPromptTemplatesModal", () => {
  it("shows an editable current system prompt above template search", async () => {
    const user = userEvent.setup()
    const onSystemPromptChange = vi.fn()

    const Wrapper = () => {
      const [systemPrompt, setSystemPrompt] =
        React.useState("Stay in character.")

      return (
        <SystemPromptTemplatesModal
          open
          onClose={vi.fn()}
          onSelect={vi.fn()}
          systemPrompt={systemPrompt}
          onSystemPromptChange={(nextPrompt) => {
            setSystemPrompt(nextPrompt)
            onSystemPromptChange(nextPrompt)
          }}
        />
      )
    }

    render(<Wrapper />)

    const editor = screen.getByLabelText(/current system prompt/i)
    const search = screen.getByLabelText(/search system prompts/i)

    expect(editor.compareDocumentPosition(search)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING
    )

    await user.clear(editor)
    await user.type(editor, "Speak plainly.")

    expect(editor).toHaveValue("Speak plainly.")
    expect(onSystemPromptChange).not.toHaveBeenCalled()

    fireEvent.blur(editor)

    expect(onSystemPromptChange).toHaveBeenLastCalledWith("Speak plainly.")
  })
})
