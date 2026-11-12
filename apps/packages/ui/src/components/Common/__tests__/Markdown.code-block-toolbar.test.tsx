// @vitest-environment jsdom
/**
 * CM-07 (#3101, G17): fenced code in chat answers renders through the
 * "compact" and "github" variants. Both must carry a slim header with the
 * language and a working Copy button, like the full CodeBlock does.
 */
import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

const pythonReply = [
  "Here is the fix:",
  "",
  "```python",
  "def greet(name):",
  "    return f\"hi {name}\"",
  "```"
].join("\n")

const pythonSource = 'def greet(name):\n    return f"hi {name}"'

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe.each(["github", "compact"] as const)(
  "Markdown %s code blocks",
  (codeBlockVariant) => {
    it("shows the language and a Copy code button for a python fence", () => {
      render(<Markdown message={pythonReply} codeBlockVariant={codeBlockVariant} />)

      const copy = screen.getByRole("button", { name: "Copy code" })
      const block = copy.closest("[data-code-block]") as HTMLElement
      expect(block).not.toBeNull()
      expect(within(block).getByText("python")).toBeInTheDocument()
      expect(block.querySelector("pre")).not.toBeNull()
    })

    it("copies the exact source and confirms it", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
      const writeText = vi.fn().mockResolvedValue(undefined)
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: { writeText }
      })
      render(<Markdown message={pythonReply} codeBlockVariant={codeBlockVariant} />)

      fireEvent.click(screen.getByRole("button", { name: "Copy code" }))

      await waitFor(() => expect(writeText).toHaveBeenCalledWith(pythonSource))
      expect(await screen.findByText("Copied")).toBeInTheDocument()
      await act(async () => {
        vi.advanceTimersByTime(4_000)
      })
      expect(screen.queryByText("Copied")).not.toBeInTheDocument()
    })

    it("labels a fence without a language as plain text", () => {
      render(
        <Markdown
          message={["```", "just text", "```"].join("\n")}
          codeBlockVariant={codeBlockVariant}
        />
      )

      const block = screen
        .getByRole("button", { name: "Copy code" })
        .closest("[data-code-block]") as HTMLElement
      expect(within(block).getByText("text")).toBeInTheDocument()
    })
  }
)
