import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeAll, describe, expect, it, vi } from "vitest"
import Markdown from "../Markdown"
import { useArtifactsStore } from "@/store/artifacts"
import { MarkdownPreview } from "../MarkdownPreview"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

beforeAll(async () => {
  await import("../Markdown")
})

describe("MarkdownPreview typography", () => {
  it.each(["xs", "sm", "base"] as const)(
    "keeps base prose for %s lists",
    async (size) => {
      const { container } = render(
        <MarkdownPreview
          size={size}
          content={
            "- Parent\n  - Nested\n\n1. Ordered\n\n- [x] Done\n- [ ] Pending"
          }
        />
      )
      await screen.findByText("Parent")
      expect(container.querySelector(".prose")).not.toBeNull()
      expect(container.querySelector("ul ul li")).toHaveTextContent("Nested")
      expect(container.querySelector("ol li")).toHaveTextContent("Ordered")
      expect(container.querySelectorAll('input[type="checkbox"]')).toHaveLength(
        2
      )
      expect(container.querySelector(".prose")).toHaveClass(
        "[&_.task-list-item]:list-none"
      )
    }
  )
})

it("does not offer artifacts in a hostless MarkdownPreview", async () => {
  render(<MarkdownPreview content={"```python\npreview_code = 1\n```"} />)
  await screen.findByText("preview_code")
  expect(
    screen.queryByRole("button", { name: /view code/i })
  ).not.toBeInTheDocument()
})

it("preserves default Markdown manual code artifacts", () => {
  const source = "chat_code = 1"
  render(<Markdown message={"```python\n" + source + "\n```"} />)
  fireEvent.click(screen.getByRole("button", { name: "View code" }))
  expect(useArtifactsStore.getState().active).toEqual(
    expect.objectContaining({
      kind: "code",
      language: "python",
      content: source
    })
  )
  useArtifactsStore.getState().closeArtifact()
})
