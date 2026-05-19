// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ResultsListItem } from "../ResultsListItem"
import type { ResultItemWithMediaId } from "../types"

vi.mock("antd", () => ({
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  List: {
    Item: ({ actions, children }: any) => (
      <div>
        <div>{children}</div>
        <div>{actions}</div>
      </div>
    ),
  },
  Tag: ({ children, color }: any) => (
    <span data-color={color}>{children}</span>
  ),
}))

const t = (_key: string, defaultValue?: any) =>
  typeof defaultValue === "string"
    ? defaultValue
    : defaultValue?.defaultValue ?? _key

describe("ResultsListItem status labels", () => {
  it("labels submit failures distinctly from processing failures", () => {
    const item: ResultItemWithMediaId = {
      id: "submit-1",
      status: "error",
      outcome: "submit_failed",
      type: "video",
      url: "https://example.com/talk",
      mediaId: null,
      error: "Queue unavailable",
    }

    render(
      <ResultsListItem
        item={item}
        processOnly={false}
        onDownloadJson={vi.fn()}
        onOpenMedia={vi.fn()}
        onDiscussInChat={vi.fn()}
        t={t as any}
      />
    )

    expect(screen.getByText("Not submitted")).toBeTruthy()
    expect(screen.queryByText("Failed")).toBeNull()
  })
})
