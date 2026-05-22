import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { FlowCheckDiffPanel } from "../FlowCheckDiffPanel"

vi.mock("antd", () => ({
  Alert: ({ title, description }: any) => (
    <div>
      {title}
      {description}
    </div>
  ),
  Button: ({ children, onClick, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.()} {...rest}>
      {children}
    </button>
  ),
  Radio: {
    Group: ({ value, onChange, children }: any) => (
      <div>
        {React.Children.map(children, (child: any) =>
          React.cloneElement(child, {
            groupValue: value,
            onGroupChange: onChange
          })
        )}
      </div>
    ),
    Button: ({ value, groupValue, onGroupChange, children }: any) => (
      <button type="button" onClick={() => onGroupChange?.({ target: { value } })}>
        {children}
        {groupValue === value ? "*" : ""}
      </button>
    )
  },
  Space: ({ children }: any) => <div>{children}</div>
}))

describe("FlowCheckDiffPanel", () => {
  it("renders flow-check diff and supports accept/reject actions", () => {
    const onAcceptChunk = vi.fn()
    const onRejectChunk = vi.fn()

    render(
      <FlowCheckDiffPanel
        diff={"--- original\n+++ suggested\n@@\n-Intro paragraph\n+Improved intro paragraph"}
        onAcceptChunk={onAcceptChunk}
        onRejectChunk={onRejectChunk}
      />
    )

    expect(screen.getByText("Accept")).toBeInTheDocument()
    expect(screen.getByText("Reject")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Accept"))
    fireEvent.click(screen.getByText("Reject"))

    expect(onAcceptChunk).toHaveBeenCalledWith("all")
    expect(onRejectChunk).toHaveBeenCalledWith("all")
  })

  it("renders flow-check callouts with design-system Alert wrappers", () => {
    render(
      <FlowCheckDiffPanel
        diff=""
        issues={[{ message: "Missing required section" }]}
        onAcceptChunk={vi.fn()}
        onRejectChunk={vi.fn()}
      />
    )

    const issueText = screen.getByText("Missing required section", { exact: false })
    const noDiffText = screen.getByText("Run flow-check to generate suggestions.", { exact: false })

    expect(issueText).toBeInTheDocument()
    expect(noDiffText).toBeInTheDocument()
    expect(issueText.closest("[data-ds-component='Alert']")).toBeInTheDocument()
    expect(noDiffText.closest("[data-ds-component='Alert']")).toBeInTheDocument()
  })
})
