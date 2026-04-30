import React from "react"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { PrototypeWorkspacePage } from "../PrototypeWorkspacePage"

describe("PrototypeWorkspacePage", () => {
  afterEach(() => {
    window.history.replaceState({}, "", "/")
  })

  it("defaults to owner mode when no prototype session token is present", () => {
    window.history.pushState({}, "", "/prototype-workspaces")

    render(<PrototypeWorkspacePage />)

    expect(screen.getByTestId("prototype-workspace-mode")).toHaveTextContent(
      "owner"
    )
  })

  it("switches to collaborator mode when a prototype session token is present", () => {
    window.history.pushState(
      {},
      "",
      "/prototype-workspaces?prototype_session_token=proto-123"
    )

    render(<PrototypeWorkspacePage />)

    expect(screen.getByTestId("prototype-workspace-mode")).toHaveTextContent(
      "collaborator"
    )
  })
})
