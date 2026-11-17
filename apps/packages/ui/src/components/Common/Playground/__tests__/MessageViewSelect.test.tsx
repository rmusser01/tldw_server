// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { MessageViewSelect } from "../MessageViewSelect"

it("offers the three message views with their purpose and changes only the view preference", async () => {
  const change = vi.fn()
  render(<MessageViewSelect value="transcript" onChange={change} />)
  fireEvent.mouseDown(screen.getByRole("combobox", { name: "Message view" }))
  expect(screen.getByRole("combobox", { name: "Message view" })).toHaveAttribute("aria-expanded", "true")
  expect(screen.getByText("For roleplay and character chat")).toBeInTheDocument()
  expect(screen.getByText("For sources and detailed review")).toBeInTheDocument()
  fireEvent.click(screen.getByText("Conversation"))
  expect(change).toHaveBeenCalledWith("conversation")
})
