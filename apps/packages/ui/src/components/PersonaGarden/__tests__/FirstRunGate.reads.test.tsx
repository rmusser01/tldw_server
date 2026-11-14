import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FirstRunGate } from "../FirstRunGate"

// XP-16: on /chat and /knowledge the gate can never show (source destinations
// continue without optional personalization), yet it still read
// /api/v1/persona/profiles on every load, next to the chat's own first-run
// nudge reading the same list.

const mocks = vi.hoisted(() => ({ apiSend: vi.fn() }))
vi.mock("@/services/api-send", () => ({ apiSend: mocks.apiSend }))

const personaProfileReads = () =>
  mocks.apiSend.mock.calls.filter(
    ([payload]) => payload?.path === "/api/v1/persona/profiles"
  ).length

describe("FirstRunGate persona profile reads (XP-16)", () => {
  beforeEach(() => {
    localStorage.clear()
    mocks.apiSend.mockReset().mockResolvedValue({ ok: true, data: [] })
  })

  it("does not read persona profiles on a destination where the gate never shows", async () => {
    render(
      <FirstRunGate onStartSetup={vi.fn()} allowCompletedSetup>
        <div>Chat</div>
      </FirstRunGate>
    )
    expect(screen.getByText("Chat")).toBeInTheDocument()
    await Promise.resolve()
    expect(personaProfileReads()).toBe(0)
    expect(screen.queryByTestId("first-run-gate-overlay")).not.toBeInTheDocument()
  })

  it("still reads persona profiles and offers setup on other pages", async () => {
    render(
      <FirstRunGate onStartSetup={vi.fn()}>
        <div>Notes</div>
      </FirstRunGate>
    )
    expect(await screen.findByTestId("first-run-gate-overlay")).toBeInTheDocument()
    await waitFor(() => expect(personaProfileReads()).toBe(1))
  })
})
