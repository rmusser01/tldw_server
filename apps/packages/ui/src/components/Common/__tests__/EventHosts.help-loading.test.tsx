import React from "react"
import { MemoryRouter } from "react-router-dom"
import { act, render } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import { EventOnlyHosts } from "../EventHosts"

const mocks = vi.hoisted(() => ({ loadHelp: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
vi.mock("../CommandPaletteHost", () => ({ CommandPaletteHost: () => null }))
vi.mock("../PageHelpModal", () => {
  mocks.loadHelp()
  return { PageHelpModal: () => null }
})

it("does not load closed help when the settings event hosts mount", async () => {
  await act(async () => { render(<MemoryRouter><EventOnlyHosts /></MemoryRouter>) })
  await act(async () => { await vi.dynamicImportSettled() })
  expect(mocks.loadHelp).not.toHaveBeenCalled()
})
