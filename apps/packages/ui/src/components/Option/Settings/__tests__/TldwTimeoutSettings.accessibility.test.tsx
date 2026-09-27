import React from "react"
import { cleanup, render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider } from "antd"
import { createInstance, type TFunction } from "i18next"
import { afterEach, beforeAll, expect, it, vi } from "vitest"
import settings from "@/assets/locale/en/settings.json"
import { TldwTimeoutSettings, type TldwTimeoutSettingsProps } from "../TldwTimeoutSettings"

const labels = [
  "Request Timeout (seconds)",
  "Streaming Idle Timeout (seconds)",
  "Chat Request Timeout",
  "Chat startup timeout",
  "Chat Stream Idle",
  "RAG Request Timeout",
  "Media Request Timeout",
  "Upload Request Timeout"
]
const i18n = createInstance()
const noop = () => {}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { settings } } })
})
afterEach(cleanup)

const props = (): TldwTimeoutSettingsProps => ({
  t: i18n.t as TFunction,
  message: { success: vi.fn() } as unknown as TldwTimeoutSettingsProps["message"],
  requestTimeoutSec: 10, setRequestTimeoutSec: noop,
  streamIdleTimeoutSec: 15, setStreamIdleTimeoutSec: noop,
  chatRequestTimeoutSec: 120, setChatRequestTimeoutSec: noop,
  chatStartupTimeoutSec: 120, setChatStartupTimeoutSec: noop,
  chatStreamIdleTimeoutSec: 15, setChatStreamIdleTimeoutSec: noop,
  ragRequestTimeoutSec: 120, setRagRequestTimeoutSec: noop,
  mediaRequestTimeoutSec: 60, setMediaRequestTimeoutSec: noop,
  uploadRequestTimeoutSec: 60, setUploadRequestTimeoutSec: noop,
  timeoutPreset: "balanced", setTimeoutPreset: noop
})

it.each(labels)("names and focuses the numeric input through its visible label: %s", async label => {
  const user = userEvent.setup()
  render(<ConfigProvider theme={{ token: { motion: false } }}><TldwTimeoutSettings {...props()} /></ConfigProvider>)
  await user.click(screen.getByRole("button", { name: /Advanced Timeouts/ }))

  const input = screen.getByRole("spinbutton", { name: label, exact: true })
  expect(screen.getByLabelText(label, { exact: true })).toBe(input)
  await user.click(screen.getByText(label, { selector: "label", exact: true }))
  expect(input).toHaveFocus()
})

it("keeps labels scoped to their own instance and stable when controlled values rerender", async () => {
  const user = userEvent.setup()
  const renderSettings = (requestTimeoutSec: number) => (
    <ConfigProvider theme={{ token: { motion: false } }}>
      <section aria-label="First timeouts"><TldwTimeoutSettings {...props()} requestTimeoutSec={requestTimeoutSec} /></section>
      <section aria-label="Second timeouts"><TldwTimeoutSettings {...props()} /></section>
    </ConfigProvider>
  )
  const view = render(renderSettings(10))
  for (const button of screen.getAllByRole("button", { name: /Advanced Timeouts/ })) {
    await user.click(button)
  }
  const first = within(screen.getByRole("region", { name: "First timeouts" }))
  const second = within(screen.getByRole("region", { name: "Second timeouts" }))
  const inputIds = () => screen.getAllByRole("spinbutton").map(input => input.id)
  const ids = inputIds()
  expect(ids).toHaveLength(16)
  expect(ids.every(Boolean)).toBe(true)
  expect(new Set(ids).size).toBe(16)

  view.rerender(renderSettings(42))
  expect(inputIds()).toEqual(ids)
  expect(first.getByRole("spinbutton", { name: labels[0], exact: true })).toHaveValue(42)
  expect(second.getByRole("spinbutton", { name: labels[0], exact: true })).toHaveValue(10)
  for (const label of labels) {
    const input = second.getByRole("spinbutton", { name: label, exact: true })
    expect(second.getByLabelText(label, { exact: true })).toBe(input)
    await user.click(second.getByText(label, { selector: "label", exact: true }))
    expect(input).toHaveFocus()
  }
})
