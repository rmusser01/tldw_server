import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { afterEach, describe, expect, it } from "vitest"
import type { PromptStudioStatus } from "@/services/prompt-studio"
import { QueueHealthWidget } from "../Studio/QueueHealthWidget"

const i18n = createInstance()
void i18n.init({ lng: "en", fallbackLng: "en", resources: { en: { settings: {} } },
  initImmediate: false, interpolation: { escapeValue: false } })

function mount(status?: PromptStudioStatus | null) {
  return render(<ConfigProvider theme={{ token: { motion: false } }}>
    <I18nextProvider i18n={i18n}><QueueHealthWidget status={status} /></I18nextProvider>
  </ConfigProvider>)
}

async function inspectSummary() {
  fireEvent.mouseEnter(screen.getByTestId("queue-health-widget"))
  const tooltip = await screen.findByRole("tooltip")
  await waitFor(() => expect(tooltip).toBeVisible())
  return tooltip
}

afterEach(cleanup)

describe("QueueHealthWidget with canonical Jobs percentage status", () => {
  it.each([
    ["100", 100, "100%", "text-success", "Healthy - all jobs processing normally."],
    ["85", 85, "85%", "text-danger", "Degraded - success rate dropped to 85%."],
    ["0", 0, "0%", "text-danger", "Degraded - success rate dropped to 0%."],
    ["1", 1, "1%", "text-danger", "Degraded - success rate dropped to 1%."],
    ["absent", undefined, "0%", "text-danger", "Degraded - success rate dropped to 0%."]
  ] as const)("shows canonical rate %s without a second conversion or fractional inference", async (_label, rate, text, color, summary) => {
    mount({ queue_depth: 0, processing: 0, leases: {}, success_rate: rate })
    const widget = screen.getByTestId("queue-health-widget")
    expect(within(widget).getByText(text).parentElement).toHaveClass(color)
    const tooltip = await inspectSummary()
    expect(within(tooltip).getByText(summary)).toBeVisible()
    expect(within(tooltip).getByText(text)).toBeVisible()
  })

  it("keeps recent failures ahead of backlog when canonical success is degraded", async () => {
    mount({ queue_depth: 11, processing: 1, leases: {}, success_rate: 85, by_status: { failed: 4 } })
    const tooltip = await inspectSummary()
    expect(within(tooltip).getByText("Degraded - 4 failures recently (85% success).")).toBeVisible()
    expect(within(tooltip).getByText("failed:")).toBeVisible()
    expect(within(tooltip).getByText("4")).toBeVisible()
  })

  it("retains backlog precedence with a successful canonical queue", async () => {
    mount({ queue_depth: 11, processing: 2, leases: {}, success_rate: 100 })
    expect(within(await inspectSummary()).getByText("Degraded - queue backlog at 11 jobs.")).toBeVisible()
  })

  it("retains healthy processing and processing-time details", async () => {
    mount({ queue_depth: 2, processing: 3, leases: {}, success_rate: 100, avg_processing_time_seconds: 0.25 })
    const tooltip = await inspectSummary()
    expect(within(tooltip).getByText("Healthy - 3 jobs processing normally.")).toBeVisible()
    expect(within(tooltip).getByText("250ms")).toBeVisible()
  })

  it("does not fabricate a widget when status is unavailable", () => {
    mount(null)
    expect(screen.queryByTestId("queue-health-widget")).not.toBeInTheDocument()
  })
})
