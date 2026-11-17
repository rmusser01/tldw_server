import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { Form } from "antd"
import { describe, expect, it, vi } from "vitest"

const support = vi.hoisted(() => ({
  current: {} as { topKUnavailableReason?: string; minPUnavailableReason?: string }
}))
vi.mock("@/hooks/useSelectedModelSamplingSupport", () => ({
  useSelectedModelSamplingSupport: () => support.current
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => (typeof fallback === "string" ? fallback : key)
  })
}))

import { AdvancedParamsTab } from "../AdvancedParamsTab"

const Harness = () => {
  const [form] = Form.useForm()
  return (
    <Form form={form}>
      <AdvancedParamsTab form={form} providerOptions={[]} selectedModel="openai:gpt-4o" />
    </Form>
  )
}

const openSampling = () => fireEvent.click(screen.getByText("Advanced Sampling"))
const field = (name: "topK" | "minP") =>
  document.querySelector<HTMLInputElement>(`#${name}`) ??
  document.querySelector<HTMLInputElement>(`input[id$="${name}"]`)

// #3283: Top K and Min P say why they can't be used with this model.
describe("AdvancedParamsTab sampling support", () => {
  it("disables Top K and Min P with the reason when the provider rejects them", () => {
    support.current = {
      topKUnavailableReason: "Not supported by OpenAI",
      minPUnavailableReason: "Not supported by OpenAI"
    }
    render(<Harness />)
    openSampling()
    expect(screen.getAllByText("Not supported by OpenAI")).toHaveLength(2)
    expect(field("topK")).toBeDisabled()
    expect(field("minP")).toBeDisabled()
  })

  it("leaves them usable when the provider accepts them", () => {
    support.current = {}
    render(<Harness />)
    openSampling()
    expect(screen.queryByText(/Not supported by/)).toBeNull()
    expect(field("topK")).toBeEnabled()
    expect(field("minP")).toBeEnabled()
  })
})
