import React from "react"
import { renderToStaticMarkup } from "react-dom/server"
import common from "@/assets/locale/en/common.json"
import { describe, expect, it, vi } from "vitest"
import {
  normalizeNotificationConfig,
  patchNotificationApi
} from "../antd-notification-compat"

describe("antd notification compatibility helpers", () => {
  it("maps deprecated message field to title", () => {
    expect(
      normalizeNotificationConfig({
        message: "Done",
        description: "Saved"
      })
    ).toEqual({
      title: "Done",
      description: "Saved"
    })
  })

  it("drops deprecated message field when title already exists", () => {
    expect(
      normalizeNotificationConfig({
        title: "Already set",
        message: "Legacy",
        description: "Details"
      })
    ).toEqual({
      title: "Already set",
      description: "Details"
    })
  })

  it("patches notification methods idempotently", () => {
    const open = vi.fn()
    const success = vi.fn()
    const api = { open, success }

    patchNotificationApi(api)
    patchNotificationApi(api)

    api.open({ message: "Hello" })
    api.success({ message: "Success" })

    expect(open).toHaveBeenCalledWith({ title: "Hello" })
    expect(success).toHaveBeenCalledWith({ title: "Success" })
  })
})

it.each(["message", "title"])("renders a localized %s resource label through the real React renderer", field => {
  const config = Object.freeze({ [field]: common.error, description: "Your draft is still here." })
  const normalized = normalizeNotificationConfig(config) as { title: React.ReactNode }
  expect(renderToStaticMarkup(React.createElement("div", null, normalized.title))).toBe("<div>Error</div>")
  expect(config[field]).toBe(common.error)
})
it("preserves explicit rich React titles and does not extract labels from React elements", () => {
  const title = React.createElement("span", { title: "Saved" }, "Saved")
  const normalized = normalizeNotificationConfig({ title, message: common.error }) as { title: React.ReactNode }
  expect(normalized.title).toBe(title)
  expect(renderToStaticMarkup(React.createElement("div", null, normalized.title))).toBe('<div><span title="Saved">Saved</span></div>')
})
