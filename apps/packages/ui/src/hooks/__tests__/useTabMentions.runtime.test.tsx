// @vitest-environment jsdom
/**
 * CC-11 (#3112): '@' tab mentions read browser tabs, which only the extension
 * can do. A web user who turned the old setting on got an empty "Select tab"
 * list on every '@'; outside the extension the stored setting is ignored.
 */
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useTabMentions } from "../useTabMentions"

const runtime = vi.hoisted(() => ({ extension: false }))

// The stored preference is I/O: the user turned tab mentions on.
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [true] }))
vi.mock("@/utils/browser-runtime", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/browser-runtime")>()),
  isExtensionRuntime: () => runtime.extension
}))

const query = vi.fn(async () => [
  { id: 7, title: "Release notes", url: "https://example.test/notes", status: "complete", active: false }
])

afterEach(() => {
  vi.unstubAllGlobals()
  query.mockClear()
})

const typeMention = async (extension: boolean) => {
  runtime.extension = extension
  vi.stubGlobal("browser", { tabs: { query } })
  const view = renderHook(() => useTabMentions(React.createRef<HTMLTextAreaElement>()))
  await act(async () => {
    await view.result.current.handleTextChange("Compare @rel", 12)
  })
  return view.result.current
}

describe("useTabMentions outside the extension", () => {
  it("offers open tabs for '@' in the extension", async () => {
    const mentions = await typeMention(true)
    expect(mentions.tabMentionsEnabled).toBe(true)
    expect(mentions.showMentions).toBe(true)
    expect(mentions.filteredTabs.map((tab) => tab.title)).toEqual(["Release notes"])
  })

  it("ignores the stored setting in the web app", async () => {
    const mentions = await typeMention(false)
    expect(mentions.tabMentionsEnabled).toBe(false)
    expect(mentions.showMentions).toBe(false)
    expect(query).not.toHaveBeenCalled()
  })
})
