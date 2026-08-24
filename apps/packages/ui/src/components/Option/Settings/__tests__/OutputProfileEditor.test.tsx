import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { ChatMacroSettings } from "@/services/chat-macros"
import { OutputProfileEditor } from "../OutputProfileEditor"

const mocks = vi.hoisted(() => ({
  updateChatMacroSettings: vi.fn()
}))

vi.mock("@/services/chat-macros", () => ({
  updateChatMacroSettings: mocks.updateChatMacroSettings
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

const makeSettings = (overrides: Partial<ChatMacroSettings> = {}): ChatMacroSettings => ({
  disabled_builtins: ["example"],
  user_macro_enabled: { research: true },
  output_profiles: {
    default: {
      format: "structured_sections",
      sections: ["summary", "action_items"],
      section_titles: {},
      include_branch_outputs: false
    },
    concise: {
      format: "single_response",
      sections: ["summary"],
      section_titles: {},
      include_branch_outputs: false
    }
  },
  unrelated_setting: { preserve: true },
  ...overrides
})

const success = (settings: ChatMacroSettings) => ({
  ok: true,
  status: 200,
  data: { settings }
})

const renderProfileEditor = (props: Partial<React.ComponentProps<typeof OutputProfileEditor>> = {}) => {
  const allProps: React.ComponentProps<typeof OutputProfileEditor> = {
    settings: makeSettings(),
    onSaved: vi.fn(),
    ...props
  }

  return {
    ...render(<OutputProfileEditor {...allProps} />),
    props: allProps
  }
}

describe("OutputProfileEditor", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.updateChatMacroSettings.mockImplementation((settings: ChatMacroSettings) =>
      Promise.resolve(success(settings))
    )
  })

  it("saves ordered sections and custom headings without dropping other settings", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    await user.click(screen.getByRole("button", { name: "Add section" }))
    await user.type(screen.getByLabelText("Section key 3"), "risks")
    await user.type(screen.getByLabelText("Section heading 3"), "Risk register")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() =>
      expect(mocks.updateChatMacroSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          disabled_builtins: ["example"],
          unrelated_setting: { preserve: true },
          output_profiles: expect.objectContaining({
            default: expect.objectContaining({
              sections: ["summary", "action_items", "risks"],
              section_titles: { risks: "Risk register" }
            })
          })
        })
      )
    )
  })

  it("saves single response format and branch-output inclusion", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    await user.click(screen.getByRole("button", { name: "Single response" }))
    await user.click(screen.getByLabelText("Include branch outputs"))
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() =>
      expect(mocks.updateChatMacroSettings).toHaveBeenCalledWith(
        expect.objectContaining({
          output_profiles: expect.objectContaining({
            default: expect.objectContaining({
              format: "single_response",
              include_branch_outputs: true
            })
          })
        })
      )
    )
  })

  it("moves sections while retaining stable row controls", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    await user.click(screen.getByRole("button", { name: "Move section 2 up" }))

    expect(screen.getByLabelText("Section key 1")).toHaveValue("action_items")
    expect(screen.getByLabelText("Section key 2")).toHaveValue("summary")
    expect(screen.getAllByTestId("output-profile-section-row")).toHaveLength(2)
    expect(screen.getByRole("button", { name: "Move section 1 up" })).toBeDisabled()
  })

  it("creates and deletes named profiles while protecting default", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    expect(screen.getByRole("button", { name: "Delete profile" })).toBeDisabled()

    await user.type(screen.getByLabelText("New profile name"), "review")
    await user.click(screen.getByRole("button", { name: "Add profile" }))

    expect(screen.getByLabelText("Profile")).toHaveValue("review")
    expect(screen.getByRole("button", { name: "Delete profile" })).toBeEnabled()

    await user.click(screen.getByRole("button", { name: "Delete profile" }))

    expect(screen.getByLabelText("Profile")).toHaveValue("default")
  })

  it("blocks invalid and duplicate section keys before saving", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    await user.click(screen.getByRole("button", { name: "Add section" }))
    await user.type(screen.getByLabelText("Section key 3"), "Risk register")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Section keys must use lowercase letters, numbers, and underscores."
    )
    expect(mocks.updateChatMacroSettings).not.toHaveBeenCalled()

    await user.clear(screen.getByLabelText("Section key 3"))
    await user.type(screen.getByLabelText("Section key 3"), "summary")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Section keys must be unique.")
    expect(mocks.updateChatMacroSettings).not.toHaveBeenCalled()
  })

  it("blocks profiles with more than ten sections and headings over 128 characters", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    for (let index = 0; index < 9; index += 1) {
      await user.click(screen.getByRole("button", { name: "Add section" }))
      await user.type(screen.getByLabelText(`Section key ${index + 3}`), `section_${index}`)
    }
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "A profile can contain at most 10 sections."
    )
    expect(mocks.updateChatMacroSettings).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: "Remove section 11" }))
    await user.type(screen.getByLabelText("Section heading 1"), "x".repeat(129))
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Section headings must be 128 characters or fewer."
    )
    expect(mocks.updateChatMacroSettings).not.toHaveBeenCalled()
  })

  it("validates new profile names before creating drafts", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    await user.type(screen.getByLabelText("New profile name"), "Bad name")
    await user.click(screen.getByRole("button", { name: "Add profile" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Profile names must use lowercase letters, numbers, and underscores."
    )

    await user.clear(screen.getByLabelText("New profile name"))
    await user.type(screen.getByLabelText("New profile name"), "concise")
    await user.click(screen.getByRole("button", { name: "Add profile" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("A profile with this name already exists.")
  })

  it("retains draft edits after a failed save", async () => {
    const user = userEvent.setup()
    mocks.updateChatMacroSettings.mockResolvedValueOnce({
      ok: false,
      status: 503,
      error: "Settings unavailable"
    })
    renderProfileEditor()

    await user.type(screen.getByLabelText("Section heading 1"), "Executive summary")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Settings unavailable")
    expect(screen.getByLabelText("Section heading 1")).toHaveValue("Executive summary")
    expect(screen.getByRole("button", { name: "Save profiles" })).toBeEnabled()
  })
})
