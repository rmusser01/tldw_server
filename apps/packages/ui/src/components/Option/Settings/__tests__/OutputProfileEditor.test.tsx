import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { ChatMacroSettings } from "@/services/chat-macros"
import { OutputProfileEditor } from "../OutputProfileEditor"

const mocks = vi.hoisted(() => ({
  updateChatMacroSettings: vi.fn(),
  updateChatMacroOutputProfiles: vi.fn()
}))

vi.mock("@/services/chat-macros", () => ({
  updateChatMacroSettings: mocks.updateChatMacroSettings,
  updateChatMacroOutputProfiles: mocks.updateChatMacroOutputProfiles
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
    mocks.updateChatMacroOutputProfiles.mockImplementation((output_profiles: ChatMacroSettings["output_profiles"]) =>
      Promise.resolve(success(makeSettings({ output_profiles })))
    )
  })

  it("uses the themed surface token for profile fields", () => {
    renderProfileEditor()

    expect(screen.getByLabelText("Profile")).toHaveClass("bg-surface")
  })

  it("saves ordered sections and custom headings without sending unrelated settings", async () => {
    const user = userEvent.setup()
    const { props } = renderProfileEditor()

    await user.click(screen.getByRole("button", { name: "Add section" }))
    await user.type(screen.getByLabelText("Section key 3"), "risks")
    await user.type(screen.getByLabelText("Section heading 3"), "Risk register")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() =>
      expect(mocks.updateChatMacroOutputProfiles).toHaveBeenCalledWith({
        concise: props.settings.output_profiles.concise,
        default: expect.objectContaining({
          sections: ["summary", "action_items", "risks"],
          section_titles: { risks: "Risk register" }
        })
      })
    )
    expect(mocks.updateChatMacroSettings).not.toHaveBeenCalled()
  })

  it("moves a custom heading with its renamed section key", async () => {
    const user = userEvent.setup()
    const settings = makeSettings()
    settings.output_profiles.default.section_titles = { summary: "Executive summary" }
    renderProfileEditor({ settings })

    await user.clear(screen.getByLabelText("Section key 1"))
    await user.type(screen.getByLabelText("Section key 1"), "overview")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() => expect(mocks.updateChatMacroOutputProfiles).toHaveBeenCalled())
    const [profiles] = mocks.updateChatMacroOutputProfiles.mock.calls[0] as [ChatMacroSettings["output_profiles"]]
    expect(profiles.default.sections).toEqual(["overview", "action_items"])
    expect(profiles.default.section_titles).toEqual({
      overview: "Executive summary"
    })
  })

  it("retains both row headings when a duplicate section key is corrected", async () => {
    const user = userEvent.setup()
    const settings = makeSettings()
    settings.output_profiles.default.section_titles = {
      summary: "Executive summary",
      action_items: "Action plan"
    }
    renderProfileEditor({ settings })

    await user.clear(screen.getByLabelText("Section key 1"))
    await user.type(screen.getByLabelText("Section key 1"), "action_items")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Section keys must be unique.")
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Section heading 1")).toHaveValue("Executive summary")
    expect(screen.getByLabelText("Section heading 2")).toHaveValue("Action plan")

    await user.clear(screen.getByLabelText("Section key 1"))
    await user.type(screen.getByLabelText("Section key 1"), "overview")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() => expect(mocks.updateChatMacroOutputProfiles).toHaveBeenCalled())
    const [profiles] = mocks.updateChatMacroOutputProfiles.mock.calls[0] as [ChatMacroSettings["output_profiles"]]
    expect(profiles.default.section_titles).toEqual({
      overview: "Executive summary",
      action_items: "Action plan"
    })
  })

  it("saves single response format and branch-output inclusion", async () => {
    const user = userEvent.setup()
    renderProfileEditor()

    await user.click(screen.getByRole("button", { name: "Single response" }))
    await user.click(screen.getByLabelText("Include branch outputs"))
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() =>
      expect(mocks.updateChatMacroOutputProfiles).toHaveBeenCalledWith(
        expect.objectContaining({
          default: expect.objectContaining({
            format: "single_response",
            include_branch_outputs: true
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
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()

    await user.clear(screen.getByLabelText("Section key 3"))
    await user.type(screen.getByLabelText("Section key 3"), "summary")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Section keys must be unique.")
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()
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
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: "Remove section 11" }))
    await user.type(screen.getByLabelText("Section heading 1"), "x".repeat(129))
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Section headings must be 128 characters or fewer."
    )
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()
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
    mocks.updateChatMacroOutputProfiles.mockResolvedValueOnce({
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

  it("preserves dirty profiles and selection when settings refresh", async () => {
    const user = userEvent.setup()
    const { props, rerender } = renderProfileEditor()
    await user.selectOptions(screen.getByLabelText("Profile"), "concise")
    await user.type(screen.getByLabelText("Section heading 1"), "Local heading")

    rerender(<OutputProfileEditor {...props} settings={makeSettings({ disabled_builtins: [] })} />)

    expect(screen.getByLabelText("Profile")).toHaveValue("concise")
    expect(screen.getByLabelText("Section heading 1")).toHaveValue("Local heading")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))
    await waitFor(() => expect(mocks.updateChatMacroOutputProfiles).toHaveBeenCalledWith(
      expect.objectContaining({ concise: expect.objectContaining({ section_titles: { summary: "Local heading" } }) })
    ))
  })

  it("preserves an unfinished new profile name when settings refresh", async () => {
    const user = userEvent.setup()
    const { props, rerender } = renderProfileEditor()
    await user.type(screen.getByLabelText("New profile name"), "review")

    rerender(<OutputProfileEditor {...props} settings={makeSettings()} />)

    expect(screen.getByLabelText("New profile name")).toHaveValue("review")
  })

  it("preserves profile additions and deletions when settings refresh", async () => {
    const user = userEvent.setup()
    const { props, rerender } = renderProfileEditor()
    await user.selectOptions(screen.getByLabelText("Profile"), "concise")
    await user.click(screen.getByRole("button", { name: "Delete profile" }))
    await user.type(screen.getByLabelText("New profile name"), "review")
    await user.click(screen.getByRole("button", { name: "Add profile" }))

    rerender(<OutputProfileEditor {...props} settings={makeSettings()} />)

    expect(screen.getByLabelText("Profile")).toHaveValue("review")
    expect(screen.queryByRole("option", { name: "concise" })).not.toBeInTheDocument()
  })

  it("refreshes pristine profiles and resumes refreshing after a successful save", async () => {
    const user = userEvent.setup()
    const { props, rerender } = renderProfileEditor()
    const refreshed = makeSettings()
    refreshed.output_profiles.default.section_titles = { summary: "Server heading" }
    rerender(<OutputProfileEditor {...props} settings={refreshed} />)
    expect(screen.getByLabelText("Section heading 1")).toHaveValue("Server heading")

    await user.type(screen.getByLabelText("Section heading 1"), " edited")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))
    await screen.findByRole("status")
    expect(screen.getByLabelText("Section heading 1")).toHaveValue("Server heading edited")

    rerender(<OutputProfileEditor {...props} settings={makeSettings()} />)
    expect(screen.getByLabelText("Section heading 1")).toHaveValue("")
  })

  it("prevents removing the last section", async () => {
    const user = userEvent.setup()
    renderProfileEditor()
    await user.click(screen.getByRole("button", { name: "Remove section 2" }))

    expect(screen.getByRole("button", { name: "Remove section 1" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Remove section 1" }))
    expect(screen.getByLabelText("Section key 1")).toHaveValue("summary")
  })

  it.each(["structured_sections", "single_response"] as const)("rejects an empty %s profile before saving", async (format) => {
    const user = userEvent.setup()
    const settings = makeSettings()
    settings.output_profiles.default = { ...settings.output_profiles.default, format, sections: [] }
    renderProfileEditor({ settings })
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("A profile must contain at least one section.")
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()
    expect(mocks.updateChatMacroSettings).not.toHaveBeenCalled()
  })

  it("round-trips existing backend-supported profile names without new-name validation", async () => {
    const user = userEvent.setup()
    const settings = makeSettings()
    settings.output_profiles["Review-Notes"] = settings.output_profiles.concise
    const { props } = renderProfileEditor({ settings })
    await user.selectOptions(screen.getByLabelText("Profile"), "Review-Notes")
    await user.type(screen.getByLabelText("Section heading 1"), "Review notes")
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() => expect(props.onSaved).toHaveBeenCalledWith(expect.objectContaining({
      output_profiles: expect.objectContaining({
        "Review-Notes": expect.objectContaining({ section_titles: { summary: "Review notes" } })
      })
    })))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("prevents draft mutations while a save is pending and applies the normalized response", async () => {
    const user = userEvent.setup()
    const settings = makeSettings()
    settings.output_profiles.concise = {
      format: "single_response",
      sections: ["summary", "action_items"],
      section_titles: {},
      include_branch_outputs: false
    }
    const normalizedSettings = makeSettings()
    normalizedSettings.output_profiles.concise = {
      format: "structured_sections",
      sections: ["normalized"],
      section_titles: { normalized: "Normalized heading" },
      include_branch_outputs: true
    }
    let resolveSave: ((response: ReturnType<typeof success>) => void) | undefined
    mocks.updateChatMacroOutputProfiles.mockImplementationOnce(
      () => new Promise((resolve) => {
        resolveSave = resolve
      })
    )
    const { props, rerender } = renderProfileEditor({ settings })

    await user.selectOptions(screen.getByLabelText("Profile"), "concise")
    expect(screen.getByRole("button", { name: "Delete profile" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Move section 1 down" })).toBeEnabled()
    await user.click(screen.getByRole("button", { name: "Save profiles" }))

    await waitFor(() => expect(mocks.updateChatMacroOutputProfiles).toHaveBeenCalled())
    expect(screen.getByLabelText("Profile")).toBeDisabled()
    expect(screen.getByLabelText("New profile name")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Add profile" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Delete profile" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Structured sections" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Single response" })).toBeDisabled()
    expect(screen.getByLabelText("Include branch outputs")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Add section" })).toBeDisabled()
    expect(screen.getByLabelText("Section key 1")).toBeDisabled()
    expect(screen.getByLabelText("Section heading 1")).toBeDisabled()
    expect(screen.getByRole("button", { name: "Move section 1 down" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Remove section 1" })).toBeDisabled()

    await user.type(screen.getByLabelText("Section key 1"), "mutated")
    expect(screen.getByLabelText("Section key 1")).toHaveValue("summary")

    rerender(<OutputProfileEditor {...props} settings={normalizedSettings} />)
    expect(screen.getByLabelText("Section key 1")).toHaveValue("summary")

    await act(async () => {
      resolveSave?.(success(normalizedSettings))
    })

    expect(await screen.findByLabelText("Section key 1")).toHaveValue("normalized")
    expect(screen.getByRole("button", { name: "Structured sections" })).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByLabelText("Include branch outputs")).toBeChecked()
  })
})
