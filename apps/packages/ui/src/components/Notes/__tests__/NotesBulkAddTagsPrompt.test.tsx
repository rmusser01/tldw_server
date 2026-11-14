import React from "react"
import { Modal } from "antd"
import type { ModalFuncProps } from "antd"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { promptBulkAddTags } from "../NotesBulkAddTagsPrompt"

const t = (key: string, options?: Record<string, unknown>) => {
  const template = String(options?.defaultValue ?? key)
  return template.replace(/\{\{(\w+)\}\}/g, (_, name: string) => String(options?.[name] ?? ""))
}

const openPrompt = (noteCount = 3) => {
  const confirm = vi.fn((_config: ModalFuncProps) => ({ destroy: vi.fn() }))
  const result = promptBulkAddTags(
    { confirm },
    {
      noteCount,
      suggestions: ["research", "summary"],
      renderSuggestionLabel: (tag) => <span data-testid={`suggestion-${tag}`}>{tag}</span>,
      t
    }
  )
  const config = confirm.mock.calls[0][0]
  return { confirm, config, result }
}

describe("promptBulkAddTags", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("opens through the app's themed modal API, not the static Modal.confirm", () => {
    const staticConfirm = vi.spyOn(Modal, "confirm")
    const { confirm, config } = openPrompt()

    expect(confirm).toHaveBeenCalledTimes(1)
    expect(staticConfirm).not.toHaveBeenCalled()
    expect(config.title).toBe("Add tags to selected notes")
    expect(config.okText).toBe("Add tags")
  })

  it("states that existing tags are kept", () => {
    const { config } = openPrompt(3)
    render(<>{config.content}</>)

    expect(screen.getByTestId("notes-bulk-add-tags-effect")).toHaveTextContent(
      "Adds these tags to 3 selected notes. Their existing tags are kept."
    )
  })

  it("offers existing tags as suggestions and resolves with the chosen and typed tags", async () => {
    const { config, result } = openPrompt()
    render(<>{config.content}</>)

    const input = screen.getByRole("combobox", { name: "Tags to add" })
    fireEvent.mouseDown(input)
    fireEvent.click(await screen.findByTestId("suggestion-research"))
    fireEvent.change(input, { target: { value: "brand-new" } })
    fireEvent.keyDown(input, { key: "Enter", code: "Enter", keyCode: 13 })
    await waitFor(() => {
      expect(screen.getByTestId("notes-bulk-add-tags-select")).toHaveTextContent("brand-new")
    })

    config.onOk?.()
    await expect(result).resolves.toEqual(["research", "brand-new"])
  })

  it("resolves null when cancelled", async () => {
    const { config, result } = openPrompt()
    config.onCancel?.()
    await expect(result).resolves.toBeNull()
  })

  it("falls back to the static modal outside antd's App provider", () => {
    const staticConfirm = vi
      .spyOn(Modal, "confirm")
      .mockImplementation(() => ({ destroy: vi.fn(), update: vi.fn() }))
    void promptBulkAddTags({}, { noteCount: 1, suggestions: [], t })
    expect(staticConfirm).toHaveBeenCalledTimes(1)
  })
})
