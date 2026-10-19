import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { PromptDrawer } from "../PromptDrawer"

const hookSpies = vi.hoisted(() => ({
  clearDraft: vi.fn(),
  saveDraft: vi.fn(),
  useFormDraftMock: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        if (fallbackOrOptions.defaultValue) {
          return Object.entries(fallbackOrOptions).reduce(
            (acc, [name, value]) =>
              name === "defaultValue"
                ? acc
                : acc.replace(new RegExp(`{{${name}}}`, "g"), String(value)),
            fallbackOrOptions.defaultValue
          )
        }
        return key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useFormDraft", () => ({
  useFormDraft: (options: unknown) => hookSpies.useFormDraftMock(options),
  formatDraftAge: () => "now"
}))

vi.mock("../Studio/Prompts/VersionHistoryDrawer", () => ({
  VersionHistoryDrawer: () => null
}))

describe("PromptDrawer stage 4 behavior", () => {
  const clickDrawerClose = () => {
    const closeButton = document.querySelector(
      ".ant-drawer-close"
    ) as HTMLButtonElement | null
    expect(closeButton).not.toBeNull()
    fireEvent.click(closeButton!)
  }

  const baseProps = {
    open: true,
    onClose: vi.fn(),
    mode: "create" as const,
    onSubmit: vi.fn(),
    isLoading: false,
    allTags: []
  }

  beforeEach(() => {
    hookSpies.clearDraft.mockReset()
    hookSpies.saveDraft.mockReset()
    hookSpies.useFormDraftMock.mockReset()
    hookSpies.useFormDraftMock.mockImplementation(() => ({
      hasDraft: false,
      draftData: null,
      saveDraft: hookSpies.saveDraft,
      clearDraft: hookSpies.clearDraft,
      applyDraft: vi.fn(),
      dismissDraft: vi.fn()
    }))
  })

  it("builds isolated draft storage keys using mode and prompt id", () => {
    render(<PromptDrawer {...baseProps} mode="create" />)

    const createCall = hookSpies.useFormDraftMock.mock.calls.at(-1)?.[0] as any
    expect(createCall.storageKey).toBe(`tldw-prompt-drawer-draft-${JSON.stringify(["local", "create", "new"])}`)
    expect(createCall.editId).toBeUndefined()

    render(
      <PromptDrawer
        {...baseProps}
        mode="edit"
        initialValues={{ id: "prompt-123", name: "Prompt A" }}
      />
    )

    const editCall = hookSpies.useFormDraftMock.mock.calls.at(-1)?.[0] as any
    expect(editCall.storageKey).toBe(`tldw-prompt-drawer-draft-${JSON.stringify(["local", "edit", "prompt-123"])}`)
    expect(editCall.editId).toBe("prompt-123")
  }, 15000)

  it("prompts on dirty close and supports cancel/discard paths", async () => {
    const onClose = vi.fn()
    render(<PromptDrawer {...baseProps} onClose={onClose} />)

    const nameInput = screen.getByTestId("prompt-drawer-name")
    fireEvent.change(nameInput, {
      target: { value: "Unsaved prompt" }
    })
    fireEvent.blur(nameInput)
    clickDrawerClose()

    expect(
      await screen.findByText("You have unsaved changes. Close anyway?")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("prompt-drawer-unsaved-cancel"))

    expect(onClose).not.toHaveBeenCalled()

    clickDrawerClose()
    fireEvent.click(await screen.findByTestId("prompt-drawer-unsaved-discard"))

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(hookSpies.clearDraft).toHaveBeenCalledTimes(1)
  })

  it("submits when choosing save from unsaved changes confirmation", async () => {
    const onSubmit = vi.fn()
    const onClose = vi.fn()
    render(<PromptDrawer {...baseProps} onSubmit={onSubmit} onClose={onClose} />)

    const nameInput = screen.getByTestId("prompt-drawer-name")
    fireEvent.change(nameInput, {
      target: { value: "Save before close" }
    })
    fireEvent.blur(nameInput)
    clickDrawerClose()
    fireEvent.click(await screen.findByTestId("prompt-drawer-unsaved-save"))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledTimes(1)
    })
    expect(onClose).not.toHaveBeenCalled()
    expect(hookSpies.clearDraft).toHaveBeenCalledTimes(1)
  })
})
