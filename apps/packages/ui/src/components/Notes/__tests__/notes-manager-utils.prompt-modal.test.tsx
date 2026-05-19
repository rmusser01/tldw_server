import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { promptModal } from "../notes-manager-utils"

type MockAntdInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  onPressEnter?: (event: React.KeyboardEvent<HTMLInputElement>) => void
}

const { mockConfirm } = vi.hoisted(() => ({
  mockConfirm: vi.fn()
}))

vi.mock("antd", () => ({
  Modal: {
    confirm: mockConfirm
  },
  Input: (props: MockAntdInputProps) =>
    React.createElement("input", props)
}))

const getPromptParts = () => {
  const config = mockConfirm.mock.calls[0]?.[0]
  const content = config.content as React.ReactElement<{ children?: React.ReactNode }>
  const children = React.Children.toArray(content.props.children)
  const inputElement = children[children.length - 1] as React.ReactElement<
    MockAntdInputProps
  >

  return {
    config,
    inputProps: inputElement.props
  }
}

describe("promptModal", () => {
  beforeEach(() => {
    mockConfirm.mockReset()
  })

  it("returns an empty string when submit is confirmed with blank input", async () => {
    const destroy = vi.fn()
    mockConfirm.mockImplementation(() => ({ destroy }))

    const prompt = promptModal({ title: "Assign tags" })
    const { inputProps } = getPromptParts()

    inputProps.onChange?.({
      target: { value: "   " }
    } as React.ChangeEvent<HTMLInputElement>)
    inputProps.onPressEnter?.({} as React.KeyboardEvent<HTMLInputElement>)

    await expect(prompt).resolves.toBe("")
    expect(destroy).toHaveBeenCalled()
  })

  it("returns null when the prompt is canceled", async () => {
    mockConfirm.mockImplementation(() => ({ destroy: vi.fn() }))

    const prompt = promptModal({ title: "Assign tags" })
    const { config } = getPromptParts()

    config.onCancel?.()

    await expect(prompt).resolves.toBeNull()
  })
})
