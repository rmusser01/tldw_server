// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { useComposerEvents } from "../useComposerEvents"

// The native motion-completion boundary is held; editor state remains real.
const mountOwner = () => {
  let finishClose: (() => boolean) | undefined
  const Owner = () => {
    const events = useComposerEvents({})
    React.useEffect(() => {
      if (!events.openModelSettings) finishClose = events.completeModelSettingsClose
    }, [events.openModelSettings, events.completeModelSettingsClose])
    return (events.modelSettingsPresent ?? events.openModelSettings) ? (
      <section data-testid="model-editor" data-open={String(events.openModelSettings)}>
        <button onClick={() => events.setOpenModelSettings(false)}>Save settings</button>
      </section>
    ) : null
  }
  const mounted = render(<Owner />)
  const open = () => act(() => { window.dispatchEvent(new Event("tldw:open-model-settings")) })
  return { ...mounted, open, completion: () => finishClose }
}

describe("model settings native close lifecycle", () => {
  it("keeps the closed editor mounted until its own native completion", () => {
    const owner = mountOwner()
    owner.open()
    expect(screen.getByTestId("model-editor")).toHaveAttribute("data-open", "true")
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }))
    expect(screen.getByTestId("model-editor")).toHaveAttribute("data-open", "false")
    act(() => { expect(owner.completion()?.()).toBe(true) })
    expect(screen.queryByTestId("model-editor")).toBeNull()
    owner.unmount()
  })

  it("rejects an old completion after an event reopens the editor", () => {
    const owner = mountOwner()
    owner.open()
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }))
    const previous = owner.completion()
    owner.open()
    act(() => { expect(previous?.()).toBe(false) })
    expect(screen.getByTestId("model-editor")).toHaveAttribute("data-open", "true")
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }))
    act(() => { expect(owner.completion()?.()).toBe(true) })
    expect(screen.queryByTestId("model-editor")).toBeNull()
    owner.unmount()
  })

  it("does not accept close completion after the parent unmounts", () => {
    const owner = mountOwner()
    owner.open()
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }))
    const completion = owner.completion()
    owner.unmount()
    expect(completion?.()).toBe(false)
  })
})
