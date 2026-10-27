// @vitest-environment jsdom
import React from "react"
import { Tooltip } from "antd"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { Button } from "../Button"

afterEach(cleanup)

it.each(["hover", "focus"] as const)("forwards installed Tooltip %s trigger handlers", async (trigger) => {
  render(<Tooltip title="Character settings help" trigger={trigger} mouseEnterDelay={0} mouseLeaveDelay={0}>
    <Button>Character settings</Button>
  </Tooltip>)
  const button = screen.getByRole("button", { name: "Character settings" })
  if (trigger === "hover") fireEvent.mouseEnter(button)
  else fireEvent.focus(button)
  expect(await screen.findByRole("tooltip")).toHaveTextContent("Character settings help")
})

it("preserves native label, accessibility state, form attributes, pointer handlers and ref", () => {
  const ref = React.createRef<HTMLButtonElement>()
  const onPointerDown = vi.fn()
  render(<Button ref={ref} aria-label="Save character" aria-expanded={true} aria-controls="character-panel"
    aria-busy={true} form="character-form" name="action" value="save" onPointerDown={onPointerDown}>Save</Button>)
  const button = screen.getByRole("button", { name: "Save character" })
  expect(button).toHaveAttribute("aria-expanded", "true")
  expect(button).toHaveAttribute("aria-controls", "character-panel")
  expect(button).toHaveAttribute("aria-busy", "true")
  expect(button).toHaveAttribute("form", "character-form")
  expect(button).toHaveAttribute("name", "action")
  expect(button).toHaveAttribute("value", "save")
  expect(ref.current).toBe(button)
  fireEvent.pointerDown(button)
  expect(onPointerDown).toHaveBeenCalledOnce()
})

it("keeps managed loading and legacy styling/type props without leaking them", () => {
  render(<Button type="primary" htmlType="submit" size="small" shape="pill" iconOnly loading ariaLabel="Save" aria-busy={false}>Save</Button>)
  const button = screen.getByRole("button", { name: "Save" })
  expect(button).toHaveAttribute("type", "submit")
  expect(button).toHaveAttribute("aria-busy", "true")
  expect(button).toBeDisabled()
  for (const attribute of ["size", "shape", "icononly", "loading", "variant", "htmltype"]) expect(button).not.toHaveAttribute(attribute)
})
