import { afterEach, expect, it, vi } from "vitest"
import { waitFor } from "@testing-library/react"
import type { Page } from "@playwright/test"
import { ChatPage } from "../page-objects/ChatPage"

vi.mock("../helpers", () => ({
  dispatchKeyboardShortcut: vi.fn(),
  waitForConnection: vi.fn()
}))
vi.mock("@playwright/test", async () => {
  const { expect } = await import("vitest")
  const { waitFor } = await import("@testing-library/react")
  return {
    expect: (locator: DomLocator) => ({
      toBeVisible: async () => waitFor(() => expect(locator.element).toBeVisible()),
      toBeEnabled: async () => waitFor(() => expect(locator.element).toBeEnabled()),
      toHaveAttribute: async (name: string, value: string) => expect(locator.element).toHaveAttribute(name, value)
    })
  }
})

// Only the browser boundary is replaced; disclosure decisions run unchanged.
class DomLocator {
  constructor(private readonly selector: string) {}
  get element() { return document.querySelector(this.selector) }
  first() { return this }
  filter() { return this }
  or(other: DomLocator) { return new DomLocator(`${this.selector}, ${other.selector}`) }
  async isVisible() { return !!this.element && !(this.element as HTMLElement).hidden }
  async getAttribute(name: string) { return this.element?.getAttribute(name) ?? null }
  async click() { (this.element as HTMLElement).click() }
}

function setup(mode: "casual" | "pro", expanded = false) {
  const toggleId = "composer-options-toggle"
  document.body.innerHTML = `<button data-testid="${toggleId}" aria-expanded="${expanded}">Options</button>
    <section data-testid="composer-options-panel" ${expanded ? "" : "hidden"}>Character controls</section>`
  const button = document.querySelector("button")!
  const click = vi.fn(() => {
    button.setAttribute("aria-expanded", "true")
    document.querySelector<HTMLElement>("section")!.hidden = false
  })
  button.addEventListener("click", click)
  const page = {
    locator: (selector: string) => new DomLocator(selector),
    getByRole: () => new DomLocator("[data-constructor-only]"),
    getByTestId: (id: string) => new DomLocator(`[data-testid="${id}"]`),
    reload: vi.fn()
  } as unknown as Page
  return { page, click }
}

afterEach(() => { document.body.innerHTML = "" })

it.each(["casual", "pro"] as const)("opens the current %s disclosure before character lookup", async mode => {
  const { page, click } = setup(mode)
  await new ChatPage(page).openConversationControls()
  expect(click).toHaveBeenCalledOnce()
  expect(document.querySelector("button")).toHaveAttribute("aria-expanded", "true")
  expect(page.reload).not.toHaveBeenCalled()
})

it.each(["casual", "pro"] as const)("keeps already open %s controls open", async mode => {
  const { page, click } = setup(mode, true)
  await new ChatPage(page).openConversationControls()
  expect(click).not.toHaveBeenCalled()
  expect(document.querySelector("section")).toBeVisible()
})

it("waits for the shared Tools disclosure before character lookup", async () => {
  const { page, click } = setup("casual")
  const button = document.querySelector("button")!
  button.remove()
  const opening = new ChatPage(page).openConversationControls()
  await Promise.resolve()
  expect(click).not.toHaveBeenCalled()
  document.body.prepend(button)
  await opening
  expect(click).toHaveBeenCalledOnce()
  await waitFor(() => expect(document.querySelector("section")).toBeVisible())
})

it("waits for the saved Tools preference before opening conversation controls", async () => {
  const { page, click } = setup("casual")
  const button = document.querySelector<HTMLButtonElement>("button")!
  button.disabled = true
  const opening = new ChatPage(page).openConversationControls()
  await Promise.resolve()
  expect(click).not.toHaveBeenCalled()
  button.disabled = false
  await opening
  expect(click).toHaveBeenCalledOnce()
})
