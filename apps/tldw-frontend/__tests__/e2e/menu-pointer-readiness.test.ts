import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { isMenuPointerReady } from "../../e2e/utils/menu-pointer-readiness"

let item: HTMLElement
let label: HTMLElement
let rect: ReturnType<typeof vi.fn>
let hit: ReturnType<typeof vi.fn>
let animations: ReturnType<typeof vi.fn>
const bounds = (x = 10) => ({ x, y: 20, width: 100, height: 32 }) as DOMRect
const originalAnimations = Object.getOwnPropertyDescriptor(Element.prototype, "getAnimations")
const originalHit = Object.getOwnPropertyDescriptor(document, "elementFromPoint")

beforeEach(() => {
  item = document.createElement("li")
  label = document.createElement("span")
  item.append(label)
  document.body.append(item)
  rect = vi.fn(() => bounds())
  item.getBoundingClientRect = rect
  hit = vi.fn(() => label)
  Object.defineProperty(document, "elementFromPoint", { configurable: true, value: hit })
  animations = vi.fn(() => [])
  Object.defineProperty(Element.prototype, "getAnimations", { configurable: true, value: animations })
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => { callback(0); return 1 })
})
afterEach(() => {
  item.remove()
  vi.unstubAllGlobals()
  if (originalAnimations) Object.defineProperty(Element.prototype, "getAnimations", originalAnimations)
  else Reflect.deleteProperty(Element.prototype, "getAnimations")
  if (originalHit) Object.defineProperty(document, "elementFromPoint", originalHit)
  else Reflect.deleteProperty(document, "elementFromPoint")
})

it("accepts a stationary item receiving pointer input through its label", async () => {
  expect(await isMenuPointerReady(item)).toBe(true)
  expect(hit).toHaveBeenCalledWith(60, 36)
})
it("rejects a motionless closing item with pointer events disabled", async () => {
  item.style.pointerEvents = "none"
  expect(await isMenuPointerReady(item)).toBe(false)
})
it("rejects a visible item whose pointer target is the editor behind it", async () => {
  hit.mockReturnValue(document.body)
  expect(await isMenuPointerReady(item)).toBe(false)
})
it("rejects an item while its flyout changes position", async () => {
  rect.mockReturnValueOnce(bounds()).mockReturnValue(bounds(30))
  expect(await isMenuPointerReady(item)).toBe(false)
})
it("rejects a retained zero-size menu item", async () => {
  rect.mockReturnValue({ ...bounds(), width: 0 })
  expect(await isMenuPointerReady(item)).toBe(false)
})
it.each(["running", "pending"])("rejects an ancestor with %s animation", async state => {
  animations.mockImplementation(function (this: Element) {
    return this === document.body ? [{ playState: state, pending: state === "pending" }] : []
  })
  expect(await isMenuPointerReady(item)).toBe(false)
})
