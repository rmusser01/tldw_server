import { afterAll, afterEach } from "vitest"
import { JSDOM } from "jsdom"

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost/"
})

const { window } = dom

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: window
})
Object.defineProperty(globalThis, "document", {
  configurable: true,
  value: window.document
})
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: window.navigator
})
Object.defineProperty(globalThis, "HTMLElement", {
  configurable: true,
  value: window.HTMLElement
})
Object.defineProperty(globalThis, "SVGElement", {
  configurable: true,
  value: window.SVGElement
})
Object.defineProperty(globalThis, "Element", {
  configurable: true,
  value: window.Element
})
Object.defineProperty(globalThis, "Node", {
  configurable: true,
  value: window.Node
})
Object.defineProperty(globalThis, "ShadowRoot", {
  configurable: true,
  value: window.ShadowRoot
})
Object.defineProperty(globalThis, "getComputedStyle", {
  configurable: true,
  value: window.getComputedStyle.bind(window)
})

if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false
    })
  })
}

Object.defineProperty(globalThis, "matchMedia", {
  configurable: true,
  value: window.matchMedia.bind(window)
})

if (!globalThis.ResizeObserver) {
  Object.defineProperty(globalThis, "ResizeObserver", {
    configurable: true,
    value: class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  })
}

const { cleanup } = await import("@testing-library/react")

afterEach(() => {
  cleanup()
})

afterAll(() => {
  dom.window.close()
})
