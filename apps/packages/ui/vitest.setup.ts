import "@testing-library/jest-dom/vitest"
import { cleanup } from "@testing-library/react"
import { afterAll, afterEach, beforeAll, vi } from "vitest"

const originalGetComputedStyle = window.getComputedStyle.bind(window)
const originalConsoleLog = console.log.bind(console)
const originalConsoleDebug = console.debug.bind(console)
const originalConsoleInfo = console.info.bind(console)
const originalConsoleWarn = console.warn.bind(console)
const originalConsoleError = console.error.bind(console)
const originalMutationObserver = window.MutationObserver

const TEXTAREA_STYLE_FALLBACKS: Record<string, string> = {
  "letter-spacing": "normal",
  "line-height": "20px",
  "padding-top": "0px",
  "padding-bottom": "0px",
  "padding-left": "0px",
  "padding-right": "0px",
  "font-family": "sans-serif",
  "font-weight": "400",
  "font-size": "16px",
  "font-variant": "normal",
  "text-rendering": "auto",
  "text-transform": "none",
  width: "320px",
  "text-indent": "0px",
  "border-width": "0px",
  "border-top-width": "0px",
  "border-bottom-width": "0px",
  "box-sizing": "border-box",
  "word-break": "break-word",
  "white-space": "pre-wrap"
}

const TEXTAREA_NUMERIC_PROPERTIES = new Set([
  "line-height",
  "padding-top",
  "padding-bottom",
  "padding-left",
  "padding-right",
  "font-size",
  "width",
  "text-indent",
  "border-width",
  "border-top-width",
  "border-bottom-width"
])

const TO_KABAB_CASE = /[A-Z]/g

const toKebabCase = (value: string): string =>
  value.replace(TO_KABAB_CASE, (match) => `-${match.toLowerCase()}`)

const normalizeTextareaStyleValue = (
  propertyName: string,
  value: string,
  element: Element
): string => {
  const fallback = TEXTAREA_STYLE_FALLBACKS[propertyName]
  if (!fallback) return value

  if (value && !value.includes("var(")) {
    if (!TEXTAREA_NUMERIC_PROPERTIES.has(propertyName)) return value
    if (!Number.isNaN(Number.parseFloat(value))) return value
  }

  if (propertyName === "width") {
    const width = element instanceof HTMLElement ? element.clientWidth : 0
    return `${Math.max(width || 320, 1)}px`
  }

  return fallback
}

const withTextareaStyleFallback = (
  element: Element,
  style: CSSStyleDeclaration
): CSSStyleDeclaration => {
  if (!(element instanceof HTMLTextAreaElement)) return style

  const getPropertyValue = style.getPropertyValue.bind(style)

  return new Proxy(style, {
    get(target, prop, receiver) {
      if (prop === "getPropertyValue") {
        return (name: string) =>
          normalizeTextareaStyleValue(name, getPropertyValue(name), element)
      }

      if (typeof prop === "string") {
        const raw = Reflect.get(target, prop, receiver)
        if (typeof raw === "string") {
          const propertyName = toKebabCase(prop)
          return normalizeTextareaStyleValue(propertyName, raw, element)
        }
      }

      return Reflect.get(target, prop, receiver)
    }
  })
}

const SUPPRESSED_WARNING_PATTERNS = [
  /Could not parse CSS stylesheet/i,
  /invalid value for the `height` css style property/i,
  /\[antd:\s*Notification\]\s+`message` is deprecated\./i,
  /\[antd:\s*List\]\s+The `List` component is deprecated\./i,
  /React Router Future Flag Warning/i,
  /Instance created by `useForm` is not connected to any Form element/i,
  /Not implemented:\s*navigation to another Document/i,
  /Failed to load seen stats:\s*Network error/i,
  /ResearchWorkspace render error.*chat pane crash/i,
  /Failed to create thread:\s*Default character unavailable/i,
  /Failed to persist chat message:\s*save failed/i,
  /Streaming search failed, falling back to standard search:\s*stream endpoint unavailable/i,
  /Export failed:\s*HTTP undefined: thread not found/i,
  /\[WorldBooks\]\s+Import parse failed/i,
  /\[WorldBooks\]\s+Import validation failed/i,
  /Skipped Kobold entry/i,
  /(^|\s)boom(\s|$)/i,
  /(^|\s)still broken(\s|$)/i,
  /(^|\s)prompt body crash(\s|$)/i,
  /(^|\s)chat pane crash(\s|$)/i
]

const shouldSuppressConsoleOutput = (args: unknown[]): boolean => {
  const combined = args
    .map((arg) => {
      if (typeof arg === "string") return arg
      if (arg instanceof Error) return arg.message
      return ""
    })
    .join(" ")

  return SUPPRESSED_WARNING_PATTERNS.some((pattern) => pattern.test(combined))
}

class MutationObserverFallback {
  constructor(_callback: MutationCallback) {}
  observe(): void {}
  disconnect(): void {}
  takeRecords(): MutationRecord[] {
    return []
  }
}

const resolveMutationObserver = (): typeof MutationObserver =>
  typeof originalMutationObserver === "function" ? originalMutationObserver : MutationObserverFallback

const ensureMutationObserver = (): void => {
  if (typeof window.MutationObserver === "function") return
  const replacement = resolveMutationObserver()
  ;(window as any).MutationObserver = replacement
  ;(globalThis as any).MutationObserver = replacement
}

window.getComputedStyle = ((element: Element, _pseudoElt?: string | null) =>
  withTextareaStyleFallback(element, originalGetComputedStyle(element))) as typeof window.getComputedStyle

beforeAll(() => {
  console.log = (...args: unknown[]) => {
    if (shouldSuppressConsoleOutput(args)) return
    originalConsoleLog(...args)
  }

  console.debug = (...args: unknown[]) => {
    if (shouldSuppressConsoleOutput(args)) return
    originalConsoleDebug(...args)
  }

  console.info = (...args: unknown[]) => {
    if (shouldSuppressConsoleOutput(args)) return
    originalConsoleInfo(...args)
  }

  console.warn = (...args: unknown[]) => {
    if (shouldSuppressConsoleOutput(args)) return
    originalConsoleWarn(...args)
  }

  console.error = (...args: unknown[]) => {
    if (shouldSuppressConsoleOutput(args)) return
    originalConsoleError(...args)
  }

  ensureMutationObserver()
})

afterAll(() => {
  console.log = originalConsoleLog
  console.debug = originalConsoleDebug
  console.info = originalConsoleInfo
  console.warn = originalConsoleWarn
  console.error = originalConsoleError
})

const evaluateMediaQuery = (query: string): boolean => {
  const width = window.innerWidth || 1024
  const minMatches = [...query.matchAll(/\(min-width:\s*(\d+)px\)/g)]
  const maxMatches = [...query.matchAll(/\(max-width:\s*(\d+)px\)/g)]

  const meetsMin = minMatches.every((match) => width >= Number(match[1]))
  const meetsMax = maxMatches.every((match) => width <= Number(match[1]))

  return meetsMin && meetsMax
}

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: evaluateMediaQuery(query),
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false
  })
})

if (typeof window.ResizeObserver === "undefined") {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
  ;(globalThis as any).ResizeObserver = ResizeObserverMock
}

afterEach(() => {
  if (vi.isFakeTimers()) {
    vi.useRealTimers()
  }
  ensureMutationObserver()
  cleanup()
})
