import { vi } from "vitest"

/**
 * Pin the OS "reduce motion" preference for one test. The shared vitest
 * matchMedia stub (vitest.setup.ts) answers true to every query without a
 * width condition, so tests that care about motion must say which they want.
 * Other queries keep the stub's answer. Returns a function that restores it.
 */
export function mockReducedMotion(reduce: boolean): () => void {
  const original = window.matchMedia
  window.matchMedia = vi.fn((query: string) => ({
    matches: query.includes("prefers-reduced-motion: reduce")
      ? reduce
      : Boolean(original?.(query)?.matches),
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(() => false)
  })) as unknown as typeof window.matchMedia
  return () => {
    window.matchMedia = original
  }
}
