import { vi } from "vitest"

const matchesMediaQuery = (query: string, width: number): boolean => {
  const minMatch = query.match(/min-width:\s*(\d+)px/)
  const maxMatch = query.match(/max-width:\s*(\d+)px/)
  const minOk = minMatch ? width >= Number(minMatch[1]) : true
  const maxOk = maxMatch ? width <= Number(maxMatch[1]) : true
  return (minMatch || maxMatch) ? minOk && maxOk : false
}

export const setViewport = (width: number): void => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: matchesMediaQuery(query, width),
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}
