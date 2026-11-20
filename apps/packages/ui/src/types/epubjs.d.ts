declare module "epubjs" {
  export interface NavItem {
    id?: string
    href: string
    label: string
    subitems?: NavItem[]
  }

  export interface Location {
    start: { cfi: string; href: string }
    end?: { cfi: string; href: string }
  }

  export interface Book {
    open: (input: string | ArrayBuffer, format?: string) => Promise<object>
    ready: Promise<void>
    loaded: { navigation: Promise<void> }
    navigation: { toc: NavItem[] }
    locations: {
      generate: (chars?: number) => Promise<void>
      length: () => number
      cfiFromPercentage: (percentage: number) => string | null
      percentageFromCfi: (cfi: string) => number | null
      locationFromCfi: (cfi: string) => number | null
      cfiFromLocation: (location: number) => string | null
    }
    spine: { get: (href: string) => { index: number } | null; length: number }
    search: (query: string) => Promise<Array<{ cfi: string; excerpt?: string }>>
    destroy: () => void
    renderTo: (
      element: HTMLElement,
      options?: Record<string, unknown>
    ) => Rendition
  }

  export interface Rendition {
    book: Book
    on: (event: string, handler: (...args: any[]) => void) => void
    off: (event: string, handler: (...args: any[]) => void) => void
    display: (target?: string) => void
    next: () => void
    prev: () => void
    getRange: (cfiRange: string) => Range | null
    currentLocation: () => Location | null
    themes: {
      register: (name: string, styles: Record<string, any>) => void
      select: (name: string) => void
    }
    annotations: {
      highlight: (
        cfiRange: string,
        data?: Record<string, any>,
        callback?: (...args: any[]) => void,
        className?: string,
        styles?: Record<string, string>
      ) => void
      remove: (cfiRange: string, type?: string) => void
    }
    destroy: () => void
  }

  export interface EpubOptions {
    [key: string]: unknown
  }

  export default function ePub(url?: string, options?: EpubOptions): Book
}
