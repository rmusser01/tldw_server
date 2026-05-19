import { useEffect, useLayoutEffect, useCallback, useRef } from "react"

/**
 * Platform-aware keyboard shortcut system.
 * Uses ⌘ on Mac, Ctrl on Windows/Linux.
 */

export type ShortcutModifier = "meta" | "ctrl" | "alt" | "shift"
export type ShortcutKey = string // e.g., "k", "n", "Enter", "Escape", "/"

export interface Shortcut {
  id: string
  key: ShortcutKey
  modifiers: ShortcutModifier[]
  action: () => void
  description: string
  /** Whether the shortcut listener should be active */
  enabled?: boolean
  /** Scope where this shortcut is active */
  scope?: "global" | "chat" | "settings" | "sidepanel"
  /** Whether shortcut works when input is focused */
  allowInInput?: boolean
}

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect

/**
 * Detect if running on Mac
 */
export const isMac =
  typeof navigator !== "undefined" &&
  /Mac|iPod|iPhone|iPad/.test(navigator.platform)

/**
 * Get the modifier key for the current platform
 */
export const getPlatformModifier = (): "meta" | "ctrl" =>
  isMac ? "meta" : "ctrl"

/**
 * Format a shortcut for display
 * @example formatShortcut({ key: "k", modifiers: ["meta"] }) => "⌘K" on Mac, "Ctrl+K" on Windows
 */
export function formatShortcut(shortcut: Pick<Shortcut, "key" | "modifiers">): string {
  const { key, modifiers } = shortcut
  const parts: string[] = []

  if (modifiers.includes("ctrl")) {
    parts.push(isMac ? "⌃" : "Ctrl")
  }
  if (modifiers.includes("meta")) {
    parts.push(isMac ? "⌘" : "Ctrl")
  }
  if (modifiers.includes("alt")) {
    parts.push(isMac ? "⌥" : "Alt")
  }
  if (modifiers.includes("shift")) {
    parts.push(isMac ? "⇧" : "Shift")
  }

  // Format special keys
  const keyDisplay = key === "Enter" ? "↵" :
                    key === "Escape" ? "Esc" :
                    key === "Backspace" ? "⌫" :
                    key === "ArrowUp" ? "↑" :
                    key === "ArrowDown" ? "↓" :
                    key === "ArrowLeft" ? "←" :
                    key === "ArrowRight" ? "→" :
                    key === " " ? "Space" :
                    key === "[" ? "[" :
                    key === "]" ? "]" :
                    key.toUpperCase()

  if (isMac) {
    return parts.join("") + keyDisplay
  }
  return [...parts, keyDisplay].join("+")
}

/**
 * Check if a keyboard event matches a shortcut
 */
export function matchesShortcut(
  event: KeyboardEvent,
  shortcut: Pick<Shortcut, "key" | "modifiers">
): boolean {
  const { key, modifiers } = shortcut

  // Check key (case-insensitive for letters)
  const eventKey = event.key.toLowerCase()
  const shortcutKey = key.toLowerCase()
  if (eventKey !== shortcutKey) return false

  // Check modifiers - handle platform-specific modifier
  const platformMod = getPlatformModifier()

  for (const mod of modifiers) {
    if (mod === "meta") {
      // On Mac, check metaKey; on Windows/Linux, check ctrlKey
      if (isMac && !event.metaKey) return false
      if (!isMac && !event.ctrlKey) return false
    } else if (mod === "ctrl") {
      if (!event.ctrlKey) return false
    } else if (mod === "alt") {
      if (!event.altKey) return false
    } else if (mod === "shift") {
      if (!event.shiftKey) return false
    }
  }

  // Ensure no extra modifiers are pressed
  const wantsMeta = modifiers.includes("meta")
  const wantsCtrl = modifiers.includes("ctrl")
  const wantsAlt = modifiers.includes("alt")
  const wantsShift = modifiers.includes("shift")

  // On Mac, meta handles ⌘; on Windows, ctrl handles Ctrl
  if (isMac) {
    if (event.metaKey !== wantsMeta) return false
    if (event.ctrlKey !== wantsCtrl) return false
  } else {
    // On Windows/Linux, "meta" maps to ctrl, so check combined
    const wantsMainMod = wantsMeta || wantsCtrl
    if (event.ctrlKey !== wantsMainMod) return false
    if (event.metaKey) return false // Windows key shouldn't be pressed
  }

  if (event.altKey !== wantsAlt) return false
  if (event.shiftKey !== wantsShift) return false

  return true
}

/**
 * Check if event target is an input element
 */
function isInputElement(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false
  const tagName = target.tagName.toLowerCase()
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    target.isContentEditable
  )
}

/**
 * Hook to register a single keyboard shortcut
 */
export function useShortcut(
  shortcut: Omit<Shortcut, "id"> & { id?: string },
  deps: React.DependencyList = []
) {
  const actionRef = useRef(shortcut.action)
  actionRef.current = shortcut.action

  useIsomorphicLayoutEffect(() => {
    if (shortcut.enabled === false) {
      return
    }

    const listenerOptions: AddEventListenerOptions = { capture: true }

    const handler = (event: KeyboardEvent) => {
      // Skip if in input and not allowed
      if (!shortcut.allowInInput && isInputElement(event.target)) {
        return
      }

      if (matchesShortcut(event, shortcut)) {
        event.preventDefault()
        event.stopPropagation()
        actionRef.current()
      }
    }

    const targets: Array<Document | Window> = []
    if (typeof window !== "undefined") {
      targets.push(window)
    }
    if (typeof document !== "undefined") {
      targets.push(document)
    }

    for (const target of targets) {
      target.addEventListener("keydown", handler, listenerOptions)
    }
    return () => {
      for (const target of targets) {
        target.removeEventListener("keydown", handler, listenerOptions)
      }
    }
    }, [
    shortcut.enabled,
    shortcut.key,
    JSON.stringify(shortcut.modifiers),
    shortcut.allowInInput,
    ...deps
  ])
}

/**
 * Hook to register multiple keyboard shortcuts at once
 */
export function useShortcuts(shortcuts: Shortcut[], scope?: string) {
  const shortcutsRef = useRef(shortcuts)
  shortcutsRef.current = shortcuts

  useIsomorphicLayoutEffect(() => {
    const listenerOptions: AddEventListenerOptions = { capture: true }

    const handler = (event: KeyboardEvent) => {
      for (const shortcut of shortcutsRef.current) {
        // Check scope
        if (scope && shortcut.scope && shortcut.scope !== scope) {
          continue
        }

        // Skip if in input and not allowed
        if (!shortcut.allowInInput && isInputElement(event.target)) {
          continue
        }

        if (matchesShortcut(event, shortcut)) {
          event.preventDefault()
          event.stopPropagation()
          shortcut.action()
          return
        }
      }
    }

    const targets: Array<Document | Window> = []
    if (typeof window !== "undefined") {
      targets.push(window)
    }
    if (typeof document !== "undefined") {
      targets.push(document)
    }

    for (const target of targets) {
      target.addEventListener("keydown", handler, listenerOptions)
    }
    return () => {
      for (const target of targets) {
        target.removeEventListener("keydown", handler, listenerOptions)
      }
    }
  }, [scope])
}

/**
 * Default shortcuts for the application
 */
export const DEFAULT_SHORTCUTS: Omit<Shortcut, "action">[] = [
  {
    id: "command-palette",
    key: "k",
    modifiers: ["meta"],
    description: "Open command palette",
    scope: "global",
  },
  {
    id: "new-chat",
    key: "n",
    modifiers: ["meta"],
    description: "New chat",
    scope: "chat",
  },
  {
    id: "toggle-rag",
    key: "r",
    modifiers: ["alt"],
    description: "Toggle Knowledge Search",
    scope: "chat",
  },
  {
    id: "toggle-web-search",
    key: "w",
    modifiers: ["alt"],
    description: "Toggle web search",
    scope: "chat",
  },
  {
    id: "search-chat",
    key: "/",
    modifiers: ["meta"],
    description: "Search in chat",
    scope: "chat",
  },
  {
    id: "switch-model",
    key: "e",
    modifiers: ["meta"],
    description: "Switch model",
    scope: "chat",
  },
  {
    id: "toggle-sidebar",
    key: "[",
    modifiers: ["meta"],
    description: "Toggle sidebar",
    scope: "global",
  },
  {
    id: "send-message",
    key: "Enter",
    modifiers: ["meta"],
    description: "Send message",
    scope: "chat",
    allowInInput: true,
  },
  {
    id: "open-settings",
    key: ",",
    modifiers: ["meta"],
    description: "Open settings",
    scope: "global",
  },
  {
    id: "ingest-page",
    key: "i",
    modifiers: ["meta"],
    description: "Ingest current page",
    scope: "global",
  },
  {
    id: "open-media",
    key: "m",
    modifiers: ["meta", "shift"],
    description: "Open media library",
    scope: "global",
  },
  {
    id: "show-shortcuts",
    key: "?",
    modifiers: ["meta", "shift"],
    description: "Show keyboard shortcuts",
    scope: "global",
  },
]

/**
 * Get formatted shortcut string by ID
 */
export function getShortcutById(id: string): string | undefined {
  const shortcut = DEFAULT_SHORTCUTS.find((s) => s.id === id)
  return shortcut ? formatShortcut(shortcut) : undefined
}
