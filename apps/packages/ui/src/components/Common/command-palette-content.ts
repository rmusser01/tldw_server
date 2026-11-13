import { useEffect, useState } from "react"

import type { CommandItem } from "./CommandPalette"

/** How long the palette waits after a keystroke before searching content (XP-06). */
export const COMMAND_PALETTE_CONTENT_DEBOUNCE_MS = 200
/** The most results the palette shows from each content provider. */
export const COMMAND_PALETTE_CONTENT_LIMIT = 5

/**
 * An async source of palette results, such as saved chats or notes. `search`
 * receives the trimmed query and a signal that aborts when the query changes
 * or the palette closes.
 */
export type CommandPaletteContentProvider = {
  id: string
  search: (query: string, signal: AbortSignal) => Promise<CommandItem[]>
}

type ContentResults = { query: string; items: CommandItem[] }

const EMPTY: ContentResults = { query: "", items: [] }

/**
 * Search the content providers for `query`: 200 ms after the last keystroke,
 * one AbortController per query, at most five results per provider, in
 * provider order. A failing provider contributes nothing. `pending` is true
 * from the keystroke until the results for that query arrive.
 */
export const useCommandPaletteContentResults = (
  query: string,
  providers: CommandPaletteContentProvider[] | undefined,
  enabled: boolean
): { items: CommandItem[]; pending: boolean } => {
  const trimmed = query.trim()
  const active = enabled && trimmed.length > 0 && (providers?.length ?? 0) > 0
  const [results, setResults] = useState<ContentResults>(EMPTY)

  useEffect(() => {
    if (!active || !providers) return
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      void Promise.all(
        providers.map((provider) =>
          provider
            .search(trimmed, controller.signal)
            .then((items) => items.slice(0, COMMAND_PALETTE_CONTENT_LIMIT))
            .catch(() => [] as CommandItem[])
        )
      ).then((groups) => {
        if (!controller.signal.aborted) setResults({ query: trimmed, items: groups.flat() })
      })
    }, COMMAND_PALETTE_CONTENT_DEBOUNCE_MS)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [active, providers, trimmed])

  // Results belong to one query; while another is typed they are not shown.
  const current = active && results.query === trimmed
  return { items: current ? results.items : [], pending: active && !current }
}
