import { useMemo } from "react"
import type { ThemeRgbTokenKey } from "@/themes/types"
import { useTheme } from "@/hooks/useTheme"
import { getComputedTokens } from "@/themes/runtime-tokens"

/**
 * Returns memoized hex color values for all semantic RGB theme tokens.
 * Recomputes when the active theme tokens change.
 *
 * Use this in React components that feed colors into JS APIs
 * (Cytoscape, Chart.js, Canvas, etc.) that cannot use CSS classes.
 */
export function useComputedTokens(): Record<ThemeRgbTokenKey, string> {
  const { tokens } = useTheme()

  return useMemo(() => {
    return getComputedTokens()
    // tokens is the dependency — when it changes, we re-read computed styles
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens])
}
