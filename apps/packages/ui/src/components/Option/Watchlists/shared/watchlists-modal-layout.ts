import type { CSSProperties } from "react"

export const WATCHLISTS_CONSTRAINED_MODAL_WIDTH = "100vw"
export const WATCHLISTS_CONSTRAINED_MODAL_BODY_MAX_HEIGHT = "calc(100vh - 152px)"

interface WatchlistsModalChrome {
  width: number | string
  style?: CSSProperties
  styles?: {
    body?: CSSProperties
    footer?: CSSProperties
  }
}

export const buildWatchlistsModalChrome = (
  isConstrained: boolean,
  desktopWidth: number,
  bodyStyle?: CSSProperties
): WatchlistsModalChrome => {
  const styles: WatchlistsModalChrome["styles"] = {}

  if (bodyStyle || isConstrained) {
    styles.body = isConstrained
      ? {
          ...bodyStyle,
          maxHeight: WATCHLISTS_CONSTRAINED_MODAL_BODY_MAX_HEIGHT,
          overflowY: "auto"
        }
      : bodyStyle
  }

  if (isConstrained) {
    styles.footer = {
      display: "flex",
      flexWrap: "wrap",
      gap: 8,
      justifyContent: "flex-end"
    }
  }

  return {
    width: isConstrained ? WATCHLISTS_CONSTRAINED_MODAL_WIDTH : desktopWidth,
    style: isConstrained
      ? {
          top: 0,
          maxWidth: WATCHLISTS_CONSTRAINED_MODAL_WIDTH,
          paddingBottom: 0
        }
      : undefined,
    styles: Object.keys(styles).length > 0 ? styles : undefined
  }
}
