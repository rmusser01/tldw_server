import React from "react"
import { SearchBar } from "../SearchBar"

type KnowledgeComposerProps = {
  className?: string
  autoFocus?: boolean
  showWebToggle?: boolean
  webFallbackAvailable?: boolean
  searchBlockedMessage?: string | null
  widthMode?: "compact" | "wide"
}

export function KnowledgeComposer({
  className,
  autoFocus = true,
  showWebToggle = false,
  webFallbackAvailable = true,
  searchBlockedMessage = null,
  widthMode = "compact",
}: KnowledgeComposerProps) {
  return (
    <SearchBar
      className={className}
      autoFocus={autoFocus}
      showWebToggle={showWebToggle}
      webFallbackAvailable={webFallbackAvailable}
      searchBlockedMessage={searchBlockedMessage}
      widthMode={widthMode}
    />
  )
}
