import { MessageCircle, Star, StarOff } from "lucide-react"
import { useMemo } from "react"
import { Tooltip } from "antd"
import type { PromptRowVM } from "./prompt-workspace-types"

export type PromptGalleryDensity = "rich" | "compact"

const DEFAULT_FALLBACK_NAME = "prompt"

const getInitialCharacter = (value: string): string => {
  const match = value.match(/[A-Za-z0-9]/)
  return match ? match[0].toUpperCase() : "?"
}

export const hashNameToHue = (name: string): number => {
  let hash = 0
  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0
  }
  return Math.abs(hash) % 360
}

export const getAvatarFallbackTokens = (name?: string) => {
  const normalized = (name || DEFAULT_FALLBACK_NAME).trim().toLowerCase()
  const hue = hashNameToHue(normalized || DEFAULT_FALLBACK_NAME)
  return {
    hue,
    initial: getInitialCharacter(name || ""),
    backgroundColor: `hsl(${hue} 68% 82%)`,
    color: `hsl(${hue} 45% 20%)`
  }
}

interface PromptGalleryCardProps {
  prompt: PromptRowVM
  onClick: () => void
  density?: PromptGalleryDensity
  onToggleFavorite?: (nextFavorite: boolean) => void
}

export function PromptGalleryCard({
  prompt,
  onClick,
  density = "rich",
  onToggleFavorite
}: PromptGalleryCardProps) {
  const displayName = prompt.title || "Untitled prompt"
  const previewText = (prompt.previewSystem || prompt.previewUser || "").trim()
  const displayKeywords = (prompt.keywords || [])
    .filter((kw) => kw.trim().length > 0)
    .slice(0, 3)
  const isCompact = density === "compact"
  const fallbackTokens = useMemo(
    () => getAvatarFallbackTokens(displayName),
    [displayName]
  )

  return (
    <div
      role="button"
      tabIndex={0}
      data-testid={`prompt-gallery-card-${prompt.id}`}
      className={`group flex w-full flex-col items-center gap-2 rounded-lg border border-border bg-surface p-3 transition-all motion-reduce:transition-none hover:border-primary/30 hover:bg-surface2 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 focus:ring-offset-bg ${
        isCompact ? "min-h-[168px]" : "min-h-[220px]"
      }`}
      onClick={onClick}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          onClick()
        }
      }}
      aria-label={`Click to preview ${displayName}`}
    >
      {/* Avatar */}
      <div
        className={`relative aspect-square w-full ${
          isCompact ? "max-w-[102px]" : "max-w-[120px]"
        }`}
      >
        {onToggleFavorite && (
          <Tooltip title={prompt.favorite ? "Remove favorite" : "Add favorite"}>
            <button
              type="button"
              data-testid={`prompt-gallery-favorite-${prompt.id}`}
              className={`absolute -right-1 -top-1 z-10 inline-flex items-center justify-center rounded-full border bg-surface p-1 shadow-sm transition ${
                prompt.favorite
                  ? "border-primary/40 text-primary"
                  : "border-border text-text-muted hover:text-primary"
              }`}
              aria-label={
                prompt.favorite
                  ? `Remove ${displayName} from favorites`
                  : `Add ${displayName} to favorites`
              }
              onClick={(event) => {
                event.stopPropagation()
                onToggleFavorite(!prompt.favorite)
              }}
            >
              {prompt.favorite ? (
                <Star className="h-3.5 w-3.5 fill-current" />
              ) : (
                <StarOff className="h-3.5 w-3.5" />
              )}
            </button>
          </Tooltip>
        )}
        <div
          data-testid="prompt-gallery-fallback-avatar"
          className="flex h-full w-full items-center justify-center rounded-lg ring-2 ring-border/70 group-hover:ring-primary/30"
          style={{
            backgroundColor: fallbackTokens.backgroundColor,
            color: fallbackTokens.color
          }}
        >
          <span className="text-3xl font-semibold leading-none">
            {fallbackTokens.initial}
          </span>
        </div>
        {/* Usage count badge */}
        {prompt.usageCount > 0 && (
          <Tooltip title={`${prompt.usageCount} usage(s)`}>
            <div
              data-testid={`prompt-gallery-usage-${prompt.id}`}
              className="absolute -bottom-1 -right-1 flex items-center gap-0.5 rounded-full bg-primary px-1.5 py-0.5 text-[10px] font-medium text-white shadow-sm"
            >
              <MessageCircle className="h-3 w-3" />
              <span>{prompt.usageCount > 99 ? "99+" : prompt.usageCount}</span>
            </div>
          </Tooltip>
        )}
      </div>

      {/* Title */}
      <Tooltip title={displayName} placement="bottom">
        <span className="w-full truncate text-center text-sm font-medium text-text group-hover:text-primary">
          {displayName}
        </span>
      </Tooltip>

      {prompt.kind === "recipe" && (
        <div className="flex items-center justify-center gap-1">
          <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
            Recipe
          </span>
          <span className="text-[10px] font-medium text-text-muted">
            {prompt.recipeTarget === "system" ? "System" : "User"}
          </span>
        </div>
      )}

      {/* Preview text (rich density only) */}
      {!isCompact && previewText && (
        <Tooltip title={previewText} placement="bottom">
          <p className="w-full line-clamp-2 text-center text-xs leading-5 text-text-muted">
            {previewText}
          </p>
        </Tooltip>
      )}

      {/* Keywords (rich density only) */}
      {!isCompact && displayKeywords.length > 0 && (
        <div className="mt-0.5 flex w-full flex-wrap justify-center gap-1">
          {displayKeywords.map((kw) => (
            <span
              key={kw}
              className="rounded-full border border-border/80 bg-surface2 px-2 py-0.5 text-[10px] font-medium text-text-muted"
            >
              {kw}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
