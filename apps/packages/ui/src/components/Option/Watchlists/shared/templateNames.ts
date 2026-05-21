export type WatchlistTemplateRecipeId = "briefing_md" | "newsletter_html" | "mece_md"

const BACKEND_TEMPLATE_BY_RECIPE: Record<WatchlistTemplateRecipeId, string> = {
  briefing_md: "briefing_markdown",
  newsletter_html: "newsletter_html",
  mece_md: "mece_markdown"
}

export const normalizeWatchlistTemplateName = (value?: string | null): string => {
  const trimmed = String(value || "").trim()
  if (!trimmed) return ""
  return BACKEND_TEMPLATE_BY_RECIPE[trimmed as WatchlistTemplateRecipeId] || trimmed
}
