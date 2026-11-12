declare module "@tldw/ui/entries/background" {
  const entry: unknown
  export default entry
}

declare module "@tldw/ui/entries/copilot-popup.content" {
  const entry: unknown
  export default entry
  /** Publishes the popup handler on globalThis for the lazy stub. */
  export function registerCopilotPopupHandler(): void
}

declare module "@tldw/ui/parser/default" {
  /** Heavy default parser (cheerio + Readability + Turndown). */
  export function defaultExtractContent(html: string): string
}

declare module "@tldw/ui/entries/web-clipper.content" {
  const entry: unknown
  export default entry
}

declare module "@tldw/ui/entries/hf-pull.content" {
  const entry: unknown
  export default entry
}

declare module "@tldw/ui/entries/options/main" {
  const entry: unknown
  export default entry
}

declare module "@tldw/ui/entries/sidepanel/main" {
  const entry: unknown
  export default entry
}
