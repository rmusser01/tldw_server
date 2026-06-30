export type SidepanelHandoffDocumentSource = {
  title?: string | null
  url?: string | null
}

export const buildVisibleDocumentHandoffSnippetText = (
  doc: SidepanelHandoffDocumentSource
) =>
  [
    doc.title ? `Title: ${doc.title}` : null,
    doc.url ? `URL: ${doc.url}` : null
  ]
    .filter(Boolean)
    .join("\n")
