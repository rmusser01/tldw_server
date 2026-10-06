import type { SidepanelChatHandoffPageContext } from "@/services/sidepanel-chat-handoff"

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

export const readVisiblePageContextForHandoff = async (
  documents: SidepanelHandoffDocumentSource[],
  pageContextActive: boolean,
  readActiveTabs: () => Promise<SidepanelHandoffDocumentSource[]>
): Promise<SidepanelChatHandoffPageContext | undefined> => {
  const snippets = documents.map((doc) => ({
    kind: "visible-context" as const,
    label: doc.title ?? undefined,
    text: buildVisibleDocumentHandoffSnippetText(doc)
  }))

  let activePage:
    | {
        title?: string
        url?: string
      }
    | undefined

  if (pageContextActive) {
    try {
      const tabs = await readActiveTabs()
      const activeTab = tabs.find((tab) => tab.title || tab.url)
      activePage = {
        title:
          typeof activeTab?.title === "string" &&
          activeTab.title.trim().length > 0
            ? activeTab.title
            : undefined,
        url:
          typeof activeTab?.url === "string" && activeTab.url.trim().length > 0
            ? activeTab.url
            : undefined
      }
    } catch {
      activePage = undefined
    }
  }

  const unavailable = pageContextActive && !activePage?.title && !activePage?.url
  if (!unavailable && !activePage?.title && !activePage?.url && snippets.length === 0) {
    return undefined
  }

  return {
    ...(activePage?.title ? { title: activePage.title } : {}),
    ...(activePage?.url ? { url: activePage.url } : {}),
    snippets,
    ...(unavailable ? { truncated: true } : {})
  }
}
