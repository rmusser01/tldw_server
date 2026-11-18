import React from "react"
import { MediaSectionNavigator } from "@/components/Media/MediaSectionNavigator"
import { useMediaNavigation, useMediaSectionContent, type MediaNavigationNode } from "@/hooks/useMediaNavigation"
import type { WorkspaceSource } from "@/types/workspace"
import type { ResearchWorkspaceSectionSelection } from "@/services/tldw/domains/workspace-api"

type Props = {
  source: WorkspaceSource
  selections: ResearchWorkspaceSectionSelection[]
  onChange: (sourceId: string, selections: ResearchWorkspaceSectionSelection[]) => void
  disabled: boolean
}

export function numberedChapterNodes(nodes: MediaNavigationNode[]): MediaNavigationNode[] {
  const levels = [...new Set(nodes.map(node => node.level))]
  const candidates = levels.map(level => nodes.filter(node => node.level === level &&
    ["generated_toc", "pdf_outline"].includes(node.source) && /^(?:chapter\s+)?\d+[.)]?\s+\S/i.test(node.title)))
  return candidates.filter(group => group.length > 1 && group.every((node, index) =>
    Number(node.title.match(/^(?:chapter\s+)?(\d+)/i)?.[1]) === index + 1))
    .sort((a, b) => b.length - a.length)[0] || []
}

/** Review existing reader sections before choosing the scope of a durable output. */
export function SectionScopePicker({ source, selections, onChange, disabled }: Props) {
  const [open, setOpen] = React.useState(false)
  const [previewId, setPreviewId] = React.useState<string | null>(null)
  const navigation = useMediaNavigation(source.mediaId, { enabled: open })
  const content = useMediaSectionContent(source.mediaId, previewId, { enabled: open && Boolean(previewId), format: "plain" })
  const current = selections.filter(selection => selection.source_id === source.id)
  const version = navigation.data?.navigation_version
  const chapters = numberedChapterNodes(navigation.data?.nodes || [])
  const stale = current.some(selection => version && selection.navigation_version !== version)
  return <details onToggle={event => setOpen(event.currentTarget.open)} className="rounded border border-border p-2">
    <summary className="cursor-pointer text-sm">{source.title} · {current.length} sections selected</summary>
    {stale && <p role="alert" className="text-sm text-error">The outline changed. Clear this selection and review it again.</p>}
    <button type="button" disabled={disabled || !current.length} className="my-2 text-sm text-primary" onClick={() => onChange(source.id, [])}>Clear section selection</button>
    {chapters.length > 0 && <button type="button" disabled={disabled || !version} className="m-2 text-sm text-primary"
      onClick={() => version && onChange(source.id, chapters.map(node => ({ source_id: source.id, node_id: node.id, navigation_version: version })))}>
      Select numbered chapters ({chapters.length})
    </button>}
    <MediaSectionNavigator nodes={navigation.data?.nodes || []} selectedNodeId={previewId}
      className="!w-full !min-w-0 max-h-80 overflow-auto" loading={navigation.isLoading} error={navigation.error}
      onRetry={() => void navigation.refetch()} onSelectNode={node => setPreviewId(node.id)}
      selection={{ disabled: disabled || stale, nodeIds: current.map(selection => selection.node_id),
        onToggle: node => {
          if (!version) return
          onChange(source.id, current.some(selection => selection.node_id === node.id)
            ? current.filter(selection => selection.node_id !== node.id)
            : [...current, { source_id: source.id, node_id: node.id, navigation_version: version }])
        } }} />
    {previewId && <div className="mt-2 text-sm" aria-live="polite">
      {content.isLoading ? "Loading section preview…" : content.error || content.data?.coverage_status === "unavailable" ? "Exact section range unavailable. Re-extract the source or review the outline before generating." : <>
        <strong>{content.data?.title}</strong>
        <p>{content.data?.target.target_type === "page" ? `Starts at page ${content.data.target.target_start}; confirm PDF and printed labels in the reader.` : "Exact text section; page labels appear with generated evidence when available."}</p>
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-sans">{content.data?.content.slice(0, 1500)}</pre>
        {Number(content.data?.content.length) > 1500 && <p>Preview excerpt. Generation reads the complete selected section.</p>}
      </>}
    </div>}
  </details>
}
