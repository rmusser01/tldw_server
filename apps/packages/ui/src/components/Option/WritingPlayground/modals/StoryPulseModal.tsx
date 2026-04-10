import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Button, Empty, List, Modal, Progress, Spin, Typography, message } from "antd"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { getManuscriptStructure, analyzeChapter, listManuscriptAnalyses } from "@/services/writing-playground"

type StoryPulseModalProps = {
  open: boolean
  onClose: () => void
}

const METRICS = [
  { key: "pacing", label: "Pacing", color: "#1677ff" },
  { key: "tension", label: "Tension", color: "#ff4d4f" },
  { key: "atmosphere", label: "Atmosphere", color: "#722ed1" },
  { key: "engagement", label: "Engagement", color: "#52c41a" },
]

export function StoryPulseModal({ open, onClose }: StoryPulseModalProps) {
  const activeProjectId = useWritingPlaygroundStore((state) => state.activeProjectId)
  const [analyzingChapterIds, setAnalyzingChapterIds] = useState<Set<string>>(
    () => new Set()
  )

  const {
    data: structure,
    isLoading: isStructureLoading,
    isFetching: isStructureFetching,
  } = useQuery({
    queryKey: ["manuscript-structure", activeProjectId],
    queryFn: () => getManuscriptStructure(activeProjectId!),
    enabled: open && !!activeProjectId,
  })

  const { data: analysesData, refetch: refetchAnalyses } = useQuery({
    queryKey: ["manuscript-analyses", activeProjectId, "pacing"],
    queryFn: () => listManuscriptAnalyses(activeProjectId!, { analysis_type: "pacing", include_stale: true }),
    enabled: open && !!activeProjectId,
  })

  const analyses = (analysesData as { analyses?: unknown[] } | undefined)?.analyses ?? []

  const allChapters = (() => {
    if (!structure) return []
    const chapters: Array<{ id: string; title: string }> = []
    const s = structure as { parts?: Array<{ chapters?: Array<{ id: string; title: string }> }>; unassigned_chapters?: Array<{ id: string; title: string }> } | undefined
    for (const part of s?.parts || []) {
      for (const ch of part.chapters || []) chapters.push(ch)
    }
    for (const ch of s?.unassigned_chapters || []) chapters.push(ch)
    return chapters
  })()

  const handleAnalyze = async (chapterId: string) => {
    setAnalyzingChapterIds((previous) => {
      const next = new Set(previous)
      next.add(chapterId)
      return next
    })
    try {
      await analyzeChapter(chapterId, { analysis_types: ["pacing"] })
      await refetchAnalyses()
    } catch {
      message.error("Failed to analyze chapter")
    } finally {
      setAnalyzingChapterIds((previous) => {
        const next = new Set(previous)
        next.delete(chapterId)
        return next
      })
    }
  }

  const getAnalysisForChapter = (chapterId: string) =>
    analyses.find((a: Record<string, unknown>) => a.scope_type === "chapter" && a.scope_id === chapterId)

  return (
    <Modal title="Story Pulse" open={open} onCancel={onClose} footer={null} width={700}>
      {!activeProjectId ? (
        <Empty description="Select a project first" />
      ) : isStructureLoading || (isStructureFetching && !structure) ? (
        <div className="flex justify-center py-10">
          <Spin size="small" />
        </div>
      ) : allChapters.length === 0 ? (
        <Empty description="No chapters to analyze" />
      ) : (
        <List
          dataSource={allChapters}
          renderItem={(ch: { id: string; title: string }) => {
            const analysis = getAnalysisForChapter(ch.id)
            const result = analysis?.result || {}
            const isStale = analysis?.stale
            return (
              <List.Item>
                <div className="w-full">
                  <div className="flex items-center justify-between mb-2">
                    <Typography.Text strong>{ch.title}</Typography.Text>
                    <Button
                      size="small"
                      type={analysis ? "default" : "primary"}
                      loading={analyzingChapterIds.has(ch.id)}
                      onClick={() => handleAnalyze(ch.id)}
                    >
                      {analysis ? (isStale ? "Re-analyze" : "Refresh") : "Analyze"}
                    </Button>
                  </div>
                  {analysis && !result.error ? (
                    <div className="flex flex-col gap-1">
                      {METRICS.map((m) => (
                        <div key={m.key} className="flex items-center gap-2">
                          <Typography.Text className="w-24 text-xs">{m.label}</Typography.Text>
                          <Progress
                            percent={Math.round((result[m.key] || 0) * 100)}
                            size="small"
                            strokeColor={m.color}
                            className="flex-1"
                          />
                        </div>
                      ))}
                      {result.assessment && (
                        <Typography.Text type="secondary" className="text-xs mt-1">
                          {result.assessment}
                        </Typography.Text>
                      )}
                    </div>
                  ) : analysis?.result?.error ? (
                    <Typography.Text type="danger" className="text-xs">{result.error}</Typography.Text>
                  ) : null}
                </div>
              </List.Item>
            )
          }}
        />
      )}
    </Modal>
  )
}

export default StoryPulseModal
