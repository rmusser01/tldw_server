import { useEffect, useRef, useState } from "react"
import { Button, Empty, Input, List, Spin, Typography } from "antd"
import { Search, BookOpen, Plus } from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import {
  createManuscriptCitation,
  searchManuscriptResearch,
  type ManuscriptResearchResponse,
  type ManuscriptResearchResult,
} from "@/services/writing-playground"

type ResearchTabProps = { isOnline: boolean }
type SearchSnapshot = { sceneId: string; query: string; token: symbol }

export function ResearchTab({ isOnline }: ResearchTabProps) {
  const { activeNodeId, activeNodeType } = useWritingPlaygroundStore()
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<ManuscriptResearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [citedIds, setCitedIds] = useState<Set<string>>(new Set())
  const lastSearchSnapshotRef = useRef<SearchSnapshot | null>(null)

  useEffect(() => {
    lastSearchSnapshotRef.current = null
    setSearching(false)
    setResults([])
    setCitedIds(new Set())
  }, [activeNodeId])

  const handleSearch = async () => {
    const trimmedQuery = query.trim()
    if (!trimmedQuery || !activeNodeId) return
    const snapshot = { sceneId: activeNodeId, query: trimmedQuery, token: Symbol("research-search") }
    lastSearchSnapshotRef.current = snapshot
    setSearching(true)
    try {
      const resp: ManuscriptResearchResponse = await searchManuscriptResearch(snapshot.sceneId, snapshot.query)
      if (
        lastSearchSnapshotRef.current?.token === snapshot.token
        && activeNodeId === snapshot.sceneId
      ) {
        setResults(resp.results || [])
      }
    } catch {
      if (lastSearchSnapshotRef.current?.token === snapshot.token) {
        setResults([])
      }
    } finally {
      if (lastSearchSnapshotRef.current?.token === snapshot.token) {
        setSearching(false)
      }
    }
  }

  const handleCite = async (result: ManuscriptResearchResult) => {
    const snapshot = lastSearchSnapshotRef.current
    if (!activeNodeId || !snapshot || snapshot.sceneId !== activeNodeId) return
    try {
      await createManuscriptCitation(activeNodeId, {
        source_type: result.source_type || "research",
        source_title: result.title || result.source_title || "Untitled",
        excerpt: result.snippet || result.excerpt || "",
        query_used: snapshot.query,
      })
      setCitedIds((prev) => new Set(prev).add(result.id || result.title))
    } catch (err) {
      console.error("[ResearchTab] Failed to create manuscript citation", err)
    }
  }

  if (!activeNodeId || activeNodeType !== "scene") {
    return (
      <Empty
        image={<BookOpen className="mx-auto h-8 w-8 text-gray-300" />}
        description="Select a scene to search your knowledge base"
        className="py-8"
      />
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-1">
        <Input
          size="small"
          placeholder="Search your knowledge base..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onPressEnter={handleSearch}
          prefix={<Search className="h-3 w-3 text-gray-400" />}
        />
        <Button
          size="small"
          type="primary"
          loading={searching}
          disabled={!query.trim() || !isOnline}
          onClick={handleSearch}
        >
          Search
        </Button>
      </div>

      {searching ? (
        <div className="flex justify-center py-4"><Spin size="small" /></div>
      ) : results.length > 0 ? (
        <List
          size="small"
          dataSource={results}
          renderItem={(result: ManuscriptResearchResult, index: number) => {
            const key = result.id || result.title || String(index)
            const isCited = citedIds.has(key)
            return (
              <List.Item className="!px-0 !py-2" actions={[
                <Button
                  key="cite"
                  size="small"
                  type={isCited ? "default" : "primary"}
                  icon={<Plus className="h-3 w-3" />}
                  disabled={isCited}
                  onClick={() => handleCite({ ...result, id: key })}
                >
                  {isCited ? "Cited" : "Cite"}
                </Button>
              ]}>
                <List.Item.Meta
                  title={<Typography.Text className="text-sm">{result.title || result.source_title || "Untitled"}</Typography.Text>}
                  description={
                    <Typography.Text type="secondary" className="text-xs">
                      {result.snippet || result.excerpt || ""}
                    </Typography.Text>
                  }
                />
              </List.Item>
            )
          }}
        />
      ) : query.trim() ? (
        <Empty description="No results found" className="py-4" />
      ) : null}
    </div>
  )
}

export default ResearchTab
