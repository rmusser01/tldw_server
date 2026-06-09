import { useEffect, useMemo, useState } from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { explainerApi } from "./explainerApi"
import { ExplainerChatbookExportButton } from "./ExplainerChatbookExportButton"
import { ExplainerDetailPanel } from "./ExplainerDetailPanel"
import { ExplainerGoalComposer } from "./ExplainerGoalComposer"
import { ExplainerModeTabs } from "./ExplainerModeTabs"
import { ExplainerSourcePicker } from "./ExplainerSourcePicker"
import { ExplainerTree } from "./ExplainerTree"
import type {
  ExplainerDepthPreset,
  ExplainerGrounding,
  ExplainerMode,
  ExplainerOutputIntent,
  ExplainerSelectedSource,
  ExplainerSourceCandidate
} from "./types"
import {
  getSelectedExplainerNode
} from "./tree"
import {
  useExplainerJob,
  useExplainerMutations,
  useExplainerSession,
  useExplainerSessions
} from "./useExplainerQueries"

const sourceKey = (source: Pick<ExplainerSelectedSource, "sourceId" | "sourceType">) =>
  `${source.sourceType}:${source.sourceId}`

const toSelectedSource = (source: ExplainerSourceCandidate): ExplainerSelectedSource => ({
  sourceId: source.sourceId,
  sourceType: source.sourceType,
  title: source.title,
  snapshotVersion: source.snapshotVersion ?? null,
  metadata: source.metadata ?? null
})

export const ExplainerWorkspace = () => {
  const [mode, setMode] = useState<ExplainerMode>("goal")
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [goal, setGoal] = useState("")
  const [outputIntent, setOutputIntent] = useState<ExplainerOutputIntent>("explain")
  const [depthPreset, setDepthPreset] = useState<ExplainerDepthPreset>("standard")
  const [grounding, setGrounding] = useState<ExplainerGrounding>("source_led")
  const [sourceQuery, setSourceQuery] = useState("")
  const [sourceResults, setSourceResults] = useState<ExplainerSourceCandidate[]>([])
  const [selectedSources, setSelectedSources] = useState<ExplainerSelectedSource[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [exportMessage, setExportMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const sessionsQuery = useExplainerSessions()
  const sessionQuery = useExplainerSession(selectedSessionId)
  const activeJobQuery = useExplainerJob(activeJobId, selectedSessionId)
  const mutations = useExplainerMutations()

  const session = sessionQuery.data ?? null
  const sessionSummaries = sessionsQuery.data?.items ?? []
  const selectedNode = useMemo(
    () => session ? getSelectedExplainerNode(session.nodes, session.rootNodeIds, selectedNodeId) : null,
    [session, selectedNodeId]
  )

  useEffect(() => {
    if (!selectedSessionId && sessionsQuery.data?.items?.[0]?.id) {
      setSelectedSessionId(sessionsQuery.data.items[0].id)
    }
  }, [selectedSessionId, sessionsQuery.data])

  useEffect(() => {
    if (session && !selectedNodeId) {
      setSelectedNodeId(session.rootNodeIds[0] ?? null)
      setOutputIntent(session.outputIntent)
      setDepthPreset(session.depthPreset)
      setGrounding(session.grounding)
      setSelectedSources(session.selectedSources)
    }
  }, [session, selectedNodeId])

  useEffect(() => {
    const status = activeJobQuery.data?.status
    if (status && ["completed", "failed", "cancelled"].includes(status)) {
      setActiveJobId(null)
    }
  }, [activeJobQuery.data?.status])

  const createGoalSession = async () => {
    const trimmedGoal = goal.trim()
    if (!trimmedGoal) return
    setErrorMessage(null)
    try {
      const created = await mutations.createSession.mutateAsync({
        mode: "goal",
        title: trimmedGoal,
        outputIntent,
        grounding: "open",
        depthPreset,
        selectedSources: [],
        rootPrompt: trimmedGoal
      })
      setSelectedSessionId(created.id)
      setSelectedNodeId(created.rootNodeIds[0] ?? null)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Explainer session creation failed")
    }
  }

  const createSourceSession = async () => {
    if (selectedSources.length === 0) return
    setErrorMessage(null)
    const title = selectedSources.length === 1
      ? selectedSources[0]?.title ?? "Source explainer"
      : `${selectedSources.length} source explainer`
    try {
      const created = await mutations.createSession.mutateAsync({
        mode: "sources",
        title,
        outputIntent,
        grounding,
        depthPreset,
        selectedSources,
        rootPrompt: `Explain selected sources: ${selectedSources.map((source) => source.title).join(", ")}`
      })
      setSelectedSessionId(created.id)
      setSelectedNodeId(created.rootNodeIds[0] ?? null)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Explainer session creation failed")
    }
  }

  const searchSources = async () => {
    const query = sourceQuery.trim()
    if (!query) return
    setIsSearching(true)
    setErrorMessage(null)
    try {
      setSourceResults(await explainerApi.searchSources(query))
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Source search failed")
    } finally {
      setIsSearching(false)
    }
  }

  const addSource = (source: ExplainerSourceCandidate) => {
    const selected = toSelectedSource(source)
    setSelectedSources((current) => {
      const keys = new Set(current.map(sourceKey))
      return keys.has(sourceKey(selected)) ? current : [...current, selected]
    })
  }

  const removeSource = (selected: ExplainerSelectedSource) => {
    const selectedKey = sourceKey(selected)
    setSelectedSources((current) =>
      current.filter((source) => sourceKey(source) !== selectedKey)
    )
  }

  const expandNode = async (nodeId: string) => {
    if (!session) return
    setErrorMessage(null)
    try {
      const accepted = await mutations.expandNode.mutateAsync({
        sessionId: session.id,
        nodeId,
        payload: { intent: outputIntent }
      })
      if (accepted?.jobId) {
        setActiveJobId(accepted.jobId)
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Explainer node expansion failed")
    }
  }

  const exportSession = async () => {
    if (!session) return
    setExportMessage(null)
    setErrorMessage(null)
    try {
      const response = await mutations.exportChatbook.mutateAsync({
        sessionId: session.id,
        payload: {
          name: `${session.title} Explainer Session`,
          asyncMode: true
        }
      })
      setExportMessage(response.message)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Chatbook export failed")
    }
  }

  const sessionCount = sessionsQuery.data?.total ?? sessionsQuery.data?.items?.length ?? 0
  const selectSession = (sessionId: string) => {
    setSelectedSessionId(sessionId || null)
    setSelectedNodeId(null)
    setExportMessage(null)
    setErrorMessage(null)
  }

  return (
    <main className="flex h-full min-h-0 flex-1 flex-col bg-bg text-text">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-surface px-5 py-4">
        <div>
          <h1 className="text-2xl font-semibold leading-tight text-text">Explainer</h1>
          <p className="mt-1 text-sm text-text-muted">
            {session ? `${session.title} · ${session.status}` : `${sessionCount} saved sessions`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {sessionSummaries.length > 0 ? (
            <label className="flex items-center gap-2 text-xs font-medium text-text-muted">
              <span className="sr-only">Saved Explainer sessions</span>
              <select
                aria-label="Saved Explainer sessions"
                className="h-9 max-w-[260px] rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
                value={selectedSessionId ?? sessionSummaries[0]?.id ?? ""}
                onChange={(event) => selectSession(event.target.value)}
              >
                {sessionSummaries.map((summary) => (
                  <option key={summary.id} value={summary.id}>
                    {summary.title}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {activeJobId ? (
            <span className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
              {activeJobQuery.data?.progressMessage ?? "Generation queued"}
            </span>
          ) : null}
          <ExplainerChatbookExportButton
            disabled={!session}
            isExporting={mutations.exportChatbook.isPending}
            message={exportMessage}
            onExport={exportSession}
          />
        </div>
      </header>

      <div className="border-b border-border bg-elevated px-4 py-3">
        <ExplainerModeTabs activeMode={mode} onModeChange={setMode} />
      </div>

      {mode === "goal" ? (
        <ExplainerGoalComposer
          goal={goal}
          outputIntent={outputIntent}
          depthPreset={depthPreset}
          isCreating={mutations.createSession.isPending}
          onGoalChange={setGoal}
          onOutputIntentChange={setOutputIntent}
          onDepthPresetChange={setDepthPreset}
          onCreate={createGoalSession}
        />
      ) : (
        <ExplainerSourcePicker
          query={sourceQuery}
          results={sourceResults}
          selectedSources={selectedSources}
          grounding={grounding}
          outputIntent={outputIntent}
          depthPreset={depthPreset}
          isSearching={isSearching}
          isCreating={mutations.createSession.isPending}
          onQueryChange={setSourceQuery}
          onSearch={searchSources}
          onAddSource={addSource}
          onRemoveSource={removeSource}
          onGroundingChange={setGrounding}
          onOutputIntentChange={setOutputIntent}
          onDepthPresetChange={setDepthPreset}
          onCreate={createSourceSession}
        />
      )}

      {errorMessage ? (
        <div className="flex items-center gap-2 border-b border-border bg-danger/10 px-4 py-2 text-sm font-medium text-danger">
          <AlertTriangle className="h-4 w-4" aria-hidden="true" />
          {errorMessage}
        </div>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_260px]">
        <ExplainerTree
          session={session}
          selectedNodeId={selectedNode?.id ?? null}
          onSelectNode={setSelectedNodeId}
          onExpandNode={expandNode}
        />
        <ExplainerDetailPanel
          session={session}
          node={selectedNode}
          isExpanding={mutations.expandNode.isPending}
          onExpand={expandNode}
        />
        <aside
          aria-label="Explainer session settings"
          className="hidden min-h-0 overflow-auto border-l border-border bg-surface px-4 py-4 lg:block"
        >
          <h2 className="text-sm font-semibold text-text">Session settings</h2>
          <dl className="mt-4 grid gap-3 text-sm">
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Intent
              </dt>
              <dd className="mt-1 text-text">{outputIntent}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Grounding
              </dt>
              <dd className="mt-1 text-text">{grounding.replace("_", "-")}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Depth
              </dt>
              <dd className="mt-1 text-text">{depthPreset}</dd>
            </div>
          </dl>
          <h3 className="mt-6 text-sm font-semibold text-text">Sources</h3>
          <ul className="mt-3 grid gap-2">
            {(session?.selectedSources ?? selectedSources).map((source) => (
              <li
                key={sourceKey(source)}
                className="rounded-md border border-border bg-surface2 px-3 py-2"
              >
                <p className="truncate text-sm font-medium text-text">{source.title}</p>
                <p className="text-xs text-text-muted">{source.sourceType}</p>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </main>
  )
}
