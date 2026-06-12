import { useEffect, useMemo, useRef, useState } from "react"
import { AlertTriangle, Archive, Pencil, Plus, RefreshCw, X } from "lucide-react"
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
  getExplainerDepthLabel,
  getExplainerGroundingLabel,
  getExplainerIntentLabel,
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

// API client errors append "(METHOD /path)" — transport detail users should
// never see in the error banner.
const sanitizeErrorMessage = (message: string): string =>
  message.replace(/\s*\((?:GET|POST|PUT|PATCH|DELETE)\s+\/[^)]*\)\s*$/i, "").trim()

export const ExplainerWorkspace = () => {
  const [mode, setMode] = useState<ExplainerMode>("goal")
  const [isComposing, setIsComposing] = useState(true)
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
  const [activeJobNodeId, setActiveJobNodeId] = useState<string | null>(null)
  const [exportMessage, setExportMessage] = useState<string | null>(null)
  const [exportDownloadUrl, setExportDownloadUrl] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isRenaming, setIsRenaming] = useState(false)
  const [renameDraft, setRenameDraft] = useState("")
  const [confirmingArchive, setConfirmingArchive] = useState(false)
  const detailSectionRef = useRef<HTMLElement | null>(null)

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
  const generatingNodeId = activeJobId ? activeJobNodeId : null

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

  // The composer yields the stage once a session is on screen.
  useEffect(() => {
    if (session) {
      setIsComposing(false)
    }
  }, [session?.id])

  useEffect(() => {
    const status = activeJobQuery.data?.status
    if (status && ["completed", "failed", "cancelled"].includes(status)) {
      setActiveJobId(null)
      setActiveJobNodeId(null)
    }
  }, [activeJobQuery.data?.status])

  // On stacked (narrow) layouts the outline sits above the reading pane;
  // bring the pane into view when a node is picked.
  useEffect(() => {
    if (selectedNodeId && window.innerWidth < 1024) {
      detailSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  }, [selectedNodeId])

  const clearTransientState = () => {
    setActiveJobId(null)
    setActiveJobNodeId(null)
    setExportMessage(null)
    setExportDownloadUrl(null)
    setErrorMessage(null)
    setIsRenaming(false)
    setConfirmingArchive(false)
  }

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
      setGoal("")
      setIsComposing(false)
      clearTransientState()
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(
          error instanceof Error ? error.message : "Explainer session creation failed"
        )
      )
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
      setIsComposing(false)
      clearTransientState()
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(
          error instanceof Error ? error.message : "Explainer session creation failed"
        )
      )
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
      setErrorMessage(
        sanitizeErrorMessage(error instanceof Error ? error.message : "Source search failed")
      )
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
        setActiveJobNodeId(nodeId)
      }
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(
          error instanceof Error ? error.message : "Explainer node expansion failed"
        )
      )
    }
  }

  const deleteNode = async (nodeId: string) => {
    if (!session) return
    setErrorMessage(null)
    const parentId = session.nodes[nodeId]?.parentId ?? session.rootNodeIds[0] ?? null
    try {
      await mutations.deleteNode.mutateAsync({ sessionId: session.id, nodeId })
      setSelectedNodeId(parentId)
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(
          error instanceof Error ? error.message : "Explainer node deletion failed"
        )
      )
    }
  }

  const renameSession = async () => {
    if (!session) return
    const title = renameDraft.trim()
    if (!title || title === session.title) {
      setIsRenaming(false)
      return
    }
    setErrorMessage(null)
    try {
      await mutations.updateSession.mutateAsync({ sessionId: session.id, payload: { title } })
      setIsRenaming(false)
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(
          error instanceof Error ? error.message : "Explainer session rename failed"
        )
      )
    }
  }

  const archiveSession = async () => {
    if (!session) return
    setErrorMessage(null)
    try {
      await mutations.archiveSession.mutateAsync(session.id)
      setConfirmingArchive(false)
      setSelectedSessionId(null)
      setSelectedNodeId(null)
      clearTransientState()
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(
          error instanceof Error ? error.message : "Explainer session archive failed"
        )
      )
    }
  }

  const exportSession = async () => {
    if (!session) return
    setExportMessage(null)
    setExportDownloadUrl(null)
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
      setExportDownloadUrl(response.download_url ?? null)
    } catch (error) {
      setErrorMessage(
        sanitizeErrorMessage(error instanceof Error ? error.message : "Chatbook export failed")
      )
    }
  }

  const sessionCount = sessionsQuery.data?.total ?? sessionsQuery.data?.items?.length ?? 0
  const selectSession = (sessionId: string) => {
    setSelectedSessionId(sessionId || null)
    setSelectedNodeId(null)
    clearTransientState()
  }

  const railIntent = session ? session.outputIntent : outputIntent
  const railGrounding = session ? session.grounding : grounding
  const railDepth = session ? session.depthPreset : depthPreset
  const railSources = session?.selectedSources ?? []

  return (
    <main className="flex h-full min-h-0 flex-1 flex-col bg-bg text-text">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-surface px-5 py-4">
        <div>
          <h1 className="text-2xl font-semibold leading-tight text-text">Explainer</h1>
          <p className="mt-1 text-sm text-text-muted">
            {session
              ? `${session.title} · ${session.status}`
              : sessionCount === 0
                ? "No saved explainers yet"
                : `${sessionCount} saved explainer${sessionCount === 1 ? "" : "s"}`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {isRenaming && session ? (
            <span className="flex items-center gap-2">
              <input
                aria-label="Session title"
                className="h-9 w-[240px] rounded-md border border-border bg-surface2 px-3 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-focus"
                value={renameDraft}
                autoFocus
                onChange={(event) => setRenameDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void renameSession()
                  if (event.key === "Escape") setIsRenaming(false)
                }}
              />
              <button
                type="button"
                aria-label="Save title"
                className="inline-flex h-9 items-center rounded-md bg-primary px-3 text-sm font-semibold text-white hover:bg-primaryStrong"
                onClick={() => void renameSession()}
              >
                Save
              </button>
              <button
                type="button"
                className="inline-flex h-9 items-center rounded-md border border-border bg-surface px-3 text-sm font-medium text-text hover:bg-surface2"
                onClick={() => setIsRenaming(false)}
              >
                Cancel
              </button>
            </span>
          ) : (
            <>
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
              {session ? (
                <>
                  <button
                    type="button"
                    aria-label="Rename session"
                    title="Rename session"
                    className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors hover:bg-surface2 hover:text-text"
                    onClick={() => {
                      setRenameDraft(session.title)
                      setIsRenaming(true)
                    }}
                  >
                    <Pencil className="h-4 w-4" aria-hidden="true" />
                  </button>
                  {confirmingArchive ? (
                    <button
                      type="button"
                      aria-label="Confirm archive"
                      className="inline-flex h-9 items-center gap-2 rounded-md bg-danger px-3 text-sm font-semibold text-white hover:opacity-90"
                      onClick={() => void archiveSession()}
                    >
                      <Archive className="h-4 w-4" aria-hidden="true" />
                      Confirm archive
                    </button>
                  ) : (
                    <button
                      type="button"
                      aria-label="Archive session"
                      title="Archive session"
                      className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors hover:bg-surface2 hover:text-text"
                      onClick={() => setConfirmingArchive(true)}
                    >
                      <Archive className="h-4 w-4" aria-hidden="true" />
                    </button>
                  )}
                </>
              ) : null}
            </>
          )}
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
            downloadUrl={exportDownloadUrl}
            onExport={exportSession}
          />
        </div>
      </header>

      {isComposing ? (
        <>
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
        </>
      ) : (
        <div className="flex items-center justify-between gap-3 border-b border-border bg-elevated px-4 py-2">
          <p className="text-xs text-text-muted">
            Start another explainer without leaving this one.
          </p>
          <button
            type="button"
            className="inline-flex h-8 items-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:bg-surface2"
            onClick={() => setIsComposing(true)}
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
            New explainer
          </button>
        </div>
      )}

      {errorMessage ? (
        <div className="flex items-center justify-between gap-2 border-b border-border bg-danger/10 px-4 py-2 text-sm font-medium text-danger">
          <span className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            {errorMessage}
          </span>
          <button
            type="button"
            aria-label="Dismiss error"
            className="inline-flex h-7 w-7 items-center justify-center rounded-md text-danger transition-colors hover:bg-danger/10"
            onClick={() => setErrorMessage(null)}
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_260px]">
        <ExplainerTree
          session={session}
          selectedNodeId={selectedNode?.id ?? null}
          generatingNodeId={generatingNodeId}
          onSelectNode={setSelectedNodeId}
          onExpandNode={expandNode}
        />
        <ExplainerDetailPanel
          session={session}
          node={selectedNode}
          isExpanding={mutations.expandNode.isPending}
          generatingNodeId={generatingNodeId}
          sectionRef={detailSectionRef}
          onExpand={expandNode}
          onDeleteNode={deleteNode}
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
              <dd className="mt-1 text-text">{getExplainerIntentLabel(railIntent)}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Grounding
              </dt>
              <dd className="mt-1 text-text">{getExplainerGroundingLabel(railGrounding)}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Depth
              </dt>
              <dd className="mt-1 text-text">{getExplainerDepthLabel(railDepth)}</dd>
            </div>
          </dl>
          <h3 className="mt-6 text-sm font-semibold text-text">Sources</h3>
          {railSources.length === 0 ? (
            <p className="mt-3 text-xs text-text-muted">
              {session ? "This explainer has no attached sources." : "No session loaded."}
            </p>
          ) : (
            <ul className="mt-3 grid gap-2">
              {railSources.map((source) => (
                <li
                  key={sourceKey(source)}
                  className="min-w-0 rounded-md border border-border bg-surface2 px-3 py-2"
                >
                  <p className="truncate text-sm font-medium text-text" title={source.title}>
                    {source.title}
                  </p>
                  <p className="truncate text-xs text-text-muted">{source.sourceType}</p>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </main>
  )
}
