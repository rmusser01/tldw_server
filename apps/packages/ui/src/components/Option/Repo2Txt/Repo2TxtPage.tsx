import { useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useStore } from "zustand"
import { AdvancedFilters } from "./components/AdvancedFilters"
import { FileTree } from "./components/FileTree"
import { OutputPanel } from "./components/OutputPanel"
import { ProviderSelector } from "./components/ProviderSelector"
import type { IProvider, RepoTreeNode } from "./providers/types"
import { createRepo2TxtStore } from "./store"

type ProviderKind = "github" | "local" | null

const toDisplayTree = (nodes: RepoTreeNode[]) =>
  nodes.map((node) => ({
    name: node.path.split("/").pop() ?? node.path,
    path: node.path,
    type: node.type === "blob" ? "file" : "directory"
  }))

const loadGitHubProvider = () =>
  import("./providers/GitHubProvider").then((module) => module.GitHubProvider)

const loadLocalProvider = () =>
  import("./providers/LocalProvider").then((module) => module.LocalProvider)

const loadFetchFilesWithConcurrency = () =>
  import("./fetchFilesWithConcurrency").then(
    (module) => module.fetchFilesWithConcurrency
  )

const loadFormatter = () =>
  import("./formatter/Formatter").then((module) => module.Formatter)

export function Repo2TxtPage() {
  const { t } = useTranslation(["option"])
  const storeRef = useRef(createRepo2TxtStore())
  const store = storeRef.current
  const nodes = useStore(store, (state) => state.nodes)
  const selectedPaths = useStore(store, (state) => state.selectedPaths)

  const [providerKind, setProviderKind] = useState<ProviderKind>(null)
  const [provider, setProvider] = useState<IProvider | null>(null)
  const [githubUrl, setGithubUrl] = useState("")
  const [filterText, setFilterText] = useState("")
  const [output, setOutput] = useState("")
  const [busy, setBusy] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string>("")
  const [sourceLoaded, setSourceLoaded] = useState(false)

  const filteredNodes = useMemo(() => {
    const query = filterText.trim().toLowerCase()
    if (!query) return nodes
    return nodes.filter((node) => node.path.toLowerCase().includes(query))
  }, [nodes, filterText])

  const activeOperation = useRef<AbortController | null>(null)
  useEffect(() => () => activeOperation.current?.abort(), [])

  const resetSource = () => {
    setProvider(null)
    setSourceLoaded(false)
    setOutput("")
    setStatusMessage("")
    store.getState().setNodes([])
  }

  const runOperation = async (work: (signal: AbortSignal) => Promise<void>) => {
    const controller = new AbortController()
    activeOperation.current?.abort()
    activeOperation.current = controller
    setBusy(true)
    setOutput("")
    setStatusMessage("")
    try {
      await work(controller.signal)
    } catch (error) {
      if (!controller.signal.aborted) {
        setStatusMessage(
          error instanceof Error
            ? error.message
            : t("option:repo2txt.errors.generate", {
                defaultValue: "Failed to generate output."
              })
        )
      }
    } finally {
      if (activeOperation.current === controller) {
        activeOperation.current = null
        setBusy(false)
      }
    }
  }

  const handleCancel = () => {
    activeOperation.current?.abort()
    activeOperation.current = null
    setBusy(false)
    setOutput("")
    setStatusMessage(
      t("option:repo2txt.status.cancelled", {
        defaultValue: "Cancelled. No output was generated."
      })
    )
  }

  const handleSelectProvider = (nextProvider: Exclude<ProviderKind, null>) => {
    setProviderKind(nextProvider)
    resetSource()
  }

  const handleLoadGithub = async () => {
    resetSource()
    await runOperation(async (signal) => {
      const GitHubProvider = await loadGitHubProvider()
      signal.throwIfAborted()
      const githubProvider = new GitHubProvider()
      if (!githubProvider.validateUrl(githubUrl)) {
        throw new Error(
          t("option:repo2txt.errors.invalidGithubUrl", {
            defaultValue: "Enter a valid GitHub repository URL."
          })
        )
      }
      const tree = await githubProvider.fetchTree(githubUrl, { signal })
      signal.throwIfAborted()
      setProvider(githubProvider)
      store.getState().setNodes(tree)
      setSourceLoaded(tree.length > 0)
      setStatusMessage(
        t("option:repo2txt.status.loadedTree", {
          defaultValue: "Loaded {{count}} file tree entries.",
          count: tree.length
        })
      )
    })
  }

  const handleLocalFilesSelected = async (files: FileList) => {
    resetSource()
    await runOperation(async (signal) => {
      const LocalProvider = await loadLocalProvider()
      signal.throwIfAborted()
      const localProvider = new LocalProvider()
      const firstFile = files[0]
      const looksLikeZip =
        files.length === 1 && /\.zip$/i.test(firstFile?.name ?? "")
      await localProvider.initialize(
        looksLikeZip
          ? { source: "zip", zipFile: firstFile, signal }
          : { source: "directory", files, signal }
      )
      signal.throwIfAborted()
      const tree = await localProvider.fetchTree("local://directory")
      signal.throwIfAborted()
      setProvider(localProvider)
      store.getState().setNodes(tree)
      setSourceLoaded(tree.length > 0)
      setStatusMessage(
        t("option:repo2txt.status.loadedLocal", {
          defaultValue: "Loaded {{count}} local files.",
          count: tree.length
        })
      )
    })
  }

  const handleTogglePath = (path: string) => {
    store.getState().togglePath(path)
    setOutput("")
    setStatusMessage("")
  }

  const handleGenerate = async () => {
    if (!provider) return
    const selectedNodes = nodes
      .filter((node) => node.type === "blob" && selectedPaths.has(node.path))
      .sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    if (selectedNodes.length === 0) {
      setOutput("")
      setStatusMessage(
        t("option:repo2txt.status.selectAtLeastOne", {
          defaultValue: "Select at least one file to generate output."
        })
      )
      return
    }
    await runOperation(async (signal) => {
      const fetchFilesWithConcurrency = await loadFetchFilesWithConcurrency()
      signal.throwIfAborted()
      const fileContents = await fetchFilesWithConcurrency({
        nodes: selectedNodes,
        provider,
        limit: 5,
        signal,
        onProgress: (completed, total) => {
          if (!signal.aborted)
            setStatusMessage(
              t("option:repo2txt.status.fetchingFiles", {
                defaultValue: "Fetching files ({{completed}}/{{total}})...",
                completed,
                total
              })
            )
        }
      })
      signal.throwIfAborted()
      const Formatter = await loadFormatter()
      signal.throwIfAborted()
      const formatted = await Formatter.formatAsync(
        toDisplayTree(selectedNodes),
        fileContents,
        undefined,
        signal
      )
      signal.throwIfAborted()
      const text = `${formatted.directoryTree}\n\n${formatted.fileContents}`
      setOutput(text)
      setStatusMessage(
        t("option:repo2txt.status.generatedBounded", {
          defaultValue:
            "Generated {{count}} files; {{bytes}} UTF-8 output bytes ({{tokenCount}} tokens).",
          count: selectedNodes.length,
          bytes: new TextEncoder().encode(text).byteLength,
          tokenCount: formatted.tokenCount
        })
      )
    })
  }

  const handleCopy = async () => {
    if (!output.trim()) return
    if (!navigator?.clipboard?.writeText) return
    await navigator.clipboard.writeText(output)
  }

  const handleDownload = () => {
    if (!output.trim()) return
    const blob = new Blob([output], { type: "text/plain;charset=utf-8" })
    const href = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = href
    anchor.download = "repo2txt-output.txt"
    anchor.click()
    URL.revokeObjectURL(href)
  }

  return (
    <section data-testid="repo2txt-route-root" className="space-y-4">
      <div
        data-testid="repo2txt-provider-panel"
        className="space-y-3 rounded border p-3"
      >
        <ProviderSelector
          provider={providerKind}
          githubUrl={githubUrl}
          busy={busy}
          onSelectProvider={handleSelectProvider}
          onGithubUrlChange={(value) => {
            setGithubUrl(value)
            resetSource()
          }}
          onLoadGithub={handleLoadGithub}
          onLocalFilesSelected={handleLocalFilesSelected}
        />
        <AdvancedFilters
          filterText={filterText}
          onFilterTextChange={setFilterText}
          totalCount={nodes.filter((node) => node.type === "blob").length}
          selectedCount={selectedPaths.size}
        />
        <FileTree
          nodes={filteredNodes}
          selectedPaths={selectedPaths}
          onTogglePath={handleTogglePath}
          disabled={busy}
        />
        {statusMessage && (
          <p role="status" className="text-xs text-text-subtle">
            {statusMessage}
          </p>
        )}
      </div>

      <div data-testid="repo2txt-output-panel" className="rounded border p-3">
        <OutputPanel
          output={output}
          busy={busy}
          canGenerate={sourceLoaded}
          onGenerate={handleGenerate}
          onCopy={handleCopy}
          onDownload={handleDownload}
          onCancel={handleCancel}
        />
      </div>
    </section>
  )
}
