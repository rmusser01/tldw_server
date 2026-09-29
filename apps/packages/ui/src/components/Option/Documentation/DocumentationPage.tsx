import React from "react"
import { Input, Tabs } from "antd"
import { FileText, Search } from "lucide-react"
import { useTranslation } from "react-i18next"

import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { MarkdownErrorBoundary } from "@/components/Common/MarkdownErrorBoundary"
import { PageShell } from "@/components/Common/PageShell"
import { classNames } from "@/libs/class-name"
import { withTemplateFallback } from "@/utils/template-guards"
import { textContainsQuery } from "@/utils/text-highlight"

const Markdown = React.lazy(() => import("@/components/Common/Markdown"))

export type DocumentationSource = "extension" | "server"

type DocImportMap = Record<string, () => Promise<string>>

export type DocumentationDocEntry = {
  id: string
  title: string
  source: DocumentationSource
  relativePath: string
  fullPath: string
  importPath?: string
  inlineContent?: string
  isFallback?: boolean
}

export type DocumentationDocsBySource = Record<
  DocumentationSource,
  DocumentationDocEntry[]
>

type DocumentationPageProps = {
  docsBySource?: Partial<DocumentationDocsBySource>
  loadDocContent?: (doc: DocumentationDocEntry) => Promise<string>
}

type DocLoadState = {
  status: "loading" | "loaded" | "error"
  content?: string
  title?: string
  error?: string
}

// import.meta.glob is Vite-only; use empty object in Next.js/non-Vite environments
const EXTENSION_DOC_IMPORTS: DocImportMap =
  typeof import.meta?.glob === "function"
    ? (import.meta.glob(
        "/Docs/User_Documentation/**/*.{md,mdx}",
        { query: "?raw", import: "default" }
      ) as DocImportMap)
    : {}
const SERVER_DOC_IMPORTS: DocImportMap =
  typeof import.meta?.glob === "function"
    ? (import.meta.glob(
        "/Docs/Published/**/*.{md,mdx}",
        { query: "?raw", import: "default" }
      ) as DocImportMap)
    : {}

const DOC_IMPORTS_BY_SOURCE: Record<DocumentationSource, DocImportMap> = {
  extension: EXTENSION_DOC_IMPORTS,
  server: SERVER_DOC_IMPORTS
}

const stripFrontmatter = (content: string) =>
  content.replace(/^---[\s\S]*?---\s*/, "")

const normalizeTitle = (value: string) =>
  value.replace(/\s+/g, " ").trim()

const toTitleCase = (value: string) =>
  value.replace(/\b\w/g, (char) => char.toUpperCase())

const fileTitleFromPath = (path: string) => {
  const fileName = path.split("/").pop() || path
  const baseName = fileName.replace(/\.[^.]+$/, "")
  return toTitleCase(normalizeTitle(baseName.replace(/[_-]+/g, " ")))
}

const extractTitle = (content: string, fallback: string) => {
  const stripped = stripFrontmatter(content)
  const match = stripped.match(/^#\s+(.+)$/m)
  if (match?.[1]) {
    return normalizeTitle(match[1])
  }
  return normalizeTitle(fallback)
}

const normalizePath = (path: string) => path.replace(/^\/+/, "")

const buildDocs = (
  imports: DocImportMap,
  source: DocumentationSource,
  baseDir: string
): DocumentationDocEntry[] =>
  Object.keys(imports)
    .map((path) => {
      const fullPath = normalizePath(path)
      const relativePath = fullPath.startsWith(`${baseDir}/`)
        ? fullPath.slice(baseDir.length + 1)
        : fullPath
      return {
        id: `${source}:${fullPath}`,
        title: fileTitleFromPath(relativePath),
        source,
        relativePath,
        fullPath,
        importPath: path,
        isFallback: false
      }
    })
    .sort((a, b) => a.title.localeCompare(b.title))

const resolveSelectedDocSelection = (
  docs: DocumentationDocEntry[],
  activeId: string | null
) => {
  if (activeId && docs.some((doc) => doc.id === activeId)) {
    return { selectedId: activeId, shouldSync: false }
  }
  const selectedId = docs[0]?.id ?? null
  return { selectedId, shouldSync: selectedId !== null }
}

const EXTENSION_DOCS = buildDocs(
  EXTENSION_DOC_IMPORTS,
  "extension",
  "Docs/User_Documentation"
)
const SERVER_DOCS = buildDocs(
  SERVER_DOC_IMPORTS,
  "server",
  "Docs/Published"
)

const buildFallbackDoc = (
  source: DocumentationSource,
  title: string,
  baseDir: string
): DocumentationDocEntry => ({
  id: `${source}:fallback`,
  title,
  source,
  relativePath: `${baseDir}/README.md`,
  fullPath: `${baseDir}/README.md`,
  importPath: `inline:${source}:fallback`,
  isFallback: true,
  inlineContent: `# ${title}

Documentation files were not auto-discovered for this source.

Expected path: \`${baseDir}\`

Add markdown files under ${baseDir} to populate this section.
`
})

const EXTENSION_FALLBACK_DOCS: DocumentationDocEntry[] = [
  buildFallbackDoc("extension", "Extension Documentation", "Docs/User_Documentation")
]

const SERVER_FALLBACK_DOCS: DocumentationDocEntry[] = [
  buildFallbackDoc("server", "Server Documentation", "Docs/Published")
]

const EXTENSION_DOCS_WITH_FALLBACK =
  EXTENSION_DOCS.length > 0 ? EXTENSION_DOCS : EXTENSION_FALLBACK_DOCS

const SERVER_DOCS_WITH_FALLBACK =
  SERVER_DOCS.length > 0 ? SERVER_DOCS : SERVER_FALLBACK_DOCS

const hasRealDocs = (docs: DocumentationDocEntry[]) =>
  docs.some((doc) => !doc.isFallback)

const resolveDocsBySource = (
  docsBySource?: Partial<DocumentationDocsBySource>
): DocumentationDocsBySource => ({
  extension:
    docsBySource?.extension && docsBySource.extension.length > 0
      ? docsBySource.extension
      : EXTENSION_DOCS_WITH_FALLBACK,
  server:
    docsBySource?.server && docsBySource.server.length > 0
      ? docsBySource.server
      : SERVER_DOCS_WITH_FALLBACK
})

const pickDefaultActiveSource = (
  docsBySource: DocumentationDocsBySource
): DocumentationSource => {
  if (!hasRealDocs(docsBySource.extension) && hasRealDocs(docsBySource.server)) {
    return "server"
  }
  if (hasRealDocs(docsBySource.extension)) return "extension"
  if (hasRealDocs(docsBySource.server)) return "server"
  if (docsBySource.extension.length > 0) return "extension"
  if (docsBySource.server.length > 0) return "server"
  return "extension"
}

export const DocumentationPage: React.FC<DocumentationPageProps> = ({
  docsBySource: docsBySourceOverride,
  loadDocContent
}) => {
  const { t } = useTranslation(["option", "common"])
  const docsBySource = React.useMemo(
    () => resolveDocsBySource(docsBySourceOverride),
    [docsBySourceOverride]
  )
  const [activeSource, setActiveSource] = React.useState<DocumentationSource>(() =>
    pickDefaultActiveSource(docsBySource)
  )
  const [activeDocIds, setActiveDocIds] = React.useState<
    Record<DocumentationSource, string | null>
  >(() => ({
    extension: docsBySource.extension[0]?.id ?? null,
    server: docsBySource.server[0]?.id ?? null
  }))
  const [searchQuery, setSearchQuery] = React.useState("")
  const [docStateById, setDocStateById] = React.useState<
    Record<string, DocLoadState>
  >({})
  const loadingDocIds = React.useRef(new Set<string>())

  const sourceMeta = React.useMemo(
    () => ({
      extension: {
        label: t(
          "option:documentation.sourceExtension",
          "tldw browser extension"
        ),
        path: "Docs/User_Documentation"
      },
      server: {
        label: t("option:documentation.sourceServer", "tldw_server"),
        path: "Docs/Published"
      }
    }),
    [t]
  )

  React.useEffect(() => {
    const preferredSource = pickDefaultActiveSource(docsBySource)
    setActiveSource((current) => {
      if (current === preferredSource) return current
      const currentDocs = docsBySource[current]
      if (!hasRealDocs(currentDocs) && hasRealDocs(docsBySource[preferredSource])) {
        return preferredSource
      }
      if (currentDocs.length === 0 && docsBySource[preferredSource].length > 0) {
        return preferredSource
      }
      return current
    })
  }, [docsBySource])

  const trimmedQuery = searchQuery.trim()

  const filterDocs = React.useCallback(
    (docs: DocumentationDocEntry[]) => {
      if (!trimmedQuery) return docs
      return docs.filter((doc) => {
        const docState = docStateById[doc.id]
        const title = docState?.title ?? doc.title
        const content = docState?.content ?? ""
        return textContainsQuery(`${title}\n${content}`, trimmedQuery)
      })
    },
    [docStateById, trimmedQuery]
  )

  const filteredDocsBySource = React.useMemo(
    () => ({
      extension: filterDocs(docsBySource.extension),
      server: filterDocs(docsBySource.server)
    }),
    [docsBySource, filterDocs]
  )

  const resolvedSelectionBySource = React.useMemo(
    () => ({
      extension: resolveSelectedDocSelection(
        filteredDocsBySource.extension,
        activeDocIds.extension
      ),
      server: resolveSelectedDocSelection(
        filteredDocsBySource.server,
        activeDocIds.server
      )
    }),
    [activeDocIds, filteredDocsBySource]
  )

  const selectedDocIdBySource = React.useMemo(
    () => ({
      extension: resolvedSelectionBySource.extension.selectedId,
      server: resolvedSelectionBySource.server.selectedId
    }),
    [resolvedSelectionBySource]
  )

  const selectedDocBySource = React.useMemo(
    () => ({
      extension:
        filteredDocsBySource.extension.find(
          (doc) => doc.id === selectedDocIdBySource.extension
        ) ?? null,
      server:
        filteredDocsBySource.server.find(
          (doc) => doc.id === selectedDocIdBySource.server
        ) ?? null
    }),
    [filteredDocsBySource, selectedDocIdBySource]
  )

  React.useEffect(() => {
    const sources: DocumentationSource[] = ["extension", "server"]
    const nextActiveDocIds = { ...activeDocIds }
    let shouldUpdate = false

    for (const source of sources) {
      const { selectedId, shouldSync } = resolvedSelectionBySource[source]
      if (shouldSync && selectedId !== activeDocIds[source]) {
        nextActiveDocIds[source] = selectedId
        shouldUpdate = true
      }
    }

    if (shouldUpdate) {
      setActiveDocIds(nextActiveDocIds)
    }
  }, [activeDocIds, resolvedSelectionBySource])

  const ensureDocLoaded = React.useCallback(
    async (doc: DocumentationDocEntry | null) => {
      if (!doc) return
      const existingState = docStateById[doc.id]
      if (
        existingState?.status === "loaded" ||
        existingState?.status === "loading" ||
        existingState?.status === "error"
      ) {
        return
      }
      if (loadingDocIds.current.has(doc.id)) return

      if (typeof doc.inlineContent === "string") {
        const title = extractTitle(doc.inlineContent, doc.title)
        setDocStateById((prev) => ({
          ...prev,
          [doc.id]: {
            status: "loaded",
            content: doc.inlineContent,
            title
          }
        }))
        return
      }

      const importLoader = doc.importPath
        ? DOC_IMPORTS_BY_SOURCE[doc.source][doc.importPath]
        : undefined
      if (!loadDocContent && !importLoader) {
        setDocStateById((prev) => ({
          ...prev,
          [doc.id]: {
            status: "error",
            error: "Document loader not found."
          }
        }))
        return
      }

      loadingDocIds.current.add(doc.id)
      setDocStateById((prev) => ({
        ...prev,
        [doc.id]: { status: "loading" }
      }))

      try {
        const content = loadDocContent
          ? await loadDocContent(doc)
          : await importLoader!()
        const title = extractTitle(content, doc.title)
        setDocStateById((prev) => ({
          ...prev,
          [doc.id]: {
            status: "loaded",
            content,
            title
          }
        }))
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error"
        setDocStateById((prev) => ({
          ...prev,
          [doc.id]: {
            status: "error",
            error: message
          }
        }))
      } finally {
        loadingDocIds.current.delete(doc.id)
      }
    },
    [docStateById, loadDocContent]
  )

  const activeSelectedDoc = selectedDocBySource[activeSource]

  React.useEffect(() => {
    void ensureDocLoaded(activeSelectedDoc)
  }, [activeSelectedDoc, ensureDocLoaded])

  const renderDocPane = (source: DocumentationSource) => {
    const docs = docsBySource[source]
    const filteredDocs = filteredDocsBySource[source]
    const selectedDocId = selectedDocIdBySource[source]
    const selectedDoc = selectedDocBySource[source]
    const selectedDocState = selectedDoc
      ? docStateById[selectedDoc.id]
      : undefined
    const selectedDocTitle = selectedDoc
      ? selectedDocState?.title ?? selectedDoc.title
      : ""
    const selectedDocContent = selectedDocState?.content ?? ""
    const isDocLoaded = selectedDocState?.status === "loaded"
    const isDocError = selectedDocState?.status === "error"

    if (docs.length === 0) {
      const emptyDescription = withTemplateFallback(
        t(
          "option:documentation.emptyDescription",
          "Add markdown files under {{path}} to populate this section.",
          { path: sourceMeta[source].path }
        ),
        `Add markdown files under ${sourceMeta[source].path} to populate this section.`
      )
      return (
        <FeatureEmptyState
          title={t(
            "option:documentation.emptyTitle",
            "No documentation found"
          )}
          description={emptyDescription}
          icon={FileText}
        />
      )
    }

    return (
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="lg:w-72 lg:shrink-0">
          <div className="rounded-2xl border border-border bg-surface/90 p-4">
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                {t("option:documentation.documentsLabel", "Documents")}
              </p>
              <p className="text-sm font-medium text-text">
                {sourceMeta[source].label}
              </p>
              <p className="text-xs text-text-muted font-mono">
                {sourceMeta[source].path}
              </p>
              <p className="text-xs text-text-muted">
                {withTemplateFallback(
                  t("option:documentation.resultsLabel", "{{count}} results", {
                    count: filteredDocs.length
                  }),
                  `${filteredDocs.length} results`
                )}
              </p>
            </div>
            <div className="mt-3 max-h-[60vh] space-y-2 overflow-auto pr-1">
              {filteredDocs.length > 0 ? (
                filteredDocs.map((doc) => {
                  const isActive = selectedDocId === doc.id
                  const docTitle = docStateById[doc.id]?.title ?? doc.title
                  return (
                    <button
                      key={doc.id}
                      type="button"
                      aria-pressed={isActive}
                      onClick={() =>
                        setActiveDocIds((prev) => ({
                          ...prev,
                          [source]: doc.id
                        }))
                      }
                      className={classNames(
                        "w-full rounded-xl border px-3 py-2 text-left text-sm transition",
                        isActive
                          ? "border-primary/40 bg-primary/10 text-text"
                          : "border-transparent text-text-muted hover:border-border hover:bg-surface2 hover:text-text"
                      )}
                    >
                      <div className="font-medium">{docTitle}</div>
                      <div className="mt-0.5 text-xs text-text-muted">
                        {doc.relativePath}
                      </div>
                    </button>
                  )
                })
              ) : (
                <div className="rounded-xl border border-dashed border-border px-3 py-4 text-xs text-text-muted">
                  {t(
                    "option:documentation.noMatchesList",
                    "No matches in this source."
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="flex-1">
          {selectedDoc ? (
            <div className="rounded-2xl border border-border bg-surface/90 p-5">
              <div className="flex flex-col gap-1 border-b border-border pb-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold text-text">
                      {selectedDocTitle}
                    </h2>
                    <p className="text-xs text-text-muted">
                      {selectedDoc.relativePath}
                    </p>
                  </div>
                  <span className="rounded-full bg-surface2 px-2 py-0.5 text-[11px] font-semibold text-text-muted">
                    {sourceMeta[source].label}
                  </span>
                </div>
              </div>
              <div className="pt-4">
                {isDocError ? (
                  <div className="text-sm text-text-muted">
                    {t(
                      "option:documentation.loadError",
                      "Unable to load document."
                    )}
                  </div>
                ) : isDocLoaded ? (
                  <MarkdownErrorBoundary fallbackText={selectedDocContent}>
                    <React.Suspense
                      fallback={
                        <div className="text-sm text-text-muted">
                          {t("common:loading.content", "Loading content...")}
                        </div>
                      }
                    >
                      <Markdown
                        message={selectedDocContent}
                        searchQuery={trimmedQuery}
                        className="prose break-words dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark"
                      />
                    </React.Suspense>
                  </MarkdownErrorBoundary>
                ) : (
                  <div className="text-sm text-text-muted">
                    {t("common:loading.content", "Loading content...")}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <FeatureEmptyState
              title={t(
                "option:documentation.noMatchesTitle",
                "No matches"
              )}
              description={t(
                "option:documentation.noMatchesDescription",
                "Try a different search term."
              )}
              icon={Search}
            />
          )}
        </div>
      </div>
    )
  }

  return (
    <PageShell className="py-6" maxWidthClassName="max-w-6xl">
      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold text-text">
            {t("option:documentation.title", "Documentation")}
          </h1>
          <p className="text-sm text-text-muted">
            {t(
              "option:documentation.subtitle",
              "Browse documentation for the tldw browser extension and tldw_server."
            )}
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <Input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            allowClear
            prefix={<Search className="h-4 w-4 text-text-muted" />}
            placeholder={t(
              "option:documentation.searchPlaceholder",
              "Search documentation..."
            )}
            className="bg-surface"
          />
          <p className="text-xs text-text-muted">
            {withTemplateFallback(
              t(
                "option:documentation.sourceHint",
                "Sources: {{extensionPath}} and {{serverPath}}",
                {
                  extensionPath: "Docs/User_Documentation",
                  serverPath: "Docs/Published"
                }
              ),
              "Sources: Docs/User_Documentation and Docs/Published"
            )}
          </p>
        </div>
        <Tabs
          activeKey={activeSource}
          onChange={(key) => setActiveSource(key as DocumentationSource)}
          items={[
            {
              key: "extension",
              label: `${sourceMeta.extension.label} (${docsBySource.extension.length})`,
              children: renderDocPane("extension")
            },
            {
              key: "server",
              label: `${sourceMeta.server.label} (${docsBySource.server.length})`,
              children: renderDocPane("server")
            }
          ]}
        />
      </div>
    </PageShell>
  )
}

export default DocumentationPage
