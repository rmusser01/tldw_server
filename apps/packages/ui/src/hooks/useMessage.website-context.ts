import { extractCompletedIngestJobMediaId } from "@/services/tldw/ingest-job-results"
import { formatDocs } from "@/utils/format-docs"

type WebsiteRagClient = {
  initialize: () => Promise<unknown>
  addMedia: (url: string) => Promise<unknown>
  ragSearch: (
    query: string,
    options: Record<string, unknown>
  ) => Promise<unknown>
}

type WebsiteContextSource = {
  name: unknown
  type: unknown
  mode: string
  url: string
  pageContent: string
  metadata: Record<string, unknown>
}

type WebsiteRagDocument = {
  content?: unknown
  text?: unknown
  chunk?: unknown
  metadata?: unknown
}

type ResolveWebsiteChatContextInput = {
  client: WebsiteRagClient
  embedURL: string
  embedType: string
  embedHTML: string
  embedPDF: Array<{ content: string; page: number }>
  maxWebsiteContext: number
  query: string
}

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

const asString = (value: unknown): string =>
  typeof value === "string" ? value : ""

export const resolveWebsiteChatContext = async ({
  client,
  embedURL,
  embedType,
  embedHTML,
  embedPDF,
  maxWebsiteContext,
  query,
}: ResolveWebsiteChatContextInput): Promise<{
  context: string
  source: WebsiteContextSource[]
}> => {
  try {
    await client.initialize()
    if (!embedURL) throw new Error("Website URL is unavailable")
    const ingestResponse = await client.addMedia(embedURL)
    const mediaId = Number(extractCompletedIngestJobMediaId(ingestResponse))
    if (!Number.isSafeInteger(mediaId) || mediaId <= 0) {
      throw new Error("Website ingest did not return a persisted media ID")
    }

    const ragResponse = asRecord(
      await client.ragSearch(query, {
        top_k: 4,
        sources: ["media_db"],
        include_media_ids: [mediaId],
      })
    )
    const candidateDocs =
      ragResponse.results ?? ragResponse.documents ?? ragResponse.docs
    const docs = Array.isArray(candidateDocs)
      ? (candidateDocs as WebsiteRagDocument[])
      : []
    const normalizedDocs = docs.map((doc) => {
      const metadata = asRecord(doc.metadata)
      return {
        pageContent:
          asString(doc.content) || asString(doc.text) || asString(doc.chunk),
        metadata,
      }
    })
    const context = formatDocs(normalizedDocs)
    if (context) {
      return {
        context,
        source: normalizedDocs.map((doc) => ({
          name: doc.metadata.source || doc.metadata.title || "untitled",
          type: doc.metadata.type || "unknown",
          mode: "chat",
          url: asString(doc.metadata.url),
          pageContent: doc.pageContent,
          metadata: doc.metadata,
        })),
      }
    }
  } catch (error) {
    console.error(
      "tldw ragSearch failed, falling back to inline context",
      error
    )
  }

  const context =
    embedType === "html"
      ? embedHTML.slice(0, maxWebsiteContext)
      : embedPDF
          .map((pdf) => pdf.content)
          .join(" ")
          .slice(0, maxWebsiteContext)
  return {
    context,
    source: [
      {
        name: embedURL,
        type: embedType,
        mode: "chat",
        url: embedURL,
        pageContent: context,
        metadata: {
          source: embedURL,
          url: embedURL,
        },
      },
    ],
  }
}
