import { z } from "zod"
import { bgRequest } from "./background-proxy"
import { requestScopeFields } from "./tldw/domains/service-prompts"
import type { QuickIngestOperation } from "./tldw/quick-ingest-authority"

const plain = (limit: number) =>
  z
    .string()
    .min(1)
    .max(limit)
    .refine(
      (value) =>
        Boolean(value.trim()) &&
        [...value].every(
          (character) =>
            character.charCodeAt(0) >= 32 || "\t\r\n".includes(character)
        ) &&
        !/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(
          value
        )
    )
const schema = z
  .object({
    title: plain(200),
    pages: z
      .array(z.object({ heading: plain(200), text: plain(6000) }).strict())
      .min(1)
      .max(10)
  })
  .strict()
export type DocumentDraft = z.infer<typeof schema>
export const parseDocumentDraft = (draft: unknown): DocumentDraft =>
  schema.parse(draft)

/** Model output is untrusted; admit only a complete, bounded, identity-free draft. */
export function parseGeneratedDocument(response: unknown): DocumentDraft {
  try {
    const result = z
      .object({
        choices: z
          .array(
            z.object({
              finish_reason: z.enum(["stop"]).nullable().optional(),
              message: z.object({ content: z.string() })
            })
          )
          .min(1)
      })
      .parse(response)
    const text = result.choices[0].message.content.trim()
    if (new TextEncoder().encode(text).length > 512 * 1024)
      throw new Error("oversized")
    const fenced = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i)
    return parseDocumentDraft(JSON.parse(fenced ? fenced[1] : text))
  } catch {
    throw new Error(
      "Model returned an invalid document draft. Adjust the instructions and generate again."
    )
  }
}

export function createDocumentClient(operation: QuickIngestOperation) {
  const options = (signal: AbortSignal) => {
    operation.assertCurrent()
    const abortSignal = AbortSignal.any([operation.signal, signal])
    abortSignal.throwIfAborted()
    return { ...requestScopeFields(operation.requestScope), abortSignal }
  }
  const complete = (signal: AbortSignal) => {
    operation.assertCurrent()
    signal.throwIfAborted()
  }
  const pdf = async (request: Promise<ArrayBuffer>, signal: AbortSignal) => {
    const bytes = await request
    complete(signal)
    if (
      Object.prototype.toString.call(bytes) !== "[object ArrayBuffer]" ||
      bytes.byteLength > 20 * 1024 * 1024 ||
      new TextDecoder().decode(bytes.slice(0, 5)) !== "%PDF-" ||
      !new TextDecoder().decode(bytes.slice(-1024)).trimEnd().endsWith("%%EOF")
    )
      throw new Error("The server did not return a complete PDF.")
    return new Blob([bytes], { type: "application/pdf" })
  }
  return {
    async generate(
      input: {
        instructions: string
        model: string
        provider: string
        temperature?: number
        topP?: number
        maxTokens?: number
      },
      signal: AbortSignal
    ) {
      if (!input.instructions.trim() || input.instructions.length > 4000)
        throw new Error(
          "Enter document instructions of at most 4,000 characters."
        )
      const result = await bgRequest<unknown>({
        ...options(signal),
        path: "/api/v1/chat/completions",
        method: "POST",
        timeoutMs: 180_000,
        body: {
          model: input.model,
          api_provider: input.provider,
          stream: false,
          temperature: input.temperature,
          top_p: input.topP,
          max_tokens: input.maxTokens ?? 4096,
          messages: [
            {
              role: "system",
              content:
                'Create a plain-text document. Return only one JSON object with exactly this structure: {"title":"Document title","pages":[{"heading":"Page heading","text":"Page body"}]}. Use 1-10 pages, a nonblank title and headings of at most 200 characters, and nonblank body text of at most 6000 characters per page. Keep each page short enough for one A4 page. No markup, IDs, owners, URLs, extra fields, reasoning or commentary.'
            },
            { role: "user", content: input.instructions }
          ]
        }
      })
      complete(signal)
      return parseGeneratedDocument(result)
    },
    async render(draft: DocumentDraft, signal: AbortSignal) {
      return pdf(
        bgRequest<ArrayBuffer>({
          ...options(signal),
          path: "/api/v1/media/documents/render",
          method: "POST",
          body: parseDocumentDraft(draft),
          responseType: "arrayBuffer",
          timeoutMs: 15_000
        }),
        signal
      )
    },
    async exportPdf(mediaId: number, signal: AbortSignal) {
      if (!Number.isSafeInteger(mediaId) || mediaId < 1)
        return Promise.reject(new Error("Choose a saved PDF first."))
      return pdf(
        bgRequest<ArrayBuffer>({
          ...options(signal),
          path: `/api/v1/media/${mediaId}/annotated-file`,
          method: "GET",
          responseType: "arrayBuffer",
          timeoutMs: 15_000
        }),
        signal
      )
    }
  }
}
