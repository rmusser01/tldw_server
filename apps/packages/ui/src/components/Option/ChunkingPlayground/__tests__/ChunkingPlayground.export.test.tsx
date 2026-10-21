import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { ChunkingResponse } from "@/services/chunking"
import { ChunkingPlayground } from "../index"

const mocks = vi.hoisted(() => ({ request: vi.fn(), isDesktop: true }))
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(),
  bgRequest: mocks.request
}))
vi.mock("@/hooks/useMediaQuery", () => ({ useDesktop: () => mocks.isDesktop }))
vi.mock("@/hooks/useDarkmode", () => ({ useDarkMode: () => ({ mode: "light" }) }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  })
}))

// Hand-checked65word fixture: each wNN is3chars;25/25/15 words span0..99,100..199,200..259.
const FIRST = "w01 w02 w03 w04 w05 w06 w07 w08 w09 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25"
const SECOND = "w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50"
const THIRD = "w51 w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65"
const INPUT = `${FIRST} ${SECOND} ${THIRD}`
const RESPONSE: ChunkingResponse = {
  original_file_name: "input_text.txt",
  chunks: [
    { text: FIRST, metadata: { chunk_index: 0, total_chunks: 3, word_count: 25, char_count: 99, start_char: 0, end_char: 99, method: "words", language: "en" } },
    { text: SECOND, metadata: { chunk_index: 1, total_chunks: 3, word_count: 25, char_count: 99, start_char: 100, end_char: 199, method: "words", language: "en" } },
    { text: THIRD, metadata: { chunk_index: 2, total_chunks: 3, word_count: 15, char_count: 59, start_char: 200, end_char: 259, method: "words", language: "en", extra_field: "preserve opaque metadata" } }
  ],
  applied_options: {
    method: "words", max_size: 25, overlap: 0, language: "en",
    tokenizer_name_or_path: "gpt2", adaptive: false, multi_level: false,
    semantic_similarity_threshold: 0.7, semantic_overlap_sentences: 2,
    json_chunkable_data_key: "data", enable_frontmatter_parsing: true,
    frontmatter_sentinel_key: "__tldw_frontmatter__", summarization_detail: 0.5,
    proposition_engine: "heuristic", proposition_aggressiveness: 1,
    proposition_min_proposition_length: 15, proposition_prompt_profile: "generic"
  }
}
const CAPABILITIES = {
  methods: ["words"], default_options: { method: "words", max_size: 400, overlap: 200, language: "en" },
  llm_required_methods: [], hierarchical_support: false,
  options_schema: { type: "object", properties: {
    method: { type: "string", default: "words" },
    max_size: { type: "integer", default: 400, minimum: 1 },
    overlap: { type: "integer", default: 200, minimum: 0 },
    language: { type: "string", default: "en" }
  } }
}

const originalCreate = Object.getOwnPropertyDescriptor(URL, "createObjectURL")
const originalRevoke = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL")
const clients: QueryClient[] = []
const downloads: Array<{ blob: Blob, filename: string }> = []
let currentBlob: Blob | undefined
const createUrl = vi.fn((blob: Blob) => {
  currentBlob = blob
  return "blob:completed-chunks"
})
const chunkResult = vi.fn<() => Promise<ChunkingResponse>>()

const restoreProperty = (name: "createObjectURL" | "revokeObjectURL", descriptor?: PropertyDescriptor) => {
  if (descriptor) Object.defineProperty(URL, name, descriptor)
  else Reflect.deleteProperty(URL, name)
}
const mount = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}><ChunkingPlayground /></QueryClientProvider>)
}
const numberInput = (label: string) => {
  const row = screen.getByText(label).closest(".ant-form-item")
  if (!row) throw new Error(`Missing settings row: ${label}`)
  return within(row as HTMLElement).getByRole("spinbutton")
}
const complete = async () => {
  fireEvent.change(screen.getByPlaceholderText("Paste text here..."), { target: { value: INPUT } })
  await screen.findByText("Max Size")
  fireEvent.change(numberInput("Max Size"), { target: { value: "25" } })
  fireEvent.change(numberInput("Overlap"), { target: { value: "0" } })
  fireEvent.click(screen.getByRole("button", { name: /Chunk Text$/ }))
  await screen.findByText(FIRST)
}
const readBlob = (blob: Blob): Promise<string> => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => resolve(String(reader.result))
  reader.onerror = () => reject(reader.error)
  reader.readAsText(blob)
})
const exported = async () => {
  fireEvent.click(screen.getByRole("button", { name: /Export JSON$/ }))
  await waitFor(() => expect(downloads).toHaveLength(1))
  return JSON.parse(await readBlob(downloads[0].blob)) as ChunkingResponse
}

beforeEach(() => {
  mocks.isDesktop = true
  mocks.request.mockReset().mockImplementation(async ({ path }: { path: string }) => {
    if (path === "/api/v1/chunking/capabilities") return CAPABILITIES
    if (path === "/api/v1/chunking/templates?include_builtin=true&include_custom=true") return { templates: [], total: 0 }
    if (path === "/api/v1/chunking/chunk_text") return chunkResult()
    throw new Error(`Unexpected external request: ${path}`)
  })
  chunkResult.mockReset().mockResolvedValue(structuredClone(RESPONSE))
  downloads.length = 0
  currentBlob = undefined
  createUrl.mockReset().mockImplementation((blob: Blob) => { currentBlob = blob; return "blob:completed-chunks" })
  Object.defineProperty(URL, "createObjectURL", { configurable: true, writable: true, value: createUrl })
  Object.defineProperty(URL, "revokeObjectURL", { configurable: true, writable: true, value: vi.fn() })
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    if (!currentBlob) throw new Error("Download has no Blob")
    downloads.push({ blob: currentBlob, filename: this.download })
  })
  vi.spyOn(message, "error").mockImplementation(() => undefined as unknown as ReturnType<typeof message.error>)
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  vi.restoreAllMocks()
  restoreProperty("createObjectURL", originalCreate)
  restoreProperty("revokeObjectURL", originalRevoke)
})

describe("ChunkingPlayground completed output export", () => {
  it.each([true, false])("exports exact completed65word response on desktop=%s", async (desktop) => {
    mocks.isDesktop = desktop
    mount()
    await complete()
    const result = await exported()
    expect(result).toEqual(RESPONSE)
    expect(result.chunks.map((chunk) => chunk.text.split(/\s+/).length)).toEqual([25, 25, 15])
    expect(result.chunks.map((chunk) => [chunk.metadata.start_char, chunk.metadata.end_char])).toEqual([[0, 99], [100, 199], [200, 259]])
    expect(result.chunks.map((chunk) => chunk.text).join(" ")).toBe(INPUT)
    expect(downloads[0].blob.type).toBe("application/json")
    expect(downloads[0].filename).toMatch(/^[a-z0-9-]+\.json$/)
  }, 30000)

  it("offers no output export before a completed result", () => {
    mount()
    expect(screen.queryByRole("button", { name: /Export JSON$/ })).not.toBeInTheDocument()
    expect(downloads).toHaveLength(0)
  }, 30000)

  it("removes output export when a later result is empty", async () => {
    mount()
    await complete()
    expect(screen.getByRole("button", { name: /Export JSON$/ })).toBeEnabled()
    chunkResult.mockResolvedValueOnce({ ...RESPONSE, chunks: [] })
    fireEvent.click(screen.getByRole("button", { name: /Chunk Text$/ }))
    await screen.findByText("Enter text and click 'Chunk Text' to see results")
    expect(screen.queryByRole("button", { name: /Export JSON$/ })).not.toBeInTheDocument()
    expect(downloads).toHaveLength(0)
  }, 30000)

  it("removes output export after a failed rerun without losing input", async () => {
    mount()
    await complete()
    expect(screen.getByRole("button", { name: /Export JSON$/ })).toBeEnabled()
    chunkResult.mockRejectedValueOnce(new Error("Chunk request failed; retry when available"))
    fireEvent.click(screen.getByRole("button", { name: /Chunk Text$/ }))
    await screen.findByRole("alert")
    expect(screen.getByRole("alert")).toHaveTextContent("Chunk request failed")
    expect(screen.getByPlaceholderText("Paste text here...")).toHaveValue(INPUT)
    expect(screen.queryByRole("button", { name: /Export JSON$/ })).not.toBeInTheDocument()
  }, 30000)

  it("disables retained output export while a new result is pending", async () => {
    mount()
    await complete()
    let resolve!: (value: ChunkingResponse) => void
    const pending = new Promise<ChunkingResponse>((done) => { resolve = done })
    chunkResult.mockReturnValueOnce(pending)
    fireEvent.click(screen.getByRole("button", { name: /Chunk Text$/ }))
    const button = screen.getByRole("button", { name: /Export JSON$/ })
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(downloads).toHaveLength(0)
    await act(async () => { resolve(structuredClone(RESPONSE)); await pending })
    await waitFor(() => expect(button).toBeEnabled())
  }, 30000)

  it("exports the completed server response after input and settings are edited", async () => {
    mount()
    await complete()
    fireEvent.change(screen.getByPlaceholderText("Paste text here..."), { target: { value: "different unsubmitted input" } })
    fireEvent.change(numberInput("Max Size"), { target: { value: "50" } })
    expect(numberInput("Max Size")).toHaveValue("50")
    const result = await exported()
    expect(result).toEqual(RESPONSE)
    expect(result.applied_options.max_size).toBe(25)
    expect(result.chunks.map((chunk) => chunk.text).join(" ")).toBe(INPUT)
  }, 30000)

  it("keeps a useful sanitized download failure visible and allows export recovery", async () => {
    mount()
    await complete()
    createUrl.mockImplementationOnce(() => { throw new Error("private-key=do-not-display") })
    fireEvent.click(screen.getByRole("button", { name: /Export JSON$/ }))
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent(/export|download/i)
    expect(alert).toHaveTextContent(/try again|retry/i)
    expect(alert).not.toHaveTextContent("private-key")
    expect(screen.getByText(FIRST)).toBeInTheDocument()
    expect(downloads).toHaveLength(0)
    expect(await exported()).toEqual(RESPONSE)
  }, 30000)
})
