import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { fetchChatModels, getEmbeddingModels } from "@/services/tldw-server"
import { QuickIngestAdvancedSettings } from "../QuickIngestAdvancedSettings"

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(),
  getEmbeddingModels: vi.fn()
}))

const i18n = createInstance()
void i18n.init({ lng: "en", resources: {}, initImmediate: false })

function renderAdvanced(values?: Record<string, unknown>) {
  const onChange = vi.fn()
  render(
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={new QueryClient()}>
        <QuickIngestAdvancedSettings values={values} onChange={onChange} />
      </QueryClientProvider>
    </I18nextProvider>
  )
  return onChange
}

function expandAdvanced() {
  fireEvent.click(screen.getByRole("button", { name: "Advanced options" }))
}

describe("QuickIngestAdvancedSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(fetchChatModels).mockResolvedValue([])
    vi.mocked(getEmbeddingModels).mockResolvedValue([])
  })

  it("loads model catalogs on expansion and retains a saved model outside the catalog", async () => {
    vi.mocked(getEmbeddingModels).mockResolvedValue([
      {
        name: "Listed model",
        model: "listed",
        provider: "local",
        nickname: "Listed model",
        avatar: undefined
      }
    ])
    renderAdvanced({ embedding_model: "previous-install/model" })
    expect(fetchChatModels).not.toHaveBeenCalled()
    expect(getEmbeddingModels).not.toHaveBeenCalled()

    expandAdvanced()
    fireEvent.change(screen.getByRole("searchbox", { name: "Search advanced fields" }), {
      target: { value: "embedding_model" }
    })
    expect(await screen.findByRole("combobox", { name: "Embedding Model" })).toBeInTheDocument()
    expect(screen.getByText("previous-install/model")).toBeInTheDocument()
    expect(fetchChatModels).toHaveBeenCalledOnce()
    expect(getEmbeddingModels).toHaveBeenCalledOnce()
  })

  it("filters by schema key and description and keeps a field's saved value through an empty search result", () => {
    const onChange = renderAdvanced({ chunk_size: 0 })
    expandAdvanced()
    const search = screen.getByRole("searchbox", { name: "Search advanced fields" })
    fireEvent.change(search, { target: { value: "  CHUNK_SIZE  " } })
    expect(screen.getByRole("spinbutton", { name: "Chunk Size" })).toHaveValue("0")
    expect(screen.queryByRole("spinbutton", { name: "Max Pages" })).toBeNull()
    fireEvent.change(search, { target: { value: "no-such-ingest-field" } })
    expect(screen.getByRole("status")).toHaveTextContent("No matching advanced fields.")
    fireEvent.change(search, { target: { value: "Target chunk size" } })
    expect(screen.getByRole("spinbutton", { name: "Chunk Size" })).toHaveValue("0")
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.change(screen.getByRole("spinbutton", { name: "Chunk Size" }), {
      target: { value: "64" }
    })
    expect(onChange).toHaveBeenCalledWith("chunk_size", 64)
  })

  it("unsets explicit false through a button named for its field and leaves common controls with the parent", () => {
    const onChange = renderAdvanced({ generate_embeddings: false })
    expandAdvanced()
    expect(screen.queryByLabelText("Api Name")).toBeNull()
    expect(screen.queryByLabelText("Perform Analysis")).toBeNull()
    expect(screen.getByLabelText("Unset Accept Archives")).toBeDisabled()
    expect(screen.getByLabelText("Generate Embeddings")).not.toBeChecked()
    fireEvent.click(screen.getByLabelText("Unset Generate Embeddings"))
    expect(onChange).toHaveBeenCalledWith("generate_embeddings", undefined)
  })
})
