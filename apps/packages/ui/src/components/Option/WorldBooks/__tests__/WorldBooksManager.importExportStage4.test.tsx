import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"

const DEFAULT_BLOB_IMPL = Blob

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  undoNotificationMock,
  confirmDangerMock,
  tldwClientMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  notificationMock: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  },
  undoNotificationMock: {
    showUndoNotification: vi.fn()
  },
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    createWorldBook: vi.fn(async () => ({ id: 1 })),
    updateWorldBook: vi.fn(async () => ({})),
    deleteWorldBook: vi.fn(async () => ({})),
    listCharacters: vi.fn(async () => []),
    listCharacterWorldBooks: vi.fn(async () => []),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    addWorldBookEntry: vi.fn(async () => ({})),
    updateWorldBookEntry: vi.fn(async () => ({})),
    deleteWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 0, failed_ids: [] })),
    exportWorldBook: vi.fn(async (id: number) => ({
      world_book: { name: id === 1 ? "Arcana" : "Bestiary" },
      entries: [{ keywords: [`k-${id}`], content: `entry-${id}` }]
    })),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({ world_book_id: 99, entries_imported: 1, merged: false })),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({}))
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => undoNotificationMock
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => confirmDangerMock
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: null,
  status: "success",
  isLoading: false,
  isFetching: false,
  isPending: false,
  error: null,
  refetch: vi.fn(),
  ...value
})

const makeUseMutationResult = (opts: any) => ({
  mutate: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    }
  },
  mutateAsync: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    }
  },
  isPending: false
})

const setupDownloadSpies = () => {
  const originalBlob = Blob
  const capturedBlobPayloads: string[] = []
  class CapturingBlob extends originalBlob {
    constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
      super(parts, options)
      const payload = Array.isArray(parts)
        ? parts.map((part) => String(part)).join("")
        : ""
      capturedBlobPayloads.push(payload)
    }
  }
  vi.stubGlobal("Blob", CapturingBlob as unknown as typeof Blob)

  if (typeof URL.createObjectURL !== "function") {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:world-books")
    })
  }
  if (typeof URL.revokeObjectURL !== "function") {
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
  }

  const createObjectUrlSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:world-books")
  const revokeObjectUrlSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
  const anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})

  const originalCreateElement = document.createElement.bind(document)
  const anchor = originalCreateElement("a")
  const createElementSpy = vi
    .spyOn(document, "createElement")
    .mockImplementation(((tagName: string) => {
      if (tagName.toLowerCase() === "a") return anchor
      return originalCreateElement(tagName)
    }) as typeof document.createElement)

  const restoreBlob = () => {
    vi.stubGlobal("Blob", originalBlob)
  }

  return {
    createObjectUrlSpy,
    revokeObjectUrlSpy,
    anchorClickSpy,
    createElementSpy,
    capturedBlobPayloads,
    restoreBlob
  }
}

const getLatestDownloadedJson = (capturedBlobPayloads: string[]) => {
  expect(capturedBlobPayloads.length).toBeGreaterThan(0)
  const payload = capturedBlobPayloads[capturedBlobPayloads.length - 1]
  return JSON.parse(payload)
}

describe("WorldBooksManager import/export stage-4 export actions and upload control", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useQueryClientMock.mockReturnValue({ invalidateQueries: vi.fn() })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({
          data: [
            { id: 1, name: "Arcana", description: "Main lore", enabled: true, entry_count: 2 },
            { id: 2, name: "Bestiary", description: "Creatures", enabled: true, entry_count: 1 }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.stubGlobal("Blob", DEFAULT_BLOB_IMPL)
  })

  it(
    "exports a single world book from row actions",
    async () => {
      const user = userEvent.setup()
      const {
        capturedBlobPayloads,
        revokeObjectUrlSpy,
        anchorClickSpy,
        createElementSpy,
        restoreBlob
      } = setupDownloadSpies()

      render(<WorldBooksManager />)
      // Open overflow menu for first row, then click Export JSON
      await user.click(screen.getAllByRole("button", { name: /More actions for/ })[0])
      await user.click(await screen.findByText("Export JSON"))

      await waitFor(() => {
        expect(tldwClientMock.exportWorldBook).toHaveBeenCalledWith(1)
      })
      expect(anchorClickSpy).toHaveBeenCalledTimes(1)
      expect(revokeObjectUrlSpy).toHaveBeenCalledTimes(1)

      const payload = getLatestDownloadedJson(capturedBlobPayloads)
      expect(payload).toEqual(
        expect.objectContaining({
          world_book: expect.objectContaining({ name: "Arcana" })
        })
      )

      createElementSpy.mockRestore()
      restoreBlob()
    },
    15000
  )

  it("exports all world books as a bundle from header actions", async () => {
    const user = userEvent.setup()
    const { capturedBlobPayloads, createElementSpy, restoreBlob } = setupDownloadSpies()

    render(<WorldBooksManager />)
    // Open Tools dropdown then click Export All
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Export All"))

    await waitFor(() => {
      expect(tldwClientMock.exportWorldBook).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.exportWorldBook).toHaveBeenNthCalledWith(1, 1)
    expect(tldwClientMock.exportWorldBook).toHaveBeenNthCalledWith(2, 2)

    const payload = getLatestDownloadedJson(capturedBlobPayloads)
    expect(payload).toEqual(
      expect.objectContaining({
        bundle_type: "tldw-world-books-export",
        export_mode: "all",
        world_books: expect.arrayContaining([
          expect.objectContaining({ id: 1, name: "Arcana" }),
          expect.objectContaining({ id: 2, name: "Bestiary" })
        ])
      })
    )

    createElementSpy.mockRestore()
    restoreBlob()
  })

  it("exports selected world books as a bundle from selection actions", async () => {
    const user = userEvent.setup()
    const { capturedBlobPayloads, createElementSpy, restoreBlob } = setupDownloadSpies()

    render(<WorldBooksManager />)

    const checkboxes = screen.getAllByRole("checkbox")
    await user.click(checkboxes[1])
    await user.click(screen.getByRole("button", { name: "Export selected" }))

    await waitFor(() => {
      expect(tldwClientMock.exportWorldBook).toHaveBeenCalledTimes(1)
    })
    expect(tldwClientMock.exportWorldBook).toHaveBeenCalledWith(1)

    const payload = getLatestDownloadedJson(capturedBlobPayloads)
    expect(payload).toEqual(
      expect.objectContaining({
        export_mode: "selected",
        world_books: [expect.objectContaining({ id: 1, name: "Arcana" })]
      })
    )

    createElementSpy.mockRestore()
    restoreBlob()
  }, 30000)

  it(
    "uses upload control and enforces json file constraints",
    async () => {
      const user = userEvent.setup({ applyAccept: false })
      render(<WorldBooksManager />)

      // Open Tools dropdown then click Import JSON
      await user.click(screen.getByRole("button", { name: "Tools" }))
      await user.click(await screen.findByText("Import JSON"))
      expect(screen.getByRole("button", { name: "Import world book JSON file" })).toBeInTheDocument()

      const modalTitles = await screen.findAllByText("Import World Book (JSON)")
      const modal = modalTitles[modalTitles.length - 1].closest(".ant-modal") as HTMLElement | null
      expect(modal).not.toBeNull()
      const input = modal?.querySelector('input[type="file"]') as HTMLInputElement | null
      expect(input).not.toBeNull()
      expect(input?.accept).toContain(".json")

      const badFile = new File(["{bad-json"], "bad.json", { type: "application/json" })
      ;(badFile as any).text = async () => "{bad-json"
      await user.upload(input as HTMLInputElement, badFile)

      await waitFor(() => {
        expect(screen.getByText("Selected: bad.json")).toBeInTheDocument()
      })
      expect(screen.getByRole("button", { name: "Import" })).toBeDisabled()
    },
    15000
  )
})
