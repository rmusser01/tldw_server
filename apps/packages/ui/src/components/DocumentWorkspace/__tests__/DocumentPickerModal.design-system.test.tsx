import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentPickerModal } from "../DocumentPickerModal"
import { tldwClient } from "@/services/tldw"
import { setSetting } from "@/services/settings/registry"
import { useServerOnline } from "@/hooks/useServerOnline"

const navigateMock = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock
}))

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: (value: string) => value
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: vi.fn()
}))

vi.mock("@/store/quick-ingest", () => ({
  useQuickIngestStore: (selector: (state: { recentlyIngestedDocs: unknown[] }) => unknown) =>
    selector({ recentlyIngestedDocs: [] })
}))

vi.mock("@/services/settings/registry", () => ({
  setSetting: vi.fn()
}))

vi.mock("@/services/settings/ui-settings", () => ({
  LAST_MEDIA_ID_SETTING: "lastMediaId"
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    listMedia: vi.fn(),
    searchMedia: vi.fn(),
    uploadMedia: vi.fn()
  }
}))

vi.mock("antd", () => {
  const Button = ({ children, icon, loading, onClick }: any) => (
    <button type="button" disabled={loading} onClick={onClick}>
      {icon}
      {children}
    </button>
  )

  const Input = ({ allowClear: _allowClear, prefix, value, onChange, placeholder }: any) => (
    <label>
      {placeholder}
      {prefix}
      <input aria-label={placeholder} value={value} onChange={onChange} />
    </label>
  )

  const List = Object.assign(
    ({ dataSource, renderItem }: any) => (
      <ul>{dataSource.map((item: unknown) => renderItem(item))}</ul>
    ),
    {
      Item: ({ children, actions }: any) => (
        <li>
          {children}
          {actions}
        </li>
      )
    }
  )

  const Empty = Object.assign(
    ({ description }: any) => <div>{description}</div>,
    { PRESENTED_IMAGE_SIMPLE: "simple" }
  )

  return {
    Modal: ({ open, title, children }: any) =>
      open ? (
        <div role="dialog" aria-label={title}>
          <h2>{title}</h2>
          {children}
        </div>
      ) : null,
    Tabs: ({ activeKey, items }: any) => (
      <div>{items.find((item: any) => item.key === activeKey)?.children}</div>
    ),
    Input,
    Button,
    List,
    Spin: () => <div>Loading</div>,
    Empty,
    Tag: ({ children }: any) => <span>{children}</span>,
    Switch: ({ checked, onChange }: any) => (
      <input
        aria-label="Show non-document media"
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.currentTarget.checked)}
      />
    )
  }
})

vi.mock("@plasmohq/storage", async () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => {},
  ensureConfigForRequest: async () => JSON.parse(localStorage.getItem("tldwConfig") || "null"),
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: async () => ({ id: 1 }) } }))
vi.mock("@/services/tldw-server", () => ({ getWebSearchPrompt: vi.fn(), LEGACY_SERVICE_PROMPT_DEFAULTS: {}, promptForRag: vi.fn() }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
import { quickIngestAuthority } from "@/services/tldw/quick-ingest-authority"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
let releaseAuthority: (() => void) | undefined

describe("DocumentPickerModal design-system states", () => {
  const renderModal = (props?: Partial<React.ComponentProps<typeof DocumentPickerModal>>) =>
    render(
      <DocumentPickerModal
        open
        initialTab="upload"
        onClose={vi.fn()}
        onOpenDocument={vi.fn()}
        {...props}
      />
    )

  beforeEach(async () => {
    vi.clearAllMocks()
    localStorage.clear(); sessionStorage.clear()
    localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: "https://a.test", authMode: "single-user", apiKey: "synthetic" }))
    useQuickIngestSessionStore.getState().setAuthority(null)
    releaseAuthority = quickIngestAuthority.retain()
    await waitFor(() => expect(useQuickIngestSessionStore.getState().authorityKey).toBeTruthy())
    vi.mocked(useServerOnline).mockReturnValue(true)
    vi.mocked(tldwClient.listMedia).mockResolvedValue({ items: [] })
    vi.mocked(tldwClient.searchMedia).mockResolvedValue({ items: [] })
    vi.mocked(tldwClient.uploadMedia).mockResolvedValue({ result: { id: 17 } })
  })

  afterEach(() => releaseAuthority?.())

  it("does not expose a legacy recent filename or act on its colliding ID", async () => {
    localStorage.setItem("document-workspace-recent", JSON.stringify([{ id: 7, title: "Bob-private.pdf", type: "pdf" }]))
    renderModal({ initialTab: "library" })
    expect(screen.queryByText("Bob-private.pdf")).not.toBeInTheDocument()
  })

  it("does not open a delayed upload result after authority replacement", async () => {
    let complete!: (value: unknown) => void
    vi.mocked(tldwClient.uploadMedia).mockReturnValueOnce(new Promise(resolve => { complete = resolve }))
    const onOpenDocument = vi.fn()
    const { container } = renderModal({ onOpenDocument })
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [new File(["private"], "Bob.pdf", { type: "application/pdf" })] } })
    await waitFor(() => expect(tldwClient.uploadMedia).toHaveBeenCalled())
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    await act(async () => complete({ result: { id: 7 } }))
    expect(onOpenDocument).not.toHaveBeenCalled()
  })

  it("renders the offline server-required state through the design-system Alert", () => {
    vi.mocked(useServerOnline).mockReturnValue(false)

    const { container } = renderModal()

    const offlineMessage = screen.getByText("Connect to server to use document workspace")
    expect(offlineMessage.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })

  it("renders unsupported upload errors through the design-system Alert", async () => {
    const { container } = renderModal()
    const fileInput = container.querySelector<HTMLInputElement>('input[type="file"]')

    expect(fileInput).not.toBeNull()
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["notes"], "notes.txt", { type: "text/plain" })]
      }
    })

    const errorMessage = await screen.findByText("Only PDF and EPUB files are supported.")
    expect(errorMessage.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })

  it("preserves the upload storage warning action inside the design-system Alert", async () => {
    const onClose = vi.fn()
    vi.mocked(tldwClient.uploadMedia).mockResolvedValue({
      result: {
        id: 42,
        warnings: ["original file could not be stored"]
      }
    })

    const { container } = renderModal({ onClose })
    const fileInput = container.querySelector<HTMLInputElement>('input[type="file"]')

    expect(fileInput).not.toBeNull()
    fireEvent.change(fileInput!, {
      target: {
        files: [new File(["pdf"], "paper.pdf", { type: "application/pdf" })]
      }
    })

    const warningMessage = await screen.findByText(
      "Upload finished, but the original file could not be stored. Open in Media to view extracted text, or re-upload after fixing storage."
    )
    expect(warningMessage.closest('[data-ds-component="Alert"]')).not.toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "Open in Media" }))

    await waitFor(() => {
      expect(setSetting).not.toHaveBeenCalled()
      expect(navigateMock).toHaveBeenCalledWith("/media?id=42")
      expect(onClose).toHaveBeenCalled()
    })
  })
})
