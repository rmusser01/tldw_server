import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { SourceFormModal } from "../SourceFormModal"
import { setViewport } from "../../__tests__/test-utils/viewport"

const formApi = {
  setFieldsValue: vi.fn(),
  resetFields: vi.fn(),
  validateFields: vi.fn()
}

const mocks = vi.hoisted(() => ({
  testWatchlistSource: vi.fn(),
  testWatchlistSourceDraft: vi.fn(),
  messageInfo: vi.fn(),
  messageSuccess: vi.fn(),
  messageWarning: vi.fn(),
  messageError: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, values?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!values) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = values[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("antd", () => {
  const FormComponent = ({ children }: any) => <form>{children}</form>
  FormComponent.Item = ({ label, extra, children }: any) => (
    <div>
      {label ? <label>{label}</label> : null}
      {extra ? <div>{extra}</div> : null}
      {children}
    </div>
  )
  FormComponent.useForm = () => [formApi]

  const Modal = ({
    open,
    title,
    children,
    onCancel,
    onOk,
    afterOpenChange,
    width,
    styles,
    okText,
    cancelText,
    ...rest
  }: any) => {
    const closeRef = React.useRef<HTMLButtonElement | null>(null)
    React.useEffect(() => {
      afterOpenChange?.(open)
      if (open) {
        closeRef.current?.focus()
      }
    }, [open, afterOpenChange])

    if (!open) return null
    return (
      <div
        data-testid={rest["data-testid"]}
        data-width={String(width ?? "")}
        data-body-max-height={String(styles?.body?.maxHeight ?? "")}
      >
        <h2>{title}</h2>
        <button type="button" ref={closeRef} onClick={() => onCancel?.()}>
          Close
        </button>
        {children}
        <div data-testid="source-form-footer">
          <button type="button" onClick={() => onCancel?.()}>
            {cancelText}
          </button>
          <button type="button" onClick={() => onOk?.()}>
            {okText}
          </button>
        </div>
      </div>
    )
  }

  return {
    Form: FormComponent,
    Input: ({ placeholder }: any) => <input placeholder={placeholder} />,
    Modal,
    Select: ({ options = [] }: any) => (
      <div>
        {options.map((option: any) => (
          <span key={String(option.value)}>{String(option.label)}</span>
        ))}
      </div>
    ),
    Button: ({ children, onClick, disabled }: any) => (
      <button type="button" disabled={disabled} onClick={onClick}>
        {children}
      </button>
    ),
    Alert: ({ title, message, description, action }: any) => (
      <div>
        <span>{title ?? message}</span>
        <span>{description}</span>
        {action}
      </div>
    ),
    message: {
      info: (...args: unknown[]) => mocks.messageInfo(...args),
      success: (...args: unknown[]) => mocks.messageSuccess(...args),
      warning: (...args: unknown[]) => mocks.messageWarning(...args),
      error: (...args: unknown[]) => mocks.messageError(...args)
    }
  }
})

vi.mock("@/services/watchlists", () => ({
  testWatchlistSource: (...args: unknown[]) => mocks.testWatchlistSource(...args),
  testWatchlistSourceDraft: (...args: unknown[]) => mocks.testWatchlistSourceDraft(...args)
}))

describe("SourceFormModal test-source preflight", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setViewport(1024)
    formApi.validateFields.mockResolvedValue({
      url: "https://example.com/feed.xml",
      source_type: "rss",
      scrape_item_selector: "",
      scrape_link_selector: "",
      scrape_title_selector: "",
      scrape_summary_selector: "",
      scrape_limit: null,
      source_top_n: null,
      discover_method: "auto"
    })
  })

  it("tests a saved feed and renders summary", async () => {
    mocks.testWatchlistSource.mockResolvedValue({
      items: [],
      total: 2,
      ingestable: 2,
      filtered: 0
    })

    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        initialValues={{
          id: 123,
          name: "Saved Feed",
          url: "https://example.com/feed.xml",
          source_type: "rss",
          active: true,
          tags: [],
          created_at: "2026-02-18T00:00:00Z"
        } as any}
        existingTags={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Test Feed" }))

    await waitFor(() => {
      expect(mocks.testWatchlistSource).toHaveBeenCalledWith(123, { limit: 10 })
      expect(mocks.testWatchlistSourceDraft).not.toHaveBeenCalled()
      expect(mocks.messageSuccess).toHaveBeenCalledWith(
        "Test succeeded: found 2 preview items."
      )
      expect(screen.getByText("Test Summary")).toBeInTheDocument()
    })
    const summaryAlert = screen
      .getByText("Test Summary")
      .closest("[data-ds-component='Alert']")
    expect(summaryAlert).toHaveAttribute("data-ds-component", "Alert")
  })

  it("tests unsaved draft feeds without requiring save", async () => {
    mocks.testWatchlistSourceDraft.mockResolvedValue({
      items: [],
      total: 1,
      ingestable: 1,
      filtered: 0
    })

    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Test Feed" }))

    await waitFor(() => {
      expect(mocks.testWatchlistSourceDraft).toHaveBeenCalledWith(
        {
          url: "https://example.com/feed.xml",
          source_type: "rss",
          settings: null
        },
        { limit: 10 }
      )
      expect(mocks.testWatchlistSource).not.toHaveBeenCalled()
      expect(mocks.messageSuccess).toHaveBeenCalledWith(
        "Test succeeded: found 1 preview item."
      )
    })

    expect(
      screen.getByText("Run Test Feed to validate URL/type connectivity before saving.")
    ).toBeInTheDocument()
  })

  it("passes draft source settings to preflight", async () => {
    formApi.validateFields.mockResolvedValue({
      url: "https://example.com/news",
      source_type: "site",
      scrape_item_selector: "css:article",
      scrape_link_selector: ".//a/@href",
      scrape_title_selector: "css:h2",
      scrape_summary_selector: "css:.summary",
      scrape_limit: 10,
      source_top_n: null,
      discover_method: "auto"
    })
    mocks.testWatchlistSourceDraft.mockResolvedValue({
      items: [],
      total: 1,
      ingestable: 1,
      filtered: 0
    })

    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Test Feed" }))

    await waitFor(() => {
      expect(mocks.testWatchlistSourceDraft).toHaveBeenCalledWith(
        {
          url: "https://example.com/news",
          source_type: "site",
          settings: {
            scrape_rules: {
              item_selector: "css:article",
              link_xpath: ".//a/@href",
              title_selector: "css:h2",
              summary_selector: "css:.summary",
              limit: 10
            }
          }
        },
        { limit: 10 }
      )
    })
  })

  it("submits cloned draft initial values with create semantics", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    formApi.validateFields.mockResolvedValue({
      name: "Saved Feed copy",
      url: "https://copy.example.com/feed.xml",
      source_type: "site",
      tags: ["news", "daily"],
      scrape_item_selector: "css:article",
      scrape_link_selector: ".//a/@href",
      scrape_title_selector: "css:h2",
      scrape_summary_selector: "",
      scrape_content_selector: "",
      scrape_date_selector: "",
      scrape_guid_selector: "css:[data-guid]",
      scrape_limit: 5,
      source_top_n: 3,
      discover_method: "links",
      skip_article_fetch: true
    })

    render(
      <SourceFormModal
        open
        mode="create"
        onClose={vi.fn()}
        onSubmit={onSubmit}
        initialValues={{
          name: "Saved Feed copy",
          url: "https://example.com/feed.xml",
          source_type: "site",
          tags: ["news", "daily"],
          settings: {
            retain_unowned_rule: "preserve",
            scrape_rules: {
              item_selector: "css:article",
              link_xpath: ".//a/@href",
              guid_xpath: "css:[data-guid]",
              limit: 5,
              skip_article_fetch: true
            },
            top_n: 3,
            discover_method: "links"
          }
        }}
        existingTags={[]}
      />
    )

    expect(screen.getByRole("heading", { name: "Add Source" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create" })).toBeInTheDocument()

    await waitFor(() => {
      expect(formApi.setFieldsValue).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Saved Feed copy",
          url: "https://example.com/feed.xml",
          source_type: "site",
          tags: ["news", "daily"],
          scrape_item_selector: "css:article",
          scrape_link_selector: ".//a/@href",
          scrape_guid_selector: "css:[data-guid]",
          scrape_limit: 5,
          source_top_n: 3,
          discover_method: "links",
          skip_article_fetch: true
        })
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith({
        name: "Saved Feed copy",
        url: "https://copy.example.com/feed.xml",
        source_type: "site",
        tags: ["news", "daily"],
        settings: {
          retain_unowned_rule: "preserve",
          scrape_rules: {
            item_selector: "css:article",
            link_xpath: ".//a/@href",
            title_selector: "css:h2",
            guid_xpath: "css:[data-guid]",
            limit: 5,
            skip_article_fetch: true
          },
          top_n: 3,
          discover_method: "links"
        }
      })
    })
  })

  it("renders optional source validation diagnostics from preview", async () => {
    formApi.validateFields.mockResolvedValue({
      url: "https://example.com/news",
      source_type: "site",
      scrape_item_selector: "css:article",
      scrape_link_selector: ".//a/@href",
      scrape_title_selector: "css:h2",
      scrape_limit: 10,
      source_top_n: null,
      discover_method: "auto"
    })
    mocks.testWatchlistSourceDraft.mockResolvedValue({
      items: [],
      total: 1,
      ingestable: 1,
      filtered: 0,
      diagnostics: {
        fetch_mode: "scrape_rules",
        fetch_status: 503,
        fetch_error: "HTTP 503 from list page",
        selector_warnings: ["title selector matched 0 nodes"],
        dedupe_preview_key: "guid_xpath"
      }
    })

    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Test Feed" }))

    await waitFor(() => {
      expect(screen.getByText("Validation diagnostics")).toBeInTheDocument()
      expect(screen.getByText("Fetch mode: scrape_rules")).toBeInTheDocument()
      expect(screen.getByText("Fetch status: HTTP 503")).toBeInTheDocument()
      expect(screen.getByText("Fetch issue: HTTP 503 from list page")).toBeInTheDocument()
      expect(screen.getByText("title selector matched 0 nodes")).toBeInTheDocument()
      expect(screen.getByText("Dedupe preview key: guid_xpath")).toBeInTheDocument()
    })
    const diagnosticsAlert = screen
      .getByText("Validation diagnostics")
      .closest("[data-ds-component='Alert']")
    expect(diagnosticsAlert).toHaveAttribute("data-ds-component", "Alert")
  })

  it("shows inline remediation guidance when draft preflight fails", async () => {
    mocks.testWatchlistSourceDraft.mockRejectedValue(
      new Error("invalid_youtube_rss_url: channel feed required")
    )

    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Test Feed" }))

    await waitFor(() => {
      expect(mocks.messageError).toHaveBeenCalledWith("Could not test feed preflight.")
      expect(screen.getByText("Could not test feed preflight.")).toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
      expect(
        screen.getByText(/Use a canonical YouTube feed URL \(channel_id or playlist_id\) and retry\./)
      ).toBeInTheDocument()
    })
    const errorAlert = screen
      .getByText("Could not test feed preflight.")
      .closest("[data-ds-component='Alert']")
    expect(errorAlert).toHaveAttribute("data-ds-component", "Alert")

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(mocks.testWatchlistSourceDraft).toHaveBeenCalledTimes(2)
    })
  })

  it("restores focus to the launch control when modal closes", async () => {
    const trigger = document.createElement("button")
    trigger.type = "button"
    trigger.textContent = "Open source form"
    document.body.appendChild(trigger)
    trigger.focus()

    const { rerender } = render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Close" })).toHaveFocus()
    })

    rerender(
      <SourceFormModal
        open={false}
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })

    trigger.remove()
  })

  it("uses a full-width constrained dialog with reachable primary actions", () => {
    setViewport(420)

    render(
      <SourceFormModal
        open
        onClose={vi.fn()}
        onSubmit={vi.fn()}
        existingTags={[]}
      />
    )

    const modal = screen.getByTestId("source-form-modal")
    expect(modal).toHaveAttribute("data-width", "100vw")
    expect(modal.getAttribute("data-body-max-height")).toContain("calc(100vh")
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Test Feed" })).toBeInTheDocument()
  })
})
