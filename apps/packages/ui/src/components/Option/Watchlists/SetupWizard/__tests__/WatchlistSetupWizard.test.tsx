// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  WatchlistSetupWizard,
  type WatchlistSetupCompleteResult
} from "../WatchlistSetupWizard"
import type {
  WatchlistContainer,
  WatchlistJob,
  WatchlistJobCreate,
  WatchlistSourceCreate
} from "@/types/watchlists"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("antd", () => {
  const Button = ({ children, icon, htmlType, type: _type, onClick, disabled, ...rest }: any) => (
    <button
      type={htmlType || "button"}
      onClick={() => onClick?.({ preventDefault: vi.fn(), stopPropagation: vi.fn() })}
      disabled={Boolean(disabled)}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
  const Input = ({ value, onChange, ...rest }: any) => (
    <input value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  Input.TextArea = ({ value, onChange, ...rest }: any) => (
    <textarea value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  const Modal = ({ open, title, children, footer, onCancel }: any) =>
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : "Create Watchlist"}>
        <h2>{title}</h2>
        {children}
        <div>{footer}</div>
        {onCancel ? <button type="button" onClick={() => onCancel()}>Cancel</button> : null}
      </div>
    ) : null
  const Alert = ({ title, message, description }: any) => (
    <div role="note">
      <strong>{title ?? message}</strong>
      <span>{description}</span>
    </div>
  )
  const Tag = ({ children }: any) => <span>{children}</span>
  const Switch = ({ checked, onChange, ...rest }: any) => (
    <button
      type="button"
      role="switch"
      aria-checked={Boolean(checked)}
      onClick={() => onChange?.(!checked)}
      {...rest}
    />
  )
  return { Alert, Button, Input, Modal, Switch, Tag }
})

const makeWatchlist = (overrides: Partial<WatchlistContainer> = {}): WatchlistContainer => ({
  id: 501,
  name: "Healthcare ransomware",
  description: null,
  objective: null,
  domain: "cti_osint",
  status: "active",
  priority: "high",
  tags: [],
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z",
  ...overrides
})

const makeJob = (overrides: Partial<WatchlistJob> = {}): WatchlistJob => ({
  id: 77,
  name: "Healthcare ransomware monitor",
  watchlist_id: 501,
  scope: { sources: [301, 302] },
  active: true,
  created_at: "2026-05-15T00:00:00Z",
  updated_at: "2026-05-15T00:00:00Z",
  ...overrides
})

const renderWizard = (
  overrides: Partial<React.ComponentProps<typeof WatchlistSetupWizard>> = {}
) => {
  const callbacks = {
    onCancel: vi.fn(),
    onCreateWatchlist: vi.fn().mockResolvedValue(makeWatchlist()),
    onCreateSources: vi.fn().mockResolvedValue([301, 302]),
    onCreateJob: vi.fn().mockResolvedValue(makeJob()),
    onComplete: vi.fn<(result: WatchlistSetupCompleteResult) => void>()
  }
  const props = { ...callbacks, ...overrides }

  render(
    <WatchlistSetupWizard
      open
      onCancel={props.onCancel}
      onCreateWatchlist={props.onCreateWatchlist}
      onCreateSources={props.onCreateSources}
      onCreateJob={props.onCreateJob}
      onComplete={props.onComplete}
    />
  )

  return props
}

const clickNext = () => {
  fireEvent.click(screen.getByRole("button", { name: "Next" }))
}

const fillDetails = (name = "Healthcare ransomware") => {
  clickNext()
  fireEvent.change(screen.getByLabelText("Watchlist name"), { target: { value: name } })
  fireEvent.change(screen.getByLabelText("Objective"), {
    target: { value: "Find ransomware reports affecting hospitals" }
  })
  fireEvent.change(screen.getByLabelText("Tracked scope"), {
    target: { value: "hospitals, Germany" }
  })
  clickNext()
}

afterEach(() => {
  cleanup()
})

describe("WatchlistSetupWizard", () => {
  it("renders domain presets, start modes, setup fields, and review step", () => {
    renderWizard()

    expect(screen.getByRole("dialog", { name: "Create Watchlist" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "CTI / OSINT" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "News" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "General" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Blank" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start from sources" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start from topic" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start from report goal" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "CTI / OSINT" }))
    clickNext()

    expect(
      screen.getByPlaceholderText("Track vulnerabilities, malware, actors, advisories, and source changes.")
    ).toBeInTheDocument()
    expect(
      screen.getByPlaceholderText("CVEs, ransomware families, sectors, regions, vendors")
    ).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Watchlist name"), {
      target: { value: "Healthcare ransomware" }
    })
    clickNext()
    clickNext()

    expect(screen.getByText("Review Watchlist setup")).toBeInTheDocument()
    expect(screen.getByText("cti")).toBeInTheDocument()
    expect(screen.getByText("osint")).toBeInTheDocument()
  })

  it("creates a topic-only Watchlist and completes into the sources destination", async () => {
    const callbacks = renderWizard({
      onCreateWatchlist: vi.fn().mockResolvedValue(makeWatchlist({ id: 601, domain: "news" }))
    })

    fireEvent.click(screen.getByRole("button", { name: "News" }))
    fireEvent.click(screen.getByRole("button", { name: "Start from topic" }))
    fillDetails("Election integrity")
    clickNext()
    fireEvent.click(screen.getByRole("button", { name: "Create Watchlist" }))

    await waitFor(() => {
      expect(callbacks.onCreateWatchlist).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Election integrity",
          domain: "news",
          tags: expect.arrayContaining(["news", "hospitals", "germany"])
        })
      )
    })
    expect(callbacks.onCreateSources).not.toHaveBeenCalled()
    expect(callbacks.onCreateJob).not.toHaveBeenCalled()
    expect(callbacks.onComplete).toHaveBeenCalledWith(
      expect.objectContaining({
        destination: "sources",
        watchlist: expect.objectContaining({ id: 601 })
      })
    )
  })

  it("creates a source-backed Watchlist, sources, and monitor through injected callbacks", async () => {
    const callbacks = renderWizard()

    fireEvent.click(screen.getByRole("button", { name: "Start from sources" }))
    fillDetails()
    fireEvent.change(screen.getByLabelText("Source URLs"), {
      target: { value: "https://example.com/feed.xml\nhttps://advisories.example.org/rss" }
    })
    fireEvent.change(screen.getByLabelText("Monitor name"), {
      target: { value: "Healthcare ransomware monitor" }
    })
    clickNext()
    fireEvent.click(screen.getByRole("button", { name: "Create Watchlist" }))

    await waitFor(() => {
      expect(callbacks.onCreateSources).toHaveBeenCalledWith(
        501,
        expect.arrayContaining<WatchlistSourceCreate>([
          expect.objectContaining({
            url: "https://example.com/feed.xml",
            source_type: "rss",
            active: true
          })
        ])
      )
    })
    expect(callbacks.onCreateJob).toHaveBeenCalledWith(
      501,
      expect.objectContaining<WatchlistJobCreate>({
        name: "Healthcare ransomware monitor",
        scope: { sources: [301, 302] },
        watchlist_id: 501
      })
    )
    expect(callbacks.onComplete).toHaveBeenCalledWith(
      expect.objectContaining({
        destination: "jobs",
        sourceIds: [301, 302],
        job: expect.objectContaining({ id: 77 })
      })
    )
  })

  it("creates a report-goal Watchlist without sources and routes to outputs", async () => {
    const callbacks = renderWizard({
      onCreateWatchlist: vi.fn().mockResolvedValue(makeWatchlist({ id: 701, domain: "news" }))
    })

    fireEvent.click(screen.getByRole("button", { name: "News" }))
    fireEvent.click(screen.getByRole("button", { name: "Start from report goal" }))
    fillDetails("AI policy briefing")
    fireEvent.change(screen.getByLabelText("Report goal"), {
      target: { value: "Concise briefing with source diversity" }
    })
    clickNext()
    fireEvent.click(screen.getByRole("button", { name: "Create Watchlist" }))

    await waitFor(() => {
      expect(callbacks.onCreateWatchlist).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "AI policy briefing",
          description: expect.stringContaining("Report goal: Concise briefing with source diversity")
        })
      )
    })
    expect(callbacks.onCreateSources).not.toHaveBeenCalled()
    expect(callbacks.onCreateJob).not.toHaveBeenCalled()
    expect(callbacks.onComplete).toHaveBeenCalledWith(
      expect.objectContaining({
        destination: "outputs",
        watchlist: expect.objectContaining({ id: 701 })
      })
    )
  })

  it("prevents advancing without a Watchlist name", () => {
    const callbacks = renderWizard()

    clickNext()
    fireEvent.change(screen.getByLabelText("Tracked scope"), {
      target: { value: "hospitals" }
    })
    clickNext()

    expect(screen.getByText("Add a Watchlist name.")).toBeInTheDocument()
    expect(callbacks.onCreateWatchlist).not.toHaveBeenCalled()
  })
})
