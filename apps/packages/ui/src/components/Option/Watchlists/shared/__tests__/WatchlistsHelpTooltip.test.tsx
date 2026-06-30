// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WatchlistsHelpTooltip } from "../WatchlistsHelpTooltip"
import { WATCHLISTS_HELP_DOCS, type WatchlistsHelpTopic } from "../help-docs"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue === "string" && options?.topic) {
        return defaultValue.replace("{{topic}}", String(options.topic))
      }
      if (typeof defaultValue === "string") return defaultValue
      return key
    }
  })
}))

vi.mock("antd", () => ({
  Button: ({ children, icon, ...rest }: any) => (
    <button type="button" {...rest}>
      {icon}
      {children}
    </button>
  ),
  Tooltip: ({ title, children }: any) => (
    <div>
      {children}
      <div data-testid="tooltip-content">{title}</div>
    </div>
  )
}))

describe("WatchlistsHelpTooltip", () => {
  const topics: Array<{
    topic: WatchlistsHelpTopic
    label: string
    title: string
    description: string
  }> = [
    {
      topic: "opml",
      label: "OPML feed import",
      title: "Import many feeds quickly",
      description:
        "Use OPML when moving feed lists from another reader so setup takes minutes instead of manual entry."
    },
    {
      topic: "cron",
      label: "cron scheduling",
      title: "Set a reliable monitor schedule",
      description:
        "Start with presets for daily or weekday runs. Use cron only when you need exact custom timing."
    },
    {
      topic: "ttl",
      label: "retention window",
      title: "Control how long outputs stay available",
      description:
        "Set retention to keep briefings only as long as your review workflow needs before automatic cleanup."
    },
    {
      topic: "jinja2",
      label: "Jinja2 templates",
      title: "Customize briefing format",
      description:
        "Start from a preset report template, then edit sections to shape briefing tone, structure, and audio script text."
    },
    {
      topic: "claimClusters",
      label: "claim tracking",
      title: "Track repeating claims across sources",
      description:
        "Subscribe to claim clusters to follow how the same claim evolves across feeds without manual tagging."
    }
  ]

  it("renders help tooltip content for all configured topics", () => {
    render(
      <div>
        {topics.map(({ topic }) => (
          <WatchlistsHelpTooltip key={topic} topic={topic} />
        ))}
      </div>
    )

    for (const { topic, label, title, description } of topics) {
      expect(
        screen.getByRole("button", { name: `Open help for ${label}` })
      ).toHaveAttribute("data-testid", `watchlists-help-${topic}`)
      expect(screen.getByText(title)).toBeInTheDocument()
      expect(screen.getByText(description)).toBeInTheDocument()
    }

    const learnMoreLinks = screen.getAllByRole("link", { name: "Learn more" })
    expect(learnMoreLinks).toHaveLength(topics.length)
    topics.forEach(({ topic }, index) => {
      expect(learnMoreLinks[index]).toHaveAttribute("href", WATCHLISTS_HELP_DOCS[topic])
    })
  })

  it("uses keyboard-focusable, screen-reader discoverable triggers", () => {
    render(<WatchlistsHelpTooltip topic="cron" />)
    const trigger = screen.getByRole("button", {
      name: "Open help for cron scheduling"
    })
    trigger.focus()
    expect(trigger).toHaveFocus()
    expect(trigger).toHaveAttribute("aria-label", "Open help for cron scheduling")
    expect(screen.getByRole("link", { name: "Learn more" })).toHaveAttribute(
      "href",
      "https://crontab.guru/"
    )
  })
})
