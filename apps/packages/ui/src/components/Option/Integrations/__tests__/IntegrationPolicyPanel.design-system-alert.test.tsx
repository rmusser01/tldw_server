// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { IntegrationPolicyPanel } from "../IntegrationPolicyPanel"

const noopAsync = vi.fn(async () => undefined)

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

describe("IntegrationPolicyPanel design-system alerts", () => {
  it("renders workspace policy error copy in a design-system alert", () => {
    render(
      <IntegrationPolicyPanel
        provider="slack"
        loading={false}
        errorMessage="Slack policy failed to load"
        onSave={noopAsync}
      />
    )

    expectDesignSystemAlert("Slack policy failed to load")
  })

  it("renders workspace policy unavailable copy in a design-system alert", () => {
    render(
      <IntegrationPolicyPanel
        provider="slack"
        loading={false}
        onSave={noopAsync}
      />
    )

    expectDesignSystemAlert("Slack policy is unavailable")
  })

  it("renders Telegram policy error copy in a design-system alert", () => {
    render(
      <IntegrationPolicyPanel
        provider="telegram"
        loading={false}
        errorMessage="Telegram bot failed to load"
        linkedActors={[]}
        onSave={noopAsync}
        onGeneratePairingCode={async () => ({
          ok: true,
          pairing_code: "PAIR-001",
          scope_type: "org",
          scope_id: 1,
          expires_at: "2026-05-30T12:00:00.000Z"
        })}
        onRevokeActor={noopAsync}
      />
    )

    expectDesignSystemAlert("Telegram bot failed to load")
  })

  it("renders generated Telegram pairing codes in a design-system success alert", () => {
    render(
      <IntegrationPolicyPanel
        provider="telegram"
        loading={false}
        bot={{
          ok: true,
          provider: "telegram",
          scope_type: "org",
          scope_id: 1,
          bot_username: "researchbot",
          enabled: true
        }}
        linkedActors={[]}
        pairingCode={{
          ok: true,
          pairing_code: "PAIR-001",
          scope_type: "org",
          scope_id: 1,
          expires_at: "2026-05-30T12:00:00.000Z"
        }}
        onSave={noopAsync}
        onGeneratePairingCode={async () => ({
          ok: true,
          pairing_code: "PAIR-001",
          scope_type: "org",
          scope_id: 1,
          expires_at: "2026-05-30T12:00:00.000Z"
        })}
        onRevokeActor={noopAsync}
      />
    )

    expectDesignSystemAlert("Pairing code generated")
    expectDesignSystemAlert("PAIR-001")
  })
})
