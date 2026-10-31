// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import React from "react"
import { I18nextProvider } from "react-i18next"
import { describe, expect, it, vi } from "vitest"

import english from "@/assets/locale/en/option.json"
import { getNotesGraphSuggestionCapabilities } from "@/services/note-graph-suggestions"
import NotesGraphInspector from "../NotesGraphInspector"

const transport = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: transport.request }))
vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn()
}))

const fingerprint = `sha256:${"a".repeat(64)}`

describe("unbound Notes suggestion capabilities", () => {
  it.each([{ allowed: [] }, { allowed: ["cancel"] }])(
    "parses and discloses missing decisions with supported actions %j",
    async ({ allowed }) => {
      transport.request.mockResolvedValue({
        ok: true,
        status: 200,
        headers: { ETag: `"${fingerprint}"` },
        data: {
          provider: "openai",
          model: "fixture-model",
          endpoint_origin_revision: fingerprint,
          data_boundary: "remote",
          disclosure_external: true,
          outbound_data_categories: ["selected_note_excerpts"],
          generation_available: false,
          unavailable_reason: "notes_graph_sync_not_ready",
          limits: {
            max_candidates: 30,
            max_relationships: 5,
            max_tags: 5,
            max_new_tags: 2,
            max_tag_catalog: 100,
            max_estimated_input_tokens: 24000,
            max_output_tokens: 2000,
            provider_timeout_seconds: 120,
            response_candidates: 1
          },
          allowed_actions: allowed,
          revision: fingerprint
        }
      })
      const capabilities = await getNotesGraphSuggestionCapabilities({
        noteId: "source"
      })
      expect(capabilities.allowed_actions).toEqual(allowed)
      const i18n = createInstance()
      await i18n.init({
        lng: "en",
        fallbackLng: "en",
        defaultNS: "option",
        resources: { en: { option: english } }
      })
      const suggestion = {
        id: "suggestion-1",
        run_id: "run-1",
        kind: "related_note",
        state: "pending",
        revision: 1,
        source_note_id: "source",
        target_note_id: "target",
        target_title: "Synthetic target",
        source_fingerprint: fingerprint,
        target_fingerprint: fingerprint,
        match_strength: "strong",
        rationale: "Synthetic reason",
        evidence: []
      }
      render(
        <I18nextProvider i18n={i18n}>
          <NotesGraphInspector
            graph={{
              nodes: [{ id: "note:source", type: "note", label: "Source" }],
              edges: []
            } as never}
            selectedNodeId="note:source"
            suggestionsAuthorized
            isOnline
            controller={{
              capabilities,
              suggestions: [suggestion],
              activeRun: { id: "run-1", state: "queued", cancellation_available: true },
              capabilitiesQuery: {},
              suggestionsQuery: {},
              mutations: {}
            } as never}
            onSelectNode={vi.fn()}
            onAnnounce={vi.fn()}
            onDecideSuggestion={vi.fn()}
          />
        </I18nextProvider>
      )
      fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))
      expect(screen.getByText("Suggestion decisions are unavailable for this Notes dataset.")).toBeVisible()
      expect(screen.getByRole("button", { name: /regenerate/i })).toBeDisabled()
      expect(screen.getByRole("button", { name: /^accept/i })).toBeDisabled()
      expect(screen.getByRole("button", { name: /^reject/i })).toBeDisabled()
      const cancel = screen.getByRole("button", { name: /cancel generation/i })
      if (allowed.includes("cancel")) expect(cancel).toBeEnabled()
      else expect(cancel).toBeDisabled()
    }
  )
})
