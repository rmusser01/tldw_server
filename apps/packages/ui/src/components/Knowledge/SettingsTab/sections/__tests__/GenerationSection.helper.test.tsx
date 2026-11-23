// CC-10 review: QA search no longer generates answers on its own (decision
// [2]), so the "Enable generation" switch now governs chat answers from your
// sources (it feeds the chat's RAG mode). Its helper says so.
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { GenerationSection } from "../GenerationSection"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
    i18n: { language: "en" }
  })
}))

describe("GenerationSection", () => {
  it("says Enable generation is for chat answers, and QA search answers on request", () => {
    render(
      <GenerationSection
        settings={{ ...DEFAULT_RAG_SETTINGS, enable_generation: false }}
        onUpdate={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /Answer Generation/ }))

    expect(
      screen.getByText(
        "Chat answers from your sources. QA search writes one only when you click Generate answer."
      )
    ).toBeInTheDocument()
  })

  // CC-10 review (approved): Generate answer uses these fields, so turning
  // chat generation off doesn't hide them.
  it("keeps the answer fields that Generate answer uses when generation is off", () => {
    render(
      <GenerationSection
        settings={{ ...DEFAULT_RAG_SETTINGS, enable_generation: false }}
        onUpdate={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /Answer Generation/ }))

    for (const label of [
      "Strict extractive",
      "Generation model",
      "Generation prompt",
      "Max tokens",
      "Enable abstention",
      "Enable synthesis"
    ]) {
      expect(screen.getByText(label), label).toBeInTheDocument()
    }
  })
})
