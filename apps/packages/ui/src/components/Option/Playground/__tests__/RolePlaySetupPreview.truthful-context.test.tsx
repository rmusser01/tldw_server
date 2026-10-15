import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { RolePlaySetupPreview } from "../RolePlaySetupPreview"
import { deriveRolePlayState } from "../role-play-state"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))

describe("RolePlaySetupPreview context evidence", () => {
  it("distinguishes zero extra pins from unknown inherited character lore", () => {
    const state = deriveRolePlayState({
      identity: { kind: "character", id: "4", name: "Mira" }
    })
    render(<RolePlaySetupPreview before={state} after={state} />)
    expect(screen.getByText("Additional context")).toBeInTheDocument()
    expect(
      screen.getByText("No additional pinned or external context")
    ).toBeInTheDocument()
    expect(screen.getByText("Character lore")).toBeInTheDocument()
    expect(screen.getByText("Not checked in this preview")).toBeInTheDocument()
    expect(screen.queryByText("No context")).not.toBeInTheDocument()
  })

  it("does not carry lore confirmation when staging a different character", () => {
    const before = deriveRolePlayState({
      identity: { kind: "character", id: "4", name: "Mira" }
    })
    const after = deriveRolePlayState({
      identity: { kind: "character", id: "5", name: "Ari" }
    })
    render(<RolePlaySetupPreview before={before} after={after} />)
    expect(screen.getByText("Ari")).toBeInTheDocument()
    expect(screen.getByText("Not checked in this preview")).toBeInTheDocument()
  })
})
