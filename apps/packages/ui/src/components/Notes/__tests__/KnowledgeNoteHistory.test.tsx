import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it, vi } from "vitest"
import { KnowledgeNoteHistory } from "../KnowledgeNoteHistory"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

it("lets keyboard users focus the named history scroll region while references remain inert", async () => {
  const user = userEvent.setup()
  render(<KnowledgeNoteHistory note={{
    knowledge_provenance_state: "active",
    knowledge_provenance: {
      origin: "knowledge_qa",
      question: "Original question",
      sources: [{ originalId: "source-1", mediaId: null, title: "Original source", type: "text", sourceType: "notes", excerpt: "Retained evidence. ".repeat(100), url: "https://example.com/inert" }],
    },
  }} />)
  await user.click(screen.getByText("Original source history"))
  const region = screen.getByRole("region", { name: "Original source history" })
  await user.tab()
  expect(region).toHaveFocus()
  expect(within(region).getByText("https://example.com/inert")).toBeInTheDocument()
  expect(within(region).queryByRole("link")).not.toBeInTheDocument()
})
