import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { VisualIdentityStage } from "../VisualIdentityStage"

describe("VisualIdentityStage", () => {
  it("renders the active actor expression", () => {
    render(
      <VisualIdentityStage
        actorName="Ari"
        resolution={{
          actor_kind: "character",
          actor_id: 7,
          pack_id: 1,
          pack_version_id: 2,
          expression_key: "happy",
          requested_expression_key: "happy",
          asset_id: 9,
          storage_relpath: null,
          fallback_reason: "requested",
          is_animated: false,
          content_type: "image/png",
          asset_url: "/api/v1/visual-identities/packs/1/assets/9/content"
        }}
      />
    )

    expect(screen.getByRole("img", { name: "Ari happy" })).toHaveAttribute(
      "src",
      "/api/v1/visual-identities/packs/1/assets/9/content"
    )
    expect(screen.getByText("Happy")).toBeInTheDocument()
  })

  it("renders nothing when no visual identity binding has an asset", () => {
    const { container } = render(
      <VisualIdentityStage actorName="Ari" resolution={null} />
    )

    expect(container).toBeEmptyDOMElement()
  })
})
