import React from "react"
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { BuddyStarterCatalogPicker } from "../BuddyStarterCatalogPicker"

const mocks = vi.hoisted(() => ({ detail: vi.fn(), fetch: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options: any) => options?.defaultValue
  })
}))
vi.mock("@/services/persona-visuals", () => ({
  getPersonaVisualStarterPack: mocks.detail
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { fetchWithAuth: mocks.fetch }
}))
const starter = {
  id: "search-lens-basic",
  title: "Search Lens",
  description: "A lens for research",
  renderer_type: "sprite_frames",
  complexity_tier: "basic",
  production_status: "art_ready",
  tags: [],
  states_offered: ["idle"],
  expected_asset_groups: [],
  animation_coverage_notes: [],
  license_label: "CC0"
} as any
beforeEach(() => {
  mocks.detail.mockReset().mockResolvedValue({
    ...starter,
    manifest: {
      manifest_version: 1,
      renderer_type: "sprite_frames",
      states: { idle: { animation_id: "idle" } },
      animations: {
        idle: {
          frames: [
            {
              asset_id: "sheet",
              region: { x: 64, y: 0, width: 64, height: 64 }
            }
          ]
        }
      }
    },
    assets: [
      { asset_key: "sheet", mime_type: "image/png", width: 128, height: 64 }
    ]
  })
  mocks.fetch
    .mockReset()
    .mockResolvedValue({ ok: true, data: new Uint8Array([1, 2, 3]).buffer })
  vi.stubGlobal(
    "URL",
    Object.assign(URL, {
      createObjectURL: vi.fn(() => "blob:starter-art"),
      revokeObjectURL: vi.fn()
    })
  )
})

it("shows authenticated source artwork before enabling copy as draft", async () => {
  const copy = vi.fn()
  render(
    <BuddyStarterCatalogPicker
      starterPacks={[starter]}
      onCopyStarterPack={copy}
    />
  )
  expect(screen.getByRole("button", { name: "Copy as draft" })).toBeDisabled()
  const preview = await screen.findByRole("img", {
    name: "Search Lens preview"
  })
  const image = preview.querySelector("image")!
  await waitFor(() => expect(image).toHaveAttribute("href", "blob:starter-art"))
  expect(preview).toHaveAttribute("viewBox", "64 0 64 64")
  expect(mocks.fetch).toHaveBeenCalledWith(
    "/api/v1/persona/visual-starter-packs/search-lens-basic/assets/sheet/content",
    expect.objectContaining({ responseType: "arrayBuffer" })
  )
  fireEvent.load(image)
  fireEvent.click(screen.getByRole("button", { name: "Copy as draft" }))
  expect(copy).toHaveBeenCalledWith("search-lens-basic")
})

it("keeps copy disabled and offers a retry when artwork cannot load", async () => {
  mocks.detail.mockRejectedValue(new Error("offline"))
  render(
    <BuddyStarterCatalogPicker
      starterPacks={[starter]}
      onCopyStarterPack={vi.fn()}
    />
  )
  expect(
    await screen.findByText("Preview unavailable. Retry to view this Buddy.")
  ).toBeVisible()
  expect(screen.getByRole("button", { name: "Copy as draft" })).toBeDisabled()
  expect(screen.getByRole("button", { name: "Retry preview" })).toBeVisible()
})
