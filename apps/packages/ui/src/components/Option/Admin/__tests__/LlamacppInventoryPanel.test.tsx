import React from "react"
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, within, waitFor } from "@testing-library/react"
import {
  INVENTORY_ROW_ESTIMATE,
  LlamacppInventoryPanel
} from "../LlamacppInventoryPanel"
import type { LlamacppInventoryItem } from "@/types/llamacpp-admin"

/**
 * jsdom performs no layout, so the virtualizer would always see a 0px-tall
 * scroll container and render no rows. Report the constrained scroll
 * container (`h-96` = 384px) with a real height so useVirtualizer computes
 * genuine windows against the estimate-based measurements.
 */
const SCROLL_CONTAINER_HEIGHT = 384
let originalOffsetHeight: PropertyDescriptor | undefined

beforeAll(() => {
  originalOffsetHeight = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "offsetHeight"
  )
  Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
    configurable: true,
    get(this: HTMLElement) {
      return this.classList.contains("h-96")
        ? SCROLL_CONTAINER_HEIGHT
        : (originalOffsetHeight?.get?.call(this) ?? 0)
    }
  })
})

afterAll(() => {
  if (originalOffsetHeight) {
    Object.defineProperty(
      HTMLElement.prototype,
      "offsetHeight",
      originalOffsetHeight
    )
  }
})

const buildModels = (count: number): LlamacppInventoryItem[] =>
  Array.from({ length: count }, (_, index) => ({
    model_id: `gguf:model-${index}`,
    display_name: `Inventory Model ${index}`,
    basename: `model-${index}.gguf`,
    source: "models_dir",
    path: `/models/model-${index}.gguf`,
    size_bytes: 4_000_000_000,
    modified_at: null,
    metadata: {},
    warnings: []
  }))

const renderInventory = (models: LlamacppInventoryItem[]) =>
  render(
    <LlamacppInventoryPanel
      inventory={{ models, warnings: [], scan_limited: false }}
      selectedModelId={undefined}
      activeModel={null}
      loading={false}
      registering={false}
      onSelectModel={vi.fn()}
      onRegisterPath={vi.fn()}
      onReload={vi.fn()}
    />
  )

describe("LlamacppInventoryPanel", () => {
  it("renders inventory metadata and selects by model_id", () => {
    const onSelect = vi.fn()

    render(
      <LlamacppInventoryPanel
        inventory={{
          models: [
            {
              model_id: "gguf:stable-id",
              display_name: "Mistral 7B Instruct Q4_K_M",
              basename: "mistral-7b-instruct-q4_k_m.gguf",
              source: "registered_path",
              path: "/models/mistral-7b-instruct-q4_k_m.gguf",
              size_bytes: 4_000_000_000,
              modified_at: null,
              metadata: {
                quantization: "Q4_K_M",
                parameter_hint: "7B",
                context_hint: null
              },
              warnings: ["Outside models directory but allowed."]
            }
          ],
          warnings: [],
          scan_limited: false
        }}
        selectedModelId={undefined}
        activeModel="different.gguf"
        loading={false}
        registering={false}
        onSelectModel={onSelect}
        onRegisterPath={vi.fn()}
        onReload={vi.fn()}
      />
    )

    expect(screen.getByText("Mistral 7B Instruct Q4_K_M")).toBeTruthy()
    expect(screen.getByText("registered_path")).toBeTruthy()
    expect(screen.getByText("Q4_K_M")).toBeTruthy()
    expect(screen.getByText("Outside models directory but allowed.")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Select" }))

    expect(onSelect).toHaveBeenCalledWith("gguf:stable-id")
  })

  it("keeps registered path text when registration fails", async () => {
    const onRegister = vi.fn().mockResolvedValue(false)

    render(
      <LlamacppInventoryPanel
        inventory={{
          models: [],
          warnings: [],
          scan_limited: false
        }}
        selectedModelId={undefined}
        activeModel={null}
        loading={false}
        registering={false}
        onSelectModel={vi.fn()}
        onRegisterPath={onRegister}
        onReload={vi.fn()}
      />
    )

    const input = screen.getByLabelText("Register local GGUF path") as HTMLInputElement
    fireEvent.change(input, { target: { value: "/external/model.gguf" } })
    fireEvent.click(screen.getByRole("button", { name: "Register path" }))

    await waitFor(() => {
      expect(onRegister).toHaveBeenCalledWith("/external/model.gguf")
    })
    expect(input.value).toBe("/external/model.gguf")
  })

  it("test_inventory_renders_windowed_rows", async () => {
    renderInventory(buildModels(500))

    const list = screen.getByRole("list", { name: "Local GGUF models" })
    const rows = within(list).getAllByRole("listitem")

    // A bounded window is mounted instead of all 500 model rows.
    expect(rows.length).toBeLessThanOrEqual(30)
    // Rows are keyed by model_id: the initial window covers the first models.
    const mountedIds = rows.map((row) => row.getAttribute("data-model-id"))
    expect(mountedIds).toContain("gguf:model-0")
    expect(mountedIds).not.toContain("gguf:model-100")
    // The virtualized path scrolls inside a constrained container.
    const scrollContainer = list.parentElement as HTMLElement
    expect(scrollContainer.className).toContain("overflow-y-auto")

    // Jumping the scroll offset mounts the window around model 100.
    scrollContainer.scrollTop = 100 * INVENTORY_ROW_ESTIMATE
    fireEvent.scroll(scrollContainer)

    expect(await screen.findByText("Inventory Model 100")).toBeTruthy()
    await waitFor(() => {
      const windowedIds = Array.from(
        list.querySelectorAll("li[data-model-id]")
      ).map((row) => row.getAttribute("data-model-id"))
      expect(windowedIds).toContain("gguf:model-100")
      expect(windowedIds).not.toContain("gguf:model-0")
      expect(windowedIds.length).toBeLessThanOrEqual(30)
    })
  })

  it("test_small_lists_render_plain", () => {
    renderInventory(buildModels(10))

    const list = screen.getByRole("list", { name: "Local GGUF models" })
    const rows = within(list).getAllByRole("listitem")

    // Below the virtualization threshold every row renders plainly.
    expect(rows).toHaveLength(10)
    expect(screen.getByText("Inventory Model 9")).toBeTruthy()
    // No virtualizer scroll container wraps the list.
    expect((list.parentElement as HTMLElement).className).not.toContain(
      "overflow-y-auto"
    )
    // Plain rows carry no virtualizer bookkeeping attributes.
    expect(list.querySelector("li[data-index]")).toBeNull()
  })
})
