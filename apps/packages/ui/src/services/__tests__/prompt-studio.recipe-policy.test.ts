import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import { beforeEach, expect, it, vi } from "vitest"

import { createPrompt, updatePrompt } from "../prompt-studio"

const api = vi.hoisted(() => vi.fn())
vi.mock("@/services/api-send", () => ({ apiSend: api }))
const payload = {
  project_id: 42,
  name: "Recipe",
  change_description: "Policy test",
  prompt_format: "structured" as const,
  prompt_schema_version: 2,
  prompt_definition: CLEAR_TASK_RECIPE.definition
}
const owner = "recipe-owner:sha256:" + "a".repeat(64)
beforeEach(() => api.mockReset())
it.each(["create", "update"])(
  "%s v2 refuses absent/capture ownership before apiSend",
  async (operation) => {
    for (const options of [
      undefined,
      { recipePersistence: { mode: "capture" as const } }
    ]) {
      const result =
        operation === "create"
          ? await createPrompt(payload, options)
          : await updatePrompt(101, payload, options)
      expect(result).toMatchObject({
        ok: false,
        recipePersistence: { state: "not_dispatched", actualOwnerId: null }
      })
    }
    expect(api).not.toHaveBeenCalled()
  }
)
it("preserves create idempotency overload with explicit required owner policy", async () => {
  api.mockResolvedValue({ ok: true })
  await createPrompt(payload, "normal-key", {
    recipePersistence: {
      mode: "require",
      expectedOwnerId: owner,
      localId: "exact"
    }
  })
  expect(api).toHaveBeenCalledWith(
    expect.objectContaining({
      headers: { "Idempotency-Key": "normal-key" },
      recipePersistence: {
        mode: "require",
        expectedOwnerId: owner,
        localId: "exact"
      }
    })
  )
})
it.each(["create", "update"])(
  "%s cannot bypass ownership by omitting the outer version of a v2 definition",
  async (operation) => {
    const implicitV2 = { ...payload, prompt_schema_version: undefined }
    const result =
      operation === "create"
        ? await createPrompt(implicitV2)
        : await updatePrompt(101, implicitV2)
    expect(result).toMatchObject({
      ok: false,
      recipePersistence: { state: "not_dispatched", actualOwnerId: null }
    })
    expect(api).not.toHaveBeenCalled()
  }
)
it.each(["create", "update"])(
  "%s v1 remains compatible without an owner",
  async (operation) => {
    api.mockResolvedValue({ ok: true })
    const v1 = {
      project_id: 42,
      name: "Legacy",
      change_description: "V1 policy test",
      prompt_schema_version: 1
    }
    if (operation === "create") await createPrompt(v1, "v1-key")
    else await updatePrompt(101, v1)
    expect(api).toHaveBeenCalledTimes(1)
  }
)
