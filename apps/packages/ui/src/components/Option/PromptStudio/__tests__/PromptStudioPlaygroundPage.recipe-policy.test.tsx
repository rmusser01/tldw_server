import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, expect, it, vi } from "vitest"

import { PromptStudioPlaygroundPage } from "../PromptStudioPlaygroundPage"

const api = vi.hoisted(() => vi.fn())
vi.mock("@/services/api-send", () => ({ apiSend: api }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: "connected_ok",
    hasCompletedFirstRun: true
  })
}))
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => ({ defaultProjectId: 42, pageSize: 10 })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback ?? _key
  })
}))
beforeEach(() => {
  api.mockReset()
})

it.each([1, 2])(
  "partial playground update preserves fetched v%s identity at the client policy boundary",
  async (version) => {
    const prompt = {
      id: 101,
      project_id: 42,
      name: "Fetched prompt",
      version_number: 1,
      prompt_format: "structured",
      prompt_schema_version: version,
      prompt_definition:
        version === 2
          ? CLEAR_TASK_RECIPE.definition
          : {
              schema_version: 1,
              format: "structured",
              blocks: [],
              variables: [],
              assembly_config: {}
            }
    }
    const writes: unknown[] = []
    api.mockImplementation(async (request) => {
      const path = request.path.split("?")[0]
      if (request.method === "PUT") {
        writes.push(request.body)
        return { ok: true, data: { success: true, data: prompt } }
      }
      const data = path.endsWith("/projects")
        ? [{ id: 42, name: "Workspace" }]
        : path.endsWith("/prompts/list/42")
          ? [prompt]
          : path.endsWith("/prompts/get/101")
            ? prompt
            : []
      return { ok: true, data: { success: true, data } }
    })
    const user = userEvent.setup()
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    render(
      <MemoryRouter>
        <QueryClientProvider client={client}>
          <PromptStudioPlaygroundPage />
        </QueryClientProvider>
      </MemoryRouter>
    )
    await user.type(
      await screen.findByRole("textbox", { name: "Change description" }),
      "Rename"
    )
    await user.click(screen.getByRole("button", { name: "Save new version" }))
    if (version === 1) {
      await waitFor(() => expect(writes).toHaveLength(1))
      expect(writes[0]).toMatchObject({
        prompt_format: "structured",
        prompt_schema_version: 1,
        prompt_definition: { schema_version: 1 }
      })
    } else {
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Save new version" })
        ).not.toHaveClass("ant-btn-loading")
      )
      expect(writes).toHaveLength(0)
    }
  },
  20_000
)
