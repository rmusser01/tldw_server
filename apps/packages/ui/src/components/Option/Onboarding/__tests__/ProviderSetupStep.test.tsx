// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

describe("ProviderSetupStep", () => {
  it("lets users select multiple providers and save one default", async () => {
    const saveProvider = vi.fn().mockResolvedValue({
      provider_key: "openai",
      status: "saved",
      masked_api_key: "sk-...test"
    })
    const onContinue = vi.fn()
    const { ProviderSetupStep } = await import("../steps/ProviderSetupStep")

    render(
      <ProviderSetupStep
        providers={[
          {
            provider_key: "openai",
            label: "OpenAI",
            provider_type: "hosted_api_key",
            supports_preflight: true,
            recommended_for_first_chat: true
          },
          {
            provider_key: "ollama",
            label: "Ollama",
            provider_type: "local_endpoint",
            default_base_url: "http://127.0.0.1:11434/v1",
            supports_preflight: true,
            recommended_for_first_chat: false
          }
        ]}
        onSaveProvider={saveProvider}
        onContinue={onContinue}
      />
    )

    fireEvent.click(screen.getByLabelText(/openai/i))
    fireEvent.click(screen.getByLabelText(/ollama/i))
    fireEvent.change(screen.getByLabelText(/openai api key/i), {
      target: { value: "sk-test" }
    })
    fireEvent.change(screen.getByLabelText(/default model/i), {
      target: { value: "gpt-4.1-mini" }
    })
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }))

    await waitFor(() =>
      expect(saveProvider).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_key: "openai",
          api_key: "sk-test",
          model: "gpt-4.1-mini",
          make_default: true
        })
      )
    )
    expect(screen.getByText(/saved as sk-\.\.\.test/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /continue/i }))
    expect(onContinue).toHaveBeenCalledWith({
      provider: "openai",
      model: "gpt-4.1-mini"
    })
  })
})
