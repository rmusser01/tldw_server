import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { createInstance } from "i18next";
import { I18nextProvider, initReactI18next } from "react-i18next";
import { notification } from "antd";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import common from "@/assets/locale/en/common.json";
import settings from "@/assets/locale/en/settings.json";
import ICUWithInterpolation from "@/i18n/icu-format";
import { ExecutePlayground } from "../ExecutePlayground";

const api = vi.hoisted(() => ({ execute: vi.fn() }));
vi.mock("@/services/prompt-studio", () => ({
  getPrompt: async () => ({
    ok: true,
    status: 200,
    data: {
      data: {
        id: 42,
        name: "Localized refusal",
        version_number: 2,
        user_prompt: "Source: {{source}}",
      },
    },
  }),
  getLlmProviders: async () => ({
    ok: true,
    status: 200,
    data: {
      default_provider: "ollama",
      providers: [{ name: "ollama", models: ["qwen2.5:0.5b"] }],
    },
  }),
  executePrompt: (...args: unknown[]) => api.execute(...args),
}));

class RenderBoundary extends React.Component<
  React.PropsWithChildren,
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <p>Unexpected render failure</p>
    ) : (
      this.props.children
    );
  }
}

beforeEach(() => {
  api.execute
    .mockReset()
    .mockResolvedValue({
      ok: false,
      status: 400,
      error: "Required prompt inputs are missing.",
    });
  vi.spyOn(notification, "error").mockImplementation(() => undefined);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});
afterEach(() => vi.restoreAllMocks());

async function mount() {
  const i18n = createInstance();
  await i18n
    .use(ICUWithInterpolation)
    .use(initReactI18next)
    .init({
      lng: "en",
      fallbackLng: false,
      resources: { en: { common, settings } },
      interpolation: { escapeValue: false },
    });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const mounted = render(
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={client}>
        <RenderBoundary>
          <ExecutePlayground open promptId={42} onClose={vi.fn()} />
        </RenderBoundary>
      </QueryClientProvider>
    </I18nextProvider>,
  );
  await screen.findByText("Localized refusal");
  await screen.findByText("Default: ollama", { exact: true });
  return {
    dispose: () => {
      mounted.unmount();
      client.clear();
    },
  };
}

describe("Execute refusal with shipped English ICU resources", () => {
  it.each(["server", "json"] as const)(
    "renders the %s refusal as text without crashing the page",
    async (mode) => {
      const mounted = await mount();
      try {
        if (mode === "json")
          fireEvent.change(screen.getByLabelText("Inputs (JSON)"), {
            target: { value: "{broken" },
          });
        fireEvent.click(
          screen.getByRole("button", { name: /^Execute$/ }),
        );
        const expected =
          mode === "server"
            ? "Required prompt inputs are missing."
            : "Invalid JSON format";
        await waitFor(() => {
          expect(
            screen.queryByText("Unexpected render failure"),
          ).not.toBeInTheDocument();
          const alerts = within(screen.getByRole("dialog"))
            .getAllByRole("alert")
            .filter(
              (element) =>
                element.getAttribute("data-ds-component") === "Alert",
            );
          expect(alerts).toHaveLength(1);
          expect(alerts[0]).toBeVisible();
          expect(alerts[0]).toHaveTextContent(common.error.label);
          expect(alerts[0]).toHaveTextContent(expected);
        });
        if (mode === "server") {
          expect(notification.error).toHaveBeenCalledWith({
            message: common.error.label,
            description: expected,
          });
          expect(api.execute).toHaveBeenCalledExactlyOnceWith({
            prompt_id: 42,
            inputs: {},
            provider: "ollama",
            model: "qwen2.5:0.5b",
          });
        } else expect(api.execute).not.toHaveBeenCalled();
      } finally {
        mounted.dispose();
      }
    },
  );
});
