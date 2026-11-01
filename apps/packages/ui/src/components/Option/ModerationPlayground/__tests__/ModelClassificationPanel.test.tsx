// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import React from "react";
import { createInstance } from "i18next";
import { I18nextProvider } from "react-i18next";
import englishOptions from "@/assets/locale/en/option.json";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { ConfigProvider } from "antd";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelClassificationPanel from "../ModelClassificationPanel";
import { classifyModeration } from "@/services/moderation";

const state = vi.hoisted(() => ({
  listeners: new Set<(invalidated: boolean) => void>(),
  catalog: {
    data: [
      { model: "existing-model", provider: "llama.cpp" },
      { model: "remote-model", provider: "openai" },
    ],
    isFetching: false,
    isError: false,
    refetch: vi.fn(),
  },
}));
vi.mock("@/hooks/useChatModelCatalog", () => ({
  useChatModelCatalog: () => state.catalog,
}));
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (listener: (invalidated: boolean) => void) => {
    state.listeners.add(listener);
    return () => state.listeners.delete(listener);
  },
}));
vi.mock("@/services/moderation", () => ({ classifyModeration: vi.fn() }));
const call = vi.mocked(classifyModeration);

function item() {
  return {
    id: "saved-classification",
    status: "needs_review" as const,
    phase: "input" as const,
    created_at: "2026-11-01T00:00:00Z",
    excerpt: "Unpublished access code: ROWAN-731.",
    source_type: "model_classification",
    source_id: "request",
    user_id: "41",
    safe_fields: { excerpt: true, context: true, matches: true },
    effective_policy: { advisory_only: true },
    matches: [],
    decision_history: [],
    context: {
      input_digest: "digest",
      classification: {
        provider: "llama.cpp",
        model: "existing-model",
        criterion: "Flag unpublished access codes.",
        label: "flagged" as const,
        reason: "The sample discloses an access code.",
        evidence: "ROWAN-731",
      },
    },
  };
}
function mount() {
  return render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <MemoryRouter>
        <ModelClassificationPanel
          configSnapshot={{
            serverUrl: "https://original.test",
            authMode: "single-user",
            apiKey: "panel-fixture-key",
          }}
        />
      </MemoryRouter>
    </ConfigProvider>,
  );
}
async function fill() {
  fireEvent.mouseDown(
    screen.getByRole("combobox", { name: "Classification model" }),
  );
  fireEvent.click(
    await screen.findByText("existing-model (llama.cpp)", {
      selector: ".ant-select-item-option-content",
    }),
  );
  fireEvent.change(screen.getByLabelText("Flag when"), {
    target: { value: "Flag unpublished access codes." },
  });
  fireEvent.change(screen.getByLabelText("Sample for classification"), {
    target: { value: "Unpublished access code: ROWAN-731." },
  });
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  call.mockReset();
  state.listeners.clear();
  state.catalog.data = [
    { model: "existing-model", provider: "llama.cpp" },
    { model: "remote-model", provider: "openai" },
  ];
  state.catalog.isFetching = false;
  state.catalog.isError = false;
});

describe("explicit model classification", () => {
  it("accepts the canonical llama provider returned by the server catalog", async () => {
    state.catalog.data = [{ model: "existing-model", provider: "llama" }];
    call.mockResolvedValue(item());
    mount();
    await fill();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    await waitFor(() => expect(call).toHaveBeenCalledOnce());
    expect(call.mock.calls[0][0]).toMatchObject({
      provider: "llama.cpp",
      model: "existing-model",
    });
  });

  it("saves advisory evidence with explicit consent and only a configured supported model", async () => {
    call.mockResolvedValue(item());
    mount();
    expect(
      screen.getByText(
        /sample and result are saved to the staff Review queue/i,
      ),
    ).toBeVisible();
    await fill();
    expect(screen.queryByText("remote-model (openai)")).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    expect(
      await screen.findByText("The sample discloses an access code."),
    ).toBeVisible();
    expect(screen.getByText("ROWAN-731")).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Open Review queue" }),
    ).toHaveAttribute("href", "/moderation");
    expect(call.mock.calls[0][0]).toMatchObject({
      provider: "llama.cpp",
      model: "existing-model",
      criterion: "Flag unpublished access codes.",
      text: "Unpublished access code: ROWAN-731.",
    });
    expect(call.mock.calls[0][0].request_id).toMatch(/^[0-9a-f-]{36}$/);
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    expect(call).toHaveBeenCalledTimes(1);
  });

  it("preserves uncertain input and request ID for one explicit recovery", async () => {
    call
      .mockRejectedValueOnce(new Error("Model unavailable"))
      .mockResolvedValueOnce(item());
    mount();
    await fill();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    expect(await screen.findByText(/Model unavailable/)).toBeVisible();
    const original = call.mock.calls[0][0];
    expect(screen.getByLabelText("Sample for classification")).toHaveValue(
      original.text,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Retry saved request" }),
    );
    expect(
      await screen.findByText("The sample discloses an access code."),
    ).toBeVisible();
    expect(call.mock.calls[1][0]).toEqual(original);
  });

  it("Stop aborts the owned request and ignores a completion that arrives afterward", async () => {
    const pending = deferred<ReturnType<typeof item>>();
    call.mockReturnValueOnce(pending.promise);
    mount();
    await fill();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Stop classification" }),
    );
    expect(call.mock.calls[0][1]?.abortSignal.aborted).toBe(true);
    await act(async () => pending.resolve(item()));
    expect(
      screen.queryByText("The sample discloses an access code."),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Retry saved request" }),
    ).toBeEnabled();
  });

  it("changing input starts a new receipt instead of reusing an old model verdict", async () => {
    call.mockResolvedValue(item());
    mount();
    await fill();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    await screen.findByText("The sample discloses an access code.");
    const first = call.mock.calls[0][0].request_id;
    fireEvent.change(screen.getByLabelText("Sample for classification"), {
      target: { value: "A benign fictional sample." },
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    await waitFor(() => expect(call).toHaveBeenCalledTimes(2));
    expect(call.mock.calls[1][0].request_id).not.toBe(first);
  });

  it("retires immediately on account change and blocks late results and new dispatch", async () => {
    const pending = deferred<ReturnType<typeof item>>();
    call.mockReturnValueOnce(pending.promise);
    mount();
    await fill();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    act(() => state.listeners.forEach((listener) => listener(true)));
    expect(call.mock.calls[0][1]?.abortSignal.aborted).toBe(true);
    await act(async () => pending.resolve(item()));
    expect(
      screen.queryByText("The sample discloses an access code."),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Classify and save for review" }),
    ).toBeDisabled();
    expect(call).toHaveBeenCalledTimes(1);
  });

  it("aborts and retires its listener on unmount", async () => {
    const pending = deferred<ReturnType<typeof item>>();
    call.mockReturnValueOnce(pending.promise);
    const view = mount();
    await fill();
    fireEvent.click(
      screen.getByRole("button", { name: "Classify and save for review" }),
    );
    view.unmount();
    expect(call.mock.calls[0][1]?.abortSignal.aborted).toBe(true);
    expect(state.listeners.size).toBe(0);
    await act(async () => pending.resolve(item()));
  });

  it("keeps classification disabled without a verified configured model", () => {
    state.catalog.data = [];
    mount();
    expect(
      screen.getByText(/No configured llama.cpp model is available/),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Classify and save for review" }),
    ).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Reload models" }));
    expect(state.catalog.refetch).toHaveBeenCalled();
    expect(call).not.toHaveBeenCalled();
  });

  it("distinguishes a failed model catalog from an unconfigured provider", () => {
    state.catalog.data = [];
    state.catalog.isError = true;
    mount();
    expect(screen.getByText(/Could not load configured models/)).toBeVisible();
    expect(
      screen.queryByText(/No configured llama.cpp model is available/),
    ).not.toBeInTheDocument();
    expect(call).not.toHaveBeenCalled();
  });

  it("keeps the classification controls understandable with the shipped English resources", async () => {
    const instance = createInstance();
    await instance.init({
      lng: "en",
      fallbackLng: "en",
      ns: ["option"],
      defaultNS: "option",
      resources: { en: { option: englishOptions } },
    });
    render(
      <I18nextProvider i18n={instance}>
        <ConfigProvider theme={{ token: { motion: false } }}>
          <MemoryRouter>
            <ModelClassificationPanel
              configSnapshot={{
                serverUrl: "https://original.test",
                authMode: "single-user",
                apiKey: "panel-fixture-key",
              }}
            />
          </MemoryRouter>
        </ConfigProvider>
      </I18nextProvider>,
    );
    expect(
      screen.getByRole("heading", { name: "Model classification" }),
    ).toBeVisible();
    expect(screen.getByLabelText("Flag when")).toBeVisible();
    expect(screen.getByLabelText("Sample for classification")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Classify and save for review" }),
    ).toBeVisible();
    expect(
      screen.getByText(
        /sample and result are saved to the staff Review queue/i,
      ),
    ).toBeVisible();
  });
});
