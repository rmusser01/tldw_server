import React from "react";
import { act } from "@testing-library/react";
import { Storage } from "@plasmohq/storage";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { useStoreMessageOption } from "@/store/option";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AnalysisModal } from "../AnalysisModal";

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn(),
  getChatModels: vi.fn(),

  messageSuccess: vi.fn(),
  messageError: vi.fn(),
  messageWarning: vi.fn(),
  messageInfo: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions;
      return fallbackOrOptions?.defaultValue || key;
    },
  }),
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();

  const Modal = ({
    open,
    title,
    onCancel,
    footer,
    children,
  }: React.PropsWithChildren<{
    open?: boolean;
    title?: React.ReactNode;
    onCancel?: () => void;
    footer?: React.ReactNode;
  }>) => {
    if (!open) return null;
    return (
      <div data-testid="analysis-modal">
        <h2>{title}</h2>
        <button type="button" onClick={onCancel}>
          Close
        </button>
        <div>{children}</div>
        <div>{footer}</div>
      </div>
    );
  };

  const Button = ({
    children,
    onClick,
    disabled,
    loading,
    danger: _danger,
    type: _type,
    ...rest
  }: Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "type"> & {
    loading?: boolean;
    danger?: boolean;
    type?: string;
  }) => (
    <button
      type="button"
      onClick={onClick}
      disabled={Boolean(disabled || loading)}
      data-loading={loading ? "true" : "false"}
      {...rest}
    >
      {children}
    </button>
  );

  const SelectComponent = Object.assign(
    ({
      value,
      onChange,
      children,
      ...rest
    }: React.PropsWithChildren<{
      value?: string;
      onChange?: (value: string) => void;
      "aria-label"?: string;
    }>) => (
      <select
        aria-label={rest["aria-label"] || "Model"}
        data-selected-value={value || ""}
        value={value || ""}
        onChange={(event) => onChange?.(event.target.value)}
      >
        {children}
      </select>
    ),
    {
      Option: ({
        value,
        children,
      }: React.PropsWithChildren<{ value: string }>) => (
        <option value={value}>{children}</option>
      ),
    },
  );

  const TextArea = ({
    value,
    onChange,
    ...rest
  }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) => (
    <textarea
      aria-label={rest["aria-label"]}
      value={value}
      onChange={(event) => onChange?.(event)}
      placeholder={rest.placeholder}
      readOnly={rest.readOnly}
    />
  );

  return {
    ...actual,
    Modal,
    Button,
    Select: SelectComponent,
    Input: { TextArea },
    Spin: () => <div>spinner</div>,
  };
});

vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock(
  "@plasmohq/storage/hook",
  () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook"),
);

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest,
  bgStream: mocks.bgStream,
}));

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getChatModels: mocks.getChatModels,
    getModels: mocks.getChatModels,
  },
}));

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mocks.messageSuccess,
    error: mocks.messageError,
    warning: mocks.messageWarning,
    info: mocks.messageInfo,
  }),
}));

const streamChunk = (text: string) =>
  `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}`;

describe("AnalysisModal explicit model with the mounted shared owner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    useStoreMessageOption.setState({ selectedModel: "tldw:ollama:gemma3:1b" });
    mocks.getChatModels.mockResolvedValue([
      { id: "ollama:gemma3:1b", name: "Ollama" },
      { id: "custom-openai-api:gemma-tested", name: "Verified Gemma" },
    ]);
    mocks.bgStream.mockImplementation(() =>
      (async function* () {
        yield streamChunk("Chosen model answer");
        yield "data: [DONE]";
      })(),
    );
    mocks.bgRequest.mockResolvedValue({
      processing: { analysis: "Chosen model answer" },
    });
  });
  it("keeps the explicit choice in store and storage and sends its model and provider", async () => {
    function ExistingOwner() {
      useSelectedModel();
      return null;
    }
    const view = render(
      <>
        <ExistingOwner />
        <AnalysisModal
          open
          onClose={vi.fn()}
          mediaId={42}
          mediaContent="Original source"
        />
      </>,
    );
    try {
      const select = screen.getByRole("combobox", { name: "Model" });
      await waitFor(() => expect(select).toHaveValue("tldw:ollama:gemma3:1b"));
      fireEvent.change(select, {
        target: { value: "tldw:custom-openai-api:gemma-tested" },
      });
      await act(async () => {
        await Promise.resolve();
      });
      expect(select).toHaveValue("tldw:custom-openai-api:gemma-tested");
      expect(useStoreMessageOption.getState().selectedModel).toBe(
        "tldw:custom-openai-api:gemma-tested",
      );
      expect(await new Storage().get("selectedModel")).toBe(
        "tldw:custom-openai-api:gemma-tested",
      );
      fireEvent.click(
        screen.getByRole("button", { name: "Generate Analysis" }),
      );
      await waitFor(() =>
        expect(mocks.bgStream).toHaveBeenCalledWith(
          expect.objectContaining({
            body: expect.objectContaining({
              model: "custom-openai-api:gemma-tested",
              api_provider: "custom-openai-api",
            }),
          }),
        ),
      );
    } finally {
      view.unmount();
    }
  });

  it.each([
    ["bare", "../../../Working/Language_Models/Gemma/exact-model.gguf", "custom-openai-api"],
    ["qualified", "../../../Working/Language_Models/Gemma/exact-model.gguf", "custom-openai-api"],
    ["bare", "gemma3:1b", "custom_openai_api"],
    ["bare", "vendor:exact-model", "custom-openai-api"],
  ])("preserves a configured selection through delayed %s catalog hydration (%s, %s)", async (shape, model, provider) => {
    const selected = `tldw:custom-openai-api:${model}`;
    await new Storage().set("selectedModel", selected);
    useStoreMessageOption.setState({ selectedModel: selected });
    type Catalog = Array<{ id: string; name: string; provider: string }>;
    let resolveCatalog!: (models: Catalog) => void;
    const pending = new Promise<Catalog>((resolve) => { resolveCatalog = resolve; });
    mocks.getChatModels.mockReturnValue(pending);
    render(<AnalysisModal open onClose={vi.fn()} mediaId={42} mediaContent="Original source" />);
    const select = screen.getByRole("combobox", { name: "Model" });
    expect(select).toHaveAttribute("data-selected-value", selected);
    await act(async () => {
      resolveCatalog([
        { id: "ollama:other-model", name: "Ollama first", provider: "ollama" },
        { id: shape === "qualified" ? `custom-openai-api:${model}` : model, name: "Configured Gemma", provider },
      ]);
      await pending;
    });
    expect((select as HTMLSelectElement).selectedOptions[0]?.textContent).toBe("Configured Gemma");
    expect(useStoreMessageOption.getState().selectedModel).toBe(selected);
    expect(await new Storage().get("selectedModel")).toBe(selected);
    fireEvent.click(screen.getByRole("button", { name: "Generate Analysis" }));
    await waitFor(() => expect(mocks.bgStream).toHaveBeenCalledWith(expect.objectContaining({
      body: expect.objectContaining({
        model: shape === "qualified" ? `custom-openai-api:${model}` : model,
        api_provider: "custom-openai-api",
      }),
    })));
    await waitFor(() => expect(mocks.messageSuccess).toHaveBeenCalled());
  });

  it.each([
    ["ambiguous bare selection", "tldw:shared-model", ["ollama", "custom-openai-api"]],
    ["foreign provider", "tldw:custom-openai-api:shared-model", ["ollama"]],
    ["unknown provider", "tldw:custom-openai-api:shared-model", ["unknown-vendor"]],
    ["providerless catalog", "tldw:custom-openai-api:shared-model", [undefined]],
  ])("requires a deliberate choice for %s", async (_name, selected, providers) => {
    useStoreMessageOption.setState({ selectedModel: selected });
    mocks.getChatModels.mockResolvedValue(providers.map((provider, index) => ({
      id: "shared-model", name: `Candidate ${index}`, provider,
    })));
    render(<AnalysisModal open onClose={vi.fn()} mediaId={42} mediaContent="Original source" />);
    await screen.findByRole("option", { name: "Candidate 0" });
    expect(screen.getByRole("button", { name: "Generate Analysis" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Model" })).toHaveAttribute("data-selected-value", "");
    expect(mocks.bgStream).not.toHaveBeenCalled();
    expect(useStoreMessageOption.getState().selectedModel).toBe(selected);
    if (_name === "ambiguous bare selection") {
      fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
        target: { value: "tldw:custom-openai-api:shared-model" },
      });
      fireEvent.click(screen.getByRole("button", { name: "Generate Analysis" }));
      await waitFor(() => expect(mocks.bgStream).toHaveBeenCalledWith(expect.objectContaining({
        body: expect.objectContaining({ model: "shared-model", api_provider: "custom-openai-api" }),
      })));
      await waitFor(() => expect(mocks.messageSuccess).toHaveBeenCalled());
    }
  });

  it("does not use a catalog descriptor whose qualified ID conflicts with its provider", async () => {
    useStoreMessageOption.setState({ selectedModel: "tldw:custom-openai-api:shared-model" });
    mocks.getChatModels.mockResolvedValue([
      { id: "custom-openai-api:shared-model", name: "Conflicting model", provider: "ollama" },
    ]);
    render(<AnalysisModal open onClose={vi.fn()} mediaId={42} mediaContent="Original source" />);
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByRole("button", { name: "Generate Analysis" })).toBeDisabled();
    expect(mocks.bgStream).not.toHaveBeenCalled();
  });

  it("keeps a newer explicit choice when a delayed catalog resolves", async () => {
    let resolveCatalog!: (models: Array<{ id: string; provider: string }>) => void;
    const pending = new Promise<Array<{ id: string; provider: string }>>((resolve) => { resolveCatalog = resolve; });
    mocks.getChatModels.mockReturnValue(pending);
    function ExistingOwner() {
      const { setSelectedModel } = useSelectedModel();
      return <button onClick={() => void setSelectedModel("tldw:custom-openai-api:new-choice")}>Choose newer</button>;
    }
    render(<><ExistingOwner /><AnalysisModal open onClose={vi.fn()} mediaId={42} mediaContent="Original source" /></>);
    fireEvent.click(screen.getByRole("button", { name: "Choose newer" }));
    await act(async () => {
      resolveCatalog([{ id: "gemma3:1b", provider: "ollama" }, { id: "new-choice", provider: "custom-openai-api" }]);
      await pending;
    });
    expect(screen.getByRole("combobox", { name: "Model" })).toHaveValue("tldw:custom-openai-api:new-choice");
    expect(await new Storage().get("selectedModel")).toBe("tldw:custom-openai-api:new-choice");
  });
});
