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
});
