import React from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useQuery } from "@tanstack/react-query";
import { ChatbooksPlaygroundPage } from "../ChatbooksPlaygroundPage";

const { capabilitiesMock, useQueryMock, tldwClientMock } = vi.hoisted(() => ({
  capabilitiesMock: { hasChatbooks: true },
  useQueryMock: vi.fn(),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    getChatbookExportScope: vi.fn(async () => ({
      mode: "full_account",
      categories: [],
      total_items: 0,
      pointer_only_count: 0,
      sensitive_category_count: 0,
      warning_count: 0,
      estimated_size_bytes: null,
    })),
    listChatbookExportJobs: vi.fn(async () => ({ jobs: [] })),
    listChatbookImportJobs: vi.fn(async () => ({ jobs: [] })),
    getChatbookExportJob: vi.fn(),
    getChatbookImportJob: vi.fn(),
    downloadChatbookExport: vi.fn(),
    cancelChatbookExportJob: vi.fn(),
    cancelChatbookImportJob: vi.fn(),
    cleanupChatbooks: vi.fn(),
    removeChatbookExportJob: vi.fn(),
    removeChatbookImportJob: vi.fn(),
    exportChatbook: vi.fn(),
    previewChatbook: vi.fn(),
    importChatbook: vi.fn(),
    listOpenWebUIImportScopes: vi.fn(async () => ({ scopes: [] })),
    previewOpenWebUIHydration: vi.fn(),
    createOpenWebUIHydrationJob: vi.fn(),
    getOpenWebUIHydrationJob: vi.fn(),
  },
}));

vi.mock("@tanstack/react-query", () => ({ useQuery: useQueryMock }));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | { defaultValue?: string; count?: number; label?: string },
    ) => {
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions;
      const value = defaultValueOrOptions?.defaultValue || key;
      return value
        .replace("{{count}}", String(defaultValueOrOptions?.count ?? ""))
        .replace("{{label}}", String(defaultValueOrOptions?.label ?? ""));
    },
  }),
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  const React = await import("react");
  const Select = ({
    value,
    onChange,
    options = [],
    disabled,
    className,
    mode,
    "aria-label": ariaLabel,
  }: any) => (
    <select
      aria-label={ariaLabel}
      className={className}
      disabled={disabled}
      multiple={mode === "multiple"}
      value={Array.isArray(value) ? value[0] || "" : value || ""}
      onChange={(event) => {
        const nextValue = event.target.value;
        onChange?.(
          mode === "tags" ? (nextValue ? [nextValue] : []) : nextValue,
        );
      }}
    >
      {(options as Array<{ value: string; label: React.ReactNode }>).map(
        (option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ),
      )}
    </select>
  );
  return { ...actual, Select };
});

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }));
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: capabilitiesMock }),
}));
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  }),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock,
}));
vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }));
vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock("@/components/Common/WorkspaceConnectionGate", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

if (!(globalThis as any).ResizeObserver) {
  (globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

describe("ChatbooksPlaygroundPage accessibility", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    capabilitiesMock.hasChatbooks = true;
    vi.mocked(useQuery).mockReturnValue({
      data: { items: [], total: 0 },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);
  });

  it("names common-path controls and keeps advanced choices disclosed", async () => {
    const { container } = render(<ChatbooksPlaygroundPage />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Chatbooks Backup & Import",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "Export mode" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByText("Advanced options"));
    expect(screen.getByRole("combobox", { name: "Tags" })).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "Categories" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: "Run as background job" }),
    ).toBeInTheDocument();

    const backupButton = screen.getByRole("button", { name: "Backup all" });
    expect(backupButton).toHaveClass("min-h-11");

    fireEvent.click(screen.getByRole("tab", { name: "Import" }));
    tldwClientMock.previewChatbook.mockResolvedValueOnce({
      manifest: {
        name: "Accessible archive",
        total_characters: 1,
        content_items: [
          { id: "character-1", type: "character", title: "Character" },
        ],
        account_inventory_summary: {
          counts: { account_profiles: 1, account_settings: 1, characters: 1 },
          warning_count: 0,
          sensitive_category_count: 0,
          post_write_verification: true,
        },
      },
    });
    const uploadInput = container.querySelector(
      '.ant-upload-drag input[type="file"]',
    ) as HTMLInputElement;
    fireEvent.change(uploadInput, {
      target: { files: [new File(["archive"], "backup.chatbook")] },
    });

    await waitFor(() =>
      expect(screen.getByText("What will be restored")).toBeInTheDocument(),
    );
    const importPanel = screen.getByRole("tabpanel", { name: "Import" });
    fireEvent.click(within(importPanel).getByText("Advanced options"));
    expect(
      within(importPanel).getByRole("combobox", {
        name: "Conflict resolution",
      }),
    ).toBeInTheDocument();
    expect(
      within(importPanel).getByRole("switch", { name: "Prefix imported" }),
    ).toBeInTheDocument();
    expect(
      within(importPanel).getByRole("switch", {
        name: "Run as background job",
      }),
    ).toBeInTheDocument();
    expect(
      within(importPanel).getByRole("switch", {
        name: "Characters: Include all",
      }),
    ).toBeInTheDocument();
  });

  it("uses semantic foreground hooks for upload, progress, and empty states", () => {
    const { container } = render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Import" }));

    expect(screen.getByText(/Drop a .zip or .chatbook archive/)).toHaveClass(
      "!text-text",
    );
    expect(screen.getByText("Preview before import")).toHaveClass(
      "!text-text-muted",
    );

    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));
    expect(container.querySelector(".chatbooks-semantic-empty")).not.toBeNull();
  });

  it("keeps inventory-only restore previews semantically complete", async () => {
    tldwClientMock.previewChatbook.mockResolvedValueOnce({
      manifest: {
        name: "Inventory-only archive",
        content_items: [],
        account_inventory: [
          { category: "account_profiles", label: "Account profile" },
        ],
        account_inventory_summary: {
          counts: { account_profiles: 1 },
          warning_count: 1,
          sensitive_category_count: 0,
          post_write_verification: true,
        },
      },
    });
    const { container } = render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Import" }));
    const uploadInput = container.querySelector(
      '.ant-upload-drag input[type="file"]',
    ) as HTMLInputElement;
    fireEvent.change(uploadInput, {
      target: { files: [new File(["archive"], "inventory.chatbook")] },
    });

    expect(
      await screen.findByRole("heading", {
        level: 2,
        name: "What will be restored",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Account profile · 1")).toBeInTheDocument();
    expect(
      screen.queryByText("Preview did not return item details."),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Review 1 warning"));
    expect(
      screen.getByText(
        "This archive reports warnings but did not include warning details.",
      ),
    ).toBeInTheDocument();
  });
});
