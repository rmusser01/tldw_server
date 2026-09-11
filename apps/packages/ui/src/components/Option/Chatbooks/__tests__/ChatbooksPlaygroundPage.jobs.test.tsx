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

const mocks = vi.hoisted(() => ({
  capabilities: { hasChatbooks: true },
  confirmDanger: vi.fn(),
  isDesktop: true,
  tldwClient: {
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
    listChatbookExportJobs: vi.fn(),
    listChatbookImportJobs: vi.fn(),
    getChatbookExportJob: vi.fn(),
    getChatbookImportJob: vi.fn(),
    downloadChatbookExport: vi.fn(),
    cancelChatbookExportJob: vi.fn(),
    cancelChatbookImportJob: vi.fn(),
    cleanupChatbooks: vi.fn(),
    removeChatbookExportJob: vi.fn(),
    removeChatbookImportJob: vi.fn(),
    removeFinishedChatbookJobs: vi.fn(),
    exportChatbook: vi.fn(),
    previewChatbook: vi.fn(),
    importChatbook: vi.fn(),
    listOpenWebUIImportScopes: vi.fn(async () => ({ scopes: [] })),
    previewOpenWebUIHydration: vi.fn(),
    createOpenWebUIHydrationJob: vi.fn(),
    getOpenWebUIHydrationJob: vi.fn(),
  },
}));

vi.mock("@tanstack/react-query", () => ({ useQuery: vi.fn() }));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | {
            defaultValue?: string;
            count?: number;
            name?: string;
            label?: string;
          },
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions;
      const value = fallbackOrOptions?.defaultValue || key;
      return value
        .replace("{{count}}", String(fallbackOrOptions?.count ?? ""))
        .replace("{{name}}", String(fallbackOrOptions?.name ?? ""))
        .replace("{{label}}", String(fallbackOrOptions?.label ?? ""));
    },
  }),
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  const Dropdown = ({ menu, children }: any) => (
    <div>
      {children}
      {Array.isArray(menu?.items) &&
        menu.items.filter(Boolean).map((item: any) => (
          <button key={item.key} type="button" onClick={() => item.onClick?.()}>
            {typeof item.label === "string" ? item.label : String(item.key)}
          </button>
        ))}
    </div>
  );
  const Table = (props: any) => (
    <>
      <div
        data-testid="chatbooks-table-fixed-right-count"
        data-count={
          props.columns?.filter((column: any) => column.fixed === "right")
            .length || 0
        }
      />
      <div
        data-testid="chatbooks-table-fixed-left-count"
        data-count={
          props.columns?.filter((column: any) => column.fixed === "left")
            .length || 0
        }
      />
      <actual.Table {...props} />
    </>
  );
  return { ...actual, Dropdown, Table };
});

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }));
vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => mocks.isDesktop,
}));
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: mocks.capabilities }),
}));
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  }),
}));
vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger,
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: mocks.tldwClient,
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

const exportJob = {
  job_id: "export-job-1",
  status: "completed",
  chatbook_name: "Nightly account backup",
  created_at: "2026-07-09T12:00:00Z",
  progress_percentage: 100,
  file_size_bytes: 12 * 1024,
  warnings: ["warning one", "warning two", "warning three", "warning four"],
  metadata: { post_write_verification: true },
};

const importJob = {
  job_id: "40e5ab11-ae90-47d6-bfff-e9d8f2ecf8f8",
  status: "failed",
  source_filename: "Family-backup.chatbook",
  created_at: "2026-07-09T12:01:00Z",
  error_message:
    "selected_openwebui_user_id is required for OpenWebUI DB imports at /private/imports/webui.db",
  metadata: { source_format: "openwebui_db" },
};

const sensitiveExportError =
  "POST /api/v1/chatbooks/export failed at /private/exports/account.zip authorization=secret-token";
const sanitizedExportError =
  "POST [server-endpoint] failed at [redacted-path] authorization=[redacted-secret]";

describe("ChatbooksPlaygroundPage Jobs", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.capabilities.hasChatbooks = true;
    mocks.isDesktop = true;
    mocks.confirmDanger.mockResolvedValue(true);
    mocks.tldwClient.listChatbookExportJobs.mockResolvedValue({
      jobs: [exportJob],
    });
    mocks.tldwClient.listChatbookImportJobs.mockResolvedValue({
      jobs: [importJob],
    });
    mocks.tldwClient.cleanupChatbooks.mockResolvedValue({ deleted_count: 1 });
    mocks.tldwClient.removeFinishedChatbookJobs.mockResolvedValue({
      export_jobs_removed: 51,
      import_jobs_removed: 1,
      archive_files_deleted: 51,
    });
    vi.mocked(useQuery).mockReturnValue({
      data: { items: [], total: 0 },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);
  });

  it("uses the full width and exposes trustworthy export status at a glance", async () => {
    const localeSpy = vi
      .spyOn(Date.prototype, "toLocaleString")
      .mockReturnValue("7/9/2026, 12:00:00 PM UTC");

    try {
      render(<ChatbooksPlaygroundPage />);
      fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));

      await waitFor(() => {
        expect(screen.getByText("Nightly account backup")).toBeInTheDocument();
      });
      expect(screen.queryByText("Job tracker")).not.toBeInTheDocument();
      expect(screen.getByTestId("chatbooks-jobs-layout")).not.toHaveClass(
        "lg:grid-cols-[minmax(0,1fr)_320px]",
      );
      expect(screen.getByText("12.0 KB")).toBeInTheDocument();
      expect(screen.getByText("Verified")).toBeInTheDocument();
      expect(screen.getByText("4")).toBeInTheDocument();
      expect(
        screen.getAllByText("7/9/2026, 12:00:00 PM UTC").length,
      ).toBeGreaterThan(0);
      expect(localeSpy).toHaveBeenCalledWith(
        undefined,
        expect.objectContaining({ timeZoneName: "short" }),
      );
      expect(
        screen.getByRole("button", { name: "Download" }),
      ).toBeInTheDocument();
      expect(
        screen
          .getAllByTestId("chatbooks-table-fixed-right-count")
          .every((marker) => marker.dataset.count === "1"),
      ).toBe(true);
      expect(
        screen
          .getAllByTestId("chatbooks-table-fixed-left-count")
          .every((marker) => marker.dataset.count === "1"),
      ).toBe(true);
    } finally {
      localeSpy.mockRestore();
    }
  });

  it("keeps job identity, status, and actions reachable without horizontal scrolling", async () => {
    mocks.isDesktop = false;

    render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));

    await screen.findByText("Nightly account backup");

    expect(screen.getByTestId("chatbooks-responsive-job-lists")).toBeVisible();
    expect(
      screen.queryByTestId("chatbooks-table-fixed-right-count"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Family-backup.chatbook")).toBeVisible();
    expect(screen.getAllByText("completed").length).toBeGreaterThan(0);
    expect(screen.getAllByText("failed").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Download" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Review import" })).toBeVisible();
    expect(
      screen.getByRole("button", {
        name: "More actions for Nightly account backup",
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("button", {
        name: "More actions for Family-backup.chatbook",
      }),
    ).toBeVisible();
    for (const name of [
      "Refresh",
      "Delete expired archives",
      "Remove finished job history",
    ]) {
      expect(screen.getByRole("button", { name })).toHaveClass("min-h-11");
    }
  });

  it("leads with archive identity and moves focus to import recovery", async () => {
    render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));

    await waitFor(() => {
      expect(screen.getByText("Family-backup.chatbook")).toBeInTheDocument();
    });
    expect(
      screen.getByText("Job ID: 40e5ab11-ae90-47d6-bfff-e9d8f2ecf8f8"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Choose an OpenWebUI user before importing."),
    ).toBeVisible();
    expect(screen.queryByText(/\/private\/imports/)).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "selected_openwebui_user_id is required for OpenWebUI DB imports at [redacted-path]",
      ),
    ).not.toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Review import" }));

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Import" })).toHaveAttribute(
        "aria-selected",
        "true",
      );
      expect(screen.getByTestId("chatbooks-import-recovery")).toHaveFocus();
    });
    expect(
      screen.getByText(
        "Review the import settings and choose the archive again if needed.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Drop an OpenWebUI webui\.db or \.sqlite database/),
    ).toBeVisible();

    const retryInput = document.querySelector<HTMLInputElement>(
      '.ant-upload-drag input[type="file"]',
    );
    expect(retryInput).not.toBeNull();
    fireEvent.change(retryInput as HTMLInputElement, {
      target: {
        files: [
          new File(["archive"], "Family-backup.chatbook", {
            type: "application/zip",
          }),
        ],
      },
    });
    await waitFor(() => {
      expect(
        screen.queryByTestId("chatbooks-import-recovery"),
      ).not.toBeInTheDocument();
    });
  }, 15_000);

  it("sanitizes export errors in the compact job tracker", async () => {
    mocks.tldwClient.listChatbookExportJobs.mockResolvedValue({
      jobs: [
        {
          ...exportJob,
          status: "failed",
          error_message: sensitiveExportError,
        },
      ],
    });

    render(<ChatbooksPlaygroundPage />);

    expect(await screen.findByText(sanitizedExportError)).toBeVisible();
    expect(screen.queryByText(sensitiveExportError)).not.toBeInTheDocument();
  });

  it.each([
    ["desktop", true],
    ["mobile", false],
  ])("sanitizes export errors in the %s jobs view", async (_label, isDesktop) => {
    mocks.isDesktop = isDesktop;
    mocks.tldwClient.listChatbookExportJobs.mockResolvedValue({
      jobs: [
        {
          ...exportJob,
          status: "failed",
          error_message: sensitiveExportError,
        },
      ],
    });

    render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));

    expect(await screen.findByText(sanitizedExportError)).toBeVisible();
    expect(screen.queryByText(sensitiveExportError)).not.toBeInTheDocument();
  });

  it("confirms the exact scope of archive and history deletion", async () => {
    render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));
    await screen.findByText("Nightly account backup");

    mocks.confirmDanger.mockResolvedValueOnce(false);
    fireEvent.click(
      screen.getByRole("button", { name: "Delete expired archives" }),
    );
    await waitFor(() => {
      expect(mocks.confirmDanger).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Delete expired archive files?",
          content:
            "Deletes expired Chatbook export files from server storage. Job history remains and is marked expired.",
        }),
      );
    });
    expect(mocks.tldwClient.cleanupChatbooks).not.toHaveBeenCalled();

    mocks.confirmDanger.mockClear();
    mocks.confirmDanger.mockResolvedValueOnce(false);
    fireEvent.click(
      screen.getByRole("button", { name: "Remove finished job history" }),
    );
    await waitFor(() => {
      expect(mocks.confirmDanger).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Remove finished job history?",
          content:
            "Removes all finished job records and deletes saved archive files for finished exports. Imported account data is not changed.",
        }),
      );
    });

    mocks.confirmDanger.mockClear();
    mocks.confirmDanger.mockResolvedValueOnce(false);
    const importCard = screen.getByText("Import jobs").closest(".ant-card");
    expect(importCard).not.toBeNull();
    fireEvent.click(
      within(importCard as HTMLElement).getByRole("button", { name: "Remove" }),
    );
    await waitFor(() => {
      expect(mocks.confirmDanger).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Remove this import job record?",
          content:
            "Removes this job from the Jobs list. Imported account data is not changed.",
        }),
      );
    });
    expect(mocks.tldwClient.removeChatbookImportJob).not.toHaveBeenCalled();
  }, 15_000);

  it("removes finished history through the server bulk operation", async () => {
    mocks.tldwClient.listChatbookExportJobs.mockResolvedValue({
      jobs: Array.from({ length: 2 }, (_, index) => ({
        ...exportJob,
        job_id: `export-job-${index + 1}`,
        chatbook_name: `Backup ${index + 1}`,
      })),
    });

    render(<ChatbooksPlaygroundPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }));
    await screen.findByText("Backup 2");

    fireEvent.click(
      screen.getByRole("button", { name: "Remove finished job history" }),
    );

    await waitFor(() => {
      expect(mocks.tldwClient.removeFinishedChatbookJobs).toHaveBeenCalledTimes(
        1,
      );
    });
    expect(mocks.tldwClient.removeChatbookExportJob).not.toHaveBeenCalled();
    expect(mocks.tldwClient.removeChatbookImportJob).not.toHaveBeenCalled();
  }, 15_000);
});
