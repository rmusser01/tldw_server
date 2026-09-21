import React from "react";
import { Blob as NodeBlob } from "node:buffer";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ContentReviewPage } from "../ContentReviewPage";

const state = vi.hoisted(() => ({
  draft: {} as Record<string, unknown>,
  saved: {} as Record<string, unknown>,
  updateFails: false,
  localWriteFails: false,
  owner: "alice" as string | null,
  generation: 0,
  controller: new AbortController(),
  listeners: new Set<() => void>(),
  uploadGate: null as null | Promise<void>,
  requests: [] as Array<{ path: string; servicePromptConfig?: unknown }>,
}));
vi.mock("@/services/tldw/quick-ingest-authority", () => ({
  useQuickIngestAuthority: () => React.useSyncExternalStore(callback => {
    state.listeners.add(callback); return () => { state.listeners.delete(callback) }
  }, () => state.owner),
  quickIngestAuthority: { capture: () => {
    const owner = state.owner; const signal = state.controller.signal
    return { authorityKey: owner, authorityRevision: state.generation, signal, requestScope: { config: { serverUrl: "http://owned-server", authMode: "multi-user", expectedUserId: owner } },
      isCurrent: () => !signal.aborted && state.owner === owner,
      assertCurrent: () => { if (signal.aborted || state.owner !== owner) throw new Error("Account changed") }
    }
  } }
}))
vi.mock("@/store/quick-ingest-session", () => ({ useQuickIngestSessionStore: (selector: (state: { generation: number }) => unknown) => selector({ generation: state.generation }) }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue || key,
  }),
}));
vi.mock("react-router-dom", () => ({
  useLocation: () => ({
    pathname: "/content-review",
    search: "?batch=batch-389&draft=draft-389",
  }),
  useNavigate: () => () => {},
}));
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [null, () => {}],
}));
vi.mock("@/components/Media/DiffViewModal", () => ({
  DiffViewModal: () => null,
}));
vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => async () => true,
}));
vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: async () => ({ hasChat: true }),
}));
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: async () => [] }));
vi.mock("@/db/dexie/schema", () => ({ db: {} }));
vi.mock("@/db/dexie/drafts", () => ({
  DRAFT_STORAGE_CAP_BYTES: 100000000,
  getDraftBatches: async () => [{ id: "batch-389", createdAt: 1 }],
  getDraftsByBatch: async () => [state.draft],
  getDraftById: async () => state.draft,
  upsertContentDraft: async (draft: Record<string, unknown>) => {
    if (state.localWriteFails) throw new Error("Local storage is full")
    state.draft = draft;
  },
  getDraftAsset: async () => ({
    blob: new NodeBlob(["Original content"]),
    fileName: "uat389.txt",
    mimeType: "text/plain",
  }),
  storeDraftAsset: async () => {
    throw new Error("Unexpected reattachment");
  },
}));
vi.mock("@/services/background-proxy", () => ({
  // Actual BatchMediaAddResponse contract: persisted ID is results[].db_id.
  bgUpload: async (request: { servicePromptConfig?: unknown }) => {
    state.requests.push({ path: "/api/v1/media/add", ...request })
    await state.uploadGate
    return ({
    results: [
      {
        status: "Success",
        input_ref: "uat389.txt",
        media_type: "document",
        db_id: 389,
        content: "Original content",
      },
    ],
  }) },
  bgRequest: async (request: {
    path: string;
    method: string;
    body: Record<string, unknown>;
  }) => {
    state.requests.push(request)
    if (request.path === "/api/v1/media/389" && request.method === "PUT") {
      if (state.updateFails) throw new Error("Review update failed");
      state.saved = request.body;
      return { media_id: 389, new_version: 2, message: "updated" };
    }
    if (request.path === "/api/v1/media/389/reprocess") return {};
    throw new Error(`Unexpected request: ${request.method} ${request.path}`);
  },
}));

describe("Content Review commit identity", () => {
  beforeEach(() => {
    state.owner = "alice"; state.generation += 1; state.controller = new AbortController(); state.uploadGate = null; state.requests = [];
    state.saved = {};
    state.updateFails = false;
    state.localWriteFails = false;
    state.draft = {
      id: "draft-389",
      batchId: "batch-389",
      title: "Reviewed UAT389",
      source: { kind: "file", fileName: "uat389.txt" },
      sourceAssetId: "asset-389",
      mediaType: "document",
      content: "Reviewed content",
      originalContent: "Original content",
      status: "in_progress",
      createdAt: 1,
      updatedAt: 1,
      processingOptions: { perform_analysis: false, perform_chunking: false },
    };
  });

  it("masks loaded private content immediately when the verified account is lost", async () => {
    render(<ContentReviewPage />)
    await screen.findByDisplayValue("Reviewed UAT389")
    act(() => { state.owner = null; state.generation += 1; state.controller.abort(); state.listeners.forEach(notify => notify()) })
    expect(screen.queryByDisplayValue("Reviewed UAT389")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^Commit$/ })).not.toBeInTheDocument()
    expect(state.draft.content).toBe("Reviewed content")
  })

  it("retires an in-flight commit before updating media under a replacement account", async () => {
    let release!: () => void
    state.uploadGate = new Promise<void>(resolve => { release = resolve })
    render(<ContentReviewPage />)
    await screen.findByDisplayValue("Reviewed UAT389")
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }))
    await waitFor(() => expect(state.requests).toHaveLength(1))
    act(() => { state.owner = null; state.generation += 1; state.controller.abort(); state.listeners.forEach(notify => notify()) })
    await act(async () => { release() })
    expect(state.requests.map(request => request.path)).toEqual(["/api/v1/media/add"])
    expect(state.saved).toEqual({})
    expect(state.draft.status).toBe("in_progress")
    expect(state.requests[0].servicePromptConfig).toEqual({ serverUrl: "http://owned-server", authMode: "multi-user", expectedUserId: "alice" })
  })

  it("retains a draft and allows manual retry after local storage fails (UAT401)", async () => {
    render(<ContentReviewPage />)
    await screen.findByDisplayValue("Reviewed UAT389")
    const save = screen.getByRole("button", { name: "Save draft" })
    state.localWriteFails = true
    fireEvent.click(save)
    await screen.findByText("Local storage is full")
    expect(save).not.toHaveClass("ant-btn-loading")
    expect(screen.getByDisplayValue("Reviewed content")).toBeInTheDocument()
    state.localWriteFails = false
    await act(async () => { fireEvent.click(save) })
    await waitFor(() => expect(Number(state.draft.updatedAt)).toBeGreaterThan(1))
    expect(state.draft.content).toBe("Reviewed content")
  })

  it("saves reviewed content to the db_id returned by ingestion and finalizes the draft", async () => {
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    const commitButton = screen.getByRole("button", { name: /^Commit$/ });
    await act(async () => {
      fireEvent.click(commitButton);
    });
    await waitFor(() => expect(state.draft.status).toBe("committed"));
    expect(state.saved).toEqual({
      title: "Reviewed UAT389",
      content: "Reviewed content",
    });
    expect(state.draft.status).toBe("committed");
  });

  it("keeps the draft recoverable when its final content update fails", async () => {
    state.updateFails = true;
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    const commitButton = screen.getByRole("button", { name: /^Commit$/ });
    await act(async () => {
      fireEvent.click(commitButton);
    });
    await screen.findByText("Review update failed");
    expect(commitButton).toBeEnabled();
    expect(state.draft.status).toBe("in_progress");
    expect(state.saved).toEqual({});
  });
});
