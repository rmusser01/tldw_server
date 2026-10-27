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
  ingestMessage: "",
  localWriteFails: false,
  selectedModel: null as string | null,
  owner: "alice" as string | null,
  generation: 0,
  controller: new AbortController(),
  listeners: new Set<() => void>(),
  uploadGate: null as null | Promise<void>,
  saveGate: null as null | Promise<void>,
  chatGate: null as null | Promise<void>,
  otherDraft: null as null | Record<string, unknown>,
  assetMissing: false,
  routeSearch: "?batch=batch-389&draft=draft-389",
  requests: [] as Array<{ path: string; body?: Record<string, unknown>; servicePromptConfig?: unknown; fileFieldName?: string }>,
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
    search: state.routeSearch,
  }),
  useNavigate: () => () => {},
}));
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string) => [key === "selectedModel" ? state.selectedModel : null, () => {}],
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
  getDraftsByBatch: async () => [state.draft, ...(state.otherDraft ? [state.otherDraft] : [])],
  getDraftById: async (id: string) => id === state.otherDraft?.id ? state.otherDraft : state.draft,
  upsertContentDraft: async (draft: Record<string, unknown>) => {
    await state.saveGate
    if (state.localWriteFails) throw new Error("Local storage is full")
    if (draft.id === state.otherDraft?.id) state.otherDraft = draft;
    else state.draft = draft;
  },
  updateContentDraft: async (id: string, values: Record<string, unknown>) => {
    if (id === state.otherDraft?.id) state.otherDraft = { ...state.otherDraft, ...values };
    else state.draft = { ...state.draft, ...values };
  },
  getDraftAsset: async () => state.assetMissing ? undefined : ({
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
        db_message: state.ingestMessage,
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
    if (request.path === "/api/v1/chat/completions") { await state.chatGate; return { choices: [{ message: { content: "Corrected content" } }] }; }
    if (request.path === "/api/v1/media/389" && request.method === "PUT") {
      if (state.updateFails) throw new Error("Review update failed");
      state.saved = request.body;
      return { media_id: 389, versions: [{ media_id: 389, version_number: 2 }] };
    }
    if (request.path === "/api/v1/media/389/metadata" && request.method === "PATCH") return {};
    if (request.path === "/api/v1/media/389/reprocess") return {};
    throw new Error(`Unexpected request: ${request.method} ${request.path}`);
  },
}));

describe("Content Review commit identity", () => {
  beforeEach(() => {
    state.owner = "alice"; state.generation += 1; state.controller = new AbortController(); state.uploadGate = null; state.saveGate = null; state.chatGate = null; state.otherDraft = null; state.assetMissing = false; state.routeSearch = "?batch=batch-389&draft=draft-389"; state.requests = [];
    state.saved = {};
    state.updateFails = false;
    state.ingestMessage = "";
    state.localWriteFails = false;
    state.selectedModel = null;
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

  it("preserves saved media and the draft when ingest skips an existing source", async () => {
    state.ingestMessage = "Media 'Original title' already exists. Overwrite not enabled."
    render(<ContentReviewPage />)
    await screen.findByDisplayValue("Reviewed UAT389")
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }))
    await screen.findByText("This source already exists. Saved media was preserved; this draft was not committed.")
    expect(state.requests.map(request => request.path)).toEqual(["/api/v1/media/add"])
    expect(state.saved).toEqual({})
    expect(state.draft.status).toBe("in_progress")
    expect(screen.getByDisplayValue("Reviewed content")).toBeInTheDocument()
  })

  it.each(["Reviewed UAT389", "already exists"])("finalizes a legitimate existing-source overwrite (%s)", async (title) => {
    state.draft.title = title
    state.ingestMessage = `Media '${title}' updated to new version.`
    state.draft.processingOptions = { perform_analysis: false, perform_chunking: false, overwrite_existing: true }
    render(<ContentReviewPage />)
    await screen.findByDisplayValue(title)
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }))
    await waitFor(() => expect(state.draft.status).toBe("committed"))
    expect(state.saved.content).toBe("Reviewed content")
  })

  it.each([true, false])("uploads reviewed files once through the canonical multipart field (original=%s)", async (original) => {
    if (!original) delete state.draft.sourceAssetId
    state.draft.content = "Original content"
    render(<ContentReviewPage />)
    await screen.findByDisplayValue("Reviewed UAT389")
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }))
    await waitFor(() => expect(state.draft.status).toBe("committed"))
    expect(state.requests[0].fileFieldName).toBe("files")
  })

  it.each([null, "llama.cpp/owned-model", "default"])("preserves selected model %s and omits an unselected server default (UAT403)", async (selectedModel) => {
    state.selectedModel = selectedModel;
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    const fix = screen.getByRole("button", { name: "AI fix" });
    await waitFor(() => expect(fix).toBeEnabled());
    fireEvent.click(fix);
    await waitFor(() => expect(state.draft.content).toBe("Corrected content"));
    const request = state.requests.find(item => item.path === "/api/v1/chat/completions");
    const sentBody = JSON.parse(JSON.stringify(request?.body));
    if (selectedModel === null) expect(sentBody).not.toHaveProperty("model");
    else expect(sentBody.model).toBe(selectedModel);
    expect(request?.servicePromptConfig).toEqual({ serverUrl: "http://owned-server", authMode: "multi-user", expectedUserId: "alice" });
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
  it("B09 resumes the admitted media after final content failure without uploading twice", async () => {
    state.updateFails = true;
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    const retry = screen.getByRole("button", { name: /^Commit$/ });
    fireEvent.click(retry);
    await screen.findByText("Review update failed");
    state.updateFails = false;
    await waitFor(() => expect(retry).toBeEnabled());
    expect(retry).toBeInTheDocument();
    fireEvent.click(retry);
    await waitFor(() => expect(state.draft.status).toBe("committed"));
    expect(state.requests.filter(request => request.path === "/api/v1/media/add")).toHaveLength(1);
    expect(state.saved.content).toBe("Reviewed content");
  });

  it("B09 retains edits made while an older local save is pending", async () => {
    let release!: () => void;
    state.saveGate = new Promise<void>(resolve => { release = resolve });
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    const save = screen.getByRole("button", { name: "Save draft" });
    fireEvent.click(save);
    const editor = screen.getByDisplayValue("Reviewed content");
    fireEvent.change(editor, { target: { value: "A newer unsaved edit" } });
    await act(async () => { release() });
    expect(screen.getByDisplayValue("A newer unsaved edit")).toBeInTheDocument();
    fireEvent.click(save);
    await waitFor(() => expect(state.draft.content).toBe("A newer unsaved edit"));
  });

  it("B09 saves dirty content before selecting another draft", async () => {
    state.otherDraft = { ...state.draft, id: "other", title: "Other draft", content: "Other body" };
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.change(screen.getByDisplayValue("Reviewed content"), { target: { value: "Unsaved first draft edit" } });
    fireEvent.click(screen.getByText("Other draft", { exact: true }));
    await screen.findByDisplayValue("Other draft");
    expect(state.draft.content).toBe("Unsaved first draft edit");
  });

  it("B09 never applies a delayed AI response to a different draft", async () => {
    state.otherDraft = { ...state.draft, id: "other", title: "Other draft", content: "Other body" };
    let release!: () => void;
    state.chatGate = new Promise<void>(resolve => { release = resolve });
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.click(screen.getByRole("button", { name: "AI fix" }));
    await waitFor(() => expect(state.requests.some(row => row.path === "/api/v1/chat/completions")).toBe(true));
    fireEvent.click(screen.getByText("Other draft", { exact: true }));
    await screen.findByDisplayValue("Other draft");
    await act(async () => { release() });
    expect(state.otherDraft.content).toBe("Other body");
    expect(screen.getByDisplayValue("Other body")).toBeInTheDocument();
  });

  it("B09 bulk commit uses the just-saved reviewed body", async () => {
    state.draft.status = "reviewed";
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.change(screen.getByDisplayValue("Reviewed content"), { target: { value: "Latest reviewed body" } });
    fireEvent.click(screen.getByRole("button", { name: "Commit All" }));
    await waitFor(() => expect(state.draft.status).toBe("committed"));
    expect(state.saved.content).toBe("Latest reviewed body");
  });

  it("B09 refuses blank content before admitting the original attachment", async () => {
    state.draft.content = "";
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }));
    await screen.findByText("A title and non-empty content are required before committing.");
    expect(state.requests).toEqual([]);
    expect(state.draft.status).toBe("in_progress");
  });

  it("B09 review refuses a stale editor before any canonical request", async () => {
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    state.draft = { ...state.draft, content: "Newer stored body", updatedAt: 2 };
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }));
    await screen.findByText(/Draft changed elsewhere/);
    expect(state.requests).toEqual([]);
    expect(screen.getByDisplayValue("Reviewed content")).toBeInTheDocument();
  });

  it("B09 review preserves another editor while an AI storage write settles", async () => {
    state.otherDraft = { ...state.draft, id: "other", title: "Other draft", content: "Other body" };
    let release!: () => void;
    state.saveGate = new Promise<void>(resolve => { release = resolve });
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.click(screen.getByRole("button", { name: "AI fix" }));
    await waitFor(() => expect(state.requests.some(row => row.path === "/api/v1/chat/completions")).toBe(true));
    fireEvent.click(screen.getByText("Other draft", { exact: true }));
    await screen.findByDisplayValue("Other draft");
    await act(async () => { release() });
    expect(screen.getByDisplayValue("Other body")).toBeInTheDocument();
  });

  it("B09 review holds navigation when edits arrive during its save", async () => {
    state.otherDraft = { ...state.draft, id: "other", title: "Other draft", content: "Other body" };
    let release!: () => void;
    state.saveGate = new Promise<void>(resolve => { release = resolve });
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.change(screen.getByDisplayValue("Reviewed content"), { target: { value: "First edit" } });
    fireEvent.click(screen.getByText("Other draft", { exact: true }));
    fireEvent.change(screen.getByDisplayValue("First edit"), { target: { value: "Newest edit" } });
    await act(async () => { release() });
    expect(screen.getByDisplayValue("Newest edit")).toBeInTheDocument();
  });

  it("B09 review freezes content and AI while canonical admission is pending", async () => {
    let release!: () => void;
    state.uploadGate = new Promise<void>(resolve => { release = resolve });
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }));
    await waitFor(() => expect(state.requests.some(row => row.path === "/api/v1/media/add")).toBe(true));
    const frozen = screen.getByDisplayValue("Reviewed content").hasAttribute("readonly");
    const aiDisabled = screen.getByRole("button", { name: "AI fix" }).hasAttribute("disabled");
    await act(async () => { release() });
    expect({ frozen, aiDisabled }).toEqual({ frozen: true, aiDisabled: true });
  });

  it("B09 review respects disabled chunking after content edits", async () => {
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }));
    await waitFor(() => expect(state.draft.status).toBe("committed"));
    expect(state.requests.some(row => row.path.endsWith("/reprocess"))).toBe(false);
  });

  it.each(["?batch=batch-389&draft=other", ""])("B09 final guard holds dirty edits across a draft-query history change (%s)", async search => {
    state.otherDraft = { ...state.draft, id: "other", title: "Other draft", content: "Other body" };
    const view = render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    if (search === "") {
      fireEvent.click(screen.getByText("Other draft", { exact: true }));
      await screen.findByDisplayValue("Other draft");
    }
    fireEvent.change(screen.getByDisplayValue(search === "" ? "Other body" : "Reviewed content"), { target: { value: "Keep this unsaved edit" } });
    state.routeSearch = search;
    view.rerender(<ContentReviewPage />);
    await act(async () => { await Promise.resolve() });
    expect(screen.getByDisplayValue("Keep this unsaved edit")).toBeInTheDocument();
  });

  it("B09 final guard exposes reattachment when the stored original asset disappeared", async () => {
    state.assetMissing = true;
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    await screen.findByText("Source file missing");
    expect(screen.getByRole("button", { name: "Attach file" })).toBeEnabled();
    expect(state.requests).toEqual([]);
  });

  it("B09 flushes the actual dirty editor through the existing route-leave guard", async () => {
    let guard: { when: boolean; onLeave: () => Promise<boolean> } | undefined;
    const LeaveGuard = (props: NonNullable<typeof guard>) => { guard = props; return null };
    render(<ContentReviewPage LeaveGuard={LeaveGuard} />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.change(screen.getByDisplayValue("Reviewed content"), { target: { value: "Saved before leaving" } });
    expect(guard?.when).toBe(true);
    let ready = false;
    await act(async () => { ready = await guard!.onLeave() });
    expect({ ready, content: state.draft.content }).toEqual({ ready: true, content: "Saved before leaving" });
  });

  it.each(["AI", "local save"])("B09 pending %s prevents canonical commit overlap", async work => {
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve });
    if (work === "AI") state.chatGate = gate;
    else state.saveGate = gate;
    render(<ContentReviewPage />);
    await screen.findByDisplayValue("Reviewed UAT389");
    fireEvent.click(screen.getByRole("button", { name: work === "AI" ? "AI fix" : "Save draft" }));
    if (work === "AI") await waitFor(() => expect(state.requests.some(row => row.path === "/api/v1/chat/completions")).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: /^Commit$/ }));
    await act(async () => { await Promise.resolve() });
    const admissions = state.requests.filter(row => row.path === "/api/v1/media/add").length;
    await act(async () => { release() });
    expect(admissions).toBe(0);
  });

});
