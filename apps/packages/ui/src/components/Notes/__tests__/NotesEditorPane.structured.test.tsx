import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import NotesEditorPane, { type NotesEditorPaneProps } from "../NotesEditorPane";
import {
  type CheckResult,
  createCheckResult,
  createDocumentRequest,
} from "@/services/document-editing/types";

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: () => <div />,
}));
vi.mock("@/components/Notes/TaskChecklistPreview", () => ({
  default: () => <div />,
}));
vi.mock("@/components/Notes/TaskActivityNotice", () => ({
  default: () => <div />,
}));
vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <p data-testid="markdown-render">{content}</p>
  ),
}));
const worker = vi.hoisted(() => ({
  submit: vi.fn(),
  receive: (_result: CheckResult) => undefined,
}));
vi.mock("@/services/document-editing/app-service", () => ({
  getDocumentWorkerClient: () => ({
    acquire: (receive: (result: CheckResult) => void) => {
      worker.receive = receive;
      return { submit: worker.submit, close: vi.fn(), format: vi.fn() };
    },
  }),
}));
vi.mock("@/components/Notes/StructuredNoteSource", () => ({
  StructuredNoteSource: React.forwardRef(
    ({ onReady }: { onReady: (ready: boolean) => void }, _ref) => {
      React.useEffect(() => {
        onReady(true);
        return () => onReady(false);
      }, [onReady]);
      return <div data-testid="native-source" />;
    },
  ),
}));

const noop = () => undefined;
const yes = async () => true;
const done = async () => undefined;
const base: NotesEditorPaneProps = {
  isMobileViewport: false,
  setMobileSidebarOpen: noop,
  selectedId: "a",
  showEmptyState: false,
  title: "Source",
  content: '{"a":1}',
  editorDisabled: false,
  isDirty: false,
  isOnline: true,
  loadingDetail: false,
  saving: false,
  backlinkConversationId: null,
  backlinkConversationLabel: null,
  backlinkMessageId: null,
  noteRelations: { sources: [], manualLinks: [], related: [], backlinks: [] },
  noteNeighborsLoading: false,
  noteNeighborsError: false,
  selectedNotePinned: false,
  editorMode: "edit",
  editorInputMode: "wysiwyg",
  openingLinkedChat: false,
  editorKeywords: [],
  keywordOptions: [],
  saveIndicator: "idle",
  saveIndicatorText: null,
  saveRecoveryNotice: null,
  selectedLastSavedAt: null,
  offlineStatusText: null,
  currentOfflineDraft: null,
  remoteVersionInfo: null,
  monitoringNotice: null,
  monitoringNoticeClasses: "",
  noteTasks: [],
  taskReconciliation: null,
  taskActivityEvents: [],
  taskConflictNotice: null,
  titleSuggestionLoading: false,
  canSwitchTitleStrategy: false,
  effectiveTitleSuggestStrategy: "heuristic",
  titleStrategyOptions: [],
  setTitleSuggestStrategy: noop,
  manualLinkTargetId: null,
  setManualLinkTargetId: noop,
  manualLinkSaving: false,
  manualLinkOptions: [],
  manualLinkDeletingEdgeId: null,
  assistLoadingAction: null,
  canUndoAssist: false,
  undoAssist: noop,
  shouldShowToc: false,
  tocEntries: [],
  previewContent: '{"a":1}',
  usesLargePreviewGuardrails: false,
  largePreviewReady: true,
  wysiwygHtml: "<p>conversion</p>",
  activeWikilinkQuery: null,
  wikilinkSuggestions: [],
  wikilinkSuggestionDisplayCounts: new Map(),
  wikilinkSelectionIndex: 0,
  metricSummaryText: "",
  revisionSummaryText: "",
  provenanceSummaryText: "",
  queuedOfflineDraftCount: 0,
  titleInputRef: null,
  contentTextareaRef: null,
  richEditorRef: null,
  attachmentInputRef: null,
  setTitle: noop,
  setIsDirty: noop,
  setSaveIndicator: noop,
  setMonitoringNotice: noop,
  toggleTaskCheckboxLocal: noop,
  toggleTaskCheckboxStatus: noop,
  dismissTaskActivity: done,
  inspectTaskActivity: noop,
  setEditorMode: noop,
  setEditorKeywords: noop,
  setEditorCursorIndex: noop,
  setShortcutHelpOpen: noop,
  markManualEdit: noop,
  suggestTitle: done,
  openLinkedConversation: done,
  openLinkedSource: noop,
  handleNewNote: done,
  duplicateSelectedNote: done,
  toggleNotePinned: done,
  copySelected: done,
  handleGenerateFlashcardsFromNote: noop,
  handleOpenNotesStudio: noop,
  exportSelected: noop,
  saveNote: yes,
  reloadSelectedNoteAfterConflict: done,
  saveAndStartNew: done,
  deleteNote: done,
  handleSelectNote: done,
  openGraphWorkspace: noop,
  createManualLink: done,
  removeManualLink: done,
  debouncedLoadKeywordSuggestions: noop,
  renderKeywordLabelWithFrequency: () => null,
  handleEditorInputModeChange: noop,
  applyMarkdownToolbarAction: noop,
  openAttachmentPicker: noop,
  handleAttachmentInputChange: noop,
  runAssistAction: done,
  switchStudioNoticeToMarkdown: noop,
  dismissStudioMarkdownOnlyNotice: noop,
  handleTocJump: noop,
  handlePreviewLinkClick: noop,
  handleWysiwygInput: noop,
  handleWysiwygPaste: noop,
  handleEditorChange: noop,
  handleEditorKeyDown: noop,
  handleEditorSelectionUpdate: noop,
  applyWikilinkSuggestion: noop,
};
const request = createDocumentRequest({
  key: {
    authority_id: "alice",
    document_id: "a",
    session_generation: 1,
    draft_revision: 0,
    language_revision: 0,
    request_id: 1,
  },
  text: base.content,
  language: "json",
});

describe("Notes pane structured ownership", () => {
  it("keeps default Markdown WYSIWYG and Markdown preview behavior", async () => {
    const view = render(
      <NotesEditorPane
        {...base}
        markdownAllowed
        contentLanguage={null}
        onContentLanguageChange={noop}
      />,
    );
    expect(screen.getByTestId("notes-wysiwyg-editor")).toBeInTheDocument();
    expect(screen.queryByTestId("native-source")).not.toBeInTheDocument();
    view.rerender(
      <NotesEditorPane
        {...base}
        editorInputMode="markdown"
        editorMode="preview"
        markdownAllowed
        contentLanguage={null}
        onContentLanguageChange={noop}
      />,
    );
    expect(await screen.findByTestId("markdown-render")).toBeInTheDocument();
  });
  it("opens structured source even with an old WYSIWYG preference and provides explicit controls", () => {
    render(
      <NotesEditorPane
        {...base}
        documentRequest={request}
        contentLanguage="json"
        markdownAllowed={false}
        onContentLanguageChange={noop}
      />,
    );
    expect(screen.getByTestId("native-source")).toBeInTheDocument();
    expect(
      screen.queryByTestId("notes-wysiwyg-editor"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Check document" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Format document" }),
    ).toBeInTheDocument();
  });
  it("previews future source literally instead of converting it", () => {
    const source = '<>&\r\n{"a":1}\n';
    render(
      <NotesEditorPane
        {...base}
        content={source}
        previewContent={source}
        editorMode="preview"
        contentLanguage="future"
        markdownAllowed={false}
        onContentLanguageChange={noop}
      />,
    );
    expect(screen.queryByTestId("markdown-render")).not.toBeInTheDocument();
    expect(screen.getByTestId("notes-preview-surface").textContent).toBe(
      source,
    );
  });
  it("keeps unsupported native source canonical while allowing title edits", () => {
    const changed = vi.fn();
    render(
      <NotesEditorPane
        {...base}
        content={"a\r\nb\n"}
        editorInputMode="markdown"
        contentLanguage="text"
        markdownAllowed={false}
        onContentLanguageChange={noop}
        setTitle={changed}
      />,
    );
    expect(
      screen.getByRole("textbox", { name: "Note content", exact: true }),
    ).toHaveAttribute("aria-readonly", "true");
    fireEvent.change(screen.getByRole("textbox", { name: "Note title" }), {
      target: { value: "Changed title" },
    });
    expect(changed).toHaveBeenCalledWith("Changed title");
  });
});

it.each([
  ["syntax", "checked", true],
  ["duplicate_key", "checked", true],
  ["unsupported", "unsupported", false],
  ["resource_limited", "resource_limited", false],
  ["incomplete", "checked", false],
] as const)(
  "disables current %s refusal but admits a newer draft",
  (code, state, complete) => {
    const props = {
      ...base,
      documentRequest: request,
      contentLanguage: "json",
      markdownAllowed: false,
      onContentLanguageChange: noop,
    };
    const view = render(<NotesEditorPane {...props} />);
    const format = screen.getByRole("button", { name: "Format document" });
    const check = screen.getByRole("button", { name: "Check document" });
    expect(format).toBeEnabled();
    const submitted = worker.submit.mock.calls.at(-1)![0];
    const refusal = createCheckResult(submitted, {
      state,
      complete,
      format_allowed: false,
      truncated: false,
      diagnostics: [
        { code, severity: "error", message: code, start: 0, end: 1 },
      ],
    });
    act(() => worker.receive(refusal));
    expect(format).toBeDisabled();
    expect(check).toBeEnabled();
    expect(screen.getByTestId("notes-document-status")).toHaveTextContent(
      "Format unavailable for this check",
    );
    const next = createDocumentRequest({
      ...request,
      text: '{"next":1}',
      key: { ...request.key, draft_revision: 1, request_id: 2 },
    });
    view.rerender(
      <NotesEditorPane {...props} content={next.text} documentRequest={next} />,
    );
    expect(format).toBeEnabled();
    act(() => worker.receive(refusal));
    expect(format).toBeEnabled();
    const current = worker.submit.mock.calls.at(-1)![0];
    act(() =>
      worker.receive(
        createCheckResult(current, {
          state: "checked",
          complete: true,
          format_allowed: true,
          diagnostics: [],
          truncated: false,
        }),
      ),
    );
    expect(format).toBeEnabled();
  },
);
