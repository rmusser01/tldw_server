import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import type { MessageInstance } from "antd/es/message/interface";
import { beforeEach, describe, expect, it, vi } from "vitest";
import NotesEditorHeader from "../NotesEditorHeader";
import { useNotesExport } from "../hooks/useNotesExport";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? _key,
  }),
}));
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }));
const SOURCE = '{"x":9007199254740993}\n';
const blobs: Blob[] = [];
const names: string[] = [];
const message = {
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
} as unknown as MessageInstance;
const read = (blob: Blob) =>
  new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve(
        new TextDecoder("utf-8", { ignoreBOM: true }).decode(
          reader.result as ArrayBuffer,
        ),
      );
    reader.readAsArrayBuffer(blob);
  });
function Harness({
  capture,
}: {
  capture?: () => {
    id: string;
    title: string;
    content: string;
    keywords: string[];
    content_language?: string | null;
  } | null;
}) {
  const exp = useNotesExport({
    message,
    confirmDanger: async () => true,
    t: (_key, options) => options?.defaultValue ?? _key,
    listMode: "active",
    query: "",
    effectiveKeywordTokens: [],
    total: 1,
    filteredCount: 1,
    hasActiveFilters: false,
    selectedBulkNotes: [],
    fetchFilteredNotesRaw: vi.fn(),
    selectedId: "n-1",
    title: "Data",
    content: SOURCE,
    contentLanguage: "json",
    editorKeywords: ["data"],
    captureExport: capture,
  });
  return (
    <NotesEditorHeader
      title="Data"
      selectedId="n-1"
      backlinkConversationId={null}
      backlinkConversationLabel={null}
      backlinkMessageId={null}
      sourceLinks={[]}
      editorDisabled={false}
      openingLinkedChat={false}
      editorMode="edit"
      hasContent
      canSave
      canGenerateFlashcards={false}
      canExport
      isSaving={false}
      canDelete={false}
      onOpenLinkedConversation={() => {}}
      onOpenSourceLink={() => {}}
      onChangeEditorMode={() => {}}
      onCopy={() => {}}
      onGenerateFlashcards={() => {}}
      onExport={exp.exportSelected}
      onSave={() => {}}
      onDelete={() => {}}
    />
  );
}
beforeEach(() => {
  vi.clearAllMocks();
  blobs.length = 0;
  names.length = 0;
  vi.stubGlobal("URL", {
    createObjectURL: (blob: Blob) => {
      blobs.push(blob);
      return "blob:note";
    },
    revokeObjectURL: vi.fn(),
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
    function () {
      names.push(this.download);
    },
  );
});
async function action(label: string) {
  fireEvent.click(screen.getByTestId("notes-overflow-menu-button"));
  fireEvent.click(await screen.findByText(label));
}
describe("real header source and portable actions", () => {
  it("downloads exact JSON source and a distinct portable envelope", async () => {
    render(<Harness />);
    await action("Download source");
    expect(names[0]).toBe("Data.json");
    expect(await read(blobs[0])).toBe(SOURCE);
    await action("Export note (.note.json)");
    expect(names[1]).toBe("Data.note.json");
    expect(JSON.parse(await read(blobs[1]))).toEqual({
      format_version: "tldw.note.v1",
      id: "n-1",
      title: "Data",
      content: SOURCE,
      content_language: "json",
      keywords: ["data"],
    });
  });
  it.each([null, "future/😀", undefined])(
    "captures canonical current source and mode %s at action time",
    async (language) => {
      const current = {
        id: "n-1",
        title: "Current",
        content: "\ufeffa\r\nb\nc\r",
        keywords: ["current"],
        content_language: language,
      };
      render(<Harness capture={() => current} />);
      await action("Download source");
      expect(names[0]).toBe("Current.txt");
      expect(await read(blobs[0])).toBe(current.content);
      await action("Export note (.note.json)");
      if (language === undefined) {
        expect(blobs).toHaveLength(1);
        expect(message.error).toHaveBeenCalledWith(
          "Portable export requires known language metadata",
        );
      } else {
        expect(JSON.parse(await read(blobs[1]))).toMatchObject(current);
      }
    },
  );
  it("retired authority produces no file", async () => {
    render(<Harness capture={() => null} />);
    await action("Download source");
    await action("Export note (.note.json)");
    expect(blobs).toHaveLength(0);
  });
});
