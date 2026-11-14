// @vitest-environment jsdom
import React from "react";
import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useTabMentions, type TabInfo } from "../useTabMentions";

// Stored feature preferences are I/O; selection and cleanup use the real hook.
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [false] }));

const original: TabInfo = {
  id: 1,
  title: "Submitted article",
  url: "https://example.test/original",
};
const next: TabInfo = {
  id: 2,
  title: "Next article",
  url: "https://example.test/next",
};
const mount = () =>
  renderHook(() => useTabMentions(React.createRef<HTMLTextAreaElement>()));

describe("submitted tab selection cleanup", () => {
  it("preserves documents selected for the next draft", () => {
    const view = mount();
    act(() => view.result.current.addDocument(original));
    const submitted = view.result.current.selectedDocuments;
    act(() => {
      view.result.current.clearSelectedDocuments();
      view.result.current.addDocument(next);
    });
    act(() => view.result.current.clearSubmittedDocuments(submitted));
    expect(view.result.current.selectedDocuments).toEqual([next]);
  });

  it("preserves a newer selection after a submitted tab is removed and re-added", () => {
    const view = mount();
    act(() => view.result.current.addDocument(original));
    const submitted = view.result.current.selectedDocuments;
    act(() => {
      view.result.current.removeDocument(original.id);
      view.result.current.addDocument(original);
    });
    act(() => view.result.current.clearSubmittedDocuments(submitted));
    expect(view.result.current.selectedDocuments).toEqual([original]);
  });

  it("preserves a newly extended selection", () => {
    const view = mount();
    act(() => view.result.current.addDocument(original));
    const submitted = view.result.current.selectedDocuments;
    act(() => view.result.current.addDocument(next));
    act(() => view.result.current.clearSubmittedDocuments(submitted));
    expect(view.result.current.selectedDocuments).toEqual([original, next]);
  });

  it("clears the unchanged submitted selection", () => {
    const view = mount();
    act(() => view.result.current.addDocument(original));
    const submitted = view.result.current.selectedDocuments;
    act(() => view.result.current.clearSubmittedDocuments(submitted));
    expect(view.result.current.selectedDocuments).toEqual([]);
  });

  it("keeps explicit user Clear unconditional", () => {
    const view = mount();
    act(() => view.result.current.addDocument(next));
    act(() => view.result.current.clearSelectedDocuments());
    expect(view.result.current.selectedDocuments).toEqual([]);
  });
});
