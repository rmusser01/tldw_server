import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import NoteQuickSaveModal from "../NoteQuickSaveModal";
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options: { defaultValue: string }) =>
      options.defaultValue,
  }),
}));
it.each([" <>&\r\n雪\n ", "\ufeffx\ry"])(
  "keeps unsupported native source canonical and leaves Save usable",
  (source) => {
    const saved = vi.fn(),
      changed = vi.fn();
    render(
      <NoteQuickSaveModal
        open
        title="Raw"
        content={source}
        saveText="Save source"
        onTitleChange={vi.fn()}
        onContentChange={changed}
        onSave={saved}
        onCancel={vi.fn()}
      />,
    );
    const body = screen.getByRole("textbox", { name: "Content" });
    expect(body).toHaveAttribute("aria-readonly", "true");
    expect(body.textContent).toBe(source);
    fireEvent.click(screen.getByRole("button", { name: "Save source" }));
    expect(saved).toHaveBeenCalledOnce();
    expect(changed).not.toHaveBeenCalled();
  },
);
it("keeps ordinary native LF source editing available", () => {
  const changed = vi.fn();
  render(
    <NoteQuickSaveModal
      open
      title="Raw"
      content="plain"
      onTitleChange={vi.fn()}
      onContentChange={changed}
      onSave={vi.fn()}
      onCancel={vi.fn()}
    />,
  );
  const textarea = document.querySelector("textarea")!;
  expect(textarea.value).toBe("plain");
  fireEvent.change(textarea, { target: { value: "edited\nsource" } });
  expect(changed).toHaveBeenCalledWith("edited\nsource");
});
