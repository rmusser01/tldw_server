import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { EditMessageForm } from "../EditMessageForm";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

const pending = new Set<{ promise: Promise<boolean>; resolve: () => void }>();

const deferred = () => {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<boolean>((yes, no) => {
    resolve = () => yes(true);
    reject = no;
  });
  const operation = { promise, resolve };
  pending.add(operation);
  void promise.then(
    () => pending.delete(operation),
    () => pending.delete(operation),
  );
  // The original Promise is still returned to the editor; observe rejection
  // so a broken fire-and-forget caller does not escape this test boundary.
  void promise.catch(() => undefined);
  return { promise, resolve, reject };
};

afterEach(async () => {
  const unfinished = [...pending];
  await act(async () => {
    unfinished.forEach(({ resolve }) => resolve());
    await Promise.allSettled(unfinished.map(({ promise }) => promise));
  });
  pending.clear();
  cleanup();
});

function EditorHost({
  onSubmit,
}: {
  onSubmit: (value: string, isSend: boolean) => void | boolean | Promise<boolean>;
}) {
  const [open, setOpen] = React.useState(true);
  return open ? (
    <EditMessageForm
      value="Original message"
      isBot={false}
      onSumbit={onSubmit}
      onClose={() => setOpen(false)}
    />
  ) : (
    <button onClick={() => setOpen(true)}>Edit again</button>
  );
}

it("keeps a Save draft visible through persistence, blocks duplicate submits, and permits retry after rejection", async () => {
  const first = deferred();
  const second = deferred();
  const onSubmit = vi
    .fn<(value: string, isSend: boolean) => Promise<boolean>>()
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise);
  render(<EditorHost onSubmit={onSubmit} />);
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Exact authored draft" },
  });

  fireEvent.click(screen.getByRole("button", { name: "save" }));
  fireEvent.click(screen.getByRole("button", { name: "save" }));
  fireEvent.submit(screen.getByRole("textbox").closest("form")!);
  expect(onSubmit.mock.calls).toEqual([["Exact authored draft", false, expect.any(Function), expect.any(Function)]]);
  expect(screen.getByRole("textbox")).toHaveValue("Exact authored draft");

  await act(async () => {
    first.reject(new Error("Storage unavailable"));
    await first.promise.catch(() => undefined);
  });
  expect(screen.getByRole("textbox")).toHaveValue("Exact authored draft");
  expect(screen.getByRole("alert")).toHaveTextContent("We could not confirm whether your edit was saved.");

  fireEvent.click(screen.getByRole("button", { name: "save" }));
  expect(onSubmit).toHaveBeenNthCalledWith(2, "Exact authored draft", false, expect.any(Function), expect.any(Function));
  expect(screen.getByRole("textbox")).toBeInTheDocument();
  await act(async () => second.resolve());
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
});

it("retains a rejected Save & Send draft and closes only after a successful retry", async () => {
  const first = deferred();
  const onSubmit = vi
    .fn<(value: string, isSend: boolean) => void | boolean | Promise<boolean>>()
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(undefined);
  render(<EditorHost onSubmit={onSubmit} />);
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Send this exact question" },
  });
  fireEvent.submit(screen.getByRole("textbox").closest("form")!);
  fireEvent.submit(screen.getByRole("textbox").closest("form")!);
  fireEvent.click(screen.getByRole("button", { name: "save" }));
  expect(onSubmit.mock.calls).toEqual([["Send this exact question", true, expect.any(Function), expect.any(Function)]]);
  expect(screen.getByRole("textbox")).toHaveValue("Send this exact question");

  await act(async () => {
    first.reject(new Error("Action unavailable"));
    await first.promise.catch(() => undefined);
  });
  expect(screen.getByRole("textbox")).toHaveValue("Send this exact question");
  expect(screen.getByRole("alert")).toHaveTextContent("We could not confirm whether your edit was saved.");

  await act(async () => {
    fireEvent.submit(screen.getByRole("textbox").closest("form")!);
  });
  expect(onSubmit).toHaveBeenNthCalledWith(2, "Send this exact question", true, expect.any(Function), expect.any(Function));
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
});

it("keeps a newer dirty editor when a canceled save commits its old text", async () => {
  const oldSave = deferred();
  function CommittingParent() {
    const [value, setValue] = React.useState("Original message");
    const [open, setOpen] = React.useState(true);
    return open ? (
      <EditMessageForm
        value={value}
        isBot={false}
        onSumbit={(draft) => oldSave.promise.then(() => setValue(draft))}
        onClose={() => setOpen(false)}
      />
    ) : (
      <button onClick={() => setOpen(true)}>Edit again</button>
    );
  }
  render(<CommittingParent />);
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Old draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "save" }));
  fireEvent.click(screen.getByRole("button", { name: "cancel" }));
  fireEvent.click(screen.getByRole("button", { name: "Edit again" }));
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "New draft" },
  });

  await act(async () => oldSave.resolve());
  expect(screen.getByRole("textbox")).toHaveValue("New draft");
});

it("does not close or overwrite a newer draft when an older save completes", async () => {
  const oldSave = deferred();
  const onClose = vi.fn();
  const view = render(
    <EditMessageForm
      value="Original message"
      isBot={false}
      onSumbit={() => oldSave.promise}
      onClose={onClose}
    />,
  );
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Submitted draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "save" }));
  view.rerender(
    <EditMessageForm
      value="Updated message"
      isBot={false}
      onSumbit={() => oldSave.promise}
      onClose={onClose}
    />,
  );
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "New unsaved draft" },
  });

  await act(async () => oldSave.resolve());
  expect(screen.getByRole("textbox")).toHaveValue("New unsaved draft");
  expect(onClose).not.toHaveBeenCalled();
});

it("keeps composition and Escape cancellation, then accepts synchronous Save after reopening", async () => {
  const onSubmit = vi.fn();
  render(<EditorHost onSubmit={onSubmit} />);
  const editor = screen.getByRole("textbox");
  fireEvent.compositionStart(editor);
  fireEvent.submit(editor.closest("form")!);
  expect(onSubmit).not.toHaveBeenCalled();
  expect(screen.getByRole("textbox")).toBeInTheDocument();
  fireEvent.compositionEnd(editor);
  fireEvent.keyDown(editor, { key: "Escape" });
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Edit again" }));
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "save" }));
  });
  expect(onSubmit.mock.calls).toEqual([["Original message", false, expect.any(Function), expect.any(Function)]]);
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
});

it("closes when the parent commits the submitted text before acknowledging Save", async () => {
  const save = deferred();
  function CommittingParent() {
    const [value, setValue] = React.useState("Original message");
    const [open, setOpen] = React.useState(true);
    return open ? (
      <EditMessageForm
        value={value}
        isBot={false}
        onSumbit={(draft) => {
          setValue(draft);
          return save.promise;
        }}
        onClose={() => setOpen(false)}
      />
    ) : null;
  }
  render(<CommittingParent />);
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "Committed draft" },
  });
  fireEvent.click(screen.getByRole("button", { name: "save" }));
  expect(screen.getByRole("textbox")).toHaveValue("Committed draft");

  await act(async () => save.resolve());
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
});
