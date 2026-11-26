import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { IngestWizardProvider, useIngestWizard } from "../IngestWizardContext";
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session";
import { AddContentStep } from "../AddContentStep";
import type { WizardQueueItem } from "../types";
const DOCUMENT_LIMIT = 250 * 1024 ** 2;

vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next");
  const { default: ICU } = await import("@/i18n/icu-format");
  const i18n = createInstance().use(ICU);
  await i18n.init({ lng: "en", resources: { en: { option: {} } }, defaultNS: "option" });
  return { useTranslation: () => ({ t: i18n.t }) };
});
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: {} }),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }));

function QueueProbe() {
  const { state, skipToProcessing } = useIngestWizard();
  return (
    <>
      <output data-testid="urls">
        {JSON.stringify(state.queueItems.map((item) => item.url))}
      </output>
      <output data-testid="filenames">
        {JSON.stringify(state.queueItems.map((item) => item.fileName))}
      </output>
      <output data-testid="ids">
        {JSON.stringify(state.queueItems.map((item) => item.id))}
      </output>
      <output data-testid="submitted">
        {state.processingState.perItemProgress.length}
      </output>
      <button onClick={skipToProcessing}>submit probe</button>
    </>
  );
}
const mount = () =>
  render(
    <IngestWizardProvider>
      <AddContentStep />
      <QueueProbe />
    </IngestWizardProvider>,
  );
const add = (input: string) => {
  fireEvent.change(screen.getByRole("textbox", { name: "URL input area" }), {
    target: { value: input },
  });
  fireEvent.click(screen.getByRole("button", { name: "Add URLs to queue" }));
};

const mountOneScreen = (processingItems?: WizardQueueItem[]) => render(
  <IngestWizardProvider>
    <AddContentStep oneScreen processingItems={processingItems} />
    <QueueProbe />
  </IngestWizardProvider>,
);

describe("one-screen source queue", () => {
  it("keeps different pasted texts of the same size eligible", () => {
    mountOneScreen();
    fireEvent.click(screen.getByRole("button", { name: "Text", exact: true }));
    for (const value of ["alpha", "bravo"]) {
      fireEvent.change(screen.getByRole("textbox", { name: "Pasted text input" }), { target: { value } });
      fireEvent.click(screen.getByRole("button", { name: /add pasted text/i }));
    }
    expect(screen.getByRole("status", { name: "Source counts" })).toHaveTextContent(/^2 ready$/);
  });

  it("offers one keyboard control for choosing files", () => {
    mountOneScreen();
    const zone = screen.getByTestId("qi-file-dropzone");
    expect(zone.querySelectorAll('[role="button"],button').length + Number(zone.matches('[role="button"]'))).toBe(1);
  });

  it("keeps queue and unsubmitted input across source choices", () => {
    mountOneScreen();
    expect(screen.getByTestId("qi-file-input")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Links", pressed: false }));
    add("https://example.com/a");
    const linkDraft = "https://example.com/b";
    fireEvent.change(screen.getByRole("textbox", { name: "URL input area" }), { target: { value: linkDraft } });
    fireEvent.click(screen.getByRole("button", { name: "Text" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Pasted text input" }), { target: { value: "my notes" } });
    fireEvent.click(screen.getByRole("button", { name: "Links" }));
    expect(screen.getByRole("textbox", { name: "URL input area" })).toHaveValue(linkDraft);
    expect(screen.getByTestId("urls")).toHaveTextContent("https://example.com/a");
    fireEvent.click(screen.getByRole("button", { name: "Text" }));
    expect(screen.getByRole("textbox", { name: "Pasted text input" })).toHaveValue("my notes");
    expect(screen.queryByRole("button", { name: /Configure/ })).not.toBeInTheDocument();
  });

  it("accounts for all rows, repairs a link in place, and explicitly includes a duplicate", () => {
    mountOneScreen();
    fireEvent.click(screen.getByRole("button", { name: "Links" }));
    add("https://example.com/a\nhttps://example.com/a\nbad-link");
    const ids = screen.getByTestId("ids").textContent;
    expect(screen.getByRole("status", { name: "Source counts" })).toHaveTextContent("1 ready · 1 duplicate · 1 invalid");
    fireEvent.click(screen.getByRole("button", { name: "Edit link" }));
    const edit = screen.getByRole("textbox", { name: "Correct link" });
    expect(edit).toHaveValue("bad-link");
    fireEvent.change(edit, { target: { value: "https://example.org/book.pdf" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply link" }));
    expect(screen.getByTestId("ids").textContent).toBe(ids);
    expect(screen.getByRole("status", { name: "Source counts" })).toHaveTextContent("2 ready · 1 duplicate");
    fireEvent.click(screen.getByRole("button", { name: "Include again" }));
    expect(screen.getByRole("status", { name: "Source counts" })).toHaveTextContent("3 ready");
    fireEvent.click(screen.getByText("submit probe"));
    expect(screen.getByTestId("submitted")).toHaveTextContent("3");
  });

  it("accounts for valid rows outside the submitted set", () => {
    mountOneScreen([]);
    fireEvent.click(screen.getByRole("button", { name: "Links" }));
    add("https://example.com/a");
    expect(screen.getByRole("status", { name: "Source counts" })).toHaveTextContent("0 ready · 1 outside this run");
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("eligible source queue", () => {
  it.each([
    [
      "https://example.com/a, https://example.com/b",
      ["https://example.com/a", "https://example.com/b"],
    ],
    ["https://example.com/a,b", ["https://example.com/a,b"]],
    [
      "https://example.com/a\nhttps://example.com/b",
      ["https://example.com/a", "https://example.com/b"],
    ],
  ])("parses %s without changing a URL", (input, urls) => {
    mount();
    add(input);
    expect(JSON.parse(screen.getByTestId("urls").textContent!)).toEqual(urls);
  });
  it("excludes ambiguous combined and invalid inputs from processing", () => {
    mount();
    add("https://example.com/a,https://example.com/b\ninvalid");
    fireEvent.click(screen.getByText("submit probe"));
    expect(screen.getByTestId("submitted")).toHaveTextContent("0");
  });
  it("skips duplicates unless explicitly processed again", () => {
    mount();
    add("https://example.com/a\nhttps://example.com/a");
    fireEvent.click(screen.getByText("submit probe"));
    expect(screen.getByTestId("submitted")).toHaveTextContent("1");
    fireEvent.click(screen.getByRole("button", { name: "Process again" }));
    fireEvent.click(screen.getByText("submit probe"));
    expect(screen.getByTestId("submitted")).toHaveTextContent("2");
  });
  it("never queries tabs on WebUI", () => {
    const query = vi.fn();
    vi.stubGlobal("browser", { runtime: {}, tabs: { query } });
    mount();
    expect(
      screen.queryByRole("button", { name: "Capture current tab" }),
    ).not.toBeInTheDocument();
    expect(query).not.toHaveBeenCalled();
  });
  it.each(["chrome://settings", undefined])(
    "explains inaccessible active tab %s",
    async (url) => {
      vi.stubGlobal("browser", {
        runtime: { id: "extension" },
        tabs: { query: vi.fn().mockResolvedValue([{ url }]) },
      });
      mount();
      fireEvent.click(
        screen.getByRole("button", { name: "Capture current tab" }),
      );
      expect(await screen.findByRole("alert")).toHaveTextContent(/HTTP.*HTTPS/);
      expect(screen.getByTestId("urls")).toHaveTextContent("[]");
    },
  );
  it("captures an HTTP tab into the queue", async () => {
    vi.stubGlobal("browser", {
      runtime: { id: "extension" },
      tabs: {
        query: vi
          .fn()
          .mockResolvedValue([{ url: "https://example.com/article" }]),
      },
    });
    mount();
    fireEvent.click(
      screen.getByRole("button", { name: "Capture current tab" }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("urls")).toHaveTextContent(
        "https://example.com/article",
      ),
    );
  });
});

it("ignores a captured tab after the owner changes", async () => {
  let resolveTabs!: (tabs: { url: string }[]) => void;
  const tabs = new Promise<{ url: string }[]>((resolve) => {
    resolveTabs = resolve;
  });
  vi.stubGlobal("browser", {
    runtime: { id: "extension" },
    tabs: { query: vi.fn().mockReturnValue(tabs) },
  });
  mount();
  fireEvent.click(screen.getByRole("button", { name: "Capture current tab" }));
  act(() =>
    useQuickIngestSessionStore.getState().setAuthority("different-owner"),
  );
  await act(async () => {
    resolveTabs([{ url: "https://example.com/private" }]);
    await tabs;
  });
  expect(screen.getByTestId("urls")).toHaveTextContent("[]");
  useQuickIngestSessionStore.getState().setAuthority(null);
});

describe("file rejection feedback", () => {
  it.each(["unsupported", "oversized"])(
    "explains a rejected %s file without adding it",
    (kind) => {
      mount();
      const file =
        kind === "unsupported"
          ? new File(["unsupported"], "rejected.exe", {
              type: "application/octet-stream",
            })
          : new File(["large"], "oversized.txt", { type: "text/plain" });
      if (kind === "oversized")
        Object.defineProperty(file, "size", {
          value: DOCUMENT_LIMIT + 1,
        });
      fireEvent.change(screen.getByTestId("qi-file-input"), {
        target: { files: [file] },
      });
      expect(screen.getByRole("alert")).toHaveTextContent(
        kind === "unsupported"
          ? "rejected.exe is not a supported file type"
          : "oversized.txt exceeds 250 MB quick-ingest limit",
      );
      expect(screen.getByTestId("filenames")).toHaveTextContent("[]");
    },
  );

  it("keeps accepted files and replaces rejection feedback after a valid selection", () => {
    mount();
    const first = new File(["first"], "first.txt", { type: "text/plain" });
    const rejected = new File(["rejected"], "bad.exe", {
      type: "application/octet-stream",
    });
    const second = new File(["second"], "second.txt", { type: "text/plain" });
    fireEvent.change(screen.getByTestId("qi-file-input"), {
      target: { files: [first, rejected] },
    });
    expect(screen.getByRole("alert")).toHaveTextContent(
      "bad.exe is not a supported file type",
    );
    expect(JSON.parse(screen.getByTestId("filenames").textContent!)).toEqual([
      "first.txt",
    ]);
    fireEvent.change(screen.getByTestId("qi-file-input"), {
      target: { files: [second] },
    });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(JSON.parse(screen.getByTestId("filenames").textContent!)).toEqual([
      "first.txt",
      "second.txt",
    ]);
  });
});
