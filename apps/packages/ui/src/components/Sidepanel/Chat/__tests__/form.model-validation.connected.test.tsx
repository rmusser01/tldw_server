// @vitest-environment jsdom
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  deferred,
  mountDisconnected,
  mountReady,
  persistedReadyDraft,
  ready,
} from "./form.test-harness";
import { useConnectionStore } from "@/store/connection";
import { useStoreMessageOption } from "@/store/option";

const typeDraft = async (text: string) => {
  fireEvent.change(screen.getByTestId("chat-input"), {
    target: { value: text },
  });
  await waitFor(() => expect(persistedReadyDraft()).toBe(text));
};

const submit = () =>
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));

const allowAnswer = (fixture: Awaited<ReturnType<typeof mountReady>>) =>
  fixture.provider.mockImplementation(async function* () {
    yield {
      choices: [
        { delta: { content: "synthetic answer" }, finish_reason: "stop" },
      ],
    };
  });

describe("sidebar model validation in the real Form", () => {
  it.each(["", "  ", "tldw:"])(
    "keeps missing-model feedback and draft for selection %j",
    async (model) => {
      const fixture = await mountReady({ model });
      try {
        await typeDraft("keep my draft");
        submit();
        await waitFor(() =>
          expect(screen.getByRole("alert")).toHaveTextContent(
            /noModel|select a model/i,
          ),
        );
        fixture.rerender();
        expect(screen.getByRole("alert")).toBeVisible();
        expect(screen.getByTestId("chat-input")).toHaveValue("keep my draft");
        expect(fixture.provider).not.toHaveBeenCalled();
        expect(useStoreMessageOption.getState().queuedMessages).toHaveLength(0);
      } finally {
        await fixture.cleanup();
      }
    },
  );

  it("keeps unavailable-model feedback despite a nonempty selection", async () => {
    const fixture = await mountReady({ model: "removed-model" });
    try {
      await waitFor(() =>
        expect(screen.getByTestId("chat-model-select")).not.toBeDisabled(),
      );
      await typeDraft("keep my draft");
      submit();
      await waitFor(() =>
        expect(screen.getByRole("alert")).toHaveTextContent(
          "Selected model is not available",
        ),
      );
      expect(fixture.provider).not.toHaveBeenCalled();
    } finally {
      await fixture.cleanup();
    }
  });

  it.each(["", "removed-model"])(
    "offers the real Casual picker and sends the retained draft after selection %j",
    async (model) => {
      const fixture = await mountReady({ model });
      try {
        await waitFor(() =>
          expect(screen.getByTestId("chat-model-select")).not.toBeDisabled(),
        );
        await typeDraft("keep my draft");
        submit();
        await screen.findByRole("alert");
        fireEvent.click(screen.getByTestId("chat-model-select"));
        fireEvent.click(
          await screen.findByRole("menuitem", { name: /ready-model/ }),
        );
        await waitFor(() =>
          expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
        );
        expect(screen.getByTestId("chat-input")).toHaveValue("keep my draft");
        allowAnswer(fixture);
        submit();
        await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));
        expect(JSON.stringify(fixture.provider.mock.calls[0][0])).toContain(
          "keep my draft",
        );
        await act(async () => {
          expect(
            (await Promise.allSettled(ready.submissions)).map(
              (item) => item.status,
            ),
          ).toEqual(["fulfilled", "fulfilled"]);
        });
      } finally {
        await fixture.cleanup();
      }
    },
  );

  it("clears an error after a draft edit, then validates the next Send again", async () => {
    const fixture = await mountReady({ model: "" });
    try {
      await typeDraft("keep my draft");
      submit();
      await screen.findByRole("alert");
      await typeDraft("edited draft");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      submit();
      await waitFor(() => expect(screen.getByRole("alert")).toBeVisible());
      expect(fixture.provider).not.toHaveBeenCalled();
    } finally {
      await fixture.cleanup();
    }
  });

  it("supports explicit error dismissal without discarding the draft", async () => {
    const fixture = await mountReady({ model: "" });
    try {
      await typeDraft("keep my draft");
      submit();
      fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }));
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByTestId("chat-input")).toHaveValue("keep my draft");
      expect(fixture.provider).not.toHaveBeenCalled();
    } finally {
      await fixture.cleanup();
    }
  });

  it("validates image-only input in Pro mode", async () => {
    const fixture = await mountReady({ model: "", mode: "pro" });
    try {
      expect(ready.setImage).not.toBeNull();
      await act(async () => {
        ready.setImage?.("data:image/png;base64,attachment");
      });
      submit();
      await waitFor(() =>
        expect(screen.getByRole("alert")).toHaveTextContent(
          /noModel|select a model/i,
        ),
      );
      expect(fixture.provider).not.toHaveBeenCalled();
    } finally {
      await fixture.cleanup();
    }
  });

  it("loads the ordinary catalog without activating optional audio controls", async () => {
    const fixture = await mountReady();
    try {
      await waitFor(() => expect(ready.models).toHaveBeenCalledTimes(1));
    } finally {
      await fixture.cleanup();
    }
  });

  it("does not fetch a model catalog before connection", async () => {
    const fixture = await mountDisconnected();
    try {
      await act(async () => {});
      expect(ready.models).not.toHaveBeenCalled();
    } finally {
      await fixture.cleanup();
    }
  });

  it("ignores empty input", async () => {
    const fixture = await mountReady({ model: "" });
    try {
      submit();
      await act(async () => {
        await Promise.allSettled(ready.submissions);
      });
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(fixture.provider).not.toHaveBeenCalled();
      expect(useStoreMessageOption.getState().queuedMessages).toHaveLength(0);
    } finally {
      await fixture.cleanup();
    }
  });

  it("sends with an available model and keeps the Casual picker out of the way", async () => {
    const fixture = await mountReady();
    try {
      allowAnswer(fixture);
      await typeDraft("keep my draft");
      submit();
      await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));
      expect(JSON.stringify(fixture.provider.mock.calls[0][0])).toContain(
        "keep my draft",
      );
      expect(screen.queryByTestId("chat-model-select")).not.toBeInTheDocument();
      await act(async () => {
        expect(
          (await Promise.allSettled(ready.submissions)).map(
            (item) => item.status,
          ),
        ).toEqual(["fulfilled"]);
      });
    } finally {
      await fixture.cleanup();
    }
  });

  it("preserves queued dispatch while another response is busy", async () => {
    const release = deferred<void>();
    const fixture = await mountReady();
    try {
      fixture.provider.mockImplementation(async function* () {
        await release.promise;
        yield {
          choices: [
            { delta: { content: "synthetic answer" }, finish_reason: "stop" },
          ],
        };
      });
      await typeDraft("first question");
      submit();
      await waitFor(() => expect(fixture.provider).toHaveBeenCalledTimes(1));
      await typeDraft("keep my draft");
      fireEvent.click(screen.getByRole("button", { name: "Queue request" }));
      await waitFor(() =>
        expect(useStoreMessageOption.getState().queuedMessages).toEqual([
          expect.objectContaining({ promptText: "keep my draft" }),
        ]),
      );
      expect(fixture.provider).toHaveBeenCalledTimes(1);
    } finally {
      release.resolve();
      await fixture.cleanup();
    }
  });

  it("keeps disconnected input without queuing it", async () => {
    const fixture = await mountReady();
    try {
      await typeDraft("keep my draft");
      localStorage.setItem("__tldw_force_unconfigured", "true");
      await act(async () => {
        await useConnectionStore.getState().checkOnce({ force: true });
      });
      expect(useConnectionStore.getState().state.isConnected).toBe(false);
      expect(useConnectionStore.getState().state.offlineBypass).toBe(false);
      expect(screen.getByTestId("chat-input")).not.toHaveAttribute("readonly");
      expect(screen.getByTestId("chat-input")).not.toBeDisabled();
      expect(screen.getByTestId("chat-input")).toHaveValue("keep my draft");
      const send = screen.getByRole("button", { name: "Send message" });
      expect(send).toBeDisabled();
      fireEvent.click(send);
      expect(useStoreMessageOption.getState().queuedMessages).toEqual([]);
      expect(fixture.provider).not.toHaveBeenCalled();
    } finally {
      await fixture.cleanup();
    }
  });
});
