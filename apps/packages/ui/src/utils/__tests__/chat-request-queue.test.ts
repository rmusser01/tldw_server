import { describe, expect, it } from "vitest";

import {
  blockQueuedRequest,
  buildQueuedRequest,
  moveQueuedRequestToFront,
  restoreQueuedRequests,
} from "@/utils/chat-request-queue";

describe("chat request queue helpers", () => {
  it("promotes the selected queued request while preserving the remaining order", () => {
    const first = buildQueuedRequest({ promptText: "first" });
    const second = buildQueuedRequest({ promptText: "second" });
    const third = buildQueuedRequest({ promptText: "third" });

    const reordered = moveQueuedRequestToFront(
      [first, second, third],
      third.id,
    );

    expect(reordered.map((item) => item.promptText)).toEqual([
      "third",
      "first",
      "second",
    ]);
  });

  it("marks a queued request blocked without losing its snapshot", () => {
    const item = buildQueuedRequest({
      promptText: "needs repair",
      snapshot: { selectedModel: "gpt-4o-mini", chatMode: "normal" },
    });

    const blocked = blockQueuedRequest(item, "missing_attachment");

    expect(blocked.status).toBe("blocked");
    expect(blocked.blockedReason).toBe("missing_attachment");
    expect(blocked.snapshot.selectedModel).toBe("gpt-4o-mini");
  });

  it("requires review of restored sending requests", () => {
    const restored = restoreQueuedRequests([
      buildQueuedRequest({
        promptText: "resume after refresh",
        status: "sending",
        blockedReason: "dispatch_failed",
      }),
    ]);

    expect(restored).toEqual([
      expect.objectContaining({
        promptText: "resume after refresh",
        status: "blocked",
        blockedReason: "restored_request",
      }),
    ]);
  });
});

it("keeps an existing restored blocked reason", () => {
  expect(
    restoreQueuedRequests([
      buildQueuedRequest({
        promptText: "KEEP",
        status: "blocked",
        blockedReason: "missing_attachment",
      }),
    ]),
  ).toMatchObject([
    {
      status: "blocked",
      blockedReason: "missing_attachment",
      promptText: "KEEP",
    },
  ]);
});

it("queues exact send text without retaining a separate insertion command", () => {
  const item = buildQueuedRequest({
    promptText: "  exact queued text  ",
    snapshot: {
      selectedQuickPrompt: "old insertion",
      selectedSystemPrompt: "system context",
    },
  });
  expect({
    text: item.promptText,
    quick: item.snapshot.selectedQuickPrompt,
    system: item.snapshot.selectedSystemPrompt,
  }).toEqual({
    text: "  exact queued text  ",
    quick: null,
    system: "system context",
  });
});

it("restores legacy queued text for review without replaying its insertion command", () => {
  const [item] = restoreQueuedRequests([
    {
      message: "legacy queued text",
      snapshot: {
        selectedQuickPrompt: "replayed command",
        selectedModel: "model",
      },
    },
  ]);
  expect({
    text: item.promptText,
    quick: item.snapshot.selectedQuickPrompt,
    model: item.snapshot.selectedModel,
    reason: item.blockedReason,
  }).toEqual({
    text: "legacy queued text",
    quick: null,
    model: "model",
    reason: "restored_request",
  });
});
