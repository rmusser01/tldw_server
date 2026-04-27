import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { useSmartScroll } from "../useSmartScroll";

type DockAwareSmartScroll = (
  messages: Array<{ id: string }>,
  streaming: boolean,
  threshold: number,
  options?: {
    bottomOffsetPx?: number;
  },
) => ReturnType<typeof useSmartScroll>;

type ScrollContainerHandle = {
  element: HTMLDivElement;
  get scrollTop(): number;
  set scrollTop(value: number);
};

const useSmartScrollWithDockOffset =
  useSmartScroll as unknown as DockAwareSmartScroll;

const messages = [{ id: "m-1" }];

const createScrollContainer = ({
  scrollHeight,
  clientHeight,
  initialScrollTop,
}: {
  scrollHeight: number;
  clientHeight: number;
  initialScrollTop: number;
}): ScrollContainerHandle => {
  const element = document.createElement("div") as HTMLDivElement;

  let top = initialScrollTop;
  Object.defineProperty(element, "scrollHeight", {
    configurable: true,
    get: () => scrollHeight,
  });
  Object.defineProperty(element, "clientHeight", {
    configurable: true,
    get: () => clientHeight,
  });
  Object.defineProperty(element, "scrollTop", {
    configurable: true,
    get: () => top,
    set: (value: number) => {
      top = Number(value);
    },
  });
  Object.defineProperty(element, "scrollTo", {
    configurable: true,
    value: ({ top: nextTop }: { top: number }) => {
      top = Number(nextTop);
    },
  });

  return {
    element,
    get scrollTop() {
      return top;
    },
    set scrollTop(value: number) {
      top = value;
    },
  };
};

describe("useSmartScroll dock offset preservation", () => {
  let originalRequestAnimationFrame:
    | typeof window.requestAnimationFrame
    | undefined;
  let originalCancelAnimationFrame:
    | typeof window.cancelAnimationFrame
    | undefined;

  beforeEach(() => {
    originalRequestAnimationFrame = window.requestAnimationFrame;
    originalCancelAnimationFrame = window.cancelAnimationFrame;

    const raf = (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    };

    (
      window as typeof window & { requestAnimationFrame?: typeof raf }
    ).requestAnimationFrame = raf;
    (
      globalThis as typeof globalThis & {
        requestAnimationFrame?: typeof raf;
      }
    ).requestAnimationFrame = raf;
    (
      window as typeof window & {
        cancelAnimationFrame?: (handle: number) => void;
      }
    ).cancelAnimationFrame = () => undefined;
    (
      globalThis as typeof globalThis & {
        cancelAnimationFrame?: (handle: number) => void;
      }
    ).cancelAnimationFrame = () => undefined;
  });

  afterEach(() => {
    if (originalRequestAnimationFrame) {
      (
        window as typeof window & {
          requestAnimationFrame?: typeof window.requestAnimationFrame;
        }
      ).requestAnimationFrame = originalRequestAnimationFrame;
      (
        globalThis as typeof globalThis & {
          requestAnimationFrame?: typeof globalThis.requestAnimationFrame;
        }
      ).requestAnimationFrame = originalRequestAnimationFrame;
    } else {
      delete (
        window as typeof window & {
          requestAnimationFrame?: typeof window.requestAnimationFrame;
        }
      ).requestAnimationFrame;
      delete (
        globalThis as typeof globalThis & {
          requestAnimationFrame?: typeof globalThis.requestAnimationFrame;
        }
      ).requestAnimationFrame;
    }

    if (originalCancelAnimationFrame) {
      (
        window as typeof window & {
          cancelAnimationFrame?: typeof window.cancelAnimationFrame;
        }
      ).cancelAnimationFrame = originalCancelAnimationFrame;
      (
        globalThis as typeof globalThis & {
          cancelAnimationFrame?: typeof globalThis.cancelAnimationFrame;
        }
      ).cancelAnimationFrame = originalCancelAnimationFrame;
    } else {
      delete (
        window as typeof window & {
          cancelAnimationFrame?: typeof window.cancelAnimationFrame;
        }
      ).cancelAnimationFrame;
      delete (
        globalThis as typeof globalThis & {
          cancelAnimationFrame?: typeof globalThis.cancelAnimationFrame;
        }
      ).cancelAnimationFrame;
    }
  });

  it("keeps the latest message anchored when the reserved dock offset grows at bottom", () => {
    const container = createScrollContainer({
      scrollHeight: 2000,
      clientHeight: 600,
      initialScrollTop: 1400,
    });

    const { result, rerender } = renderHook(
      ({ bottomOffsetPx }) =>
        useSmartScrollWithDockOffset(messages, false, 120, { bottomOffsetPx }),
      {
        initialProps: { bottomOffsetPx: 80 },
      },
    );

    act(() => {
      result.current.containerRef.current = container.element;
    });

    act(() => {
      rerender({ bottomOffsetPx: 160 });
    });

    expect(container.scrollTop).toBe(1480);
  });

  it("preserves the current scroll position when the reserved dock offset grows while scrolled up", () => {
    const container = createScrollContainer({
      scrollHeight: 2000,
      clientHeight: 600,
      initialScrollTop: 900,
    });

    const { result, rerender } = renderHook(
      ({ bottomOffsetPx }) =>
        useSmartScrollWithDockOffset(messages, false, 120, { bottomOffsetPx }),
      {
        initialProps: { bottomOffsetPx: 80 },
      },
    );

    act(() => {
      result.current.containerRef.current = container.element;
    });

    act(() => {
      container.scrollTop = 980;
      rerender({ bottomOffsetPx: 160 });
    });

    expect(container.scrollTop).toBe(980);
  });
});
