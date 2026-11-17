import { useRef, useEffect, useState, useCallback } from "react";
import { smoothScrollBehavior } from "@/utils/reduced-motion";

type SmartScrollOptions = {
  bottomOffsetPx?: number;
  /**
   * Keep following the bottom when the transcript grows without a new
   * message, for example when the final markdown, table or code block of a
   * reply renders after streaming ends.
   */
  pinOnContentResize?: boolean;
};

type ScrollableMessage =
  | { id?: unknown; isBot?: unknown; role?: unknown }
  | null
  | undefined;

/** Whether the reader can see the latest content, and whether there is anything to scroll. */
type ViewportState = { atBottom: boolean; scrollable: boolean };

/** An unmounted or empty scroller shows everything there is. */
const SETTLED_VIEWPORT: ViewportState = { atBottom: true, scrollable: false };

const sameMessage = (left: ScrollableMessage, right: ScrollableMessage) => {
  if (left === right) return true;
  if (!left || !right) return false;
  return left.id != null && left.id === right.id;
};

const isUserMessage = (message: ScrollableMessage) =>
  !!message &&
  message.role !== "system" &&
  (message.role === "user" || message.isBot === false);

/**
 * True when `next` is `previous` with messages appended at the end and at
 * least one of them is the user's: the reader has just sent something.
 * Loading older messages above, or swapping to another chat, is not a send.
 */
export const isUserSendAppend = (
  previous: readonly unknown[],
  next: readonly unknown[],
) => {
  if (previous.length === 0 || next.length <= previous.length) return false;
  const anchor = previous.length - 1;
  if (
    !sameMessage(
      previous[anchor] as ScrollableMessage,
      next[anchor] as ScrollableMessage,
    )
  ) {
    return false;
  }
  return next
    .slice(previous.length)
    .some((message) => isUserMessage(message as ScrollableMessage));
};

export const useSmartScroll = (
  messages: any[],
  streaming: boolean,
  threshold: number = 100,
  options: SmartScrollOptions = {},
) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isAutoScrollEnabled, setAutoScrollEnabledState] = useState(true);
  const autoScrollPermission = useRef(true);
  const setIsAutoScrollEnabled = useCallback((enabled: boolean) => {
    autoScrollPermission.current = enabled;
    setAutoScrollEnabledState(enabled);
  }, []);
  const [viewport, setViewport] = useState<ViewportState>(SETTLED_VIEWPORT);
  const [hasUnseenMessages, setHasUnseenMessages] = useState(false);
  const scrollTimeout = useRef<NodeJS.Timeout | null>(null);
  const lastScrollTop = useRef(0);
  const lastScrollHeight = useRef(0);
  const lastClientHeight = useRef(0);
  const isScrollingProgrammatically = useRef(false);
  const previousMessagesRef = useRef<any[]>(messages);
  const wasStreamingRef = useRef(streaming);
  const bottomOffsetPx = Math.max(0, Math.round(options.bottomOffsetPx ?? 0));
  const bottomOffsetPxRef = useRef(bottomOffsetPx);
  const previousBottomOffsetPxRef = useRef(bottomOffsetPx);

  useEffect(() => {
    bottomOffsetPxRef.current = bottomOffsetPx;
  }, [bottomOffsetPx]);

  const isAtBottom = useCallback(
    (offsetPx: number = bottomOffsetPxRef.current) => {
      const container = containerRef.current;
      if (!container) return true;

      const { scrollTop, scrollHeight, clientHeight } = container;
      return scrollHeight - scrollTop - clientHeight - offsetPx <= threshold;
    },
    [threshold],
  );

  const applyViewport = useCallback((next: ViewportState) => {
    setViewport((previous) =>
      previous.atBottom === next.atBottom &&
      previous.scrollable === next.scrollable
        ? previous
        : next,
    );
    if (next.atBottom) setHasUnseenMessages(false);
  }, []);

  /** Re-reads the scroller so the "latest is in view" state is current. */
  const syncViewport = useCallback(() => {
    const container = containerRef.current;
    if (!container) {
      applyViewport(SETTLED_VIEWPORT);
      return;
    }
    applyViewport({
      atBottom: isAtBottom(),
      scrollable: container.scrollHeight > container.clientHeight,
    });
  }, [applyViewport, isAtBottom]);

  const pauseAutoScroll = useCallback(() => {
    setIsAutoScrollEnabled(false);
    if (scrollTimeout.current) {
      clearTimeout(scrollTimeout.current);
      scrollTimeout.current = null;
    }
    syncViewport();
  }, [setIsAutoScrollEnabled, syncViewport]);

  const scrollToBottom = useCallback((smooth: boolean = false) => {
    const container = containerRef.current;
    if (!container) return;

    isScrollingProgrammatically.current = true;

    if (typeof container.scrollTo === "function") {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: smooth ? smoothScrollBehavior() : "auto",
      });
    } else {
      container.scrollTop = container.scrollHeight;
    }

    setTimeout(
      () => {
        isScrollingProgrammatically.current = false;
      },
      smooth ? 300 : 50,
    );
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    lastScrollTop.current = container.scrollTop;
    lastScrollHeight.current = container.scrollHeight;
    lastClientHeight.current = container.clientHeight;
    // While following, the follow-up scroll reports the position; reading it
    // now could flash "not at the latest" for a frame.
    if (!autoScrollPermission.current) syncViewport();

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const movedUp = scrollTop < lastScrollTop.current;
      // When the content shrank or the scroller grew, the browser clamps
      // scrollTop; that drop is layout, not the reader scrolling up.
      const layoutShrank =
        scrollHeight < lastScrollHeight.current ||
        clientHeight > lastClientHeight.current;

      lastScrollTop.current = scrollTop;
      lastScrollHeight.current = scrollHeight;
      lastClientHeight.current = clientHeight;

      if (isScrollingProgrammatically.current) {
        // Mid-animation positions are not the reader's; only record arrival.
        if (isAtBottom()) syncViewport();
        return;
      }

      syncViewport();

      if (movedUp && !layoutShrank) {
        setIsAutoScrollEnabled(false);
      }

      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current);
      }

      scrollTimeout.current = setTimeout(() => {
        if (isAtBottom()) {
          setIsAutoScrollEnabled(true);
        }
      }, 300);
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    const resizeObserver =
      typeof ResizeObserver === "function"
        ? new ResizeObserver(() => syncViewport())
        : null;
    resizeObserver?.observe(container);

    return () => {
      container.removeEventListener("scroll", handleScroll);
      resizeObserver?.disconnect();
      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current);
      }
    };
  }, [isAtBottom, setIsAutoScrollEnabled, syncViewport]);

  // A send brings the reader to their message and the reply, even when they
  // had scrolled up to read; other new messages below them are flagged.
  useEffect(() => {
    const previous = previousMessagesRef.current;
    previousMessagesRef.current = messages;
    if (previous === messages) return;

    if (isUserSendAppend(previous, messages)) {
      setIsAutoScrollEnabled(true);
      setHasUnseenMessages(false);
      return;
    }
    if (
      messages.length > previous.length &&
      !autoScrollPermission.current &&
      !isAtBottom()
    ) {
      setHasUnseenMessages(true);
    }
  }, [messages, isAtBottom, setIsAutoScrollEnabled]);

  // A reply that finishes while the reader is scrolled up is new to them.
  useEffect(() => {
    const wasStreaming = wasStreamingRef.current;
    wasStreamingRef.current = streaming;
    if (
      wasStreaming &&
      !streaming &&
      !autoScrollPermission.current &&
      !isAtBottom()
    ) {
      setHasUnseenMessages(true);
    }
  }, [streaming, isAtBottom]);

  useEffect(() => {
    if (streaming && isAutoScrollEnabled) {
      requestAnimationFrame(() => {
        if (autoScrollPermission.current) scrollToBottom(false);
      });
    }
  }, [streaming, isAutoScrollEnabled, scrollToBottom]);

  useEffect(() => {
    if (messages.length === 0) {
      setIsAutoScrollEnabled(true);
      applyViewport(SETTLED_VIEWPORT);
      return;
    }

    if (isAutoScrollEnabled && !isAtBottom()) {
      requestAnimationFrame(() => {
        if (autoScrollPermission.current) scrollToBottom(!streaming);
      });
      return;
    }

    syncViewport();
  }, [
    messages,
    isAutoScrollEnabled,
    scrollToBottom,
    streaming,
    isAtBottom,
    setIsAutoScrollEnabled,
    applyViewport,
    syncViewport,
  ]);

  useEffect(() => {
    const container = containerRef.current;
    const previousBottomOffsetPx = previousBottomOffsetPxRef.current;
    previousBottomOffsetPxRef.current = bottomOffsetPx;

    if (!container) return;
    const bottomOffsetDeltaPx = bottomOffsetPx - previousBottomOffsetPx;
    if (bottomOffsetDeltaPx === 0) return;

    const wasPinnedToBottom = isAtBottom(previousBottomOffsetPx);
    if (wasPinnedToBottom) {
      const nextScrollTop = container.scrollTop + bottomOffsetDeltaPx;
      container.scrollTop = Math.max(0, nextScrollTop);
    }
    lastScrollTop.current = container.scrollTop;
    syncViewport();
  }, [bottomOffsetPx, isAtBottom, syncViewport]);

  const pinOnContentResize = options.pinOnContentResize === true;
  const hasMessages = messages.length > 0;
  useEffect(() => {
    const container = containerRef.current;
    if (!pinOnContentResize || !container) return;
    if (typeof ResizeObserver === "undefined") return;

    let lastScrollHeight = container.scrollHeight;
    let lastClientHeight = container.clientHeight;
    const resizeObserver = new ResizeObserver(() => {
      const nextScrollHeight = container.scrollHeight;
      const previousScrollHeight = lastScrollHeight;
      const previousClientHeight = lastClientHeight;
      lastScrollHeight = nextScrollHeight;
      lastClientHeight = container.clientHeight;
      if (nextScrollHeight <= previousScrollHeight) return;
      if (!autoScrollPermission.current) return;
      // Follow the growth only if the reader was at the bottom before it; a
      // short upward scroll can be missed while a programmatic scroll settles.
      const distanceFromBottomBeforeGrowth =
        previousScrollHeight -
        container.scrollTop -
        previousClientHeight -
        bottomOffsetPxRef.current;
      if (distanceFromBottomBeforeGrowth > threshold) return;
      scrollToBottom(false);
    });
    resizeObserver.observe(container);
    Array.from(container.children).forEach((child) =>
      resizeObserver.observe(child),
    );

    // Watch transcript blocks that render later without rebuilding the
    // observer on every streamed token.
    const childObserver =
      typeof MutationObserver === "undefined"
        ? null
        : new MutationObserver((records) => {
            records.forEach((record) => {
              record.addedNodes.forEach((node) => {
                if (node instanceof Element) resizeObserver.observe(node);
              });
            });
          });
    childObserver?.observe(container, { childList: true });

    return () => {
      childObserver?.disconnect();
      resizeObserver.disconnect();
    };
  }, [hasMessages, pinOnContentResize, scrollToBottom, threshold]);

  const autoScrollToBottom = useCallback(() => {
    setIsAutoScrollEnabled(true);
    setHasUnseenMessages(false);
    scrollToBottom(true);
  }, [scrollToBottom, setIsAutoScrollEnabled]);

  // Hidden (true) when the latest is in view and either we are following the
  // conversation or there is nothing to scroll.
  const isAutoScrollToBottom =
    viewport.atBottom && (isAutoScrollEnabled || !viewport.scrollable);

  return {
    containerRef,
    isAutoScrollToBottom,
    autoScrollToBottom,
    pauseAutoScroll,
    /** New messages or a finished reply arrived below the reader. */
    hasUnseenMessages,
  };
};
