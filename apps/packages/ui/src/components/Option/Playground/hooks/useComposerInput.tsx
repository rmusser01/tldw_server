import React from "react";
import useDynamicTextareaSize from "~/hooks/useDynamicTextareaSize";
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText";
import { handleChatInputKeyDown } from "@/utils/key-down";
import { PASTED_TEXT_CHAR_LIMIT } from "@/utils/constant";
import { isFirefoxTarget } from "@/config/platform";
import { createComposerPerfTracker } from "@/utils/perf/composer-perf";
import { createRenderPerfTracker } from "@/utils/perf/render-profiler";
import type { CollapsedRange } from "@/hooks/playground";

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UseComposerInputDeps {
  /** Textarea ref shared with parent */
  textareaRef: React.RefObject<HTMLTextAreaElement>;
  /** Message collapse controls from useMessageCollapse */
  isMessageCollapsed: boolean;
  setIsMessageCollapsed: (collapsed: boolean) => void;
  collapsedRange: CollapsedRange | null;
  setCollapsedRange: (range: CollapsedRange | null) => void;
  hasExpandedLargeText: boolean;
  setHasExpandedLargeText: (expanded: boolean) => void;
  collapseLargeMessage: (
    value: string,
    options?: { force?: boolean; range?: CollapsedRange },
  ) => void;
  restoreCollapseState: (
    value: string,
    metadata?: {
      wasExpanded?: boolean;
      collapsedRange?: CollapsedRange | null;
    },
  ) => void;
  getCollapsedDisplayMeta: (message: string, range: CollapsedRange) => any;
  getDisplayCaretFromMessage: (caret: number, meta: any) => number;
  getMessageCaretFromDisplay: (
    displayCaret: number,
    meta: any,
    options?: { prefer?: string },
  ) => number;
  normalizeCollapsedRange: (
    range: CollapsedRange,
    length: number,
  ) => CollapsedRange;
  expandLargeMessage: (options?: { force?: boolean }) => void;
  pendingCaretRef: React.MutableRefObject<number | null>;
  lastDisplaySelectionRef: React.MutableRefObject<{
    start: number;
    end: number;
  } | null>;
  pendingCollapsedStateRef: React.MutableRefObject<{
    message: string;
    range: CollapsedRange;
    caret: number;
  } | null>;
  pointerDownRef: React.MutableRefObject<boolean>;
  selectionFromPointerRef: React.MutableRefObject<boolean>;
  /** Tab mentions */
  tabMentionsEnabled: boolean;
  handleTextChange: (value: string, cursorPosition: number) => void;
  /** Pro mode */
  isProMode: boolean;
  /** Sticky dock-specific textarea cap */
  textareaMaxHeightOverride?: number;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useComposerInput(deps: UseComposerInputDeps) {
  const {
    textareaRef,
    isMessageCollapsed,
    setIsMessageCollapsed,
    collapsedRange,
    setCollapsedRange,
    hasExpandedLargeText,
    setHasExpandedLargeText,
    collapseLargeMessage,
    restoreCollapseState,
    getCollapsedDisplayMeta,
    getDisplayCaretFromMessage,
    getMessageCaretFromDisplay,
    normalizeCollapsedRange,
    expandLargeMessage,
    pendingCaretRef,
    lastDisplaySelectionRef,
    pendingCollapsedStateRef,
    pointerDownRef,
    selectionFromPointerRef,
    tabMentionsEnabled,
    handleTextChange,
    isProMode,
    textareaMaxHeightOverride,
  } = deps;

  // --- Shared text/draft/focus primitive ---
  // See apps/packages/ui/src/components/Chat/composer/hooks/useComposerText.ts.
  // `form`, `draftSaved`, and `textAreaFocus` live there; auto-resize is
  // wired separately below so we can size against `messageDisplayValue`.
  const composerText = useComposerText({
    draftKey: "tldw:playgroundChatDraft",
    textareaRef,
    isProMode,
    maxHeight: textareaMaxHeightOverride,
    getDraftMetadata: () => ({
      wasExpanded: hasExpandedLargeText,
      collapsedRange: collapsedRange
        ? { start: collapsedRange.start, end: collapsedRange.end }
        : null,
    }),
    restoreWithMetadata: (value, metadata) => {
      // `useComposerText` already wrote the message into the form; layer on
      // the Playground-specific collapse-state restore.
      restoreCollapseState(value, metadata as any);
    },
  });
  const { form, textAreaFocus, draftSaved, textareaMaxHeight } = composerText;

  const setFieldValueRef = React.useRef(form.setFieldValue);
  React.useEffect(() => {
    setFieldValueRef.current = form.setFieldValue;
  }, [form.setFieldValue]);

  // --- Perf tracking ---
  const composerPerfTrackerRef = React.useRef(
    createComposerPerfTracker({
      enabled: Boolean((globalThis as any).__TLDW_CHAT_PERF__),
    }),
  );
  const renderPerfTrackerRef = React.useRef(
    createRenderPerfTracker({
      enabled: Boolean((globalThis as any).__TLDW_CHAT_PERF__),
    }),
  );

  const markComposerPerf = React.useCallback((label: string) => {
    return composerPerfTrackerRef.current.start(label);
  }, []);

  const onComposerRenderProfile =
    React.useCallback<React.ProfilerOnRenderCallback>(
      (id, phase, actualDuration, baseDuration, startTime, commitTime) => {
        renderPerfTrackerRef.current.onRender(
          String(id),
          phase,
          actualDuration,
          baseDuration,
          startTime,
          commitTime,
        );
      },
      [],
    );

  const measureComposerPerf = React.useCallback(
    <T,>(label: string, fn: () => T): T => {
      const end = markComposerPerf(label);
      try {
        return fn();
      } finally {
        end();
      }
    },
    [markComposerPerf],
  );

  const wrapComposerProfile = React.useCallback(
    (id: string, node: React.ReactNode): React.ReactNode => {
      if (!renderPerfTrackerRef.current.isEnabled()) return node;
      return (
        <React.Profiler id={id} onRender={onComposerRenderProfile}>
          {node}
        </React.Profiler>
      );
    },
    [onComposerRenderProfile],
  );

  // Expose perf APIs on window
  React.useEffect(() => {
    const inputTracker = composerPerfTrackerRef.current;
    const renderTracker = renderPerfTrackerRef.current;
    if (!inputTracker.isEnabled() || typeof window === "undefined") return;
    (window as any).__TLDW_CHAT_PERF_SNAPSHOT__ = () => inputTracker.snapshot();
    (window as any).__TLDW_CHAT_PERF_CLEAR__ = () => {
      inputTracker.clear();
      renderTracker.clear();
    };
    (window as any).__TLDW_CHAT_RENDER_PERF_SNAPSHOT__ = () =>
      renderTracker.snapshot();
    (window as any).__TLDW_CHAT_RENDER_PERF_SUMMARY__ = () =>
      renderTracker.summarize();
    (window as any).__TLDW_CHAT_RENDER_PERF_CLEAR__ = () =>
      renderTracker.clear();
    return () => {
      delete (window as any).__TLDW_CHAT_PERF_SNAPSHOT__;
      delete (window as any).__TLDW_CHAT_PERF_CLEAR__;
      delete (window as any).__TLDW_CHAT_RENDER_PERF_SNAPSHOT__;
      delete (window as any).__TLDW_CHAT_RENDER_PERF_SUMMARY__;
      delete (window as any).__TLDW_CHAT_RENDER_PERF_CLEAR__;
    };
  }, []);

  // --- Message value helpers ---
  const restoreMessageValue = React.useCallback(
    (
      value: string,
      metadata?: {
        wasExpanded?: boolean;
        collapsedRange?: CollapsedRange | null;
      },
    ) => {
      setFieldValueRef.current("message", value);
      restoreCollapseState(value, metadata);
    },
    [restoreCollapseState],
  );

  const setMessageValue = React.useCallback(
    (
      nextValue: string,
      options?: {
        collapseLarge?: boolean;
        forceCollapse?: boolean;
        collapsedRange?: CollapsedRange;
      },
    ) => {
      form.setFieldValue("message", nextValue);
      if (options?.collapseLarge) {
        collapseLargeMessage(nextValue, {
          force: options?.forceCollapse,
          range: options?.collapsedRange,
        });
      }
    },
    [collapseLargeMessage, form.setFieldValue],
  );

  // --- Display value ---
  const collapsedDisplayMeta = React.useMemo(() => {
    const message = form.values.message || "";
    if (!message || !collapsedRange) return null;
    return getCollapsedDisplayMeta(message, collapsedRange);
  }, [collapsedRange, form.values.message, getCollapsedDisplayMeta]);

  const messageDisplayValue = React.useMemo(() => {
    const message = form.values.message || "";
    if (!message) return "";
    if (!isMessageCollapsed || !collapsedDisplayMeta) return message;
    return collapsedDisplayMeta.display;
  }, [collapsedDisplayMeta, form.values.message, isMessageCollapsed]);

  // Reset collapse state when message is short
  React.useEffect(() => {
    const message = form.values.message || "";
    if (!message || message.length <= PASTED_TEXT_CHAR_LIMIT) {
      setIsMessageCollapsed(false);
      setHasExpandedLargeText(false);
      setCollapsedRange(null);
    }
  }, [form.values.message]);

  // --- Textarea sizing ---
  // Sized against `messageDisplayValue` (not the raw form value) so the
  // textarea shrinks when the paste-collapse overlay swaps in a short label.
  useDynamicTextareaSize(textareaRef, messageDisplayValue, textareaMaxHeight);

  // --- Collapsed caret sync ---
  const syncCollapsedCaret = React.useCallback(
    (options?: {
      message?: string;
      range?: CollapsedRange | null;
      caret?: number;
    }) => {
      if (!isMessageCollapsed) return;
      const pendingState = pendingCollapsedStateRef.current;
      const message =
        options?.message ?? pendingState?.message ?? form.values.message ?? "";
      const range = options?.range ?? pendingState?.range ?? collapsedRange;
      if (!range) return;
      if (!message) return;
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) return;
        const meta = getCollapsedDisplayMeta(message, range);
        const selection =
          lastDisplaySelectionRef.current ??
          (el.selectionStart !== null
            ? {
                start: el.selectionStart ?? 0,
                end: el.selectionEnd ?? el.selectionStart ?? 0,
              }
            : null);
        const hasSelection = selection
          ? selection.start !== selection.end
          : false;
        let caret =
          options?.caret ?? pendingState?.caret ?? pendingCaretRef.current;
        if (caret === undefined || caret === null) {
          if (selection && hasSelection) {
            const start = Math.max(
              0,
              Math.min(selection.start, meta.display.length),
            );
            const end = Math.max(
              0,
              Math.min(selection.end, meta.display.length),
            );
            el.focus();
            el.setSelectionRange(start, end);
            pendingCollapsedStateRef.current = null;
            return;
          }
          if (selection) {
            const displayCaret = Math.max(
              0,
              Math.min(selection.start, meta.display.length),
            );
            const prefer =
              displayCaret > meta.labelStart && displayCaret < meta.labelEnd
                ? "after"
                : undefined;
            caret = getMessageCaretFromDisplay(displayCaret, meta, { prefer });
          } else {
            caret = meta.messageLength;
          }
        }
        if (caret > meta.rangeStart && caret < meta.rangeEnd) {
          caret = meta.rangeEnd;
        }
        caret = Math.max(0, Math.min(caret, meta.messageLength));
        pendingCaretRef.current = caret;
        pendingCollapsedStateRef.current = null;
        const displayCaret = getDisplayCaretFromMessage(caret, meta);
        el.focus();
        el.setSelectionRange(displayCaret, displayCaret);
      });
    },
    [
      collapsedRange,
      form.values.message,
      getDisplayCaretFromMessage,
      getCollapsedDisplayMeta,
      isMessageCollapsed,
      textareaRef,
    ],
  );

  // Sync caret after collapsed range changes
  React.useEffect(() => {
    if (!isMessageCollapsed || !collapsedRange) return;
    if (!pendingCollapsedStateRef.current && pendingCaretRef.current === null) {
      const el = textareaRef.current;
      if (el) {
        lastDisplaySelectionRef.current = {
          start: el.selectionStart ?? 0,
          end: el.selectionEnd ?? el.selectionStart ?? 0,
        };
      }
    }
    syncCollapsedCaret();
  }, [
    collapsedRange,
    form.values.message,
    isMessageCollapsed,
    syncCollapsedCaret,
  ]);

  // --- Collapsed editing helpers ---
  const commitCollapsedEdit = React.useCallback(
    (
      nextValue: string,
      nextCaret: number,
      nextRange: CollapsedRange | null,
    ) => {
      const shouldCollapse = nextValue.length > PASTED_TEXT_CHAR_LIMIT;
      const range = shouldCollapse
        ? normalizeCollapsedRange(
            nextRange ?? { start: 0, end: nextValue.length },
            nextValue.length,
          )
        : null;
      pendingCaretRef.current = nextCaret;
      pendingCollapsedStateRef.current = range
        ? { message: nextValue, range, caret: nextCaret }
        : null;
      setMessageValue(nextValue, {
        collapseLarge: shouldCollapse,
        forceCollapse: shouldCollapse,
        collapsedRange: range ?? undefined,
      });
      if (range) {
        syncCollapsedCaret({ message: nextValue, range, caret: nextCaret });
        return;
      }
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.focus();
        el.setSelectionRange(nextCaret, nextCaret);
      });
    },
    [normalizeCollapsedRange, setMessageValue, syncCollapsedCaret, textareaRef],
  );

  const replaceCollapsedRange = React.useCallback(
    (
      currentValue: string,
      meta: ReturnType<typeof getCollapsedDisplayMeta>,
      editStart: number,
      editEnd: number,
      replacement: string,
    ) => {
      const safeStart = Math.max(0, Math.min(editStart, currentValue.length));
      const safeEnd = Math.max(
        safeStart,
        Math.min(editEnd, currentValue.length),
      );
      const nextValue =
        currentValue.slice(0, safeStart) +
        replacement +
        currentValue.slice(safeEnd);
      const nextCaret = safeStart + replacement.length;
      const overlapsBlock =
        safeStart < meta.rangeEnd && safeEnd > meta.rangeStart;
      if (overlapsBlock) {
        commitCollapsedEdit(nextValue, nextCaret, null);
        return;
      }
      const delta = replacement.length - (safeEnd - safeStart);
      const nextRng =
        safeEnd <= meta.rangeStart
          ? { start: meta.rangeStart + delta, end: meta.rangeEnd + delta }
          : { start: meta.rangeStart, end: meta.rangeEnd };
      commitCollapsedEdit(nextValue, nextCaret, nextRng);
    },
    [commitCollapsedEdit],
  );

  // --- Textarea event handlers ---
  const [typing, setTyping] = React.useState<boolean>(false);

  const handleCompositionStart = React.useCallback(() => {
    if (!isFirefoxTarget) setTyping(true);
  }, []);

  const handleCompositionEnd = React.useCallback(() => {
    if (!isFirefoxTarget) setTyping(false);
  }, []);

  const handleTextareaMouseDown = React.useCallback(() => {
    if (isMessageCollapsed) {
      pointerDownRef.current = true;
      selectionFromPointerRef.current = true;
    }
  }, [isMessageCollapsed]);

  const handleTextareaMouseUp = React.useCallback(() => {
    pointerDownRef.current = false;
    if (selectionFromPointerRef.current) {
      requestAnimationFrame(() => {
        selectionFromPointerRef.current = false;
      });
    }
  }, []);

  const handleTextareaChange = React.useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const endPerf = markComposerPerf("input:textarea-change");
      try {
        if (isMessageCollapsed) return;
        form.getInputProps("message").onChange(e);
        if (tabMentionsEnabled && textareaRef.current) {
          handleTextChange(
            e.target.value,
            textareaRef.current.selectionStart || 0,
          );
        }
      } finally {
        endPerf();
      }
    },
    [
      isMessageCollapsed,
      form,
      tabMentionsEnabled,
      textareaRef,
      handleTextChange,
      markComposerPerf,
    ],
  );

  const handleTextareaSelect = React.useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      lastDisplaySelectionRef.current = {
        start: textarea.selectionStart ?? 0,
        end: textarea.selectionEnd ?? textarea.selectionStart ?? 0,
      };
    }
    if (isMessageCollapsed && collapsedRange) {
      const message = form.values.message || "";
      if (!message || !textarea) return;
      const meta =
        collapsedDisplayMeta ??
        getCollapsedDisplayMeta(message, collapsedRange);
      const selectionStart = textarea.selectionStart ?? meta.labelStart;
      const selectionEnd = textarea.selectionEnd ?? selectionStart;
      const displayStart = Math.min(selectionStart, selectionEnd);
      const displayEnd = Math.max(selectionStart, selectionEnd);
      const hasSelection = displayStart !== displayEnd;
      const selectionTouchesLabel =
        displayStart < meta.labelEnd && displayEnd > meta.labelStart;
      const fromPointer = selectionFromPointerRef.current;
      selectionFromPointerRef.current = false;
      if (hasSelection) {
        pendingCaretRef.current = null;
        return;
      }
      const caretInsideLabel =
        displayStart > meta.labelStart && displayStart < meta.labelEnd;
      if (selectionTouchesLabel && fromPointer && caretInsideLabel) {
        pendingCaretRef.current = meta.rangeEnd;
        expandLargeMessage({ force: true });
        return;
      }
      const prefer =
        caretInsideLabel &&
        (pendingCaretRef.current ?? meta.rangeEnd) <= meta.rangeStart
          ? "before"
          : "after";
      const caret = getMessageCaretFromDisplay(displayStart, meta, {
        prefer: caretInsideLabel ? prefer : undefined,
      });
      pendingCaretRef.current = caret;
      if (caretInsideLabel) {
        syncCollapsedCaret({ caret });
      }
      return;
    }
    if (tabMentionsEnabled && textareaRef.current) {
      handleTextChange(
        textareaRef.current.value,
        textareaRef.current.selectionStart || 0,
      );
    }
  }, [
    textareaRef,
    isMessageCollapsed,
    collapsedRange,
    form.values.message,
    collapsedDisplayMeta,
    getCollapsedDisplayMeta,
    expandLargeMessage,
    getMessageCaretFromDisplay,
    syncCollapsedCaret,
    tabMentionsEnabled,
    handleTextChange,
  ]);

  return {
    form,
    typing,
    // Message value helpers
    setMessageValue,
    restoreMessageValue,
    messageDisplayValue,
    collapsedDisplayMeta,
    textAreaFocus,
    // Collapsed editing
    syncCollapsedCaret,
    commitCollapsedEdit,
    replaceCollapsedRange,
    // Textarea handlers
    handleCompositionStart,
    handleCompositionEnd,
    handleTextareaMouseDown,
    handleTextareaMouseUp,
    handleTextareaChange,
    handleTextareaSelect,
    // Perf
    markComposerPerf,
    measureComposerPerf,
    onComposerRenderProfile,
    wrapComposerProfile,
    // Draft persistence
    draftSaved,
  };
}
