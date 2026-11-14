import React from "react";
import { useTranslation } from "react-i18next";
import { useStoreMessageOption } from "@/store/option";
import { getVariable } from "@/utils/select-variable";

type Props = {
  form: {
    values: { message: string };
    getValues: () => { message: string };
    getRevision: () => number;
  };
  setMessageValue: (value: string) => void;
  ready: boolean;
  destinationKey: string;
  captureGuard: () => () => boolean;
  textareaRef: React.RefObject<HTMLTextAreaElement>;
};

type Review = {
  text: string;
  commandRevision: number;
  destinationKey: string;
  isOwned: () => boolean;
  handled: boolean;
};

export const useQuickPromptInsertion = ({
  form,
  setMessageValue,
  ready,
  destinationKey,
  captureGuard,
  textareaRef,
}: Props) => {
  const prompt = useStoreMessageOption((state) => state.selectedQuickPrompt);
  const commandRevision = useStoreMessageOption(
    (state) => state.quickPromptRevision,
  );
  const setQuickPrompt = useStoreMessageOption(
    (state) => state.setSelectedQuickPrompt,
  );
  const [review, setReview] = React.useState<Review | null>(null);
  const mounted = React.useRef(false);
  const view = React.useRef({ ready, destinationKey });
  view.current = { ready, destinationKey };
  const selection = React.useRef<{
    element: HTMLTextAreaElement;
    initiator: Element | null;
    review: Review;
    value: string;
    inputRevision: number;
    range: ReturnType<typeof getVariable>;
  } | null>(null);
  React.useLayoutEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Acquire a one-time command into this composer; leaving cannot replay it.
  React.useLayoutEffect(() => {
    if (!ready || !prompt) return;
    const isOwned = captureGuard();
    const current = useStoreMessageOption.getState();
    if (
      !isOwned() ||
      current.quickPromptRevision !== commandRevision ||
      current.selectedQuickPrompt !== prompt
    )
      return;
    const inputRevision = form.getRevision();
    const text = form.getValues().message;
    if (!setQuickPrompt(null, { expectedRevision: commandRevision })) return;
    const consumed = useStoreMessageOption.getState();
    if (
      !isOwned() ||
      !view.current.ready ||
      view.current.destinationKey !== destinationKey ||
      consumed.quickPromptRevision !== commandRevision + 1 ||
      consumed.selectedQuickPrompt !== null ||
      form.getRevision() !== inputRevision ||
      form.getValues().message !== text
    )
      return;
    setReview({
      text: prompt,
      commandRevision: commandRevision + 1,
      destinationKey,
      isOwned,
      handled: false,
    });
  }, [
    ready,
    prompt,
    commandRevision,
    destinationKey,
    captureGuard,
    setQuickPrompt,
    form,
  ]);

  const isCurrent = () => {
    const current = useStoreMessageOption.getState();
    return (
      mounted.current &&
      review !== null &&
      !review.handled &&
      view.current.ready &&
      view.current.destinationKey === review.destinationKey &&
      review.isOwned() &&
      current.quickPromptRevision === review.commandRevision &&
      current.selectedQuickPrompt === null
    );
  };
  const reviewIsCurrent = isCurrent();
  React.useLayoutEffect(() => {
    if (review && !reviewIsCurrent) {
      review.handled = true;
      setReview((current) => (current === review ? null : current));
    }
  }, [review, reviewIsCurrent]);

  const renderedInputRevision = form.getRevision();
  const renderedText = form.values.message;
  const inputIsCurrent = () =>
    form.getRevision() === renderedInputRevision &&
    form.getValues().message === renderedText;
  const accept = (append: boolean) => {
    if (!isCurrent() || !review || !inputIsCurrent()) return;
    review.handled = true;
    const offset =
      append && renderedText.length > 0 ? renderedText.length + 2 : 0;
    const value = offset ? renderedText + "\n\n" + review.text : review.text;
    const word = getVariable(review.text);
    const element = textareaRef.current;
    const initiator =
      typeof document === "undefined" ? null : document.activeElement;
    setMessageValue(value);
    if (element)
      selection.current = {
        element,
        initiator,
        review,
        value,
        inputRevision: form.getRevision(),
        range: word
          ? { ...word, start: word.start + offset, end: word.end + offset }
          : null,
      };
    setReview((current) => (current === review ? null : current));
  };
  React.useLayoutEffect(() => {
    const pending = selection.current;
    selection.current = null;
    if (!pending) return;
    const current = useStoreMessageOption.getState();
    const active =
      typeof document === "undefined" ? null : document.activeElement;
    const ownsFocus =
      active === pending.initiator ||
      (typeof document !== "undefined" &&
        active === document.body &&
        pending.initiator !== null &&
        !pending.initiator.isConnected);
    if (
      !mounted.current ||
      !view.current.ready ||
      view.current.destinationKey !== pending.review.destinationKey ||
      !pending.review.isOwned() ||
      current.quickPromptRevision !== pending.review.commandRevision ||
      current.selectedQuickPrompt !== null ||
      form.getRevision() !== pending.inputRevision ||
      form.getValues().message !== pending.value ||
      textareaRef.current !== pending.element ||
      pending.element.value !== pending.value ||
      !ownsFocus
    )
      return;
    pending.element.focus();
    if (pending.range)
      pending.element.setSelectionRange(pending.range.start, pending.range.end);
  });

  return {
    promptReview: reviewIsCurrent ? review!.text : null,
    overwrite: () => accept(false),
    append: () => accept(true),
    cancel: () => {
      if (!isCurrent() || !review || !inputIsCurrent()) return;
      review.handled = true;
      setReview((current) => (current === review ? null : current));
    },
  };
};

export const QuickPromptInsertionReview = (props: Props) => {
  const { t } = useTranslation(["option", "common"]);
  const insertion = useQuickPromptInsertion(props);
  const id = React.useId();
  if (insertion.promptReview === null) return null;
  return (
    <section
      aria-labelledby={id}
      className="mb-2 w-full min-w-0 rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
    >
      <h3 id={id} className="font-medium">
        {t("option:promptInsert.confirmTitle", "Use prompt in chat?")}
      </h3>
      <p className="text-text-muted">
        {t(
          "option:promptInsert.reviewDescription",
          "Your draft remains here. Review this prompt, then overwrite your message, append it, or cancel.",
        )}
      </p>
      <pre
        className="my-2 max-h-24 overflow-auto whitespace-pre-wrap break-words font-sans"
        aria-label={t("option:promptInsert.preview", "Prompt preview")}
      >
        {insertion.promptReview}
      </pre>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={insertion.overwrite}
          className="min-h-11 rounded border border-border px-3 py-1 focus-visible:ring-2 focus-visible:ring-focus"
        >
          {t("option:promptInsert.overwrite", "Overwrite message")}
        </button>
        <button
          type="button"
          onClick={insertion.append}
          className="min-h-11 rounded border border-border px-3 py-1 focus-visible:ring-2 focus-visible:ring-focus"
        >
          {t("option:promptInsert.append", "Append")}
        </button>
        <button
          type="button"
          onClick={insertion.cancel}
          className="min-h-11 rounded border border-border px-3 py-1 focus-visible:ring-2 focus-visible:ring-focus"
        >
          {t("common:cancel", "Cancel")}
        </button>
      </div>
    </section>
  );
};
