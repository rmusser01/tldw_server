import { useStoreMessageOption } from "@/store/option";

const normalizeCount = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.trunc(value))
    : 0;

export const getCockpitMessageCount = (
  messages: ReadonlyArray<unknown> | null | undefined,
  history: ReadonlyArray<unknown> | null | undefined,
  renderedCount?: number | null,
) =>
  Math.max(
    messages?.length ?? 0,
    history?.length ?? 0,
    normalizeCount(renderedCount),
  );

export const formatCockpitMessageCount = (
  value: unknown,
  count: number,
) => {
  const fallback = count === 1 ? "{{count}} message" : "{{count}} messages";
  const candidate = typeof value === "string" ? value.trim() : "";
  const embeddedCount = candidate.match(/\d+/)?.[0];
  const template =
    candidate.includes("{{count}}") ||
    (embeddedCount != null && Number(embeddedCount) === count)
      ? candidate
      : fallback;

  return template.replace(/\{\{\s*count\s*\}\}/g, String(count));
};

export const useCockpitMessageCount = (fallbackCount: number) => {
  const messageCount = useStoreMessageOption(
    (state) => state.messages?.length ?? 0,
  );
  const historyCount = useStoreMessageOption(
    (state) => state.history?.length ?? 0,
  );
  return Math.max(normalizeCount(fallbackCount), messageCount, historyCount);
};
