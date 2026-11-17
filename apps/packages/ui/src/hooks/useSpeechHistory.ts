import { useCallback, type SetStateAction } from "react";
import { useStorage } from "@plasmohq/storage/hook";
import { useChatDraftOwner } from "@/hooks/useChatDraftOwner";

type OwnedHistory<T> = { ownerKey: string; items: T[] };

export function useSpeechHistory<T>(onBoundary: () => void, storageKey = "speechPlaygroundHistory") {
  const { ownerKey, isCurrent } = useChatDraftOwner(onBoundary);
  // Legacy rows have no proven owner; preserve them without assigning them to a login.
  const [stored, setStored] = useStorage<OwnedHistory<T> | null>(
    `${storageKey}:${ownerKey || "unresolved"}`,
    null,
  );
  const items =
    ownerKey &&
    isCurrent() &&
    stored?.ownerKey === ownerKey &&
    Array.isArray(stored.items)
      ? stored.items
      : [];
  const setItems = useCallback(
    async (next: SetStateAction<T[]>) => {
      if (!ownerKey || !isCurrent()) return;
      await setStored((previous) => {
        const priorItems =
          previous?.ownerKey === ownerKey && Array.isArray(previous.items)
            ? previous.items
            : [];
        return {
          ownerKey,
          items: typeof next === "function" ? next(priorItems) : next,
        };
      });
    },
    [ownerKey, isCurrent, setStored],
  );
  return [items, setItems, { ownerKey, isCurrent }] as const;
}
