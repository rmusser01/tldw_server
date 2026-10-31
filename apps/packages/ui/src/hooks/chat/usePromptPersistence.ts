import React from "react";
import { useStorage } from "@plasmohq/storage/hook";

type LogE2EDebug = (
  key: "syncSystem" | "storeSystem",
  payload: Record<string, unknown>,
) => void;

const noopLogE2EDebug: LogE2EDebug = () => {};

/**
 * Syncs the selected system prompt between
 * Zustand store and browser storage with ref-based dedup.
 */
export const usePromptPersistence = ({
  selectedSystemPrompt,
  setSelectedSystemPrompt,
  logE2EDebug = noopLogE2EDebug,
}: {
  selectedSystemPrompt: string | null;
  setSelectedSystemPrompt: (prompt: string) => void;
  logE2EDebug?: LogE2EDebug;
}) => {
  const [storedSystemPrompt, setStoredSystemPrompt] = useStorage<string | null>(
    "selectedSystemPrompt",
    null,
  );
  const storedSystemPromptRef = React.useRef<string | null>(storedSystemPrompt);
  const hydratedSystemPromptFromStorageRef = React.useRef(false);

  // Storage → Zustand sync
  React.useEffect(() => {
    if (
      !hydratedSystemPromptFromStorageRef.current &&
      storedSystemPrompt &&
      storedSystemPrompt === selectedSystemPrompt
    ) {
      hydratedSystemPromptFromStorageRef.current = true;
      storedSystemPromptRef.current = storedSystemPrompt;
      return;
    }

    if (
      !hydratedSystemPromptFromStorageRef.current &&
      storedSystemPrompt &&
      storedSystemPrompt !== selectedSystemPrompt
    ) {
      logE2EDebug("syncSystem", {
        storedSystemPrompt,
        selectedSystemPrompt,
      });
      storedSystemPromptRef.current = storedSystemPrompt;
      hydratedSystemPromptFromStorageRef.current = true;
      setSelectedSystemPrompt(storedSystemPrompt);
    }
  }, [
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    storedSystemPrompt,
    logE2EDebug,
  ]);

  // Zustand → Storage sync
  React.useEffect(() => {
    const nextValue = selectedSystemPrompt ?? null;
    if (nextValue === storedSystemPromptRef.current) {
      return;
    }
    logE2EDebug("storeSystem", {
      nextValue,
      storedSystemPromptRef: storedSystemPromptRef.current,
    });
    storedSystemPromptRef.current = nextValue;
    setStoredSystemPrompt(nextValue);
  }, [selectedSystemPrompt, setStoredSystemPrompt, logE2EDebug]);
};
