import React from "react";
import { useStorage } from "@plasmohq/storage/hook";

type LogE2EDebug = (
  key: "syncSystem" | "syncQuick" | "storeSystem" | "storeQuick",
  payload: Record<string, unknown>,
) => void;

const noopLogE2EDebug: LogE2EDebug = () => {};

/**
 * Syncs selectedSystemPrompt and selectedQuickPrompt between
 * Zustand store and browser storage with ref-based dedup.
 */
export const usePromptPersistence = ({
  selectedSystemPrompt,
  setSelectedSystemPrompt,
  selectedQuickPrompt,
  setSelectedQuickPrompt,
  logE2EDebug = noopLogE2EDebug,
}: {
  selectedSystemPrompt: string | null;
  setSelectedSystemPrompt: (prompt: string) => void;
  selectedQuickPrompt: string | null;
  setSelectedQuickPrompt: (prompt: string | null) => void;
  logE2EDebug?: LogE2EDebug;
}) => {
  const [storedSystemPrompt, setStoredSystemPrompt] = useStorage<string | null>(
    "selectedSystemPrompt",
    null,
  );
  const [storedQuickPrompt, setStoredQuickPrompt] = useStorage<string | null>(
    "selectedQuickPrompt",
    null,
  );
  const storedSystemPromptRef = React.useRef<string | null>(storedSystemPrompt);
  const storedQuickPromptRef = React.useRef<string | null>(storedQuickPrompt);
  const hydratedSystemPromptFromStorageRef = React.useRef(false);
  const hydratedQuickPromptFromStorageRef = React.useRef(false);

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
  }, [selectedSystemPrompt, setSelectedSystemPrompt, storedSystemPrompt, logE2EDebug]);

  React.useEffect(() => {
    if (
      !hydratedQuickPromptFromStorageRef.current &&
      storedQuickPrompt &&
      storedQuickPrompt === selectedQuickPrompt
    ) {
      hydratedQuickPromptFromStorageRef.current = true;
      storedQuickPromptRef.current = storedQuickPrompt;
      return;
    }

    if (
      !hydratedQuickPromptFromStorageRef.current &&
      storedQuickPrompt &&
      storedQuickPrompt !== selectedQuickPrompt
    ) {
      logE2EDebug("syncQuick", {
        storedQuickPrompt,
        selectedQuickPrompt,
      });
      storedQuickPromptRef.current = storedQuickPrompt;
      hydratedQuickPromptFromStorageRef.current = true;
      setSelectedQuickPrompt(storedQuickPrompt);
    }
  }, [selectedQuickPrompt, setSelectedQuickPrompt, storedQuickPrompt, logE2EDebug]);

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

  React.useEffect(() => {
    const nextValue = selectedQuickPrompt ?? null;
    if (nextValue === storedQuickPromptRef.current) {
      return;
    }
    logE2EDebug("storeQuick", {
      nextValue,
      storedQuickPromptRef: storedQuickPromptRef.current,
    });
    storedQuickPromptRef.current = nextValue;
    setStoredQuickPrompt(nextValue);
  }, [selectedQuickPrompt, setStoredQuickPrompt, logE2EDebug]);
};
