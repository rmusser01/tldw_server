import React from "react";
import { getDocumentWorkerClient } from "@/services/document-editing/app-service";
import type { DocumentCheckScheduler } from "@/services/document-editing/scheduler";
import type {
  CheckResult,
  DocumentRequest,
  FormatResult,
} from "@/services/document-editing/types";

type Status =
  | CheckResult["state"]
  | "pending"
  | "idle"
  | "formatted"
  | "unchanged"
  | "blocked";
const identity = (request: DocumentRequest | null) =>
  request ? JSON.stringify(request.key) : null;
const authority = (request: DocumentRequest | null) =>
  request
    ? JSON.stringify([
        request.key.authority_id,
        request.key.document_id,
        request.key.session_generation,
      ])
    : null;

/** Diagnostics own only a transient lease, never the note's draft/save state. */
export function useDocumentDiagnostics(
  request: DocumentRequest | null,
  composing: boolean,
) {
  const current = React.useRef({ request, composing });
  current.current = { request, composing };
  const manualIdentity = React.useRef<string | null>(null);
  const lease = React.useRef<DocumentCheckScheduler | null>(null);
  const [outcome, setOutcome] = React.useState<{
    identity: string;
    result: CheckResult;
    status: Status;
    manual: boolean;
  } | null>(null);
  const scope = authority(request);
  const key = identity(request);
  const previous = React.useRef<DocumentRequest | null>(null);

  React.useLayoutEffect(() => {
    const request = current.current.request;
    if (!request || composing || !scope) return;
    const owned = getDocumentWorkerClient().acquire((result) => {
      if (
        lease.current === owned &&
        !current.current.composing &&
        identity(current.current.request) === JSON.stringify(result.key)
      ) {
        setOutcome({
          identity: JSON.stringify(result.key),
          result,
          status: result.state,
          manual: manualIdentity.current === JSON.stringify(result.key),
        });
      }
    });
    lease.current = owned;
    return () => {
      if (lease.current === owned) lease.current = null;
      owned.close();
    };
    // Source revisions share this lease; scope/composition restoration acquires anew.
  }, [scope, composing]);

  React.useLayoutEffect(() => {
    const request = current.current.request;
    const old = previous.current;
    if (identity(old) !== identity(request)) manualIdentity.current = null;
    previous.current = composing ? null : request;
    if (!request || composing) return;
    const immediate =
      authority(old) !== scope || old?.language !== request.language;
    lease.current?.submit(request, immediate);
  }, [key, scope, composing]);

  const checkNow = React.useCallback(() => {
    const value = current.current;
    if (!value.request || value.composing) return;
    setOutcome(null);
    manualIdentity.current = identity(value.request);
    lease.current?.submit(value.request, true);
  }, []);
  const format = React.useCallback(
    async (
      apply?: (result: FormatResult) => boolean,
    ): Promise<FormatResult | null> => {
      const value = current.current;
      const owned = lease.current;
      if (!value.request || value.composing || !owned) return null;
      setOutcome(null);
      const result = await owned.format(value.request);
      if (
        !result ||
        owned !== lease.current ||
        current.current.composing ||
        identity(current.current.request) !== JSON.stringify(result.key)
      )
        return null;
      const accepted =
        !["formatted", "unchanged"].includes(result.state) ||
        !apply ||
        apply(result);
      setOutcome({
        identity: JSON.stringify(result.key),
        result: result.check,
        status: accepted ? result.state : "blocked",
        manual: true,
      });
      return accepted ? result : null;
    },
    [],
  );
  const visible =
    outcome?.identity === key && request && !composing ? outcome : null;
  return {
    status: visible?.status ?? ((request ? "pending" : "idle") as Status),
    announce: visible?.manual ?? false,
    diagnostics: visible?.result.diagnostics ?? [],
    result: visible?.result ?? null,
    format,
    checkNow,
  };
}
