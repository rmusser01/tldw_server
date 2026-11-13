import { useCallback, useEffect, useRef, useState } from 'react';
import { tldwClient, type ScopedRequestOptions } from '@/services/tldw/TldwApiClient';
import type { ApiDataTableGenerateResponse } from '@/services/tldw/data-tables';
import {
  createServicePromptScopeLease,
  resolveServicePromptScope,
} from '@/services/service-prompts';
import { pollDataTableJob } from '@/utils/data-tables-jobs';
import type { DataTable } from '@/types/data-tables';

type Operation = {
  controller: AbortController;
  receipt?: Promise<ApiDataTableGenerateResponse>;
  options?: ScopedRequestOptions;
  cancelRequested: boolean;
  cancellation?: Promise<void>;
  release?: () => void;
};

/** Retain the start ACK so even an early Cancel can retire the exact captured-owner job. */
function cancelOperation(operation: Operation, acknowledged?: () => void): Promise<void> {
  if (operation.cancellation) return operation.cancellation;
  operation.cancelRequested = true;
  operation.cancellation = (async () => {
    if (operation.receipt) {
      const receipt = await operation.receipt;
      await tldwClient.cancelDataTableJob(receipt.job_id, operation.options);
    }
    acknowledged?.();
    operation.controller.abort();
  })().catch((error) => {
    operation.cancelRequested = false;
    operation.cancellation = undefined;
    throw error;
  });
  return operation.cancellation;
}

export function useDataTableGeneration(enabled = true, tableId?: string | null) {
  const active = useRef<Operation | null>(null);
  const [isGenerating, setGenerating] = useState(false);
  const [isCancelling, setCancelling] = useState(false);
  const retire = useCallback(() => {
    const operation = active.current;
    active.current = null;
    if (!operation) return;
    operation.controller.abort();
    operation.release?.();
    void cancelOperation(operation).catch(() => {
      console.error('Data table cleanup could not cancel its captured-owner job');
    });
  }, []);
  useEffect(() => {
    if (!enabled) retire();
    setGenerating(false);
    setCancelling(false);
    return retire;
  }, [enabled, tableId, retire]);

  const run = useCallback(
    async (
      start: (options: ScopedRequestOptions) => Promise<ApiDataTableGenerateResponse>,
      rowsLimit: number,
      callbacks: { onComplete: (table: DataTable) => void; onError: (error: unknown) => void }
    ): Promise<void> => {
      if (active.current) return;
      const operation: Operation = { controller: new AbortController(), cancelRequested: false };
      active.current = operation;
      setGenerating(true);
      const lease = createServicePromptScopeLease(operation.controller.signal);
      const scopeAborted = () => {
        if (active.current !== operation) return;
        retire();
        setGenerating(false);
        setCancelling(false);
      };
      operation.release = () => {
        lease.signal.removeEventListener('abort', scopeAborted);
        lease.release();
      };
      lease.signal.addEventListener('abort', scopeAborted, { once: true });
      let accepted = false;
      let terminal = false;
      let keepForCancellation = false;
      try {
        const scope = await resolveServicePromptScope({ signal: lease.signal });
        if (active.current !== operation || operation.cancelRequested || lease.signal.aborted)
          return;
        if (scope.clientPrincipalVerified === false)
          throw new Error('Authenticated account is unresolved');
        lease.bind(scope);
        operation.options = { requestScope: { config: scope.config, userId: scope.userId } };
        // Do not abort the start response: its ACK identifies the server work to cancel.
        operation.receipt = start(operation.options);
        const receipt = await operation.receipt;
        if (!receipt.job_id || !receipt.table?.uuid)
          throw new Error('Generation job was not created');
        accepted = true;
        if (operation.cancellation) await operation.cancellation.catch(() => {});
        if (active.current !== operation || operation.cancelRequested || lease.signal.aborted)
          return;
        const status = await pollDataTableJob({
          jobId: receipt.job_id,
          fetchJob: (id) =>
            tldwClient.getDataTableJob(id, {
              ...operation.options,
              signal: lease.signal,
            }),
          signal: lease.signal,
        });
        if (operation.cancellation) await operation.cancellation.catch(() => {});
        if (active.current !== operation || operation.cancelRequested || lease.signal.aborted)
          return;
        terminal = ['completed', 'failed', 'cancelled'].includes(
          String(status.status).toLowerCase()
        );
        if (String(status.status).toLowerCase() !== 'completed')
          throw new Error(status.error_message || 'Generation failed');
        const publication = status.result?.publication_status;
        if (publication && publication !== 'ready')
          throw new Error(publication === 'conflict'
            ? 'Your saved table changed. Your edits were preserved. Generate again to use the latest version.'
            : publication === 'pending'
              ? 'Generation finished, but saving is not confirmed yet. Reopen the table to check its saved content.'
              : 'The generated result could not be saved. Your saved table is unchanged. Try generating again.');
        const table = await tldwClient.getDataTable(
          status.table_uuid || receipt.table.uuid,
          { rows_limit: rowsLimit },
          { ...operation.options, signal: lease.signal }
        );
        if (operation.cancellation) await operation.cancellation.catch(() => {});
        if (active.current !== operation || operation.cancelRequested || lease.signal.aborted)
          return;
        if (!table) throw new Error('No table data in response');
        callbacks.onComplete(table);
      } catch (error) {
        if (operation.cancellation) await operation.cancellation.catch(() => {});
        if (active.current !== operation || operation.cancelRequested || lease.signal.aborted)
          return;
        keepForCancellation = accepted && !terminal;
        callbacks.onError(error);
      } finally {
        if (!keepForCancellation) operation.release();
        if (active.current === operation && !keepForCancellation) {
          active.current = null;
          setGenerating(false);
          setCancelling(false);
        }
      }
    },
    [retire]
  );

  const cancel = useCallback(async (onCancelled: () => void) => {
    const operation = active.current;
    if (!operation) return;
    setCancelling(true);
    try {
      await cancelOperation(operation, () => {
        if (active.current === operation) {
          onCancelled();
          active.current = null;
          operation.controller.abort();
          operation.release?.();
          setGenerating(false);
          setCancelling(false);
        }
      });
    } catch (error) {
      if (active.current === operation) throw error;
    } finally {
      if (active.current === operation) setCancelling(false);
    }
  }, []);
  return { run, cancel, isGenerating, isCancelling };
}
