import React from 'react';
import { Badge } from '@web/components/ui/Badge';
import type { VNAssetPromptPreview } from '@web/types/vn-assets';

export interface PromptPreviewProps {
  preview?: VNAssetPromptPreview | null;
}

export default function PromptPreview({ preview }: PromptPreviewProps) {
  if (!preview) {
    return (
      <div className="rounded-md bg-bg p-3 text-sm text-text-muted">
        Prompt preview will appear after a slot is selected.
      </div>
    );
  }

  const omittedEntries = Object.entries(preview.omitted_source_counts ?? {});
  const totalTokens = preview.token_estimates?.total;

  return (
    <div className="grid gap-3 rounded-md bg-bg p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="neutral">
          {typeof totalTokens === 'number' ? `${totalTokens} tokens` : 'No token estimate'}
        </Badge>
        {preview.warnings.map((warning) => (
          <Badge key={warning} variant="warning">
            {warning}
          </Badge>
        ))}
      </div>
      {omittedEntries.length > 0 && (
        <div className="flex flex-wrap gap-2 text-xs text-text-muted">
          {omittedEntries.map(([source, count]) => (
            <span key={source}>{source}: {count}</span>
          ))}
        </div>
      )}
      <div className="max-h-36 overflow-auto rounded-md bg-surface2 p-3 text-sm">
        <p>{preview.prompt}</p>
        {preview.negative_prompt && (
          <p className="mt-2 text-text-muted">{preview.negative_prompt}</p>
        )}
      </div>
    </div>
  );
}
