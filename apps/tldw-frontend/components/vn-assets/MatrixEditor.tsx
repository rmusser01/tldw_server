import React, { useEffect, useMemo, useState } from 'react';
import { Grid2X2, Wand2 } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import PromptPreview from '@web/components/vn-assets/PromptPreview';
import type { VNAssetPromptPreview, VNAssetStarterMatrix } from '@web/types/vn-assets';

export interface MatrixEditorProps {
  matrix?: VNAssetStarterMatrix | null;
  selectedPackId?: number | null;
  isApplying?: boolean;
  promptPreview?: VNAssetPromptPreview | null;
  onApplyMatrix?: (matrixKey: string, overrides: Record<string, unknown>) => void;
}

function formatAssetTypes(assetTypes: string[]): string {
  if (assetTypes.length === 0) return 'No asset types';
  return assetTypes.join(', ');
}

function plannedAssetLabel(count: number): string {
  return `${count} planned ${count === 1 ? 'asset' : 'assets'}`;
}

export default function MatrixEditor({
  matrix,
  selectedPackId,
  isApplying = false,
  promptPreview,
  onApplyMatrix,
}: MatrixEditorProps) {
  const [variantCountValue, setVariantCountValue] = useState('1');

  useEffect(() => {
    setVariantCountValue('1');
  }, [matrix?.key]);

  const variantCount = useMemo(() => {
    const parsed = Number(variantCountValue);
    if (!Number.isFinite(parsed)) return 1;
    return Math.max(1, Math.floor(parsed));
  }, [variantCountValue]);

  const plannedAssets = matrix ? matrix.planned_output_count * variantCount : 0;

  const handleApply = () => {
    if (!matrix || !selectedPackId) return;
    onApplyMatrix?.(matrix.key, { variant_count: variantCount });
  };

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Grid2X2 aria-hidden className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Starter matrix</h2>
        </div>
        <Badge variant="info">{plannedAssetLabel(plannedAssets)}</Badge>
      </div>

      {matrix ? (
        <div className="grid gap-4">
          <div className="flex flex-wrap items-center gap-2 text-sm text-text-muted">
            <span>{matrix.slot_count} slots</span>
            <span aria-hidden>·</span>
            <span>{formatAssetTypes(matrix.asset_types)}</span>
          </div>
          <Input
            label="Variants per slot"
            min={1}
            type="number"
            value={variantCountValue}
            onChange={(event) => setVariantCountValue(event.target.value)}
          />
          <Button
            className="gap-2"
            disabled={!selectedPackId}
            loading={isApplying}
            onClick={handleApply}
            type="button"
            variant="secondary"
          >
            <Wand2 aria-hidden className="h-4 w-4" />
            Apply starter matrix
          </Button>
          <PromptPreview preview={promptPreview} />
        </div>
      ) : (
        <p className="text-sm text-text-muted">Starter matrix metadata is unavailable.</p>
      )}
    </section>
  );
}
