// These documented conflicts reject generation before replacement work is created.
export const VN_GENERATION_REJECTION_MESSAGES = {
  vn_asset_recipe_unavailable: 'Original generation settings are unavailable. Start generation to use current settings.',
  vn_asset_recipe_invalid: 'Original generation settings cannot be read. Start generation to use current settings.',
  vn_asset_recipe_slot_mismatch: 'This slot was not in the selected batch. Refresh generation status and retry the failed slot.',
  vn_asset_retry_source_unavailable: 'No failed generation batch is available for Retry. Refresh generation status or start generation.',
  vn_asset_retry_source_active: 'Original generation work is still queued or running. Wait for it to finish, then refresh generation status before Retry.',
  vn_asset_retry_override_conflict: 'Retry uses the original settings. Use Regenerate or Start generation for changed settings.',
  vn_asset_execution_recipe_invalid: 'The original backend selection cannot be read. Start generation to use current settings.',
} as const;

export type VNGenerationRejectionCode = keyof typeof VN_GENERATION_REJECTION_MESSAGES;

export function asVNGenerationRejectionCode(value: unknown): VNGenerationRejectionCode | undefined {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(VN_GENERATION_REJECTION_MESSAGES, value)
    ? value as VNGenerationRejectionCode : undefined;
}

export function isVNGenerationRejected(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const { status, errorCode } = error as { status?: number; errorCode?: string };
  return (status !== undefined && [403, 404, 422].includes(status)) ||
    (status === 409 && asVNGenerationRejectionCode(errorCode) !== undefined);
}
