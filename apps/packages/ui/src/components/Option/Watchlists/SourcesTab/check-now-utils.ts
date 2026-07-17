export const normalizeSourceIds = (sourceIds: Array<number | string | null | undefined>): number[] =>
  Array.from(
    new Set(
      sourceIds
        .map((id) => Number(id))
        .filter((id) => Number.isInteger(id) && id > 0)
    )
  )

export const shouldConfirmMultiSourceCheck = (
  sourceIds: Array<number | string | null | undefined>
): boolean => normalizeSourceIds(sourceIds).length > 1
