export type SharedWorkspaceRouteMode =
  | { kind: "local" }
  | { kind: "shared-invalid" }
  | { kind: "shared-valid"; shareId: number };

export const parseSharedWorkspaceRoute = (
  search: string,
): SharedWorkspaceRouteMode => {
  const values = new URLSearchParams(search).getAll("shared");
  if (values.length === 0) return { kind: "local" };
  if (values.length !== 1 || !/^[1-9][0-9]*$/.test(values[0])) {
    return { kind: "shared-invalid" };
  }

  const shareId = Number(values[0]);
  return Number.isSafeInteger(shareId)
    ? { kind: "shared-valid", shareId }
    : { kind: "shared-invalid" };
};
