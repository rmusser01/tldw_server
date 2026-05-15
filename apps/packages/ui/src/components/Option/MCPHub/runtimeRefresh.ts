export const MCP_RUNTIME_QUERY_FAMILIES = [
  ["mcp-tools"],
  ["mcp-tool-catalogs"],
  ["mcp-tool-modules"],
  ["mcp-health"]
] as const

type RuntimeQueryClient = {
  invalidateQueries: (input: { queryKey: readonly string[] }) => Promise<unknown> | unknown
}

export const invalidateMcpRuntimeQueries = async (
  queryClient: RuntimeQueryClient
) => {
  await Promise.all(
    MCP_RUNTIME_QUERY_FAMILIES.map((queryKey) =>
      queryClient.invalidateQueries({ queryKey })
    )
  )
}
