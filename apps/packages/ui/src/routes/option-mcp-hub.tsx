import OptionLayout from "~/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { McpHubPage } from "@/components/Option/MCPHub"

const OptionMcpHub = () => {
  return (
    <RouteErrorBoundary routeId="mcp-hub" routeLabel="MCP Hub">
      <OptionLayout>
        <PageShell className="flex-1 min-h-0" maxWidthClassName="max-w-full">
          <McpHubPage />
        </PageShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionMcpHub
