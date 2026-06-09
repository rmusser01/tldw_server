import OptionLayout from "@web/components/layout/WebLayout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ExplainerWorkspace } from "@/components/Option/Explainer/ExplainerWorkspace"

const OptionExplainer = () => {
  return (
    <RouteErrorBoundary routeId="explainer" routeLabel="Explainer">
      <OptionLayout>
        <PageShell
          className="flex h-full min-h-0 w-full flex-1 overflow-hidden"
          maxWidthClassName="max-w-full"
        >
          <ExplainerWorkspace />
        </PageShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionExplainer
