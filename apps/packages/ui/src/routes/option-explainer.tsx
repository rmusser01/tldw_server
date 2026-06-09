import OptionLayout from "~/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ExplainerWorkspace } from "@/components/Option/Explainer/ExplainerWorkspace"

const OptionExplainer = () => {
  return (
    <OptionLayout>
      <RouteErrorBoundary routeId="explainer" routeLabel="Explainer">
        <PageShell
          className="flex h-full min-h-0 w-full flex-1 overflow-hidden"
          maxWidthClassName="max-w-full"
        >
          <ExplainerWorkspace />
        </PageShell>
      </RouteErrorBoundary>
    </OptionLayout>
  )
}

export default OptionExplainer
