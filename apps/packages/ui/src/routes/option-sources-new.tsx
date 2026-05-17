import OptionLayout from "@/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { SourceForm } from "@/components/Option/Sources/SourceForm"
import { SourcesAvailabilityGate } from "@/components/Option/Sources/SourcesAvailabilityGate"
import { useSearchParams } from "react-router-dom"

export default function OptionSourcesNew() {
  const [searchParams] = useSearchParams()
  const preset = searchParams.get("preset")
  const sourceFormPreset = preset === "notes-folder-sync" ? preset : undefined

  return (
    <RouteErrorBoundary routeId="sources-new" routeLabel="Sources">
      <OptionLayout>
        <SourcesAvailabilityGate maxWidthClassName="max-w-4xl">
          <PageShell className="py-6" maxWidthClassName="max-w-4xl">
            <SourceForm mode="create" preset={sourceFormPreset} />
          </PageShell>
        </SourcesAvailabilityGate>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}
