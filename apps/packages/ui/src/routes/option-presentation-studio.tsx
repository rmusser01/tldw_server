import OptionLayout from "@/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { PresentationStudioIndex } from "@/components/Option/PresentationStudio/PresentationStudioIndex"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"

export default function OptionPresentationStudio() {
  const online = useServerOnline()
  const { capabilities, loading } = useServerCapabilities()

  let content = <PresentationStudioIndex />
  if (!online) {
    content = (
      <section className="rounded-lg border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Presentation Studio</h1>
        <p className="mt-2 text-sm text-text-muted">Server is offline. Connect to use Presentation Studio.</p>
      </section>
    )
  } else if (!loading && capabilities && !capabilities.hasPresentationStudio) {
    content = (
      <section className="rounded-lg border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Presentation Studio</h1>
        <p className="mt-2 text-sm text-text-muted">Presentation Studio is not available on this server.</p>
      </section>
    )
  }

  return (
    <RouteErrorBoundary routeId="presentation-studio" routeLabel="Presentation Studio">
      <OptionLayout>
        {content}
      </OptionLayout>
    </RouteErrorBoundary>
  )
}
