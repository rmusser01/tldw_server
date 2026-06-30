import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { CompatibilityRedirect } from "@/components/Option/AudioStudio/CompatibilityRedirect"

const OptionAudiobookStudio = () => {
  return (
    <RouteErrorBoundary routeId="audiobook-studio" routeLabel="Audiobook Studio">
      <OptionLayout>
        <CompatibilityRedirect />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAudiobookStudio
