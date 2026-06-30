import OptionLayout from "@web/components/layout/WebLayout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ModerationReviewShell } from "@/components/Option/ModerationReview"

const OptionModerationReview = () => {
  return (
    <RouteErrorBoundary routeId="moderation-review" routeLabel="Moderation Review">
      <OptionLayout>
        <PageShell className="py-6" maxWidthClassName="max-w-7xl">
          <ModerationReviewShell />
        </PageShell>
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionModerationReview
