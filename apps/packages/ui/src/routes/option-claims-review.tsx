import OptionLayout from "~/components/Layouts/Layout"
import { PageShell } from "@/components/Common/PageShell"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ClaimsReviewShell } from "@/components/Option/ClaimsReview/ClaimsReviewShell"

const OptionClaimsReview = () => (
  <RouteErrorBoundary routeId="claims-review" routeLabel="Claims Review">
    <OptionLayout>
      <PageShell className="py-6" maxWidthClassName="max-w-7xl">
        <ClaimsReviewShell />
      </PageShell>
    </OptionLayout>
  </RouteErrorBoundary>
)
export default OptionClaimsReview
