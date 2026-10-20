import OptionLayout from "@web/components/layout/WebLayout"
import { RouteLeaveGuard } from "@/entries/shared/route-leave-guard"
import ContentReviewPage from "@/components/ContentReview/ContentReviewPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionContentReview = () => {
  return (
    <RouteErrorBoundary routeId="content-review" routeLabel="Content Review">
      <OptionLayout>
        <ContentReviewPage LeaveGuard={RouteLeaveGuard} />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionContentReview
