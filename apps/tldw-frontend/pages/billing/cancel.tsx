import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function BillingCancelPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Hosted Billing Redirects Live In The Private Distribution"
      description="The hosted checkout cancel route is not part of the OSS web client."
      plannedPath="/billing/cancel"
    />
  )
}
