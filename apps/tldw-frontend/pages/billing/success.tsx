import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function BillingSuccessPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Hosted Billing Redirects Live In The Private Distribution"
      description="The hosted checkout success route is not part of the OSS web client."
      plannedPath="/billing/success"
    />
  )
}
