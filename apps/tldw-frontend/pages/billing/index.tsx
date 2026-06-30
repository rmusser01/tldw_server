import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function BillingPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Hosted Billing Lives In The Private Distribution"
      description="The OSS web client does not ship the hosted subscription and invoice surface. Self-host deployments should manage commercial billing outside this public frontend."
      plannedPath="/billing"
    />
  )
}
