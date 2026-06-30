import { HostedOnlyRoutePlaceholder } from "@web/components/navigation/HostedOnlyRoutePlaceholder"

export default function AccountPage() {
  return (
    <HostedOnlyRoutePlaceholder
      title="Hosted Account Pages Live In The Private Distribution"
      description="The OSS web client does not ship the hosted account surface. Self-host operators can manage users and auth through the local server and admin flows."
      plannedPath="/account"
    />
  )
}
