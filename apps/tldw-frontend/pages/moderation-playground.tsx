import { RouteRedirect } from "@web/components/navigation/RouteRedirect"

export default function ModerationPlaygroundRedirectPage() {
  return (
    <RouteRedirect
      to="/moderation/rules"
      title="Moderation Playground has moved"
      description="Content Rules now contains moderation policy, blocklist, override, and testing configuration."
    />
  )
}
