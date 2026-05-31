import { RouteRedirect } from '@web/components/navigation/RouteRedirect';

export default function PrivilegesRedirectPage() {
  return (
    <RouteRedirect
      to="/settings"
      title="Privileges moved to settings"
      description="Role and permission controls now live in the settings area."
    />
  );
}
