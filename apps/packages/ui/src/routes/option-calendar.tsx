import OptionLayout from "@/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { CalendarPage } from "@/components/Option/Calendar/CalendarPage"

const OptionCalendar = () => (
  <RouteErrorBoundary routeId="calendar" routeLabel="Calendar">
    <OptionLayout>
      <CalendarPage />
    </OptionLayout>
  </RouteErrorBoundary>
)

export default OptionCalendar
