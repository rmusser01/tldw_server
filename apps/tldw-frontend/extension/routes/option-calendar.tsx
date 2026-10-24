import OptionLayout from "@web/components/layout/WebLayout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { CalendarPage } from "@/components/Option/Calendar/CalendarPage"

const OptionCalendar = () => {
  return (
    <RouteErrorBoundary routeId="calendar" routeLabel="Calendar">
      <OptionLayout>
        <CalendarPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionCalendar
