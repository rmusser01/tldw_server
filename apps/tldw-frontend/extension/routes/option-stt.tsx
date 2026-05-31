import OptionLayout from "@web/components/layout/WebLayout"
import SttPlaygroundPage from "@/components/Option/STT/SttPlaygroundPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionStt = () => {
  return (
    <RouteErrorBoundary routeId="stt" routeLabel="STT Playground">
      <OptionLayout>
        <SttPlaygroundPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionStt
