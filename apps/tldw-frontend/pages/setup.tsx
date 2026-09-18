import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Setup = dynamic(() => import("@/routes/option-setup"), { ssr: false })

export default withPageTitle("Setup", Setup)
