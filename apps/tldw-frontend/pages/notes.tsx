import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Notes = dynamic(() => import("@/routes/option-notes"), { ssr: false })

export default withPageTitle("Notes", Notes)
