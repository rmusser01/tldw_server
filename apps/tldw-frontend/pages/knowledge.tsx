import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Knowledge = dynamic(() => import("@/routes/option-knowledge"), { ssr: false })

export default withPageTitle("Knowledge", Knowledge)
