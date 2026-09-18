import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Characters = dynamic(() => import("@/routes/option-characters"), { ssr: false })

export default withPageTitle("Characters", Characters)
