import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Prompts = dynamic(() => import("@/routes/option-prompts"), { ssr: false })

export default withPageTitle("Prompts", Prompts)
