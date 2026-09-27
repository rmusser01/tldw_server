import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Media = dynamic(() => import("@/routes/option-media"), { ssr: false })

export default withPageTitle("Media", Media)
