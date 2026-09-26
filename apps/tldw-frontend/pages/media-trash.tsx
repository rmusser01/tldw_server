import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const MediaTrash = dynamic(() => import("@/routes/option-media-trash"), { ssr: false })

export default withPageTitle("Trash", MediaTrash)
