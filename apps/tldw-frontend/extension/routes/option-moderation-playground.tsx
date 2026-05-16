import { Navigate } from "react-router-dom"
import { MODERATION_RULES_PATH } from "@/routes/route-paths"

const OptionModerationPlayground = () => {
  return <Navigate to={MODERATION_RULES_PATH} replace />
}

export default OptionModerationPlayground
