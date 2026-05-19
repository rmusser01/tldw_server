import OptionLayout from "@web/components/layout/WebLayout"
import SpeechPlaygroundPage from "@/components/Option/Speech/SpeechPlaygroundPage"

const OptionTts = () => {
  return (
    <OptionLayout>
      <SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />
    </OptionLayout>
  )
}

export default OptionTts
