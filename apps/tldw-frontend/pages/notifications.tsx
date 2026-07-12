import dynamic from 'next/dynamic'

export default dynamic(() => import('@web/components/notifications/NotificationsRoute'), { ssr: false })
