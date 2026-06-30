import dynamic from 'next/dynamic';

export default dynamic(() => import('@web/components/vn-play/VNPlayWorkspace'), { ssr: false });
