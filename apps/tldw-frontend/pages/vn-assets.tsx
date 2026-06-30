import dynamic from 'next/dynamic';

export default dynamic(() => import('@web/components/vn-assets/VNAssetsWorkbench'), { ssr: false });
