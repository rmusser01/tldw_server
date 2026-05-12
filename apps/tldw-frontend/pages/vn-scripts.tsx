import dynamic from 'next/dynamic';

export default dynamic(() => import('@web/components/vn-scripts/VNScriptsWorkbench'), {
  ssr: false,
});
