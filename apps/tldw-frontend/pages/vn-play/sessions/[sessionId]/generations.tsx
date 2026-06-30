import dynamic from 'next/dynamic';
import { useRouter } from 'next/router';

const VNPlayWorkspace = dynamic(() => import('@web/components/vn-play/VNPlayWorkspace'), { ssr: false });

function parseSessionId(value: string | string[] | undefined): number | undefined {
  const raw = Array.isArray(value) ? value[0] : value;
  if (!raw) return undefined;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export default function VNPlayGenerationInspectorRoute() {
  const router = useRouter();
  const sessionId = parseSessionId(router.query.sessionId);

  return <VNPlayWorkspace generationInspectorRoute initialSessionId={sessionId} />;
}
