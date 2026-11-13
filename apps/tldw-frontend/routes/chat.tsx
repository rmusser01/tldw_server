import dynamic from 'next/dynamic';
import Head from 'next/head';

const ChatRoute = dynamic(() => import('@web/components/ChatPageContent'), { ssr: false });

export default function ChatPage() {
  return (
    <>
      <Head><title>Chat | tldw</title></Head>
      <ChatRoute />
    </>
  );
}
