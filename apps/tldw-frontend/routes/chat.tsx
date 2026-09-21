import dynamic from 'next/dynamic';
import Head from 'next/head';
import { useActiveChatTitle } from '@/hooks/useActiveChatTitle';

const ChatRoute = dynamic(() => import('@/routes/option-chat'), { ssr: false });

export default function ChatPage() {
  const { title } = useActiveChatTitle();
  return (
    <>
      <Head>
        <title>{title ? `${title} | tldw` : 'Chat | tldw'}</title>
      </Head>
      <ChatRoute />
    </>
  );
}
