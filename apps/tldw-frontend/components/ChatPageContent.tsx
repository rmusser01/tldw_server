import Head from 'next/head';
import { useActiveChatTitle } from '@/hooks/useActiveChatTitle';

import ChatRoute from '@/routes/option-chat';

export default function ChatPageContent() {
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
