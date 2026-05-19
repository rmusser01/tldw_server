'use client';

import { useEffect } from 'react';
import * as Sentry from '@sentry/nextjs';

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  useEffect(() => {
    console.error('Global error:', error);
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en">
      <head>
        <style dangerouslySetInnerHTML={{ __html: `
          .ge-page { min-height:100vh; display:flex; align-items:center; justify-content:center; background:#f8f9fa; padding:1rem; font-family:system-ui,-apple-system,sans-serif; color:#111827; }
          .ge-card { max-width:28rem; width:100%; background:white; border-radius:0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.1); padding:2rem; text-align:center; }
          .ge-icon { width:3rem; height:3rem; margin:0 auto 1rem; border-radius:50%; background:#fee2e2; display:flex; align-items:center; justify-content:center; }
          .ge-title { font-size:1.25rem; font-weight:600; margin-bottom:0.5rem; }
          .ge-desc { color:#6b7280; margin-bottom:1.5rem; font-size:0.875rem; }
          .ge-detail { background:#f3f4f6; border-radius:0.375rem; padding:0.75rem; margin-bottom:1.5rem; }
          .ge-error { color:#dc2626; font-size:0.875rem; font-weight:500; }
          .ge-digest { color:#9ca3af; font-size:0.75rem; margin-top:0.25rem; }
          .ge-actions { display:flex; gap:0.5rem; }
          .ge-btn { flex:1; padding:0.5rem 1rem; border-radius:0.375rem; cursor:pointer; font-size:0.875rem; font-weight:500; border:none; }
          .ge-btn-primary { background:#3b82f6; color:white; }
          .ge-btn-secondary { background:white; color:#374151; border:1px solid #d1d5db; }
          @media(prefers-color-scheme:dark){
            .ge-page { background:#1a1a2e; color:#e0e0e0; }
            .ge-card { background:#2d2d44; box-shadow:0 1px 3px rgba(0,0,0,0.3); }
            .ge-icon { background:#5b2121; }
            .ge-desc { color:#a0a0b0; }
            .ge-detail { background:#3a3a55; }
            .ge-error { color:#f87171; }
            .ge-digest { color:#8888a0; }
            .ge-btn-primary { background:#3b82f6; color:white; }
            .ge-btn-secondary { background:#3a3a55; color:#e0e0e0; border:1px solid #555570; }
          }
        ` }} />
      </head>
      <body>
        <div className="ge-page">
          <div className="ge-card">
            <div className="ge-icon">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#dc2626"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                <path d="M12 9v4" />
                <path d="M12 17h.01" />
              </svg>
            </div>

            <h1 className="ge-title">Critical Error</h1>
            <p className="ge-desc">
              A critical error occurred. Please reload the page.
            </p>

            <div className="ge-detail">
              <p className="ge-error">
                {error.message || 'Unknown error'}
              </p>
              {error.digest && (
                <p className="ge-digest">
                  Error ID: {error.digest}
                </p>
              )}
            </div>

            <div className="ge-actions">
              <button onClick={reset} className="ge-btn ge-btn-primary">
                Try Again
              </button>
              <button onClick={() => (window.location.href = '/')} className="ge-btn ge-btn-secondary">
                Go Home
              </button>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
