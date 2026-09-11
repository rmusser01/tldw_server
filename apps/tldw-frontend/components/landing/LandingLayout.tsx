import React from "react"
import Head from "next/head"
import Link from "next/link"
import { cn } from "@web/lib/utils"

interface LandingLayoutProps {
  children: React.ReactNode
  title: string
  description: string
  segment: "journalists" | "researchers" | "osint"
}

export function LandingLayout({ children, title, description, segment }: LandingLayoutProps) {
  return (
    <>
      <Head>
        <title>{title} | tldw</title>
        <meta name="description" content={description} />
        <meta property="og:title" content={title} />
        <meta property="og:description" content={description} />
      </Head>

      <div className="min-h-screen bg-bg text-text">
        {/* Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-bg/80 backdrop-blur-md border-b border-border">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <Link href="/" className="text-xl font-bold text-primary">
              tldw
            </Link>
            <div className="flex items-center gap-6">
              <Link href="/for/journalists" className={cn(
                "text-sm hover:text-primary transition-colors",
                segment === "journalists" && "text-primary font-medium"
              )}>
                Journalists
              </Link>
              <Link href="/research" className={cn(
                "text-sm hover:text-primary transition-colors",
                segment === "researchers" && "text-primary font-medium"
              )}>
                Researchers
              </Link>
              <Link href="/for/osint" className={cn(
                "text-sm hover:text-primary transition-colors",
                segment === "osint" && "text-primary font-medium"
              )}>
                OSINT
              </Link>
              <a
                href="https://github.com/rmusser01/tldw_server"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm hover:text-primary transition-colors"
              >
                GitHub
              </a>
              <Link
                href="/login"
                className="px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                Get Started
              </Link>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="pt-16">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-border py-12 mt-24">
          <div className="max-w-6xl mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div>
                <h4 className="font-bold mb-4">tldw</h4>
                <p className="text-sm text-text-muted">
                  Self-hosted transcription and knowledge management.
                  Your data, your infrastructure.
                </p>
              </div>
              <div>
                <h4 className="font-medium mb-4">Product</h4>
                <ul className="space-y-2 text-sm text-text-muted">
                  <li><Link href="/features" className="hover:text-primary">Features</Link></li>
                  <li><Link href="/pricing" className="hover:text-primary">Pricing</Link></li>
                  <li><Link href="/docs" className="hover:text-primary">Documentation</Link></li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-4">Use Cases</h4>
                <ul className="space-y-2 text-sm text-text-muted">
                  <li><Link href="/for/journalists" className="hover:text-primary">Journalists</Link></li>
                  <li><Link href="/research" className="hover:text-primary">Researchers</Link></li>
                  <li><Link href="/for/osint" className="hover:text-primary">OSINT</Link></li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-4">Source & Community</h4>
                <ul className="space-y-2 text-sm text-text-muted">
                  <li><a href="https://github.com/rmusser01/tldw_server" className="hover:text-primary">GitHub</a></li>
                  <li><Link href="/docs/self-hosting" className="hover:text-primary">Self-Hosting Guide</Link></li>
                  <li><Link href="/docs/contributing" className="hover:text-primary">Contributing</Link></li>
                </ul>
              </div>
            </div>
            <div className="mt-12 pt-8 border-t border-border text-sm text-text-muted">
              <p>Server GPL-3.0-only. Frontend source-available. No telemetry. No data collection.</p>
            </div>
          </div>
        </footer>
      </div>
    </>
  )
}
