import { NextResponse } from 'next/server';

const appVersion = process.env.npm_package_version ?? process.env.APP_VERSION ?? '0.0.0';

export async function GET(): Promise<NextResponse> {
  return NextResponse.json(
    {
      status: 'ok',
      timestamp: new Date().toISOString(),
      version: appVersion,
    },
    {
      status: 200,
      headers: { 'Cache-Control': 'no-store' },
    }
  );
}
