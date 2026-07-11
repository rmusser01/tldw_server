import type { NextApiRequest, NextApiResponse } from 'next';
import { getDeploymentMode, resolveRuntimeAuthPolicy } from './runtime-auth-policy';

type RuntimeConfigResponse = {
  runtimeAuth:
    | {
        available: true;
        authMode: 'single-user';
        transport: 'cookie-session';
      }
    | {
        available: false;
      };
  networking: {
    deploymentMode: string;
    serverUrl: string;
  };
};

const unavailable = (): RuntimeConfigResponse => ({
  runtimeAuth: { available: false },
  networking: {
    deploymentMode: getDeploymentMode(),
    serverUrl: '',
  },
});

export default function handler(req: NextApiRequest, res: NextApiResponse<RuntimeConfigResponse>) {
  res.setHeader('Cache-Control', 'no-store, max-age=0');

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.status(405).end();
    return;
  }

  const policy = resolveRuntimeAuthPolicy(req);
  if (!policy.available) {
    res.status(200).json(unavailable());
    return;
  }

  res.status(200).json({
    runtimeAuth: {
      available: true,
      authMode: 'single-user',
      transport: 'cookie-session',
    },
    networking: {
      deploymentMode: getDeploymentMode(),
      serverUrl: '',
    },
  });
}
