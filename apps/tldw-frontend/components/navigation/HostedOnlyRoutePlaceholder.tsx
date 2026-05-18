import React from 'react';
import { isHostedTldwDeployment } from '@/services/tldw/deployment-mode';
import { RoutePlaceholder, type RoutePlaceholderProps } from './RoutePlaceholder';

type HostedOnlyRoutePlaceholderProps = Omit<
  RoutePlaceholderProps,
  'primaryCtaHref' | 'primaryCtaLabel'
>;

export const HostedOnlyRoutePlaceholder: React.FC<HostedOnlyRoutePlaceholderProps> = (props) => {
  const hostedMode = isHostedTldwDeployment();

  return (
    <RoutePlaceholder
      {...props}
      primaryCtaHref={hostedMode ? '/login' : '/settings/tldw'}
      primaryCtaLabel={hostedMode ? 'Open Login' : 'Open Local Auth Settings'}
    />
  );
};

export default HostedOnlyRoutePlaceholder;
