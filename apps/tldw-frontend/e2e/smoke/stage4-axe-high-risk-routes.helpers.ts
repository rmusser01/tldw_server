export type RedirectDisposition =
  | {
      shouldSkip: false;
    }
  | {
      shouldSkip: true;
      message: string;
    };

type RedirectDispositionInput = {
  routePath: string;
  finalPath: string;
  mayRedirectWhenUnavailable?: boolean;
  navigationObservedDuringScan?: boolean;
};

export type Stage4HighRiskRoute = {
  path: string;
  name: string;
  rationale: string;
  acceptablePaths?: string[];
  requiresSeededAuth?: boolean;
  mayRedirectWhenUnavailable?: boolean;
};

type RouteMetadataLookup = (path: string) =>
  | {
      smoke: 'include' | 'exclude' | 'manual';
      surface: string;
      availability: string[];
    }
  | undefined;

const delegatedRouteSurfaces = new Set(['legacy_alias', 'redirect', 'deprecated']);

export function getStage4HighRiskRouteGovernanceProblems(
  routes: Stage4HighRiskRoute[],
  getRouteMetadata: RouteMetadataLookup
): string[] {
  return routes.flatMap((route) => {
    const problems: string[] = [];
    const metadata = getRouteMetadata(route.path);

    if (!metadata) {
      problems.push(`${route.path} is missing route metadata`);
    } else {
      if (!metadata.availability.includes('web')) {
        problems.push(`${route.path} metadata does not include web availability`);
      }

      if (metadata.smoke === 'exclude') {
        problems.push(`${route.path} metadata is excluded from smoke coverage`);
      }

      if (delegatedRouteSurfaces.has(metadata.surface)) {
        problems.push(`${route.path} metadata delegates ownership to ${metadata.surface}`);
      }
    }

    if (route.rationale.trim().length < 20) {
      problems.push(`${route.path} is missing an explicit Stage 4 Axe rationale`);
    }

    return problems;
  });
}

export function getRedirectDispositionForA11yScan({
  routePath,
  finalPath,
  mayRedirectWhenUnavailable,
  navigationObservedDuringScan = false,
}: RedirectDispositionInput): RedirectDisposition {
  if (!mayRedirectWhenUnavailable) {
    return { shouldSkip: false };
  }

  if (!navigationObservedDuringScan && finalPath === routePath) {
    return { shouldSkip: false };
  }

  if (finalPath !== routePath) {
    return {
      shouldSkip: true,
      message: `Route ${routePath} redirected to ${finalPath}; feature is unavailable in this runtime`,
    };
  }

  return {
    shouldSkip: true,
    message: `Route ${routePath} reloaded during accessibility scan; feature is unavailable in this runtime`,
  };
}
