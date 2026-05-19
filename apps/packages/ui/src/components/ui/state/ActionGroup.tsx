import React from "react";
import { Button } from "@/components/Common/Button";
import { cn } from "@/libs/utils";

export interface StateAction {
  label: React.ReactNode;
  ariaLabel?: string;
  onClick?: () => void;
  loading?: boolean;
  disabled?: boolean;
  ref?: React.Ref<HTMLButtonElement>;
  "data-testid"?: string;
}

export interface ActionGroupProps {
  primaryAction?: StateAction;
  secondaryActions?: StateAction[];
  className?: string;
  "data-testid"?: string;
}

export function ActionGroup({
  primaryAction,
  secondaryActions = [],
  className,
  "data-testid": dataTestId,
}: ActionGroupProps) {
  if (!primaryAction && secondaryActions.length === 0) {
    return null;
  }

  return (
    <div
      className={cn("flex flex-wrap items-center gap-2", className)}
      data-testid={dataTestId}
    >
      {primaryAction ? (
        <Button
          ref={primaryAction.ref}
          variant="primary"
          onClick={primaryAction.onClick}
          loading={primaryAction.loading}
          disabled={primaryAction.disabled}
          ariaLabel={primaryAction.ariaLabel}
          data-testid={primaryAction["data-testid"]}
        >
          {primaryAction.label}
        </Button>
      ) : null}
      {secondaryActions.map((action, index) => (
        <Button
          key={index}
          ref={action.ref}
          variant="outline"
          onClick={action.onClick}
          loading={action.loading}
          disabled={action.disabled || action.loading}
          ariaLabel={action.ariaLabel}
          data-testid={action["data-testid"]}
        >
          {action.label}
        </Button>
      ))}
    </div>
  );
}
