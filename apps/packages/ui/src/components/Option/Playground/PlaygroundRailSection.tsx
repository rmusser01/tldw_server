import React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cockpitRailStyles } from "./playground-cockpit-rail-styles";

export type PlaygroundRailSectionProps = {
  label: string;
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
};

export const PlaygroundRailSection = ({
  label,
  title,
  children,
  defaultOpen = true,
}: PlaygroundRailSectionProps) => {
  const { t } = useTranslation("playground");
  const [open, setOpen] = React.useState(defaultOpen);
  const headingId = React.useId();
  const panelId = React.useId();
  const toggleLabel = open
    ? t("cockpit.collapseRailSection", `Collapse ${title}`, { title })
    : t("cockpit.expandRailSection", `Expand ${title}`, { title });

  return (
    <section
      className={cockpitRailStyles.section}
      aria-label={label}
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <h2 id={headingId} className={cockpitRailStyles.heading}>
          {title}
        </h2>
        <button
          type="button"
          className={cockpitRailStyles.collapseAction}
          aria-label={toggleLabel}
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? (
            <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
          )}
        </button>
      </div>
      <div
        id={panelId}
        aria-hidden={!open}
        className={open ? undefined : "hidden"}
      >
        {children}
      </div>
    </section>
  );
};
