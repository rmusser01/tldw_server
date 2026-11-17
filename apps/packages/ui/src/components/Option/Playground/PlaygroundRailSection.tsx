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
  const panelId = React.useId();
  const toggleLabel = open
    ? t("cockpit.collapseRailSection", `Collapse ${title}`, { title })
    : t("cockpit.expandRailSection", `Expand ${title}`, { title });
  return (
    <section className={cockpitRailStyles.section} aria-label={label}>
      <h2 className={cockpitRailStyles.heading} aria-label={title}>
        <button
          type="button"
          className="flex min-h-11 w-full items-center justify-between gap-2 rounded text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          aria-label={toggleLabel}
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((value) => !value)}
        >
          <span>{title}</span>
          {open ? (
            <ChevronDown
              className="h-3.5 w-3.5 text-text-muted"
              aria-hidden="true"
            />
          ) : (
            <ChevronRight
              className="h-3.5 w-3.5 text-text-muted"
              aria-hidden="true"
            />
          )}
        </button>
      </h2>
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
