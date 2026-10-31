import React from "react";
import { beforeEach, vi } from "vitest";

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  const wrap =
    (Component: typeof actual.Modal | typeof actual.Drawer) =>
    (props: Record<string, unknown>) => (
      <actual.ConfigProvider theme={{ token: { motion: false } }}>
        <Component
          {...props}
          {...(Component === actual.Modal
            ? {
                transitionName: "",
                maskTransitionName: "",
                styles: {
                  ...(props.styles as object),
                  wrapper: { position: "fixed" },
                },
              }
            : {})}
        />
      </actual.ConfigProvider>
    );
  return { ...actual, Modal: wrap(actual.Modal), Drawer: wrap(actual.Drawer) };
});

// jsdom has no layout. AntD uses geometry to filter its actual tabbable nodes.
beforeEach(() => {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
    function () {
      const hidden = this.closest('[hidden], [style*="display: none"]');
      return new DOMRect(0, 0, hidden ? 0 : 100, hidden ? 0 : 40);
    },
  );
});
