import { describe, expect, it } from "vitest";
import { replaceUserDisplayNamePlaceholders } from "../chat-display-name";

describe("user and character display names", () => {
  it("renders user and character placeholders with distinct identities and literal dollar names", () => {
    expect(
      replaceUserDisplayNamePlaceholders(
        "{{user}} / {{ username }} / {{CHAR}} / {{ character }}",
        "Viewer $&",
        "Guide $'",
      ),
    ).toBe("Viewer $& / Viewer $& / Guide $' / Guide $'");
  });
  it("leaves character placeholders unresolved when only the user identity is known", () => {
    expect(
      replaceUserDisplayNamePlaceholders("{{user}} meets {{char}}", "Viewer"),
    ).toBe("Viewer meets {{char}}");
  });
  it("renders a known character even when the user name is absent", () => {
    expect(
      replaceUserDisplayNamePlaceholders(
        "{{user}} meets {{char}}",
        "",
        "Guide",
      ),
    ).toBe("User meets Guide");
  });
  it.each(["", " \t\n "])(
    "uses the backend-compatible fallback for all user aliases with a blank name (%j)",
    (name) => {
      expect(
        replaceUserDisplayNamePlaceholders(
          "{{user}} / {{ USERNAME }} / {{user_name}} / {{ you }} / {{me}} / {{char}} / {{unknown}}",
          name,
        ),
      ).toBe("User / User / User / User / User / {{char}} / {{unknown}}");
    },
  );
  it("prefers an explicit trimmed display name without trimming source text", () => {
    expect(
      replaceUserDisplayNamePlaceholders(
        "  {{ me }} meets {{ character }}\n  exact  ",
        "  Viewer $&  ",
        "  Guide $'  ",
      ),
    ).toBe("  Viewer $& meets Guide $'\n  exact  ");
  });
});
