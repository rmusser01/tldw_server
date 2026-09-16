# Screenshot inspection

All3 retained PNGs were inspected and copied byte-for-byte. No credentials are visible.

| File | Observed result |
| --- | --- |
| uat099-final-native-scrolled.png | Desktop source, analysis and statistics occupy the bounded scrolling panel; source is at the captured saved position. |
| uat099-final-native-reloaded.png | Same source position remains after normal reload; paired native record establishes scrollTop160 and GET-only observation. |
| uat099-final-native-mobile.png | Right-hand source text and controls extend beyond390px and are visibly clipped. This confirms UAT101; mobile acceptance is not passed. |

Screenshots corroborate the paired geometry/traffic records. They do not establish all possible viewport or timing behavior.
