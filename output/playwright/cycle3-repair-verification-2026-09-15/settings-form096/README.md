# Settings Form lifecycle repair

TASK13260.37 / UAT096, integrated-dev branch. This bundle records the reviewed automated repair; native logout re-verification and the full fresh UAT remain pending.

Normal browser logout succeeded but emitted the Ant Form disconnected-instance warning. The retained native action and exact warning excerpt establish that observation. The automated regression reproduces the initial configuration hydration lifetime with actual Ant Form and actual WebUI storage; it does not claim to reproduce the entire native logout scheduling.

The minimal change stages accepted configuration fields until their Form mounts, retaining the existing configuration generation guard and auth handlers. The original source fails the permanent hydration regression with the exact warning; the corrected implementation and independent review each pass57 tests across5 suites. These overlapping runs are not additive. Pending cookie/manual logout, failed logout and current-account draft controls are included. Lint has0 errors and33 unchanged warnings. Bandit does not apply to the TypeScript-only scope. The final combined compiler check is parent-owned and pending in this checkpoint.

The frozen manifest binds source/test/task and the original replay inputs at its recorded timestamp. Task notes may subsequently change as verification proceeds; the production hash must remain exact for acceptance. Original probe and source bytes are retained; text log/report trailing whitespace is normalized. All indexed files are scanned for the14 isolated runtime credential values and JWT/private-key patterns before writing. The native screenshot is not needed to establish a console-only warning.
