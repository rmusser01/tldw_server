# UAT162 bounded design and plan
Task TASK-13260.99. Root authorized label repair; runtime/browser/commit/tracker remain root-owned.

## Root cause and design
All eight numeric fields in TldwTimeoutSettings have visible sibling labels, but no htmlFor/id link. Native receipt .tmp/uat152-153-final-native-20260916/balanced-fields-and-labels.txt confirms empty IDs and labels arrays. Use one React.useId prefix plus a unique descriptive suffix per field; associate the existing translated label via htmlFor. Ant Design Input forwards id to the actual input. No timeout values, setters, presets, save behavior or translation changes.

Existing conventions read: ConversationTab numericFieldId, LongformDraftEditor inputIdPrefix, AdvancedPanel PerfField.

## Stage 1: mounted RED
Status: Complete.
Create focused test with real TldwTimeoutSettings, Ant Design Collapse/Input and real i18next English resources. Failure target: a missing or incorrect label association causes accessible-name query/label-click focus failure. Test all eight labels. Two instances plus rerender prove instance-specific stable input IDs and correct local label targeting.

## Stage 2: minimal GREEN
Status: Complete.
Add one useId and eight paired htmlFor/id attributes. No other production edits.

## Stage 3: verification and review
Status: Author verification complete; independent review/native acceptance pending.
Run focused suite plus existing tldw.timeouts.form integration suite (preset/save/reload/request budgets), scoped ESLint, diff check, compiler comparison if needed, and Bandit with truthful TS coverage limitation. Retain exact owned hashes/report. Independent reviewer and root-native accessibility acceptance required before closure.
