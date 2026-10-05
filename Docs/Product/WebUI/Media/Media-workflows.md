# Add, read and recover Media

Media Inspector and Multi-Item Review share the import wizard in the WebUI and browser extension. Configure your server connection first. In the extension, Capture current tab accepts an active HTTP or HTTPS page; internal browser and extension pages cannot be captured. Paste a URL instead when capture is unavailable.

## Add one source or a batch

Choose **Add media**, paste one URL per line, and choose **Add URLs**, or browse/drop files. The first-use URL field also passes its URL into the same wizard. An unambiguous comma followed by another URL separates sources; a comma inside one URL is preserved. Check the queue before processing.

The wizard distinguishes added inputs from sources eligible for this run. Invalid, duplicate and deselected inputs stay visible with their exclusion reason. A repeated URL is excluded by default. Choose **Process again** explicitly when intentional reprocessing is needed. Configure and Review use the same eligible set; Results explains the exclusions alongside succeeded, failed, saved and cancelled counts.

Choose Quick, Standard or Deep for the processing work. These settings apply to **This run**. Saved defaults are a separate choice for future runs. Processing depth does not grant replacement permission: the explicit replacement setting is preserved when switching presets. Review states whether matching saved sources may be replaced. It does not identify affected saved IDs unless the server has supplied that evidence.

Start processing after checking the sources and permissions. Minimize or close the wizard to continue using Media; reopen the current import to resume its existing progress and results. Cancel stops the active run. A retry targets only eligible retryable failures and preserves earlier successful outcomes. After a reload, a missing local file must be attached again before retry; the browser does not persist File data.

**Saved**, **Processing** and **Ready for Knowledge** are different states. A server-confirmed saved ID can be opened in Media. Requested analysis or chunking alone does not prove indexing or Knowledge readiness. The interface labels readiness as unconfirmed when affirmative server evidence is absent.

## Continue a saved batch

Results offers **Review these N saved items** for the successful stored IDs, including an ordinary batch. It opens exactly that set in Multi-Item Review. From the extension sidebar, saved review, individual Media and Knowledge actions open the full extension page; WebUI and full extension pages navigate in place. Extracted content that was not saved, failed submissions and cancelled items do not enter the saved set. Conference imports retain their existing collection identity and retry behavior.

**Recent imports** appears in the primary Inspector and its empty-library view from submission onward. Expand it to recognize sources by host/file name, count, status and time. **Resume import** opens the current session; **Refresh import** reads its known server batches; **Review N saved items** reopens its saved set. Batch IDs remain optional diagnostics.

History is limited to ten metadata records in this browser tab's existing session storage. It survives a reload of that tab, but is not a server-wide history or a cross-device archive. It contains concise labels and IDs, not full content, uploaded files or source URL credentials/query strings. Records and actions belong to the verified server/account. An unread storage record is preserved if a transient read fails; retrying hydration can recover it. A server/account change prevents old records or delayed responses from appearing in the new context.

## Read and select

In Inspector, opening a result on a narrow screen chooses **Content**. **Back to results** returns to the list. Bulk mode provides explicit checkboxes, a persistent selected count, and **Open selection**. Selection survives pagination: two items from page one and two from page two form a four-item set. **Select this page** applies to the visible page; it does not mean every matching server item.

In Multi-Item Review, clicking a row and pressing Enter preview the same source. Use the named checkbox to change selection; Space on it selects without changing the preview. The displayed heading and navigation identify either **Result preview** or **Selected reading**. Starting selected reading preserves the complete ordered set across pages; returning to preview preserves its separate context.

A metadata/action set may exceed thirty items. Reading loads and renders at most thirty at a time. The reading-window count identifies the range and total; use **Next reading window** or **Previous reading window** for larger sets. Batch tags, export, reprocess and trash apply to the full selected set. **Side-by-side reading** describes layout; **Compare content** compares extracted content. Reading and Selection actions are grouped separately in Options.

## Move to Trash and restore

Inspector asks for confirmation before **Move N items to trash**. Cancel makes no deletion request. Successful items leave the selection; failed items remain available for retry and are reported separately. Use **Open Trash** to inspect and restore Media. Permanent deletion is a separate operation.

Notes selected in Inspector retain their own kind and captured version. A successful bulk Note move can offer **Restore N notes** using the captured deleted versions. Partial restore failures remain retryable. After reload, recover Notes in **Notes Manager → Trash**; Media Trash contains Media and does not serve as Notes Trash. Changing server/account invalidates delayed mutations and recovery callbacks.
