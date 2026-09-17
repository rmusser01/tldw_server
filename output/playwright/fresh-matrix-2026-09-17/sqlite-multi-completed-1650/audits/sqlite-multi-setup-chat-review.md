# SQLite-multi setup and Chat independent audit

Reviewed 2026-09-17T15:29:28.226605+00:00. Frozen revision attributed by parent run context: `8f8774e6c868b304a96d95ab82e28389c129a78b`. Scope is the20 authorized wrapper-redacted/native receipts below. No private files, credentials, source code, browser, runtime, inference, tests or tasks were accessed or changed. Only this ignored report was written.

## Verdict

**Bounded setup/account creation and Alice's ordinary two-turn Chat + failed-turn Retry pass.** Provider setup includes a disclosed operator configuration/restart adaptation. Image attachment/local guard/rendered reload pass within the stated boundary; no vision pass. **Natural expiry is parked and not accepted by this audit.** No full authentication lifecycle, reciprocal isolation or full-matrix acceptance is inferred.

### Setup, accounts and provider adaptation

- `setup-guide.txt` shows the multi-user First-time setup guide and Sign in, rather than a single-user local-provider wizard.
- `admin-page.txt` records authenticated `uat_admin`, ID1, from15:03:12.931Z onward. Its initially loading Users/Roles tables do not establish missing data: later settled create receipts list actual users/roles.
- Ordinary visible Create user forms create **Alice ID2** at **15:04:31.861Z**, and **Bob ID3** at **15:05:19.220Z**; both successful admin endpoint responses are **200**, with role `user`, active=true and verified=true. Their settled user-list responses preserve those identities. This does not yet prove reciprocal content isolation.
- `operator-provider.json` at15:09:59.659Z explicitly records supported operator configuration of only `API.default_api`, `Local-API.llama_api_IP`, and `Local-API.llama_model`, discovery200 against existing local `http://127.0.0.1:9099/v1`, and the actual Gemma GGUF identifier. It supplies before/after config hashes and says source unchanged/planned same-profile restart. This is an operator-assisted setup outcome, not an entirely UI-only provider configuration claim.
- Parent run context identifies the subsequent API as **45087**; the supplied operator receipt is a pre-restart intent record, so these20 inputs alone do not independently bind that process's launch/PID. `provider-ready.txt` at15:11:00.648Z shows one usable provider, the real LLaMa.cpp model and server-routed Auto defaults; Alice's later actual requests prove the configured service works.
- Native admin Logout is recorded at15:12:31.875Z. Normal visible Alice Sign in returns auth200 at15:13:03.259Z with expiresIn1800, followed by repeated identity200 confirming **Alice2**. The immediate login splash says sign-in is required transiently; settled Chat at15:13:22.168Z has Alice identity, composer and Healthy LLaMa.cpp. No settled auth failure is inferred from that early splash.

### Real two-turn conversation and canonical persistence

Conversation: `246b9dc3-1eda-4f9e-a2b4-61ca132ec789`; actual provider `llama`, configured Gemma GGUF, save_to_db=true.

| Turn | Actual request / response | Persisted result |
|---|---|---|
| Remember ORBIT-742 | POST15:13:48.570Z;20015:13:49.132Z; body-read complete15:13:50.189Z; client `pa_ae37-cef7-d7a-06f3` | Assistant `ORBIT-742`. |
| Recall the code | POST15:14:27.838Z;20015:14:28.168Z; body-read complete15:14:30.337Z; client `pa_3b7a-18cf-503-552a` | Assistant `ORBIT-742`, with actual prior user/assistant context in the request. |

`chat-two-turn-reload.txt` records an actual reload but ends just as the new canonical request starts. The later **15:14:31.874Z GET200**, retained in `chat-failure-settled.txt`, proves the settled **five unique version1 canonical rows**. The earlier sampling cutoff is not a persistence failure.

### Availability rejection and exact native Retry

- One-shot fault receipt at **15:14:52.099Z** records the original llama/Gemma target and effective `ollama` / `uat-deliberately-unavailable-20260917`, with **one real continued backend request and no response fulfillment**. Actual backend response **400 at15:14:52.219Z** is `model_not_available`.
- This is a real **backend availability rejection**, not an upstream generation/network outage or a fabricated response. The UI retains the prior conversation and failed user turn, but displays generic server-health guidance, consistent with existing UX232.
- Native **Retry same model** sends15:15:37.957Z and returns **200 at15:15:38.290Z**, body-read complete15:15:40.224Z. The successful UI answer is `ORBIT-742`; the same command reloads and finds that third answer again.
- Independent structural comparison of the failed request's original intended payload and Retry finds **only metadata differs**: Retry adds `tldw_retry_failed_turn:true`. Conversation, provider/model, complete five-message context, final user prompt, temperature/top_p/penalties, streaming/save options and original client ID **`pa_45c2-be7e-94d-4e2d`** are unchanged.
- `chat-retry-reloaded.txt` is sampled before its canonical read finishes. The allowlisted later `ingest-start-minimize-resume.txt` retains that read: **GET200 at15:15:41.623Z**, seven unique version1 canonical rows, the original five rows exactly unchanged plus one successful user/assistant pair. The later file's ingestion activity is outside this audit.

| Canonical role/order | ID |
|---|---|
| System | `aa1c8f6f-0465-42c0-b9e3-0b626883fdfc` |
| User1 | `19689a9f-4411-4410-8186-e9e2c23ea6a1` |
| Assistant1 | `a7e562bc-f893-4ade-862c-48826ea35993` |
| User2 | `6db15852-f543-45d1-b78a-da6c97773d0a` |
| Assistant2 | `bbc0aeeb-ef67-483d-966f-86a38fa2ea09` |
| Retried User3 | `81be61b1-5962-403f-a15c-c3b27ffef2ed` |
| Assistant3 | `d05add70-10be-49e4-bb96-8dc762e39cad` |

There is no duplicated failed canonical pair in this settled read. Local error rendering is not counted as a canonical assistant.

### Image guard and visible reload

- `image-upload.txt` records native file chooser upload of the source archive's public `apps/tldw-frontend/public/icon/128.png`. `image-guard-retry.txt` records a real Send and native Retry same model, resulting in one image user turn and guard response variant **2 of2**.
- Completion-request count is **4 before /4 after**: this image action sent no inference request. The visible message truthfully says image support is not confirmed and suggests an image-capable model or text-only conversation.
- After actual reload at15:17:19.297Z, `image-reloaded.txt` preserves the image/guard pair and measures the Chat image element: **Uploaded Image, natural128×128, complete=true**. This supports decoded-image rendering, not byte-identical upload proof or vision inference.
- Separate image conversation is `503ca276-4ab5-41b1-a104-65fc547b4d00`. The sampled canonical empty reads are timestamped15:16:58.299/.300Z, **before** the reload; do not mislabel them as settled post-reload canonical readbacks. The local post-reload display is directly observed. No genuine hidden-tab state or upstream-outage acceptance follows.

### Natural expiry — pending

- The separate child context uses the normal guide → Sign in flow. Successful Alice2 login at **15:07:07.498Z** advertises **1800 seconds**. Later identity/Notes responses are200, including Notes/collections/keywords at15:07:09.830–.841Z.
- The immediate post-login Notes snapshot's not-connected state is superseded by those actual200 reads and the settled parked UI; it does not establish a durable connection defect.
- `expiry-parked.txt` at **15:07:52.271Z** closes the child page and reports **0 open pages /0 service workers**, preserving the context for return. Its planned return time is **15:37:17.498Z** (issuance +1800s +10s). The retained command does not manipulate tokens or clocks.
- These inputs contain **no post-deadline return, natural refresh/rejection, renewed identity or recovery result**. Elapsed time alone cannot close this boundary. Parent owns the child context and the later observation; expiry remains pending regardless of when this report is read.

## Limits

This audit covers a bounded multi-user setup and ordinary Alice Chat. It does not establish fresh-machine dependency installation, complete multi-user auth/outage/isolation matrix, successful vision, genuinely hidden tabs, source ingestion/RAG, Character, five-card Study, or an upstream generation failure. No model reasoning is reproduced. Private helper/log links inside wrappers were not opened. Input bytes were rechecked unchanged at report creation.

## Reviewed input hashes

Paths below are relative to `.tmp/uat-next-matrix-20260916/native/sqlite-multi/`; exact bytes, no normalization.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `admin-logout.txt` | 736 | `535675900cc2a39f272f91e1c683441e8359305c0495feb027c7d871c73b6234` |
| `admin-page.txt` | 6146 | `a7b689193a37010f452415125e35e61d13fffa892c6808cd08080d905cce3473` |
| `alice-chat-initial.txt` | 4975 | `fc2cded2b93eb68dfba933b67d16ce32bff8d7b841c5b9740bc1b4327b97fa0c` |
| `alice-created-redacted.txt` | 4970 | `200084cf46b58c2bb0cf2394bcb0cb62a0399ee36c3a6d2affce6e7fac931bf0` |
| `alice-login-redacted.txt` | 2443 | `32ce216f61e810dbbb1a4573f74aa12adbb3a03bb1d0144c0c3fb645e6a8378a` |
| `bob-created-redacted.txt` | 6623 | `c9e4b8d29541ebb067b9175fb90be43d99c888c164ab7db0f0528830ce17298f` |
| `chat-failure-settled.txt` | 15711 | `1dcd799ad83c9d02465f6cb2457f7f0c24e64b81e764639488e49c1571d1981b` |
| `chat-retry-reloaded.txt` | 89989 | `33e2e3de6d1d3a6f30667198abfb337fea711004512e31358431a2b0f59841c9` |
| `chat-turn1-result.txt` | 62275 | `cbaafcdf695c6a052d976db7d788854a492751bce047b9ebc04ab60168a51c90` |
| `chat-two-turn-reload.txt` | 71316 | `91259ebb796aaaa75dde26da90b82cf17b70dff05314270508c97c9a4e3a6a65` |
| `expiry-guide-signin.txt` | 1016 | `1833b3f3751897521684003e0cf619ba1fc642436b78d030e99e24ee44703adc` |
| `expiry-login-redacted.txt` | 5710 | `11ea802ce1d1ce597f2e9ccf728fb96e812da7d51ea8e5672e04f6bc43ba7e91` |
| `expiry-parked.txt` | 7464 | `3b02c1ca2e87377f884531df9f814243cfc0f3ce73e9836dfa275f3d414a1d75` |
| `image-guard-retry.txt` | 6028 | `43fcf72efd5df87026552578b1162051c931fcf1d7d099d9db23f168635dda0c` |
| `image-reloaded.txt` | 6615 | `bd973863f7e58368d8f914a147e907ed14660660be2f3a271e12afa945d255ec` |
| `image-upload.txt` | 10580 | `ab3c106d1684fc187f9e9ed624bf180ed8a957f9377c8cac9e1daf1906478c64` |
| `ingest-start-minimize-resume.txt` | 13042 | `5662676f08587ff184f95f09efa05e9a0775d66b10591f12f98327f2afb30de2` |
| `operator-provider.json` | 782 | `511b44e42e38e8ffacd1f7b96f880862f84de2662ff54dc1a391d4b8d8219b13` |
| `provider-ready.txt` | 9478 | `c832da88f461c9f15953faf8c7f9b5d1d9c443229f2c3dd92e33e89f4e35b68e` |
| `setup-guide.txt` | 1537 | `b4e1fcb7729423f0a1d5358ce9f150c6492e51ff849cbe135522c9e85b975e0a` |
