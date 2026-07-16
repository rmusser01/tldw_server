# Setting up a Local LLM

## Connect tldw_server to a local endpoint

Start the model server so it listens on an address reachable from the `tldw_server` process. Binding a model server to `127.0.0.1` exposes it only inside that host or container; use the LAN/container interface when the API runs elsewhere, and protect the service with firewall rules and authentication where supported.

Configure the provider in `tldw_Server_API/Config_Files/config.txt`. Common examples are:

```ini
[Local-API]
# llama.cpp on another LAN host
llama_api_IP = http://192.168.1.50:8080/v1
llama_model = local-model.gguf

# A Docker Compose service reachable by service DNS
vllm_api_IP = http://vllm:8000/v1
vllm_model = organization/model-name

# A Tailscale address; a resolvable MagicDNS hostname also works
ooba_api_IP = http://100.90.80.70:5000/v1
ooba_model = local-model
```

Use the base path expected by the provider, commonly `/v1` for OpenAI-compatible servers. The configured scheme, hostname, and effective port form one exact trusted origin. tldw permits that origin on loopback, private LAN, Docker, or approved overlay addresses while keeping the global SSRF policy intact. Redirects cannot change the configured origin, DNS changes fail closed, and metadata/link-local targets remain blocked.

The shared model catalog powers both the WebUI and browser extension. After setup reports `status="saved"`, the saving context writes a unique invalidation marker to the existing shared model-cache record. Open WebUI tabs and extension sidepanel/background contexts observe that marker, clear both model-cache layers once, and fetch current backend metadata on their next selector request. The saving context ignores its own storage echo, and later model records remain local cache data rather than cross-context update messages.

### Troubleshooting readiness

Check the provider's `readiness_reason_code` in the model/provider metadata response or setup diagnostics:

- `egress_blocked`: the origin does not exactly match the trusted configuration, a denylist matched, or DNS returned a forbidden special-use address. Do not work around it by globally allowing private addresses.
- `endpoint_unreachable`: the API process could not resolve or connect to the endpoint. Test DNS and connectivity from the API host/container. For Docker Desktop, a host service may be reachable as `host.docker.internal`; in Compose, prefer the service name on a shared network.
- `auth_failed`: the local server rejected the configured credential.
- `endpoint_error`: the server returned an error or incompatible response. Confirm its OpenAI-compatible URL and inspect its logs.
- `model_discovery_unavailable`: the server's model-list response is unsupported. Configure the provider's explicit `*_model` field.
- `no_models_reported`: discovery worked but the server reported no loaded models. Load a model or set the explicit model field.

If a previous workaround set `[Egress] block_private=false`, return it to `true` only after auditing unrelated webhooks, workflows, and integrations that might depend on that global setting. tldw_server will not rewrite the value automatically.

https://github.com/ggerganov/llama.cpp/blob/cddae4884c853b1a7ab420458236d666e2e34423/examples/quantize/README.md#L27

- **Setting up Local LLM Runner**
  - **Llama.cpp**
    - **Linux & Mac**
      1. `git clone https://github.com/ggerganov/llama.cpp`
      2. `make` in the `llama.cpp` folder
      3. `./server -m ../path/to/model -c <context_size> -ngl <layers-to-offload-to-gpu>`
        * Example: `./server -m ../path/to/model -c 8192 -ngl 999` - This will run the model with a context size of 8192 tokens and offload all layers to the GPU.
    - **Windows**
      1. `git clone https://github.com/ggerganov/llama.cpp`
      2. Download + Run: https://github.com/skeeto/w64devkit/releases
      3. cd to `llama.cpp` folder make` in the `llama.cpp` folder
      4. `server.exe -m ..\path\to\model -c <context_size>`
        * Example: `./server -m ../path/to/model -c 8192 -ngl 999` - This will run the model with a context size of 8192 tokens and offload all layers to the GPU.
    - **tldw managed llama.cpp WebUI**
      1. Build or install `llama-server`, then open the tldw WebUI at `/admin/llamacpp`.
      2. In **Readiness**, set the executable path, models directory, allowed paths, and default host/port. Some changes require restarting the tldw API server before the active handler sees them.
      3. In **Assets**, register an existing GGUF or mmproj file, or preview and confirm a local folder import. Local registration/import only updates the managed asset inventory; it does not create a profile, start a runtime, or change Chat routing.
      4. In **Profiles**, create a durable runtime profile. Profiles store mode, model asset, optional mmproj projector, host/port, structured server arguments, provider alias, tags, autostart, and bounded restart policy. Profile state is stored by the backend in `llamacpp_profiles.json` next to the active tldw config file.
      5. For multimodal or vision profiles, select a matching mmproj asset. The backend rejects missing or conflicting projector definitions, but hardware and VRAM fit messages stay warnings rather than hard blockers.
      6. In **Runtime instances**, start the profile you want. Autostart profiles are reconciled on server startup, paused profiles stay paused, and restart attempts are bounded by the saved policy.
      7. Use **Use in Chat** only after the desired runtime is running. This explicit action points the llama.cpp provider endpoint at that runtime; starting a profile alone does not silently rewire Chat.
      8. Remote downloads and future catalog workflows live in the asset acquisition flow. They are not part of profile launch and do not automatically create profiles or start runtimes.
  - **Kobold.cpp** - c/p'd from: https://github.com/LostRuins/koboldcpp/wiki
    - **Windows**
      1. Download from here: https://github.com/LostRuins/koboldcpp/releases/latest
      2. `Double click KoboldCPP.exe and select model OR run "KoboldCPP.exe --help" in CMD prompt to get command line arguments for more control.`
      3. `Generally you don't have to change much besides the Presets and GPU Layers. Run with CuBLAS or CLBlast for GPU acceleration.`
      4. `Select your GGUF or GGML model you downloaded earlier, and connect to the displayed URL once it finishes loading.`
    - **Linux**
      1. `On Linux, we provide a koboldcpp-linux-x64 PyInstaller prebuilt binary on the releases page for modern systems. Simply download and run the binary.`
        * Alternatively, you can also install koboldcpp to the current directory by running the following terminal command: `curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64 && chmod +x koboldcpp`
      2. When you can't use the precompiled binary directly, we provide an automated build script which uses conda to obtain all dependencies, and generates (from source) a ready-to-use a pyinstaller binary for linux users. Simply execute the build script with `./koboldcpp.sh dist` and run the generated binary.
  - **oobabooga - text-generation-webui** - https://github.com/oobabooga/text-generation-webui
    1. Clone or download the repository.
      * Clone: `git clone https://github.com/oobabooga/text-generation-webui`
      * Download: https://github.com/oobabooga/text-generation-webui/releases/latest -> Download the `Soruce code (zip)` file -> Extract -> Continue below.
    2. Run the `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, or `start_wsl.bat` script depending on your OS.
    3. Select your GPU vendor when asked.
    4. Once the installation ends, browse to http://localhost:7860/?__theme=dark.
  - **Exvllama2**
- **Setting up a Local LLM Model**
  1. microsoft/Phi-3-mini-128k-instruct - 3.8B Model/7GB base, 4GB Q8 - https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
    * GGUF Quants: https://huggingface.co/pjh64/Phi-3-mini-128K-Instruct.gguf
  2. Meta Llama3-8B - 8B Model/16GB base, 8.5GB Q8  - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    * GGUF Quants: https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF

### LLMs for Offline/Private Use
- For offline LLM usage, I recommend the following models in no particular order past the first
- All these models minus Command-R/+ can be ran on a single 12GB VRAM GPU, or 12GB of system RAM at a much slower speed.
- Either way, I recommend using the Q4 GGUF versions of the models, as they are the most efficient and fastest to load, while still maintaining their accuracy.
- So for Mistral-Nemo-Instruct-2407, you'd want to download `Mistral-Nemo-Instruct-2407-Q4_K_M.gguf` - notice the `Q4` in the name.
    1. Samantha-Mistral-instruct-7B-Bulleted-Notes - https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes_GGUF
       * Reason being is that its 'good enough', otherwise would recommend Mistral-Nemo-Instruct2407. Very likely Nemo will prove to be better. Time will tell.
    2. Mistral-Nemo-Instruct-2407
       *  https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 / GGUF: https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF
    3. Microsoft Phi-3-mini-4k-Instruct
       * https://huggingface.co/microsoft/Phi-3-mini-4k-instruct / GGUF: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
       * Also the 128k Context version: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct / Abliterated GGUF: https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF
    4. Cohere Command-R+
       * https://huggingface.co/cohere-ai/Command-R-plus / GGUF: https://huggingface.co/XelotX/c4ai-command-r-plus-XelotX-XelotX-iQuants
    5. Cohere Command-R (non-plus version)
       * https://huggingface.co/CohereForAI/c4ai-command-r-v01 / GGUF: https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF
    6. Phi-3-Medium-4k-Instruct
       * https://huggingface.co/microsoft/Phi-3-medium-4k-instruct / Abliterated GGUF:https://huggingface.co/failspy/Phi-3-medium-4k-instruct-abliterated-v3
         * Also the 128k Context version: https://huggingface.co/microsoft/Phi-3-medium-128k-instruct / GGUF: https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF
    6. Hermes-2-Theta-Llama-3-8B
       * https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B / GGUF: https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF
    7. Yi-1.5-34B-Chat-16k
       * https://huggingface.co/01-ai/Yi-1.5-34B-Chat-16K / GGUF: https://huggingface.co/mradermacher/Yi-1.5-34B-Chat-16K-GGUF
