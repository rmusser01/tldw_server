# Setting up a Local LLM

# FIXME - last updated?

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