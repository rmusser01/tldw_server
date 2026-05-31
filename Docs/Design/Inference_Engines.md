# Inference Engines

## Introduction


### HuggingFace Transformers
https://huggingface.co/docs/transformers/main/en/conversations

### Llama.cpp
https://www.youtube.com/watch?v=vW30o4U9BFE
https://unsloth.ai/blog/dynamic-4bit
https://github.com/iuliaturc/gguf-docs
https://github.com/ggml-org/llama.cpp/discussions/15709
https://www.aleksagordic.com/blog/vllm



### Llamafile


### TabbyAPI

w
### vLLM



### Link Dump

 Not sure who needs to know this, but I just reduced my vLLM cold start time by over 50% just by loading the pytorch cache as a volume in my docker compose:

volumes:
- ./vllm_cache:/root/.cache/vllm

The next time it starts, it will still compile but sub sequent starts will read the cache and skip the compile. Obviously if you change your config or load a different model, it will need to do another one-time compile.
https://docs.vllm.ai/en/latest/examples/offline_inference/save_sharded_state.html

https://lmsys.org/blog/2025-05-05-large-scale-ep/

https://huggingface.co/blog/vlms-2025
https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html
https://github.com/intel/auto-round
https://github.com/codelion/optillm/tree/main
https://kaitchup.substack.com/p/how-well-does-qwen3-handle-4-bit
https://github.com/intel/neural-compressor/tree/v3.2
https://southbridge-research.notion.site/Entropixplained-11e5fec70db18022b083d7d7b0e93505
https://arxiv.org/abs/2506.06105
https://arxiv.org/abs/2506.12928v1
https://docs.vllm.ai/en/latest/examples/offline_inference/save_sharded_state.html
https://github.com/LMCache/LMCache
https://medium.com/@damianvtran/the-anatomy-of-a-modern-llm-0347afd72514

https://ngrok.com/blog/quantization
https://www.baseten.co/inference-engineering/digital-download/
https://arxiv.org/abs/2508.20893
https://damek.github.io/random/basic-facts-about-gpus/
https://bentoml.com/llm/
https://cvw.cac.cornell.edu/gpu-architecture
https://allenai.org/blog/flexolmo
    https://github.com/allenai/FlexOlmo
    https://www.datocms-assets.com/64837/1752084947-flexolmo-5.pdf
https://kserve.github.io/website/latest/modelserving/v1beta1/llm/huggingface/kv_cache_offloading/
https://sakana.ai/ab-mcts/
https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
https://arxiv.org/abs/2507.21509
https://kaitchup.substack.com/p/using-gguf-models-optimize-your-inference
https://www.tilderesearch.com/blog/sparse-attn
https://github.com/codelion/optillm/tree/main/optillm/plugins/deepthink
https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms?utm_medium=email
https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch?utm_medium=email
https://arxiv.org/abs/2405.09673
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/serving/vertex_ai_tgi_gemma_multi_lora_adapters_deployment.ipynb
https://github.com/Infini-AI-Lab/UMbreLLa
https://github.com/codelion/pts
https://huggingface.co/blog/andthattoo/dpab-a
https://arxiv.org/abs/2502.20604
https://www.anyscale.com/blog/continuous-batching-llm-inference
https://huggingface.co/blog/Kseniase/testtimecompute
https://eqimp.github.io/hogwild_llm/
https://transformer-circuits.pub/2025/attribution-graphs/biology.html
https://arxiv.org/pdf/2502.17601
https://www.reddit.com/r/LocalLLaMA/comments/1k0tkca/massive_5000_tokens_per_second_on_2x3090/
https://github.com/HazyResearch/minions
https://github.com/NVIDIA/NeMo-Inspector
https://arxiv.org/html/2504.17999v2
https://arxiv.org/html/2403.06988v1
