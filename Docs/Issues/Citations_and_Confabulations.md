# Citations & Confabulations

## Table of Contents
1. [Citations](#citations)
2. [Confabulations](#confabulations)
3. [References](#references)


https://huggingface.co/spaces/LISA-Kadi/LISA-demo/tree/main
https://arxiv.org/abs/2412.07965
https://arxiv.org/abs/2501.08292
https://huggingface.co/QuantFactory/gemma-7b-aps-it-GGUF
https://huggingface.co/google/gemma-7b-aps-it
https://docs.anthropic.com/en/docs/build-with-claude/citations
https://mattyyeung.github.io/deterministic-quoting
https://github.com/tangzhy/RealCritic
https://huggingface.co/papers/2502.09604
https://github.com/voidism/SelfCite
https://www.reddit.com/r/LocalLLaMA/comments/1j5lym7/lightweight_hallucination_detector_for_local_rag/
https://liveideabench.com/
https://www.archgw.com/blogs/detecting-hallucinations-in-llm-function-calling-with-entropy-and-varentropy
https://github.com/lechmazur/confabulations
https://python.langchain.com/docs/how_to/qa_citations/
https://oumi.ai/blog/posts/introducing-halloumi
https://github.com/Liyan06/MiniCheck
https://github.com/jacobwarren/Latent-Space-Verification-for-Self-Correcting-LLMs
https://huggingface.co/jacobpwarren/Qwen2.5-7B-Latent_Verification/discussions
https://arxiv.org/abs/2504.07527
https://arxiv.org/abs/2504.00824
https://arxiv.org/abs/2503.22216
https://allenai.org/blog/olmotrace
https://arxiv.org/abs/2504.18114

RAG
  https://www.lycee.ai/blog/rag-ragallucinations-and-how-to-fight-them
  https://huggingface.co/PleIAs/Pleias-Nano
  https://arxiv.org/abs/2412.11536
  https://cloud.google.com/generative-ai-app-builder/docs/check-grounding
  https://cloud.google.com/generative-ai-app-builder/docs/grounded-gen
  https://arxiv.org/html/2412.15189v1#S6
  https://aclanthology.org/2024.fever-1.10/
  https://arxiv.org/pdf/2412.15189
  https://huggingface.co/papers/2408.12060
  https://primer.ai/research/rag-v-divide-and-conquer-with-factual-claims/
  https://arxiv.org/abs/2411.06037
  https://www.sciencedirect.com/science/article/abs/pii/S0306457320309675
  https://github.com/Huffon/factsumm
  https://diamantai.substack.com/p/graph-rag-explained?utm_campaign=post&triedRedirect=true
  https://arxiv.org/abs/2410.07176

Finetuning: 
- https://eugeneyan.com/writing/finetuning/

``````
On the source pane, give us the ability to enter in proper bibliographic information for a source in a variety of formats (MLA, Chicago etc.) When pulling the source name for citations, use this information instead and properly cite the footnote. You should still include the hotlink to the specific location in the material, or perhaps inside the footnote, just directly quote the relavent paragraph. When I am working with uploaded books, I can always just search my original digital copy to get the page number. 

You could also have a bibliography tab on the sources which will generate an annotated bibliography from the same material. Having the ability for an agent to properly generate and fill out the bibliography itself would be nice of course, but I would be more than happy to just enter it in myself if the result was citations that were actually useful when taken outside of NotebookLM by copy/paste.
```



----------------------------------------------------------------------------------------------------------------
### <a name="citations"></a> Citations
- **101**
- Unsorted
    - https://mattyyeung.github.io/deterministic-quoting#7-conclusion-is-this-really-ready-for-healthcare
  https://github.com/sunnynexus/RetroLLM
    - https://github.com/MadryLab/context-cite

Abstractive Proposition Segmentation
  https://arxiv.org/abs/2406.19803
  https://huggingface.co/google/gemma-2b-aps-it
  https://ritvik19.medium.com/papers-explained-244-gemma-aps-8fac1838b9ef

Anthropic:
```
Draft a press release for our new cybersecurity product, AcmeSecurity Pro, using only information from these product briefs and market reports.
<documents>
{{DOCUMENTS}}
</documents>

After drafting, review each claim in your press release. For each claim, find a direct quote from the documents that supports it. If you can’t find a supporting quote for a claim, remove that claim from the press release and mark where it was removed with empty [] brackets.
```

Attributions
  https://github.com/aws-samples/llm-based-advanced-summarization/blob/main/detect_attribution.ipynb


Long context generation
  https://arxiv.org/pdf/2408.15518
  https://arxiv.org/pdf/2408.14906
  https://arxiv.org/pdf/2408.15496
  https://arxiv.org/pdf/2408.11745
  https://arxiv.org/pdf/2407.14482
  https://arxiv.org/pdf/2407.09450
  https://arxiv.org/pdf/2407.14057
  https://www.turingpost.com/p/longrag
  https://www.turingpost.com/p/deepseek
  https://arxiv.org/pdf/2408.07055
----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### <a name="confabulations"></a> Confabulations


Benchmarks
  https://github.com/lechmazur/confabulations/
  https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard
  https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard
  https://osu-nlp-group.github.io/AttributionBench/
  Fake News
    https://arxiv.org/abs/2412.14686
  FACTS
    https://www.kaggle.com/facts-leaderboard
    https://storage.googleapis.com/deepmind-media/FACTS/FACTS_grounding_paper.pdf
    https://deepmind.google/discover/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/
Halogen
  https://arxiv.org/pdf/2501.08292

Detecting Hallucinations using Semantic Entropy:
    https://www.nature.com/articles/s41586-024-07421-0
    https://github.com/jlko/semantic_uncertainty
    https://github.com/jlko/long_hallucinations
    https://arxiv.org/abs/2406.15927
    https://www.amazon.science/publications/knowledge-centric-hallucination-detection
    https://www.amazon.science/publications/hallumeasure-fine-grained-hallucination-measurement-using-chain-of-thought-reasoning
    https://arxiv.org/abs/1910.12840
    https://aclanthology.org/2020.emnlp-main.750/

Evals:
- https://github.com/yanhong-lbh/LLM-SelfReflection-Eval
  - https://eugeneyan.com/writing/evals/
  https://github.com/confident-ai/deepeval/tree/99aae8ebc09093b8691c7bd6791f6927385cafa8/deepeval/metrics/hallucination

Explainability
  https://towardsdatascience.com/explaining-llms-for-rag-and-summarization-067e486020b4

Research
  https://github.com/EdinburghNLP/awesome-hallucination-detection
  https://www.lycee.ai/blog/rag-ragallucinations-and-how-to-fight-them
  https://thetechoasis.beehiiv.com/p/eliminating-hallucinations-robots-imitate-us
  https://llm-editing.github.io/
  https://cleanlab.ai/blog/trustworthy-language-model/
  General
    https://arxiv.org/pdf/2410.19385
    https://arxiv.org/pdf/2409.18475
    https://arxiv.org/pdf/2406.02543
    https://arxiv.org/abs/2407.19825
    https://arxiv.org/abs/2407.16604
    https://arxiv.org/abs/2407.16557
    https://arxiv.org/abs/2412.04235
  Attention/Long Context
    https://arxiv.org/abs/2407.13481
    https://arxiv.org/pdf/2407.03651
  CoV
    https://arxiv.org/pdf/2309.11495
  KnowledgeGraph
    https://arxiv.org/abs/2408.07852
  Mutual Reasoning
    https://arxiv.org/abs/2408.06195
  Self-Reasoning
    https://arxiv.org/abs/2407.19813
    https://arxiv.org/abs/2412.14860
  Detecting Hallucinations
    https://arxiv.org/abs/2410.22071
    https://arxiv.org/abs/2410.02707
    https://arxiv.org/abs/2411.14257
  Reflective thinking
    https://arxiv.org/html/2404.09129v1
    https://github.com/yanhong-lbh/LLM-SelfReflection-Eval
  Semantic Entropy
    https://www.nature.com/articles/s41586-024-07421-0
    https://arxiv.org/abs/2406.15927
  Software Packages
    https://arxiv.org/abs/2406.10279
  TruthX
    https://arxiv.org/abs/2402.17811
  Working memory
    https://arxiv.org/abs/2412.18069
  HALVA
    https://research.google/blog/halva-hallucination-attenuated-language-and-vision-assistant/
Long Form Factuality - Google
  https://github.com/google-deepmind/long-form-factuality
  https://deepmind.google/research/publications/85420/



LLM As Judge:
  https://arxiv.org/pdf/2404.12272
  https://arize.com/blog/breaking-down-evalgen-who-validates-the-validators/
  https://huggingface.co/vectara/hallucination_evaluation_model
  https://arxiv.org/pdf/2404.12272
  https://arize.com/blog/breaking-down-evalgen-who-validates-the-validators/


Lynx/patronus
- https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model
- https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-Q4_K_M-GGUF
- https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/community/patronus-lynx.md
- https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/examples/configs/patronusai/prompts.yml
- https://arxiv.org/html/2407.08488v1
- https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/examples/configs/patronusai/prompts.yml
- https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/community/patronus-lynx.md
- https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-Q4_K_M-GGUF
- https://arxiv.org/abs/2407.08488




----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
### <a name="references"></a> References


----------------------------------------------------------------------------------------------------------------