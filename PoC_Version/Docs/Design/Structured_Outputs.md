# Structured Outputs

https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
https://towardsdatascience.com/diving-deeper-with-structured-outputs-b4a5d280c208
https://generativeai.pub/building-multi-agent-llm-systems-with-pydanticai-framework-a-step-by-step-guide-to-create-ai-5e41fbba2608
https://github.com/openai/openai-structured-outputs-samples
https://github.com/BoundaryML/baml
https://github.com/jcrist/msgspec
https://arxiv.org/abs/2502.14905

## Introduction
This page serves as documentation regarding the structured outputs within tldw and provides context/justification for the decisions made within the module.

## Structured Outputs
- Structured outputs are useful for generating structured data from unstructured text.

### Use Cases
1. File Creation
   - .ical file (calendar file creation)
   - .json file (structured data)
   - .csv file (Anki Flash cards + structured data)
   - .xml file
   - .yaml file
   - .toml file
   - 
2. Data Extraction
   - https://github.com/yobix-ai/extractous
   - Can use structured outputs for data extraction from unstructured text. Though why isn't this talked about/even mentioned in any of the papers about RAG or writeups on RAG implementations? hmmmm......
3. Data Generation
   - Can use structured outputs for data generation from unstructured text.
   - Could come in handy for RPGs/Text-based games reliant on world building/lore generation.


### Implementation
- Integration for file creation
- Look at using for ETL pipeline
- Support/integration for content creation pipelines for RPG campaigns, etc.


Process
   https://python.plainenglish.io/generating-perfectly-structured-json-using-llms-all-the-time-13b7eb504240

Tools
   https://python.useinstructor.com/
   https://github.com/mlc-ai/xgrammar
   https://github.com/guidance-ai/guidance
   https://github.com/boundaryml/baml
   https://docs.pydantic.dev/latest/
   https://github.com/outlines-dev/outlines
   https://github.com/Dan-wanna-M/formatron/tree/master
   https://github.com/whyhow-ai/knowledge-table
   https://github.com/guardrails-ai/guardrails
   https://arena-ai.github.io/structured-logprobs/

Examples
   https://github.com/dottxt-ai/demos/tree/main/lore-generator
   https://github.com/dottxt-ai/demos/tree/main/logs
   https://github.com/dottxt-ai/demos/tree/main/earnings-reports
   https://github.com/dottxt-ai/demos/tree/main/its-a-smol-world
   https://github.com/dottxt-ai/cursed/tree/main/scp


Reliability/Quality of:
   https://dylancastillo.co/posts/say-what-you-mean-sometimes.html
   https://blog.dottxt.co/say-what-you-mean.html

Papers
   https://arxiv.org/html/2408.02442v1 - Structured Outputs harms reasoning capabilities


Gemini
   https://ai.google.dev/gemini-api/docs/structured-output?lang=python

### Link Dump:
