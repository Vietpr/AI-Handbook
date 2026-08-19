---
title: "LLM Training vs Inference: Two Very Different Systems"
description: "Pretraining, instruction tuning and next-token generation explained as separate stages with different engineering constraints."
domain: "Generative AI"
topic: "LLMs"
type: "Concept"
section: "Handbook"
language: "en"
level: "Intermediate"
priority: "High"
order: 4
pubDate: 2026-08-19
readingTime: 15
featured: false
draft: false
prerequisites:
  - Transformers
  - Probability
---
People often talk about "the LLM" as one thing, but training and serving a language model are different computational problems.

## Pretraining learns next-token prediction

An autoregressive language model receives a sequence and learns to predict the next token at each position.

```text
"the sky is" → predict "blue"
```

Across enormous datasets, this simple objective forces the model to learn useful statistical structure about language, code and many domains represented in the data.

The model is not trained with a database of explicit facts. Knowledge is distributed across learned parameters and activated by context.

## Instruction tuning changes interaction behavior

A pretrained model may be capable but awkward to use. Instruction tuning trains on examples of prompts and desired responses so the model learns a more useful interaction format.

Additional alignment techniques can shape helpfulness, safety and preference behavior.

These stages usually adapt a pretrained capability rather than create all knowledge from zero.

## Inference generates one token at a time

At runtime, autoregressive generation is sequential:

```text
prompt
  ↓
next token
  ↓
prompt + token
  ↓
next token
  ↓
...
```

This sequential dependency makes inference latency fundamentally different from training throughput.

## The KV cache avoids repeated attention work

During generation, previous key/value projections can be cached so the model does not recompute them from scratch for every new token.

This improves speed but consumes memory proportional to context length and active sequences.

That trade-off is central to LLM serving systems.

## Sampling controls how probabilities become text

The model outputs a probability distribution over next tokens. Decoding chooses from that distribution.

Common controls include:

- greedy decoding;
- temperature;
- top-k;
- nucleus / top-p sampling.

These do not change the model's learned knowledge. They change how the probability distribution is converted into a concrete sequence.

## Context is temporary working memory

During inference, the model can condition on tokens provided in the current context window. Prompt instructions, retrieved documents, tool outputs and conversation history all compete for that finite space.

That is why LLM application architecture becomes a context-management problem.

## Training constraints and product constraints differ

Training cares about large-scale optimization, data quality, distributed compute and checkpointing.

Inference cares about latency, throughput, memory, batching, context length, cost and reliability.

Application engineering adds another layer: retrieval, tools, permissions, state, evaluation and product behavior.

Separating these layers prevents architecture discussions from treating every problem as something the base model should solve alone.
