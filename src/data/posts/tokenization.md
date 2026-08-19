---
title: "Tokenization: How Text Becomes Model Input"
description: "Why language models operate on tokens rather than words, how subword tokenization works and what token boundaries affect."
domain: "Generative AI"
topic: "Tokenization"
type: "Concept"
section: "Handbook"
language: "en"
level: "Beginner"
priority: "Essential"
order: 1
pubDate: 2026-08-19
readingTime: 11
featured: false
draft: false
prerequisites: []
---
Language models do not directly read words or characters. They operate on sequences of discrete token IDs.

Tokenization is the conversion layer between human text and the symbols the model was trained to process.

## Why not use whole words?

A vocabulary containing every possible word would be enormous and still fail on new names, typos, code and multilingual text.

Character-level modeling avoids unknown words but creates much longer sequences.

Subword tokenization is a compromise: common patterns can be represented compactly while rare strings can be decomposed into smaller pieces.

## Text becomes IDs

A simplified pipeline looks like:

```text
"retrieval works"
      ↓ tokenizer
["retr", "ieval", " works"]
      ↓ vocabulary lookup
[4812, 938, 2201]
```

Those IDs are then mapped to vectors before entering the transformer.

The exact segmentation depends on the tokenizer. Do not assume one token equals one word.

## Token boundaries affect cost and context

Model context windows and API usage are usually measured in tokens. Two texts with the same number of characters can consume different token counts.

Code, uncommon names and some languages may tokenize differently from ordinary English prose.

This matters when designing:

- prompt budgets;
- chunk sizes;
- truncation policies;
- memory compression;
- cost estimates.

## Tokens affect model behavior

The model learns statistics over token sequences. Awkward tokenization can make exact spelling, identifiers or arithmetic harder because the underlying symbol boundaries do not align with the conceptual unit a human sees.

A product code may split into many pieces. A rare Japanese term may be represented differently from a common phrase. These details can matter in retrieval, generation and fine-tuning.

## Tokenization is fixed at model training time

For a pretrained model, the tokenizer is part of the model interface. You cannot casually replace it without changing how learned embeddings and model parameters correspond to token IDs.

Fine-tuning typically keeps the original tokenizer unless the model is deliberately adapted in a more fundamental way.

## What to remember

When debugging an LLM system, inspect tokenization when:

- context length seems surprising;
- exact identifiers are unreliable;
- multilingual behavior differs;
- prompts are unexpectedly expensive;
- truncation removes useful information.

Tokens are not an implementation detail. They define the units on which the language model operates.
