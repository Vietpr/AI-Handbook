---
title: "Attention: Let Each Token Decide What Matters"
description: "A geometric and operational explanation of queries, keys, values and why attention changed sequence modeling."
domain: "Generative AI"
topic: "Attention"
type: "Concept"
section: "Learn"
language: "en"
level: "Intermediate"
priority: "Essential"
order: 2
pubDate: 2026-08-19
readingTime: 15
featured: true
draft: false
prerequisites:
  - Vectors and matrices
  - Tokenization
---
Attention gives each token a way to gather information from other tokens based on learned relevance.

The core idea is easier to understand if you ignore the full transformer for a moment.

## Three views of each token

From each token representation, the model produces three vectors:

- **query** - what information am I looking for?
- **key** - what kind of information do I offer?
- **value** - what information should be passed along if I am selected?

These are learned linear projections of the same underlying token state.

## Queries compare with keys

For one token, its query is compared with every candidate key using a dot product.

```text
query(token i)
   ├─ compare → key(token 1)
   ├─ compare → key(token 2)
   ├─ compare → key(token 3)
   └─ ...
```

The resulting scores are normalized into attention weights. Larger weights mean the current token will take more information from that position.

## Values are mixed using the weights

The output is a weighted combination of value vectors:

```text
attention weights
      ×
value vectors
      ↓
contextualized representation
```

So attention is not only "looking at" another token. It is a content-dependent information-routing mechanism.

## Why scale the dot product?

As vector dimension grows, raw dot products can become large. The standard scaled dot-product attention divides by the square root of key dimension before softmax.

The practical reason is optimization stability: without scaling, softmax can become extremely sharp and gradients can become less useful.

## Multi-head attention learns multiple relationships

Instead of one attention operation over the full representation, transformers use several heads with different learned projections.

Different heads can specialize in different patterns, although interpreting individual heads as clean human concepts is risky.

The important architectural effect is that the model has multiple parallel subspaces for routing information.

## Causal masking changes what can be seen

Autoregressive language models must not use future tokens to predict the next token. A causal mask blocks attention from looking ahead.

```text
position 1 → can see 1
position 2 → can see 1..2
position 3 → can see 1..3
```

This preserves the next-token prediction setup during training.

## Attention is powerful, not magical memory

A model can attend only to information available in its current context. Attention does not create long-term memory by itself.

If a fact is outside the context window, another system must retrieve or summarize it before attention can use it.

That distinction becomes crucial in RAG and agent memory architectures.

## The reusable mental model

Think of attention as:

> learned relevance scoring + weighted information mixing.

That model is enough to understand why attention appears in transformers, cross-modal models and many retrieval-like neural architectures.
