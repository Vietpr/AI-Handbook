---
title: "Transformers: A Stack for Building Contextual Representations"
description: "How embeddings, attention, feed-forward layers, residual paths and normalization combine into the transformer architecture."
domain: "Generative AI"
topic: "Transformers"
type: "Concept"
section: "Handbook"
language: "en"
level: "Intermediate"
priority: "Essential"
order: 3
pubDate: 2026-08-19
readingTime: 16
featured: false
draft: false
prerequisites:
  - Tokenization
  - Attention
  - Gradient descent
---
A transformer is not one operation. It is a repeated architecture for turning token representations into increasingly contextual representations.

## Start with token representations

Token IDs are mapped to embeddings. The model also needs information about position because pure attention does not inherently know token order.

The result is a sequence of vectors:

```text
token 1 → vector
 token 2 → vector
 token 3 → vector
```

## A transformer block has two main computations

A simplified block looks like:

```text
input states
    ↓
self-attention
    ↓
residual + normalization
    ↓
feed-forward network
    ↓
residual + normalization
    ↓
output states
```

Implementation details vary, especially around normalization placement, but the conceptual roles remain similar.

## Attention mixes information across positions

Self-attention allows each token position to gather context from other positions.

This is the sequence-interaction step.

## The feed-forward network transforms each position

After tokens exchange information through attention, a small neural network is applied independently to each position.

You can think of this as giving each token representation a richer nonlinear transformation after contextual information has been collected.

## Residual connections preserve information flow

Instead of replacing a representation completely, residual paths add transformed information back to the previous state.

This makes deep networks easier to optimize and allows layers to refine representations rather than rebuild them from scratch.

## Normalization stabilizes the computation

Normalization keeps activations in a manageable range and improves training dynamics. Modern transformer variants differ in exact normalization strategy, but it remains a critical part of deep stable training.

## Stack many blocks

A transformer model repeats this block many times.

Early layers may capture local or syntactic patterns; deeper layers can build more task-relevant or semantic representations. Real behavior is distributed across the network rather than cleanly assigned one concept per layer.

## Encoder, decoder and decoder-only variants

The original transformer architecture uses an encoder and decoder. Modern LLMs are often decoder-only because next-token generation fits naturally with causal self-attention.

Other tasks still use encoder-only or encoder-decoder architectures.

The shared foundation is attention-based contextual representation.

## Why transformers scaled

Several properties made transformers attractive:

- parallel processing across training tokens compared with recurrent models;
- flexible long-range interactions;
- architecture that scales with data and compute;
- reusable pretraining across many downstream tasks.

They are not free of limitations: attention cost grows quickly with sequence length, context is finite and inference can be expensive.

## The mental stack

When you hear "transformer," visualize:

```text
tokens
  ↓
embeddings + position
  ↓
[ attention → feed-forward ] × many layers
  ↓
contextual token states
  ↓
task-specific output
```

That is the skeleton underneath modern language models.
