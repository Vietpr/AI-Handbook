---
title: "Context Engineering: Decide What the Model Gets to See"
description: "Prompts are only one part of the problem. Context engineering manages instructions, state, retrieval, tools and memory under a finite budget."
domain: "Generative AI"
topic: "Context Engineering"
type: "Concept"
section: "Handbook"
language: "en"
level: "Intermediate"
priority: "Essential"
order: 6
pubDate: 2026-08-19
readingTime: 16
featured: true
draft: false
prerequisites:
  - LLM training and inference
  - RAG
---
Prompt engineering asks how to phrase instructions. Context engineering asks a larger question:

> **What information should be available to the model for this decision, in what form and at what time?**

That framing is more useful for production LLM systems.

## The context window is a scarce workspace

A model has a finite context window. Into that space, an application may need to fit:

```text
system policy
user request
conversation history
workflow state
retrieved evidence
tool results
memory
output constraints
```

Adding more information is not always better. Irrelevant context increases cost, can dilute important instructions and may introduce contradictions.

## Context has different authority levels

Not all tokens should be treated as equally trustworthy.

A useful architecture distinguishes:

- product/system policy;
- verified structured state;
- retrieved source material;
- user-provided claims;
- model-generated hypotheses.

Mixing these into one unstructured transcript makes provenance and control harder.

## State is not conversation history

A workflow may need facts such as selected store, date range, report type or confirmed intent. These should often live as explicit state rather than being reconstructed from raw chat every turn.

```text
conversation
    ↓ parse / update
structured state
    ↓
context builder
    ↓
model
```

This reduces avoidable ambiguity.

## Retrieval should be conditional

Not every question needs RAG. Not every tool result belongs in context. Context engineering includes deciding **whether** a source is needed before deciding how to inject it.

This can reduce latency and prevent irrelevant evidence from steering the model.

## Compression is lossy by design

Long conversations and documents may need summarization. A summary preserves some information and discards other information.

Good compression therefore depends on future use. A generic summary may lose exact dates, decisions or unresolved issues that a workflow still needs.

Structured memory and task-aware summaries are often safer than repeatedly compressing arbitrary text.

## Context assembly should be observable

When an answer is wrong, engineers should be able to inspect:

```text
what instructions were present?
what state was carried over?
what evidence was retrieved?
what tool outputs were included?
what was omitted due to budget?
```

Without this trace, teams compensate by adding more prompt text and hoping behavior improves.

## Context engineering is architecture

As LLM applications become more capable, the hard problem shifts from writing one perfect prompt to coordinating many information sources around the model.

The model remains important, but the application determines what reality the model gets to see.
