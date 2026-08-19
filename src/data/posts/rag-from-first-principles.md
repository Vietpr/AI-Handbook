---
title: "RAG From First Principles: Retrieve Evidence, Then Generate"
description: "A clean architecture for retrieval-augmented generation, what retrieval actually contributes and the failure modes hidden by simple diagrams."
domain: "Generative AI"
topic: "RAG"
type: "Concept"
section: "Handbook"
language: "en"
level: "Intermediate"
priority: "Essential"
order: 5
pubDate: 2026-08-19
readingTime: 17
featured: true
draft: false
prerequisites:
  - Embeddings
  - Transformers
---
Retrieval-augmented generation gives a language model access to external evidence at inference time.

The important idea is not "put documents in a vector database." The important idea is **construct the right evidence context for the current question**.

## The minimal RAG loop

```text
user question
      ↓
query representation
      ↓
retrieve candidate evidence
      ↓
select / rerank / filter
      ↓
construct model context
      ↓
LLM generates answer
```

Each arrow can fail independently.

## Retrieval solves a different problem from generation

The generator is good at synthesizing language from available context. It should not be expected to reliably reproduce private, changing or highly specific knowledge from model parameters.

Retrieval supplies relevant external information when it is needed.

That separation improves freshness, provenance and controllability.

## Chunking defines the retrieval unit

Documents are usually split into chunks before indexing. Chunking decides what the retriever can return as one unit.

Too small:

- context can lose relationships;
- answers need information spread across many chunks.

Too large:

- retrieval becomes less precise;
- context budget is wasted;
- one relevant sentence drags in unrelated text.

Chunking should follow document structure and retrieval tasks, not an arbitrary token number alone.

## Dense retrieval is not always enough

Embeddings capture semantic similarity well, but exact identifiers, rare terms and product codes may be better served by lexical search.

Real systems often combine:

- metadata filters;
- BM25 / lexical retrieval;
- dense vector retrieval;
- reranking;
- relation expansion.

The right stack is determined by observed retrieval failures.

## Retrieval quality and answer quality are different

A correct document can be retrieved and still be ignored or misinterpreted by the model. Conversely, a fluent answer can hide missing evidence.

Evaluate at least two layers:

```text
Did retrieval find the needed evidence?
                ↓
Did generation use that evidence correctly?
```

Blending them into one pass/fail score makes debugging harder.

## RAG is context engineering

The final prompt often contains more than retrieved chunks:

- system instructions;
- user request;
- conversation state;
- retrieved documents;
- tool outputs;
- structured metadata.

RAG therefore becomes part of a broader context architecture. Retrieval is valuable only if selected evidence enters the model in a form the model can use.

## Common failure modes

RAG systems frequently fail because of:

- wrong query interpretation;
- missing metadata filters;
- poor chunk boundaries;
- top-k too small or too large;
- embedding mismatch with domain language;
- outdated index contents;
- weak reranking;
- conflicting retrieved evidence;
- model answering beyond the evidence.

The vector database is rarely the whole story.

## The durable mental model

RAG is an **evidence pipeline**.

Treat retrieval, selection, context construction and generation as explicit stages with separate traces and evaluation. Once you do that, RAG becomes much easier to improve systematically.
