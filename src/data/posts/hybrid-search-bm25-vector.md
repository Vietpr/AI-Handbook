---
title: "Hybrid Search: Why BM25 and Embeddings Need Each Other"
description: "A retrieval mental model for combining exact lexical evidence with semantic similarity."
domain: "AI Systems"
topic: "Retrieval"
type: "Concept"
section: "Learn"
language: "en"
level: "Intermediate"
pubDate: 2026-08-17
readingTime: 12
featured: true
draft: false
prerequisites:
  - Embeddings
  - Cosine similarity
---
Dense retrieval is good at meaning. Lexical retrieval is good at exact evidence. Real search systems frequently need both.

## Two different questions

BM25 asks roughly: **do the query terms provide strong evidence that this document is relevant?**

Vector search asks: **does this document live near the query in representation space?**

Those are related questions, not identical ones.

## Where dense retrieval loses

Exact identifiers, product codes, error messages and rare names can be semantically uninteresting while being operationally decisive.

A vector can smooth away the detail you care about.

## Where lexical retrieval loses

Natural language varies. A document can answer a question without sharing the user's wording. Synonyms, paraphrases and multilingual queries expose the limits of token overlap.

## Fusion

A practical hybrid pipeline keeps the retrievers independent long enough to preserve their strengths:

```text
query
 ├── lexical retriever ──→ ranked list A
 └── dense retriever   ──→ ranked list B
                              │
                        rank fusion
                              ↓
                         final candidates
```

Reciprocal Rank Fusion is attractive because it combines ranks without pretending the raw scores from different retrievers live on the same scale.

## Retrieval is still an evaluation problem

Adding another retriever is not automatically an improvement. Build a query set, label useful evidence and measure what each retrieval stage contributes.

The architecture should be justified by failure cases, not by the number of search techniques in the diagram.
