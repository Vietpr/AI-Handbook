---
title: "Agent Memory Is a Context Architecture Problem"
description: "Why durable memory is less about storing messages and more about deciding what becomes context, when, and for whom."
domain: "Generative AI"
topic: "Memory"
type: "Concept"
section: "Handbook"
language: "en"
level: "Advanced"
pubDate: 2026-08-19
updatedDate: 2026-08-19
readingTime: 15
featured: true
draft: false
prerequisites:
  - Context windows
  - Embeddings
  - Agent state
---
Most diagrams reduce agent memory to a database attached to an LLM. That picture is useful, but it hides the harder problem.

A memory system has to decide **what deserves to survive**, **how it is represented**, **who can read it**, and **when it should re-enter the model context**.

## State is not memory

State is what the workflow needs to continue correctly right now. Memory is information that may remain useful after the current step, turn, or session is gone.

```text
request
  ↓
working state ──→ model context
  │
  └── candidate memories
           ↓
      memory policy
           ↓
      durable store
```

If every state field becomes memory, the system accumulates noise. If nothing becomes memory, every new turn starts from zero.

## Four decisions every memory layer makes

### 1. Selection

A raw conversation is not yet a memory. Selection decides which observations are durable enough to keep.

Useful candidates can include stable preferences, verified facts, decisions, unresolved tasks and summaries of long-running work.

### 2. Representation

The same event can be stored as raw text, a structured fact, a compressed summary, an embedding, or several of these at once.

The representation determines what retrieval can do later.

### 3. Retrieval

Retrieval is not only semantic similarity. Time, entity, scope, workflow state and permissions can be as important as embedding distance.

### 4. Injection

Even a correct memory can hurt if it enters the wrong prompt. Injection is a context-budget and relevance decision.

## Shared memory changes the architecture

In a multi-capability product, separate memory per feature creates artificial amnesia. A report flow may know something the free-chat flow cannot see, even though both are serving the same user and task domain.

The better abstraction is usually a shared memory layer with explicit scopes:

| Scope | Example | Typical lifetime |
|---|---|---|
| Turn | tool result | seconds |
| Session | current analysis | minutes |
| User | preference | months |
| Workspace | shared decision | months |
| Entity | store / project fact | variable |

## The useful mental model

Think of memory as a **context compiler**. Storage is only one stage. The actual product is the policy that converts a large history of possible information into the smallest context that lets the model make the next correct decision.

That is why memory quality is ultimately evaluated at the behavior layer, not by how many vectors were successfully written.
