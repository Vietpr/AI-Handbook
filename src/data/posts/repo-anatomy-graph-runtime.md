---
title: "Repository Anatomy: A Minimal Agent Graph Runtime"
description: "The core abstractions hiding underneath graph-based agent frameworks, rebuilt as a small execution model."
domain: "Generative AI"
topic: "Agents"
type: "Concept"
section: "Learn"
language: "en"
level: "Advanced"
pubDate: 2026-08-13
readingTime: 18
featured: true
draft: false
prerequisites:
  - State machines
  - Tool calling
  - Async Python
---
Graph-based agent libraries can look sophisticated because they expose many production features at once. Strip them down and the core runtime is surprisingly small.

## The problem the graph solves

An agent workflow has steps, shared state and decisions about where execution should go next.

```text
state
  ↓
node A ── condition ──→ node B
  │                     │
  └──── condition ─────→ node C
```

A graph makes the control flow explicit instead of burying it inside one large prompt loop.

## The minimum abstraction

A tiny runtime needs only a few concepts:

```python
class Graph:
    nodes: dict[str, Callable]
    edges: dict[str, list[str]]
    router: Callable[[str, dict], str]

    async def run(self, start, state):
        current = start
        while current != "END":
            patch = await self.nodes[current](state)
            state.update(patch)
            current = self.router(current, state)
        return state
```

Real frameworks add persistence, retries, streaming, interrupts, tracing, concurrency and human approval. Those features matter enormously in production, but they are layers around the execution idea above.

## Why state is the center

Nodes are often ordinary functions. Edges are often ordinary decisions. The difficult part is defining state that can evolve without turning into an untyped bag of everything the system has ever seen.

A useful state schema answers three questions:

- What does the next node actually need?
- Which fields are authoritative?
- Which fields are transient versus durable?

## Where the abstraction leaks

Graph frameworks are less magical when routing depends on fuzzy model judgment, when state grows without discipline, or when every node can mutate everything.

At that point, the graph is visible but the semantics are still hidden.

## When I would use one

Use a graph when execution genuinely has branches, resumable checkpoints, human interrupts or multiple tools with explicit orchestration.

For a two-step deterministic pipeline, a graph can be more abstraction than architecture.
