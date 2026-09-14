---
title: "Tool Calling and Agents: Give the Model Actions, Not Omnipotence"
description: "A grounded model of tools, agent loops, state and why reliable control flow matters more than calling everything an agent."
domain: "Generative AI"
topic: "Agents"
type: "Concept"
section: "Learn"
language: "en"
level: "Intermediate"
priority: "High"
order: 7
pubDate: 2026-08-19
readingTime: 16
featured: false
draft: false
prerequisites:
  - Context engineering
  - LLM inference
---
A language model produces text. Tool calling extends that interface so the model can request structured actions such as searching data, running code or calling an API.

An agent is usually a system that repeatedly combines model decisions with tools and state until a task reaches a stopping condition.

## A tool is an explicit capability boundary

A good tool has:

- a clear name and purpose;
- typed inputs;
- constrained outputs;
- deterministic permission checks;
- observable execution.

The model proposes a call. Application code decides whether and how that call actually executes.

```text
model decision
     ↓
validated tool request
     ↓
application / API / database
     ↓
structured result
     ↓
model context
```

This boundary is important for safety and correctness.

## The smallest agent loop

A minimal loop can be expressed as:

```text
observe state
    ↓
model chooses next action
    ↓
execute tool or finish
    ↓
update state
    ↺
```

Frameworks add retries, persistence, streaming, human approval, tracing and concurrency. Those features matter, but they sit around this core control loop.

## State should remain explicit

If the only source of truth is the conversation transcript, the model must infer workflow state repeatedly.

Explicit state can hold:

- selected entities;
- confirmed date ranges;
- completed steps;
- tool results;
- retry counts;
- pending approvals.

That gives the application deterministic control where determinism is available.

## Routing is probabilistic when the model decides

An LLM router can misclassify intent. A production design should therefore think in terms of confidence, recovery and bounded consequence rather than assuming routing is perfect.

Useful patterns include:

- deterministic routing for obvious structured actions;
- model routing for ambiguous semantic cases;
- confirmation when the cost of a wrong route is high;
- repair paths when tools fail;
- bounded rounds to prevent loops.

## More agents do not imply more intelligence

Splitting one task across many model roles can increase context duplication, latency and coordination errors.

Use multiple agents when the decomposition creates real isolation or parallel value, not because a diagram looks sophisticated.

Often a single orchestrator with well-designed tools and explicit state is easier to evaluate and maintain.

## Reliability lives outside the model too

Security, tenant isolation, permissions, schema validation, idempotency and read/write boundaries should be enforced in code.

Do not ask a language model to "remember" invariants that the application can guarantee deterministically.

## A practical definition

Instead of asking whether a system is a "true agent," ask:

> What decisions are delegated to the model, what actions can it take, what state persists, and what boundaries remain deterministic?

That description tells you far more about the architecture.
