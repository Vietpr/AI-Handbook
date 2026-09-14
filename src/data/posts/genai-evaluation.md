---
title: "Evaluating Generative AI: From Vibes to Failure Taxonomies"
description: "How to evaluate RAG, tool use and generated answers with layered metrics, golden cases, traces and targeted human review."
domain: "Generative AI"
topic: "Evaluation"
type: "Concept"
section: "Learn"
language: "en"
level: "Intermediate"
priority: "Essential"
order: 8
pubDate: 2026-08-19
readingTime: 17
featured: false
draft: false
prerequisites:
  - Model evaluation
  - RAG
  - Tool calling
---
Generative systems are harder to evaluate than classifiers because many outputs can be acceptable and failures can occur at several hidden stages.

The solution is not to give up on measurement. It is to **decompose the behavior you care about**.

## Start with the task contract

Before choosing a judge model or metric, define what a correct result means.

For a data-analysis assistant, the contract may include:

- correct time period;
- correct entity/store scope;
- correct data source;
- correct arithmetic;
- evidence-backed explanation;
- no unsupported claims;
- required output format.

These criteria are more useful than a vague "response quality" score.

## Separate deterministic checks from semantic judgment

Some behaviors can be checked exactly:

```text
HTTP success
schema valid
correct tenant
SQL read-only
required fields present
known arithmetic correct
```

Others require semantic evaluation:

```text
answer addresses the question
reasoning is supported by evidence
summary preserves the important finding
```

Use deterministic checks wherever possible and reserve LLM/human judgment for genuinely semantic properties.

## Evaluate the pipeline, not only the final answer

For RAG or agents, inspect stages:

```text
intent / route
    ↓
retrieval / tool selection
    ↓
arguments / query
    ↓
tool result
    ↓
context construction
    ↓
final response
```

If you only score the final text, a retrieval failure and a reasoning failure look identical.

## Golden datasets should evolve from failures

A useful golden set is not just a static benchmark. It should accumulate representative product behaviors and real failure cases.

When production reveals a new bug:

```text
failure
  ↓
reproducible testcase
  ↓
failure category
  ↓
fix
  ↓
regression test
```

Over time, the suite becomes product memory.

## LLM-as-a-judge needs calibration

Judge models are useful for scaling semantic review, but they are not ground truth.

Validate judge behavior against human-labeled subsets. Use clear rubrics. Prefer evidence-based criteria over open-ended scoring. Track disagreements.

If a judge is unreliable for a category, do not hide that under an aggregate average.

## Slice results by failure mode

Overall pass rate can hide important regressions. Useful slices include:

- intent;
- language;
- tool path;
- data source;
- multi-turn versus single-turn;
- difficulty;
- entity type;
- failure taxonomy.

The purpose of evaluation is not only ranking versions. It is locating where the system should improve.

## Production feedback closes the loop

Offline evaluation cannot cover every future user behavior. Production traces, support issues and explicit user feedback should feed new cases back into the test suite.

That creates a continuous loop between real behavior and controlled measurement.

## The durable principle

Do not ask for one number that proves the system is good.

Build an evaluation system that can answer:

> **What failed, where did it fail, how often does it matter, and did the latest change actually fix that class of failure without breaking another one?**
