---
title: "Model Evaluation: Measure the Failure You Actually Care About"
description: "Accuracy, precision, recall, F1, ROC, regression error and the deeper problem of matching metrics to decisions."
domain: "Machine Learning"
topic: "Evaluation"
type: "Concept"
section: "Handbook"
language: "en"
level: "Beginner"
priority: "Essential"
order: 6
pubDate: 2026-08-19
readingTime: 14
featured: false
draft: false
prerequisites:
  - Probability
  - Statistics
---
Evaluation is not the final step after modeling. It is the mechanism that defines what counts as improvement.

A poor metric can push development toward the wrong system even when optimization and implementation are flawless.

## Start from the decision

Suppose a classifier predicts whether a transaction is suspicious. Four outcomes exist:

```text
                 actual positive   actual negative
pred positive        TP                FP
pred negative        FN                TN
```

Which error matters more depends on the application.

Precision asks: among predicted positives, how many were correct?

Recall asks: among actual positives, how many did we find?

Neither is universally better.

## Accuracy can hide class imbalance

If only 1% of examples are positive, predicting "negative" every time gives 99% accuracy and solves nothing.

This is why evaluation should reflect the operational class distribution and the value of each outcome.

## F1 is a compromise, not a law

F1 combines precision and recall into one number. It is useful when both matter and you want a compact comparison.

But it silently assumes a particular balance between them. If missing a positive is ten times more costly than investigating a false alarm, F1 may not match the product objective.

## Threshold-free curves still need decisions

ROC and precision-recall curves show performance across thresholds. They are useful for understanding trade-offs, especially before choosing a decision threshold.

But the deployed system eventually acts at some threshold or ranking cutoff. Evaluate that operating point too.

## Regression needs error distributions

Metrics such as MAE and RMSE summarize numeric prediction error differently. RMSE punishes large errors more strongly because it squares them.

Do not stop at one aggregate. Inspect residuals and error by important slices.

## Offline metrics are proxies

A retrieval model can improve recall@k while the final user experience gets worse. A language model can score higher on a judge while becoming less useful for a particular workflow.

The chain is often:

```text
component metric
      ↓
system behavior
      ↓
product outcome
```

Each arrow needs validation.

## Build an evaluation hierarchy

A mature AI system usually needs several layers:

1. **Unit/invariant tests** — deterministic rules that must never break.
2. **Component metrics** — retrieval, classification, extraction, latency.
3. **End-to-end cases** — realistic user tasks.
4. **Slice analysis** — language, intent, segment, data source, difficulty.
5. **Production monitoring** — drift, failures and user outcomes.

A single score cannot replace this structure.

## A metric is a contract

Every metric encodes a claim about what behavior matters. Before optimizing it, make that claim explicit.
