---
title: "Logistic Regression: From Scores to Class Probabilities"
description: "Why a linear decision boundary plus a sigmoid becomes a powerful baseline for binary classification."
domain: "Machine Learning"
topic: "Classification"
type: "Concept"
section: "Handbook"
language: "en"
level: "Beginner"
priority: "High"
order: 3
pubDate: 2026-08-19
readingTime: 11
featured: false
draft: false
prerequisites:
  - Linear regression
  - Probability
---
Classification asks for discrete decisions, but optimization works more smoothly with continuous scores. Logistic regression connects the two.

## Start with a linear score

The model first computes:

```text
z = w · x + b
```

`z` can be any real number. Large positive values indicate evidence toward one class; large negative values indicate evidence toward the other.

## The sigmoid maps scores into 0 to 1

The sigmoid compresses the real line into a range that can be interpreted as a probability under the model.

```text
very negative z → probability near 0
z = 0           → probability 0.5
very positive z → probability near 1
```

A threshold such as `0.5` can then convert probability into a class decision.

## The threshold belongs to the product

The default threshold is not sacred.

For fraud detection, missing a true fraud case may be more expensive than investigating an extra false positive. For an automated blocking system, false positives may be extremely costly.

So model probability and decision threshold should be separated:

```text
model → probability → business threshold → action
```

## Cross-entropy rewards probabilistic correctness

Classification commonly uses log loss / cross-entropy. It heavily penalizes being confidently wrong and rewards assigning high probability to the correct outcome.

This makes the training objective align better with probability estimation than squared error would.

## The boundary is still linear

Despite the nonlinear sigmoid, the decision boundary in the original feature space is linear.

That is both a limitation and a feature. Logistic regression is easy to interpret, stable and often a strong baseline. If it performs almost as well as a complex model, the simpler model may be the better engineering choice.

## Where it fails

Logistic regression struggles when useful boundaries are strongly nonlinear and the features do not expose that structure.

Feature engineering can help. So can moving to trees or neural networks.

## Why it remains important

Logistic regression teaches a pattern reused across machine learning:

1. produce a score;
2. turn the score into a probability distribution;
3. optimize likelihood-like loss;
4. make a downstream decision with a threshold.

Understanding that separation makes later classifiers easier to reason about.
