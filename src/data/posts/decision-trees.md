---
title: "Decision Trees: Learning by Splitting the Feature Space"
description: "Understand recursive splits, impurity, overfitting and why tree ensembles are so strong on structured data."
domain: "Machine Learning"
topic: "Trees"
type: "Concept"
section: "Learn"
language: "en"
level: "Beginner"
priority: "High"
order: 4
pubDate: 2026-08-19
readingTime: 12
featured: false
draft: false
prerequisites:
  - Basic probability
---
Decision trees learn a prediction by repeatedly asking simple questions about features.

```text
age < 30?
├─ yes → income < 50k?
│        ├─ yes → class A
│        └─ no  → class B
└─ no  → class B
```

The power comes from composing simple splits into nonlinear decision regions.

## A split tries to create purer groups

At each node, the algorithm considers candidate feature thresholds and chooses a split that improves an objective.

For classification, common criteria measure class impurity. For regression, splits may reduce variance or squared error.

The exact formula matters less than the intuition: **a good split creates child groups that are easier to predict than the parent group**.

## Trees naturally model interactions

A linear model adds feature contributions. A tree can express rules like:

> feature A matters only when feature B is above a threshold.

This makes trees effective for structured/tabular datasets where nonlinear interactions and mixed feature scales are common.

## A single tree overfits easily

If allowed to keep splitting, a tree can isolate tiny subsets or individual training examples. Training error falls while generalization worsens.

Controls such as maximum depth, minimum samples per leaf and pruning limit this complexity.

## Ensembles make trees much stronger

Two major strategies dominate:

- **bagging / random forests**: train many diverse trees and average their predictions;
- **boosting**: train trees sequentially so later trees focus on errors left by earlier ones.

Modern gradient-boosted trees are difficult baselines to beat on many tabular problems.

## Feature importance needs caution

Tree libraries often expose feature-importance scores, but these can be biased or misleading. Importance says a feature helped prediction under the fitted model; it does not prove causal influence.

Permutation-based analysis or SHAP-style explanations can add perspective, but interpretation still depends on the data-generating process.

## When trees are a good first choice

Consider tree-based models when you have:

- structured tables;
- nonlinear feature interactions;
- missing values or heterogeneous feature scales;
- limited need for learned representations from raw images/text;
- a strong requirement for fast baselines.

Deep learning is not automatically the default for every dataset. On tabular data, trees often deserve to be tested early.
