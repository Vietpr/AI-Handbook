---
title: "Generalization, Overfitting and Regularization"
description: "Why fitting training data is not the goal, how overfitting appears and what regularization is really trying to control."
domain: "Machine Learning"
topic: "Generalization"
type: "Concept"
section: "Handbook"
language: "en"
level: "Beginner"
priority: "Essential"
order: 5
pubDate: 2026-08-19
readingTime: 13
featured: false
draft: false
prerequisites:
  - What machine learning means
---
A model is valuable because it works on examples it did not directly train on. That property is **generalization**.

Overfitting is what happens when the model learns the training set more specifically than the underlying task requires.

## Training performance can be misleading

Imagine two models:

```text
Model A: train 99%, test 72%
Model B: train 91%, test 88%
```

Model A fits the observed training data better. Model B is probably the better predictive model.

The gap between training and validation behavior is one signal of overfitting.

## Capacity changes what can be memorized

A model with more capacity can represent more complicated functions. That can help capture real structure, but it can also capture noise and accidental correlations.

Overfitting is therefore not simply "the model is too big." It is a relationship between model capacity, dataset size, noise, task complexity and training procedure.

## Regularization changes the preference among solutions

Regularization discourages certain solutions even if they fit training data well.

Examples include:

- L1 or L2 penalties on weights;
- limiting tree depth;
- dropout;
- data augmentation;
- early stopping;
- weight decay;
- architectural constraints.

These methods differ mechanically, but they share a purpose: **bias learning toward solutions expected to transfer better**.

## Validation is part of the training process

Hyperparameters are often selected using validation performance. That means the validation set indirectly influences model development.

Repeatedly tuning against the same validation set can itself lead to overfitting. A final untouched test set remains useful when you need an honest estimate after model selection.

## Leakage can look like amazing generalization

Some failures are worse than ordinary overfitting. If information from the future, target label or test split leaks into training features, metrics can look excellent while the deployed model collapses.

Always check whether a feature would truly be available at prediction time.

## Distribution shift changes the game

A model can generalize well to data drawn from the same distribution as its test set and still fail when production behavior changes.

Examples:

- new customer segments;
- a new camera type;
- different language patterns;
- policy changes;
- seasonal behavior.

Generalization is always relative to a distribution.

## The practical objective

Do not ask only:

> How low can I make training loss?

Ask:

> What structure do I expect to remain true when the model meets new data?

Regularization, validation design and evaluation slices are all attempts to answer that second question.
