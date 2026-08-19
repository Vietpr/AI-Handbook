---
title: "Linear Regression: The Smallest Useful Learning Model"
description: "Use linear regression to understand features, parameters, residuals, loss and what a model is really fitting."
domain: "Machine Learning"
topic: "Regression"
type: "Concept"
section: "Handbook"
language: "en"
level: "Beginner"
priority: "High"
order: 2
pubDate: 2026-08-19
readingTime: 10
featured: false
draft: false
prerequisites:
  - Vectors and matrices
  - Gradient descent
---
Linear regression is simple enough to inspect completely, which makes it one of the best models for learning what supervised training actually does.

## The model is a weighted combination of features

For several features:

```text
prediction = w1*x1 + w2*x2 + ... + wn*xn + bias
```

The weights are parameters learned from data. Their job is to turn input features into a prediction.

In vector form, the same idea becomes a dot product plus bias.

## Residuals are the errors left behind

For each example:

```text
residual = prediction - target
```

A good fit should leave residuals that are small and, ideally, free from obvious structure. If residuals systematically grow with the target or change across groups, the model is missing something.

## Mean squared error rewards small residuals

A common objective squares each residual and averages them. Squaring makes large errors expensive and removes sign cancellation.

This is useful but not neutral. If your data contains extreme outliers, squared error can let those cases dominate the fit.

The loss function encodes priorities.

## Coefficients are useful, but interpretation needs care

A positive weight means that, holding other included features fixed, increasing that feature increases the model prediction.

That does **not** automatically imply causation. Correlated features, omitted variables and dataset design can all distort interpretation.

## Linear does not mean useless

Linear models can work surprisingly well when:

- relationships are approximately additive;
- features already encode useful nonlinear structure;
- interpretability matters;
- data is limited;
- you need a strong baseline.

They are also fast to train and easy to debug.

## Why learn this before neural networks

Linear regression exposes the full learning pipeline without hiding it behind architecture complexity:

```text
features
  ↓
weighted prediction
  ↓
residual
  ↓
loss
  ↓
gradient
  ↓
updated weights
```

Neural networks keep the same outer loop. They replace the simple linear mapping with many learned transformations.

If this pipeline is clear, bigger models become extensions rather than a new universe.
