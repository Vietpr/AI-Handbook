---
title: "Linear Regression From Scratch"
description: "Implement the full learning loop with arrays, gradients and no machine-learning framework."
domain: "Machine Learning"
topic: "Regression"
type: "Concept"
section: "Learn"
language: "en"
level: "Beginner"
pubDate: 2026-08-10
readingTime: 10
featured: false
draft: false
prerequisites:
  - Vectors
  - Derivatives
---
Linear regression is simple enough to fit in a few lines and rich enough to expose the entire learning loop.

## Model

For one feature, the prediction is a line parameterized by a weight and bias.

The interesting part is not the formula. It is how the parameters move from a bad guess toward a better one.

## Training loop

```python
w = 0.0
b = 0.0

for step in range(1000):
    y_hat = w * x + b
    error = y_hat - y

    dw = 2 * mean(error * x)
    db = 2 * mean(error)

    w -= lr * dw
    b -= lr * db
```

That loop already contains the pattern reused by much larger models: forward computation, objective, gradient and parameter update.

## Why build this manually

Framework APIs are designed to remove bookkeeping. That is excellent for shipping code and sometimes bad for building intuition.

Once the small version is obvious, automatic differentiation feels like an acceleration of a known process rather than magic.
