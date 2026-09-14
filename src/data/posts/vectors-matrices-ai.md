---
title: "Vectors and Matrices: The Language Models Compute In"
description: "A practical mental model for vectors, matrices, dot products and why so much of AI reduces to transforming representations."
domain: "Foundations"
topic: "Linear Algebra"
type: "Concept"
section: "Learn"
language: "en"
level: "Beginner"
priority: "Essential"
order: 1
pubDate: 2026-08-19
readingTime: 12
featured: false
draft: false
prerequisites: []
---
A large part of modern AI can be understood as a sequence of transformations applied to vectors. Images, words, users and intermediate model states all eventually become numerical representations that can be compared and transformed.

You do not need all of linear algebra before learning AI. You do need a strong mental model for **vectors, matrices, dot products and dimensions**.

## A vector is a representation

A vector is an ordered list of numbers:

```text
x = [2.1, -0.4, 0.7]
```

Geometrically, that can be viewed as a point or direction in a three-dimensional space. In AI, the more useful interpretation is usually: **these numbers are a representation of something**.

A row of tabular features can be a vector. A token embedding is a vector. A hidden state inside a transformer is a vector.

The dimensions do not have to correspond to concepts a human can name. What matters is what relationships the representation makes easy for the model to use.

## A matrix transforms many dimensions at once

A matrix is a rectangular array of numbers. Multiplying a vector by a matrix produces another vector:

```text
input vector
    ↓
[ matrix transform ]
    ↓
output vector
```

A linear layer in a neural network is fundamentally this operation plus a bias:

```python
y = W @ x + b
```

If `x` has 128 dimensions and `W` maps 128 dimensions to 512, then the output lives in a 512-dimensional representation space.

The matrix is learned because its entries determine which combinations of input dimensions become useful output features.

## The dot product measures alignment

For two equal-length vectors, the dot product multiplies corresponding dimensions and sums them.

```python
dot = sum(a_i * b_i for a_i, b_i in zip(a, b))
```

It appears everywhere because it is both simple and expressive. Attention uses dot products to compare queries and keys. Similarity systems often derive cosine similarity from a normalized dot product.

A larger dot product usually means the vectors point more strongly in compatible directions, although scale also matters.

## Shape is part of the meaning

When debugging AI code, tensor shape is often as important as tensor value.

```text
batch × tokens × hidden_dimension
```

For a transformer, a shape like `32 × 128 × 768` might mean:

- 32 sequences in the batch
- 128 token positions per sequence
- 768 numbers representing each token

Many architecture bugs are really shape misunderstandings: a dimension is pooled too early, transposed incorrectly or mixed with another axis.

## Why this matters later

Once vectors and matrices are comfortable, several advanced ideas stop looking mysterious:

- embeddings become points in representation space;
- neural layers become learned transformations;
- convolution becomes a structured linear operation plus nonlinearity;
- attention becomes learned vector comparison and weighted mixing;
- vector search becomes nearest-neighbor retrieval.

The goal is not to manually multiply large matrices. It is to recognize **what is represented, what transformation is being applied and which dimension means what**.

## Keep this mental model

When you see a model component, ask three questions:

1. What does the input vector represent?
2. How does this operation transform that representation?
3. What relationship should become easier to express after the transformation?

That framing scales surprisingly far—from linear regression to transformers.
