---
title: "Where Linear Models Give Up"
description: "Where linear models stop improving without hand-engineered features, and why that approach does not scale to images or audio."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "where-linear-models-give-up"
order: 1
pubDate: 2026-08-19
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will see in numbers where a linear model can only get further when people prepare its features for it, understand why inventing features by hand does not scale to images or audio, and grasp the question this whole chapter sets out to answer: can a model find its own way of representing data instead of relying on people.

## 1. Four rows of data that defeat logistic regression

This table has four rows, two columns and 0/1 labels. It is small enough to understand at a glance:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | 0 |

The rule: the label is 1 when *exactly one* of the two columns equals 1. This is **XOR** (exclusive or) - a rule any child picks up in three seconds.

Give it to the logistic regression from Machine Learning · Lesson 03:

```python
from sklearn.linear_model import LogisticRegression
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 1, 1, 0]
print(LogisticRegression().fit(X, y).score(X, y))   # → 0.5
print(LogisticRegression().fit(X, y).predict(X))    # → [0 0 0 0]
```

**Accuracy 0.5.** Not on new data - on the very four rows it just learned. The model gives up and predicts 0 for everything.

The reason is geometric. Logistic regression draws a **straight line** and declares: this side is 1, that side is 0. Try drawing one yourself:

```
 x₂
  │
1 ┤  ●(0,1)          ○(1,1)
  │
  │
0 ┤  ○(0,0)          ●(1,0)
  └──┼─────────────────┼──── x₁
     0                 1

  ● = label 1     ○ = label 0
```

The two label-1 points sit at *opposite* corners, and so do the two label-0 points. No straight line can separate one pair from the other - however you draw it, one point ends up on the wrong side. This is not about a weak algorithm or too little data; it is a limit of the *form* of the model.

> 🔧 **Try it now:** change the label of the last row from 0 to 1 (turning it into OR) and run it again. What is the accuracy?
> **1.0.** With OR, the three label-1 points gather on one side and the label-0 point sits alone in the (0,0) corner - one straight line separates them cleanly. Same four rows, same two columns, same model: the only thing that changed is the *shape* of the rule.

## 2. The old fix: a person invents a new column for the machine

Machine Learning · Lesson 11 already gave the answer to this situation - **feature engineering**. For XOR, adding one column does it: $x_3 = x_1 \times x_2$.

| $x_1$ | $x_2$ | $x_1 x_2$ | $y$ |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | **1** |
| 1 | 0 | 0 | **1** |
| 1 | 1 | **1** | 0 |

Now label 0 corresponds to "the first two columns sum to 0 **or** the third column equals 1" - separable by a plane in three-dimensional space:

```python
X3 = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]
print(LogisticRegression().fit(X3, y).score(X3, y))          # → 0.75
print(LogisticRegression(C=10).fit(X3, y).score(X3, y))      # → 1.0
```

A small detail in the first line is worth pausing on. The new column has made the data separable by a plane, but `LogisticRegression()` comes with regularization `C=1.0` by default, and with only four rows of data that penalty is enough to pull the coefficients down, so the model cannot yet build the plane it needs and still gets one row wrong. Loosening the penalty with `C=10` is enough for 100%. This has nothing to do with XOR - it just repeats what Machine Learning · Lesson 06 said: a library default is a choice, not a truth.

The problem is solved, but notice who solved it: **you**. You looked at the table, recognized the rule as XOR, and came up with the idea that multiplying the two columns would capture it. The model only fit coefficients for the column you prepared.

On real data, it is not that easy. The `make_moons` dataset is two interleaved arcs - you met it in Machine Learning · Lesson 09:

| Model | Test accuracy |
|---|---|
| Logistic regression | 0.8767 |
| Logistic + degree-2 polynomial features | 0.8733 |
| Logistic + degree-10 polynomial features | 0.9667 |

Read this table a little more carefully. Adding degree-2 polynomials - most people's first reflex - **does not help**, and is even slightly worse. You have to go all the way to degree 10 to do well. But degree 10 on 2 columns produces 66 columns, most of them useless, and you found that number 10 by trial and error.

And that is only two-dimensional data, which you can plot and see with your own eyes.

## 3. Where the old fix runs out of road

A 224×224 pixel color image is **150,528 numbers**. Try applying exactly the process above to it.

Step one: look at the data to come up with features. But look at what? Column 74,311 is the green-channel brightness of a pixel somewhere in the image. It has no meaning of its own. Unlike `income` or `years of work` in the previous chapter - columns you understand immediately and know how to combine - pixel 74,311 says nothing at all, and it says nothing different when the cat in the image moves three pixels to the right.

What *does* carry meaning in an image lives at a much higher level: an edge, a corner, a patch of striped fur, a pointed ear. None of these is a column in a table. Each is a **combination** of hundreds of pixels, and that combination must still be recognizable when the object shifts, tilts, or the lighting changes.

For many years, people did exactly that by hand. Computer vision before 2012 had an entire subfield devoted to designing **feature descriptors** - SIFT, HOG, SURF - human-designed formulas that extract a few thousand more "meaningful" numbers from a raw image, which are then fed into a classifier such as the SVM from Machine Learning · Lesson 08. This approach worked, and was once the standard. But it came at two costs: each field needs its own descriptors designed by experts in that field, and the quality ceiling is capped by the designer's imagination.

The natural question arises here, and it is the question of this whole chapter:

> If people do not know which features to create, can the model **find on its own** a way to represent the data?

The approach that answers "yes" to that question is called **representation learning**: instead of giving the model prepared features, we give it raw data and let the training process itself find a useful representation.

## 4. One hidden layer is enough to learn XOR

Back to the four rows from section 1, but this time instead of adding the $x_1 x_2$ column ourselves, we place an **intermediate layer** between input and output - a few neurons, each learning its own combination of $x_1$ and $x_2$:

```
   x₁ ──┐
        ├──▶ [ hidden layer: 4 neurons ]──▶ output ──▶ y
   x₂ ──┘
        ↑                    ↑
   raw data           combinations the MACHINE finds,
                      not ones you came up with
```

```python
import torch
from torch import nn

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([0., 1., 1., 0.])
torch.manual_seed(0)

net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.1)
lossf = nn.BCEWithLogitsLoss()

for _ in range(2000):
    opt.zero_grad()
    loss = lossf(net(X).squeeze(), y)
    loss.backward()
    opt.step()

print(torch.sigmoid(net(X).squeeze()).detach().numpy().round(4))
```

Result: `[0. 1. 1. 0.]`, with loss down to 0.00002. All four rows correct - and this time **nobody told it about the $x_1 x_2$ column**. The four neurons in the middle layer found combinations good enough to separate the two classes on their own.

That is the whole idea of this chapter, shrunk down to four rows of data.

> ⚠️ **A small note:** do not read this result as "just add a hidden layer and you are done". Rerun exactly the code above with different `manual_seed` values, changing only the number of neurons in the hidden layer, and count how many times XOR is solved correctly out of 20 tries:
>
> | Hidden neurons | 2 | 3 | 4 | 8 |
> |---|---|---|---|---|
> | Seeds that solve it | 7/20 | 17/20 | 19/20 | 20/20 |
>
> Two neurons is the theoretical minimum, and in practice it fails more than half the time - training gets stuck in a poor solution depending on the random starting point. Neural networks have no closed-form solution like linear regression; they have to walk down a mountain in the fog (Foundations · Lesson 11), and the starting point has a real effect. Lesson 08 returns to this under the name **initialization**.

## 5. How much does it buy on real data?

XOR is a toy. Try it on **FashionMNIST**: 70,000 28×28 grayscale images of clothing, 10 classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot). It is the same size as handwritten-digit MNIST but considerably harder - and it will stay with you throughout this chapter.

Two models, same data, same 20 epochs, same optimization algorithm:

| Model | Parameters | Best validation accuracy |
|---|---|---|
| Softmax regression (linear, no hidden layer) | 7,850 | 0.8396 |
| Plus 1 hidden layer of 256 neurons | 203,530 | **0.8675** |

The hidden layer brings **2.8 percentage points**, in exchange for 26 times as many parameters.

Let us be honest about this number. It is real and reproducible, but it is not a leap - and that is worth saying right in the first lesson of the chapter. Throwing more parameters at an architecture that does not suit the data only gets you this far. What makes the big difference is **an architecture that fits the shape of the data**: lesson 09 will bring a network that knows its input is an *image* - that neighboring pixels are related, and that an edge is still an edge no matter which corner of the frame it sits in.

Three properties make this direction worth following, even though the gain in the first lesson is still modest:

- **It scales by itself.** Same idea, more layers, more neurons, more data - no need to rethink the feature set from scratch.
- **It transfers across fields.** The same framework runs on images, audio and text, whereas SIFT is only for images.
- **What it learns can be reused.** The features a network learns from millions of images can be carried over to another problem with only a few thousand images - that is lesson 11.

## Lesson summary

- A linear model draws a straight boundary, so there are simple rules it cannot learn: four-row XOR drops logistic regression to 0.5 accuracy on the training data itself.
- The Machine Learning chapter's fix is for people to create more features. This is correct but depends on you *guessing right* which features are needed: on `make_moons`, degree-2 polynomials do not help (0.8733), and you have to hunt up to degree 10 to reach 0.9667.
- For images, audio and text that approach runs out of road: individual pixels carry no meaning of their own, what carries meaning are higher-level combinations, and they must be recognizable even when the object shifts or the lighting changes.
- The question of the whole chapter: can a model **find its own representation** instead of being handed one? The approach that answers "yes" is called representation learning.
- One hidden layer solves XOR without anyone supplying the $x_1 x_2$ column - but with 2 neurons, 13/20 tries fail, because the network has to search rather than use a closed-form formula.
- On FashionMNIST, adding one hidden layer raises accuracy from 0.8396 to 0.8675 with 26 times the parameters. A real but moderate gain - the big leap comes from an architecture that fits the data, not from adding parameters.

## Self-check questions

1. Why can logistic regression not learn XOR, while it can learn OR? Answer with geometry, not with formulas.
2. You add the $x_1 x_2$ column to the XOR table and the model reaches 100%. Who really solved the problem in this case, and what does that say about how well this approach scales?
3. On `make_moons`, degree-2 polynomial features give a *worse* result than plain logistic regression. Offer one explanation, and how would you verify it?
4. Why does "look at the data and come up with features" work for an income table but not for a 224×224 image?
5. A network with 2 hidden neurons solves XOR for 7/20 seeds. What does that tell you about how neural networks are trained, in contrast with the linear regression of the previous chapter?
6. Adding a hidden layer of 256 neurons to FashionMNIST gains only 2.8 accuracy points but costs 26 times the parameters. In a real project, what else would you need to know before concluding whether this trade-off is worth it?

## Further reading

**Core sources for this lesson:**

| Source | What to read |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chapter 5.1 *Multilayer Perceptrons* - section 5.1.1 explains why stacking many linear layers still gives only a linear function, and why activation functions are needed; the introduction of chapter 1 on representation learning |
| **An Introduction to Statistical Learning (ISLP)** | Chapter 10.1 - the single-hidden-layer neural network and how it builds derived features from the inputs |
| **Foundations · Lessons 11, 13, 14** | Gradient descent and the matter of the starting point; capacity; overfitting - three things that come back throughout this chapter |

**Additional online sources (free):**

- [FashionMNIST - the original Zalando Research repository](https://github.com/zalandoresearch/fashion-mnist) - a description of the dataset and a benchmark table for many models, handy for comparing against the numbers you run yourself.
- [TensorFlow Playground](https://playground.tensorflow.org/) - drag and drop the number of layers and neurons and watch the decision boundary change shape in real time; try the XOR dataset in the left corner.
- [Krizhevsky, Sutskever, Hinton - ImageNet Classification with Deep Convolutional Neural Networks (NeurIPS 2012)](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) - the work that marks where learned features overtook hand-designed features on ImageNet.
- [Lowe - Distinctive Image Features from Scale-Invariant Keypoints (IJCV 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) - the SIFT paper, to see what a hand-designed feature descriptor looks like.
- [Bengio, Courville, Vincent - Representation Learning: A Review and New Perspectives (2013)](https://arxiv.org/abs/1206.5538) - the survey that gave a name to the very question in section 3.

> **Next lesson:** [A Neuron and Its Activation Function](../a-neuron-and-its-activation-function/) - this lesson talks about a "hidden layer" without opening it up to see what is inside. The next lesson takes a neuron apart - it turns out you have met it before, under a different name - and answers why, without an activation function, stacking any number of layers is useless.
