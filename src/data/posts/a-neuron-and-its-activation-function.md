---
title: "A Neuron and Its Activation Function"
description: "What is inside a neuron, why stacking layers is pointless without an activation function, how to pick one for hidden layers, and how it differs from the function on the output layer."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "a-neuron-and-its-activation-function"
order: 2
pubDate: 2026-08-19
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will know what is inside a neuron (and recognize that you met it in the previous chapter under another name), understand through both algebra and numbers why stacking any number of layers is useless without an activation function, be able to choose an activation function for hidden layers, and tell it apart from the function on the output layer - the spot where beginners get confused most often.

## 1. Taking a neuron apart

Lesson 01 called the middle layer a "hidden layer" and moved on without opening it up. Let us open it now. A **neuron** does exactly two things, in order:

```
 x₁ ──w₁──┐
 x₂ ──w₂──┤     z = w₁x₁ + w₂x₂ + ... + b        a = f(z)
  ⋮       ├──▶ ────────────────────────  ──▶  ──────────  ──▶ output
 xₙ ──wₙ──┘        (1) weighted sum                (2) activation
           b                                    function
```

**Step 1 - the weighted sum.** Multiply each input by a weight, add them all up, then add an intercept $b$. This is exactly the dot product from Foundations · Lesson 07, and also exactly the plumber's formula `fee = 50 + 100 × hours` from Machine Learning · Lesson 02.

**Step 2 - the activation function.** Push the number $z$ through a nonlinear function $f$. Section 3 explains why this step must not be dropped.

That is all. A neuron is not a simulation of a nerve cell; it is a multiply-and-add followed by bending the result.

## 2. You have met it already: a sigmoid neuron is logistic regression

Take exactly one neuron, make its activation function the **sigmoid** $\sigma(z) = 1/(1+e^{-z})$, and train it with log loss. Write out the formula:

$$\hat{p} = \sigma(w_1 x_1 + w_2 x_2 + \dots + w_p x_p + b)$$

That is *word for word* the logistic regression of Machine Learning · Lesson 03. Not "nearly the same" - one model, two names.

Let us verify it numerically on the familiar `adult` dataset (48,842 records, 89 columns after preprocessing):

```python
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

sk = LogisticRegression(max_iter=3000, C=1.0).fit(Xp, y)      # scikit-learn

neuron = nn.Linear(Xp.shape[1], 1)                            # exactly ONE neuron
opt = torch.optim.LBFGS(neuron.parameters(), max_iter=500)
lossf = nn.BCEWithLogitsLoss()
lam = 1.0 / (2 * len(y))                                      # matches sklearn's C=1

def closure():
    opt.zero_grad()
    loss = lossf(neuron(Xt).squeeze(), yt) + lam * (neuron.weight ** 2).sum()
    loss.backward()
    return loss

opt.step(closure)
```

| | scikit-learn | One neuron in PyTorch |
|---|---|---|
| Accuracy | 0.8522 | 0.8523 |

Correlation between the 89 coefficients of the two: **0.9803**. Correlation between the probabilities they predict for the 48,842 records: **1.0000**, with a maximum difference of 0.088.

Does that 0.852 look familiar? It is exactly the accuracy that Machine Learning · Lesson 03 reported for logistic regression on the same dataset. You have not learned a new model - you have just called an old one by a new vocabulary.

> ⚠️ **A small note:** the two sets of coefficients correlate at 0.98 rather than 1.00, even though the probabilities match exactly. The reason is that `adult` has many mutually correlated columns after one-hot encoding (Machine Learning · Lesson 02 calls this multicollinearity): many different sets of coefficients produce the same predictions, and the two optimization algorithms stop at different sets. When comparing two models, comparing *predictions* is more reliable than comparing *coefficients*.

If one neuron is just logistic regression, then what is new lies in **placing many neurons side by side and stacking them on top of each other**. That is lesson 03. But first there is a more important question to answer.

## 3. Without an activation function, stacking any number of layers is useless

Suppose we drop step 2 and keep only the weighted sum. Two layers in sequence:

$$h = W_1 x + b_1 \qquad\qquad \hat{y} = W_2 h + b_2$$

Substitute the first into the second:

$$\hat{y} = W_2 (W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{\text{call it } W'} x + \underbrace{(W_2 b_1 + b_2)}_{\text{call it } b'} = W' x + b'$$

The result is still **a single linear map** - exactly of the form $Wx + b$. Whether you stack 2 layers, 10 layers or 100 layers, the product of the matrices is still one matrix, and the model can still only draw straight boundaries. All the effort of adding layers is swallowed by a single matrix multiplication.

Let us measure it on FashionMNIST, a network with two hidden layers of 256 neurons, 10 epochs:

| Hidden-layer activation function | Validation accuracy |
|---|---|
| **None** (only `Linear`) | 0.8283 |
| Sigmoid | 0.7611 |
| Tanh | 0.8452 |
| **ReLU** | **0.8549** |

The first row is evidence for the algebra above: without an activation function, a two-hidden-layer network with 270,000 parameters gives **0.8283** - while the 7,850-parameter softmax regression from lesson 01 gives 0.8317 with the same number of epochs. Add 260,000 parameters and gain nothing, because at heart it is still exactly one linear model.

A nonlinear activation function is the only thing that makes stacking layers meaningful.

> 🔧 **Try it now:** why must the activation function be **nonlinear**? What if we choose $f(z) = 3z$?
> Substitute: $h = 3(W_1x + b_1)$, then $\hat{y} = W_2 h + b_2 = (3W_2W_1)x + (3W_2b_1 + b_2)$ - once again of the form $W'x + b'$. A linear function nested in a linear function is still a linear function; the constant 3 is simply absorbed into the weights. It has to be a function that *bends* - with a kink, or with saturation - for nesting to produce something new.

## 4. Three activation functions you will meet

```
   Sigmoid σ(z)              Tanh(z)                  ReLU(z)
 1 ┤      ╭────         1 ┤       ╭────         3 ┤        ╱
   │     ╱                │      ╱                 │      ╱
   │    ╱               0 ┼─────┼─────           1 ┤    ╱
   │   ╱                  │    ╱                   │  ╱
 0 ┼──╯                -1 ┤────╯                 0 ┼────────
   └──┼──────              └──┼──────              └──┼──────
      0                       0                       0
 flat at both ends     flat at both ends,      kink at 0,
 output (0, 1)         output (−1, 1), symmetric   straight when positive
```

**Sigmoid** squeezes every number into the interval $(0, 1)$. It was the classic activation function of the 1990s and is now almost never used in hidden layers - the table above shows why: 0.7611, worse even than the model with no activation function. The reason lies at the two ends of the graph: when $z$ is large or small, the curve lies flat and the slope is close to 0, so the learning signal nearly dies out as it propagates back through many layers. Lesson 08 measures this phenomenon and gives it its proper name.

**Tanh** has the same shape but outputs the interval $(-1, 1)$ and is symmetric about 0. Better than sigmoid, still flat at both ends.

**ReLU** - *rectified linear unit* - is suspiciously simple: $\text{ReLU}(z) = \max(0, z)$. Negative values become 0, positive values pass through unchanged. It wins in the table above, and the three reasons are all practical: its derivative on the positive branch is always 1, so the activation function itself does not shrink the learning signal as it passes through many layers (the weight matrices can still shrink or blow it up - that is lesson 08's topic); computing it is just a comparison; and the fact that it outputs exactly 0 on the negative branch means the network uses only some of its neurons for each input.

> ⚠️ **A small note:** ReLU has a flaw called **dying ReLU**. If the weights are pushed to a point where $z$ is always negative for all data, that neuron always outputs 0; the derivative of ReLU there is also 0, so no gradient flows back to that neuron's own weights and it stops learning by itself. Saying it is "dead for good" is a bit of an overstatement: the earlier layers are still being updated, so this neuron's input distribution can change and push $z$ positive again; weight decay or the optimizer's momentum can also do that. It is just that it has no way to rescue itself, so in practice most dead neurons stay dead. Measured on FashionMNIST after 5 epochs, counting the neurons that never activate on 1,000 validation images:
>
> | Learning rate | Dead neurons |
> |---|---|
> | 0.1 | 2 / 256 |
> | 0.5 | **31 / 256** |
>
> A large learning rate pushes the weights in overshooting steps: from 2 dead neurons up to 31, that is 12% of the neurons in the layer. Variants such as **Leaky ReLU** (a small slope on the negative branch instead of 0) were created to fix exactly this.

## 5. The output-layer function is a different matter

This is where beginners often get confused, so it gets a section of its own.

The "activation function" in the four sections above refers to **hidden layers** - where its job is to bend things so that stacking layers is meaningful. At the **output layer** the job is entirely different: to bring the number into exactly the form the problem needs. Choose by problem, not by habit:

| Problem | Output layer | Function | Paired loss function |
|---|---|---|---|
| Binary classification | 1 neuron | **sigmoid** | log loss / BCE |
| Multiclass classification | 1 neuron per class | **softmax** | cross-entropy |
| Regression | 1 neuron | **nothing** | MSE / MAE |

Three takeaways from the table:

- **Sigmoid is still alive, but at the output layer.** It was dropped from hidden layers because it is flat at both ends, but at the output layer "squeeze into (0, 1)" is exactly what we need - a probability.
- **Softmax is the multiclass version of sigmoid**, just as Machine Learning · Lesson 03 said: one score per class, exponentiate, then divide by the total to get numbers that sum to 1.
- **For regression, put nothing there.** A house price can be any positive number; sticking a sigmoid there caps the output in the interval (0, 1).

> ⚠️ **A small note:** in PyTorch, `nn.CrossEntropyLoss` and `nn.BCEWithLogitsLoss` **already include softmax/sigmoid internally**. This means your network must output raw numbers (called **logits**), and must not apply softmax itself before passing them to the loss function - doing so applies it twice, and the model will learn very badly without raising any error. That is why every classification network in this chapter ends with a bare `nn.Linear`, with no function after it. This merging is not just for convenience: it also makes the computation more numerically stable than doing the two steps separately.

## Lesson summary

- A neuron does two things: a weighted sum of its inputs (the dot product, Foundations · Lesson 07), then pushes the result through a nonlinear activation function.
- A neuron with a sigmoid is logistic regression - not similar, but the same model: on `adult`, scikit-learn gives 0.8522 and a PyTorch neuron gives 0.8523, with a probability correlation of 1.0000.
- Without an activation function, two linear layers collapse into one: $W_2(W_1x + b_1) + b_2 = W'x + b'$. Measured: a two-hidden-layer network without activation functions gives 0.8283, on par with a 7,850-parameter linear model.
- In hidden layers, ReLU is the usual starting choice (0.8549) because its positive-branch derivative is 1 and it is very cheap to compute; sigmoid is clearly worse (0.7611) because it is flat at both ends.
- ReLU can die when the learning rate is too large: 31/256 neurons stop activating at lr = 0.5, compared with 2/256 at lr = 0.1.
- The **output-layer** function is chosen by problem, not by habit: sigmoid for binary, softmax for multiclass, nothing for regression.
- In PyTorch, the classification loss functions already contain softmax/sigmoid - the network must output raw logits, otherwise it is applied twice.

## Self-check questions

1. Write down the two things a neuron does, in the right order. Which one did you meet in the Machine Learning chapter, and under what name?
2. Prove with algebra that two `Linear` layers in sequence, with no activation function between them, are equivalent to exactly one `Linear` layer.
3. A colleague uses $f(z) = 2z + 1$ as the hidden-layer activation function and wonders why their deep network does no better than linear regression. Explain it to them.
4. Why is sigmoid dropped from hidden layers but still used at the output layer? How do those two roles differ?
5. Your model predicts house prices and you put a sigmoid at the output layer. What will happen to the expensive houses?
6. After training, you find that 40% of the neurons in a hidden layer never activate on the validation set. What is the diagnosis, and what do you try first?

## Further reading

**Core sources for this lesson:**

| Source | What to read |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chapter 5.1 *Multilayer Perceptrons* - section 5.1.2 *Activation Functions*: graphs and derivatives of ReLU, sigmoid, tanh; section 5.1.1 proves that stacking linear layers still yields a linear function |
| **An Introduction to Statistical Learning (ISLP)** | Chapter 10.1 - neurons and activation functions in a single-hidden-layer network, with a figure comparing sigmoid with ReLU |
| **Machine Learning · Lesson 03** | Logistic regression, sigmoid, softmax and log loss - section 2 of this lesson is that very lesson seen from another angle |

**Additional online sources (free):**

- [`torch.nn` - activation functions in PyTorch](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) - the full list with formulas and graphs.
- [`nn.CrossEntropyLoss` - PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - read carefully the line saying this function already combines `LogSoftmax` internally, exactly the warning in section 5.
- [Glorot, Bordes, Bengio - Deep Sparse Rectifier Neural Networks (AISTATS 2011)](https://proceedings.mlr.press/v15/glorot11a.html) - the paper that made ReLU the default choice, with a discussion of sparsity and of neurons that never activate.
- [Maas, Hannun, Ng - Rectifier Nonlinearities Improve Neural Network Acoustic Models (ICML 2013)](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) - the paper introducing Leaky ReLU, the fix for the dying ReLU in section 4.

> **Next lesson:** [Stacking Layers: Networks Learn Their Own Features](../stacking-layers-networks-learn-their-own-features/) - now that you know what a neuron does and why activation functions are needed, the remaining question is what stacking them into layers adds - and how much the word "deep" in deep learning actually buys.
