---
title: "Why Deep Networks Are Hard to Train"
description: "Why a 20-layer network can fail to learn anything, tracing the cause to signal size layer by layer, and two fixes, proper initialization and batch normalization, with measurements for each."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "why-deep-networks-are-hard-to-train"
order: 8
pubDate: 2026-08-22
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand why a 20-layer network can learn nothing at all, be able to trace the cause back to the size of the signal passing through each layer, and know two fixes - choosing the right initialization and adding batch normalization - along with the measurements for each.

## 1. A 20-layer network, accuracy 0.1000

Lesson 03 stopped at an uncomfortable fact: a network with 4 hidden layers did worse than one with 3. Push on to 20 layers and it is no longer a matter of "worse":

| Network | train loss after 8 epochs | validation accuracy |
|---|---|---|
| 5 hidden layers, ReLU | 0.4198 | 0.8166 |
| **20 hidden layers, ReLU** | **2.3028** | **0.1000** |

An accuracy of 0.1000 on 10 classes is random guessing, and 2.3028 is exactly $\ln(10)$ - the number we met in Lesson 06: the model spreads probability evenly across all 10 classes and distinguishes nothing. It has 20 layers, over 300 thousand parameters, runs 8 epochs without errors, and learns **exactly nothing**.

What is worth noting: this is not overfitting (train loss is also 2.3028), not a wrong learning rate (the 5-layer network with the same learning rate works fine), not a lack of data (still 60,000 samples). Being *deeper*, by itself, was enough to kill it.

## 2. Tracing the cause: the signal shrinks layer by layer

Lesson 04 measured the gradient flowing backward. Doing it again on the 20-layer network:

| Configuration | Layer 1 | Layer 10 | Last layer |
|---|---|---|---|
| ReLU, default initialization | **3.3 · 10⁻⁸** | 1.2 · 10⁻⁶ | 3.9 · 10⁻² |
| Sigmoid, default initialization | **2.3 · 10⁻¹⁷** | 2.4 · 10⁻¹⁰ | 6.1 · 10⁻¹ |

With sigmoid, the gradient reaching the first layer is $2.3 \times 10^{-17}$. That number still exists, and a 32-bit float can still represent it - the breakage happens at the next step. The update step is $\eta g$; with a learning rate of 0.1 it equals $2.3 \times 10^{-18}$, while a weight of ordinary size, say 0.05, has a **local resolution** in 32-bit floating point of roughly $3.7 \times 10^{-9}$ - more than a billion times larger. Adding an amount smaller than the gap between two adjacent representable numbers rounds the result back to exactly where it was: the weight **does not change at all**. So the first layer is not "learning slowly", it is barely learning. This is the **vanishing gradient**.

The mechanism is the compounding multiplication of the chain rule (Lesson 04). The gradient reaching layer $k$ is the product of the derivatives from the last layer back:

$$\frac{\partial L}{\partial W_k} \sim \frac{\partial L}{\partial \text{out}} \times \underbrace{f'(z_n) \cdot W_n \times \dots \times f'(z_{k+1}) \cdot W_{k+1}}_{\text{the deeper, the more factors}}$$

Multiply 20 numbers together and the result is very sensitive: if each one is just slightly below 1, the product rushes to 0; slightly above 1 and the product blows up (the **exploding gradient**, which usually shows up as the loss turning into `nan`).

But saying "the derivative is below 1" is only half the story, because the formula above also contains $W$. Let us measure the *forward* direction instead of the backward one - pass a batch through an untrained 20-layer network and look at the standard deviation of the activations after each layer:

| Initialization | std after layer 1 | layer 10 | layer 20 |
|---|---|---|---|
| PyTorch default | 0.1642 | 0.0272 | 0.0314 |
| Xavier / Glorot | 0.3076 | 0.0390 | 0.0506 |
| **He / Kaiming** | 0.3318 | **0.2877** | **0.2940** |

This is the real root. With the default initialization, the signal shrinks from 0.16 to 0.027 after only 10 layers - each layer multiplies by a factor below 1, and after 20 of them very little is left. Once the forward signal has died out, the backward gradient dies with it.

**He / Kaiming** keeps the standard deviation almost unchanged across all 20 layers (0.33 → 0.29). That is exactly what it was designed to do.

## 3. The first fix: choosing the right initialization

The idea behind both Xavier and He is to choose the initial variance of the weights so that the signal **neither shrinks nor grows** as it passes through a layer. The difference between the two: Xavier is derived for functions symmetric around 0 such as tanh, while He is derived specifically for ReLU - because ReLU throws away the negative half, it needs larger weights to make up for exactly the part that was lost.

```python
for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
```

The 20-layer ReLU network, 8 epochs, trying several learning rates:

| Initialization | lr = 0.1 | lr = 0.03 | lr = 0.01 |
|---|---|---|---|
| PyTorch default | 0.1000 | - | - |
| N(0; 1) - too large | `nan` | - | - |
| N(0; 0.01) - too small | 0.1000 | - | - |
| Xavier / Glorot | 0.5358 | 0.3163 | 0.1000 |
| **He / Kaiming** | `nan` | **0.8419** | 0.8237 |

From 0.1000 up to **0.8419** - same architecture, same data, same optimization algorithm. The only thing that changed is the random numbers at initialization.

Two rows deserve a careful read. **N(0; 1) gives `nan`** - weights that are too large make the signal grow through each layer until it overflows: that is the exploding gradient. **N(0; 0.01) gives 0.1000** - weights that are too small, the signal dies out: vanishing. Two ends of the same axis, and both Xavier and He are ways of computing the balance point in the middle.

> ⚠️ **A small note:** notice that the cell **He with lr = 0.1 also gives `nan`**, while lr = 0.03 gives the best result in the table. Do not read this as "He is unstable". Initialization and learning rate interact: He deliberately keeps the signal larger through the layers, so the gradients reaching the early layers are larger too, and a step of lr = 0.1 becomes too much. When you change the initialization, re-tune the learning rate - this is one of the most tightly coupled pairs of hyperparameters in deep learning.

## 4. The second fix: normalizing right in the middle of the network

Initialization only takes care of the *start*. After a few hundred update steps, the weights have changed and the statistics of the signal inside the network drift again. **Batch normalization** forces those statistics back into place after each layer, throughout training.

For each batch of data, at each layer, it takes the output, subtracts the mean and divides by the standard deviation *of that same batch* - exactly the `StandardScaler` operation from Machine Learning · Lesson 11, but applied in the middle of the network instead of at the input, and recomputed for every batch. It then multiplies by two learned parameters ($\gamma$, $\beta$) so the network still has the freedom to choose a different scale if it needs to.

> ⚠️ **A small note:** *why* BatchNorm helps is still debated to this day. The original 2015 paper explained it with the concept of "internal covariate shift" - the input distribution of each layer keeps drifting as the earlier layers change. Santurkar et al. (2018) measured again and found that explanation does not hold up: they deliberately injected noise to make the distribution drift more, yet the network with BatchNorm still trained well, and they argued that what BatchNorm really provides is a smoother loss surface, which lets it tolerate larger learning rates. Nobody disputes the experimental results in the table below; only the explanation of the mechanism is unsettled. Both papers are in the Further reading section.

```python
nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), ...)
```

| Number of layers | Without BatchNorm | With BatchNorm |
|---|---|---|
| 5 layers | 0.8166 | **0.8790** |
| **20 layers** | **0.1000** | **0.8300** |

The 20-layer network jumps from learning nothing at all to 0.8300. Looking back at the gradient table in section 2 shows why:

| Configuration | Layer 1 | Layer 10 | Last layer |
|---|---|---|---|
| Plain ReLU | 3.3 · 10⁻⁸ | 1.2 · 10⁻⁶ | 3.9 · 10⁻² |
| **ReLU + BatchNorm** | **50.0** | **2.95** | **0.64** |

Every layer now receives a gradient in a usable range. Notice that the order is even reversed - the first layer has the *largest* gradient instead of the smallest - but all three are within a few orders of magnitude of each other, not a million times apart as before.

And look at the 5-layer row: BatchNorm raises 0.8166 to 0.8790 even though that network was already learning normally. It is not just a lifebuoy for deep networks - it often helps healthy networks too, because it allows larger learning rates and makes training less sensitive to initialization.

> ⚠️ **A small note:** BatchNorm is a layer that behaves differently in the two modes, just as Lesson 05 warned. During training it uses the mean and standard deviation of the **current batch**, while gradually updating a set of running statistics. In `eval()` it **stops using the statistics of the batch being predicted** and uses those running statistics instead - not because there are no batches at prediction time (predicting in batches of 32 or 128 images is perfectly normal), but so that the result for one sample does not depend on whichever samples happen to share its batch. Forgetting `model.eval()` with BatchNorm is therefore even more harmful than forgetting it with dropout: the same image will give different results depending on which images it shares a batch with. And because it relies on batch statistics during training, batches that are too small make the statistics very noisy and BatchNorm much less stable - how small is "too small" depends on the problem and the architecture; there is no universal threshold. In that case people use variants such as LayerNorm, which is also the normalization used in the Transformer that the Generative AI chapter will introduce.

> 🔧 **Try it now:** both He initialization and BatchNorm rescue the 20-layer network. If you could only choose one for a new project, which would it be?
> In practice people use **both**, because they handle two different phases: initialization handles the starting point, BatchNorm handles the whole journey once the weights have drifted. If forced to choose one, BatchNorm is more robust - it keeps renormalizing, so it forgives both a poor initialization and a slightly-off learning rate, just as the 5-layer table shows it helps even when the network was already fine. But it costs extra computation and adds another place to slip up (`model.eval()`), while initialization is completely free.

## 5. One more piece, saved for a later lesson

The three fixes above are enough to get a 20-layer network running. But they are not enough for truly deep networks - the kind with 50, 100 or 150 layers.

The fix at that scale is the **skip connection**: giving the signal a direct path that jumps over a few layers, instead of forcing it to squeeze through every layer one by one. The gradient then has a short route to flow back to the early layers without being multiplied through dozens of factors. That is the idea behind the **ResNet** architecture, and Lesson 09 will meet it when discussing networks for images.

## Lesson summary

- A 20-layer ReLU network with default initialization gives a train loss of 2.3028 and accuracy of 0.1000 - exactly random guessing. Being deeper, by itself, is enough to kill a network.
- The cause is the compounding multiplication of the chain rule: the gradient reaching layer $k$ is a product of dozens of factors, so slightly below 1 it vanishes, slightly above 1 it explodes into `nan`.
- The root can be measured in the forward pass: with default initialization, the standard deviation of the activations shrinks from 0.16 to 0.027 after 10 layers; with He/Kaiming it holds steady at 0.33 → 0.29 across all 20 layers.
- Changing only the initialization takes the 20-layer network from accuracy 0.1000 to **0.8419**. Too large an initialization gives `nan`, too small gives 0.1000 - Xavier and He are ways of computing the balance point.
- Initialization and learning rate are coupled: He gives `nan` at lr = 0.1 but the best result in the table at lr = 0.03. If you change the initialization, re-tune the learning rate.
- **BatchNorm** renormalizes the signal in the middle of the network, batch by batch: the 20-layer network goes from 0.1000 to 0.8300, and even the already-healthy 5-layer network rises from 0.8166 to 0.8790.
- BatchNorm behaves differently in `train()` and `eval()`, and loses its effect when the batch is too small.

## Self-check questions

1. Your network has a train loss stuck at 2.30 on a 10-class problem, and it has 30 layers. Give a diagnosis and the first two things you would try.
2. Why is "the derivative is below 1" only half the explanation for the vanishing gradient? Where is the other half, and which table in the lesson measures it?
3. Initialization with N(0; 1) gives `nan`, while N(0; 0.01) gives accuracy 0.1000. In what way are these two failures opposites?
4. He/Kaiming uses a larger variance than Xavier. Why does ReLU need larger weights than tanh?
5. You switch from the default initialization to He and the model immediately produces `nan`. Is this evidence that He is worse? What do you do next?
6. Why is forgetting `model.eval()` with BatchNorm even more dangerous than with dropout? Describe concretely what happens to a single prediction.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chapter 5.4 *Numerical Stability and Initialization* - vanishing/exploding gradients and Xavier initialization, exactly the content of sections 2-3; chapter 8.5 *Batch Normalization* |
| **Foundations · Lesson 09** | The chain rule - the compounding multiplication in section 2 is exactly that, repeated across 20 layers |
| **Machine Learning · Lesson 11** | `StandardScaler` - BatchNorm is that same operation applied in the middle of the network |

**Additional online sources (free):**

- [Glorot & Bengio - Understanding the difficulty of training deep feedforward neural networks (AISTATS 2010)](https://proceedings.mlr.press/v9/glorot10a.html) - the Xavier paper, with plots of standard deviation by layer just like the table in section 2.
- [He et al. - Delving Deep into Rectifiers (ICCV 2015)](https://arxiv.org/abs/1502.01852) - the He/Kaiming paper, explaining why ReLU needs a different factor from Xavier.
- [Ioffe & Szegedy - Batch Normalization (ICML 2015)](https://arxiv.org/abs/1502.03167) - the original BatchNorm paper.
- [Santurkar et al. - How Does Batch Normalization Help Optimization? (NeurIPS 2018)](https://arxiv.org/abs/1805.11604) - the rebuttal showing that Ioffe & Szegedy's original explanation was not right, a nice example of a technique working before it was properly understood.
- [`torch.nn.init` - documentation](https://pytorch.org/docs/stable/nn.init.html) - every available initialization method.

> **Next lesson:** [CNNs: Networks That Learn to See](../cnn-networks-that-learn-to-see/) - the past eight lessons all flattened 28×28 images into 784-vectors and threw them straight into `nn.Linear` - that is, they discarded the information about which pixel sits next to which. The next lesson uses an architecture that knows its input is an image, and this time the gain is anything but modest.
