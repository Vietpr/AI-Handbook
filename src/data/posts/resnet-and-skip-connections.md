---
title: "ResNet and Skip Connections"
description: "How skip connections help deep networks train, how to distinguish depth degradation from overfitting, and what a small experiment reproduces and misses."
domain: "Computer Vision"
section: "Learn"
language: "en"
translationKey: "resnet-and-skip-connections"
order: 3
pubDate: 2026-08-26
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will be able to tell **degradation with depth** apart from overfitting using a single column of numbers, understand what problem a skip connection turns each block's job into, and see an experiment that reproduces half of the phenomenon but not the other half - along with the reason why.

## 1. A debt from the Deep Learning chapter

Deep Learning · Lesson 08 showed that a 20-layer fully connected network with default initialization gives an accuracy of 0.1000 - exactly random guessing - and that it can be fixed with He initialization or BatchNorm. That lesson mentioned **skip connections** as a third fix, but never measured them, and never measured anything on a convolutional network.

This lesson pays off that debt, because one question has not been answered: **once we already have good initialization and BatchNorm, what happens if we stack even deeper?**

Lesson 02 stopped at VGG-16 with 13 convolutional layers. The natural answer is to keep stacking - 20, 34, 50 layers. But what happened when people did that took the whole field by surprise.

## 2. Degradation is not overfitting

The experiment is designed to change **exactly one thing**. Both networks have the same number of blocks, the same channel counts, the same BatchNorm, the same SGD with momentum 0.9 and a cosine schedule, the same 6 epochs, the same 32×32 Imagenette data. The only difference: one has the addition `y = F(x) + x` at the end of each block, the other does not.

The parameter budgets are almost equal - the skip connections add only about 0.1% extra parameters for the projections that change dimensions:

| Network | Parameters | Train loss | Accuracy |
|---|---|---|---|
| Plain-20 | 269,722 | 1.4001 | 0.5246 |
| **ResNet-20** | 272,474 | **1.1671** | **0.5717** |
| Plain-44 | 658,586 | 2.1247 | **0.2003** |
| **ResNet-44** | 661,338 | **1.7607** | **0.3740** |

The column to look at first is **not** accuracy but **train loss**, and lesson 06 of the Deep Learning chapter taught exactly this order: always read train loss first.

**The 44-layer plain network has a train loss of 2.1247.** With 10 classes, a uniform guess gives $\ln(10) = 2.3026$. That network can barely learn **the data it has already seen**. This is the decisive diagnosis: if it were overfitting, train loss would be *low* and only accuracy would be bad. Here both are bad.

So this phenomenon **is not overfitting**, but **degradation with depth**: a deeper network is harder to *optimize*, not more prone to memorizing.

**And that is very hard to accept in theory.** The 44-layer network contains the 20-layer network as a special case - the last 24 blocks only need to learn the identity function exactly and the two networks give identical results. That means the 44-layer network **cannot, in principle, be worse**. But gradient descent does not find that solution: from 0.5246 down to 0.2003, 32.4 points worse despite having 2.4 times as many parameters.

**Skip connections help at both depths, and they help in the right place.** Train loss is lower at both 20 layers (1.1671 versus 1.4001) and 44 layers (1.7607 versus 2.1247). At depth 44, the accuracy gap is **17.4 points** for two networks whose parameter counts differ by 0.1%.

## 3. Where this experiment does not reproduce

The table above has one detail that contradicts the ResNet paper, and it should be stated rather than ignored: **ResNet-44 is also worse than ResNet-20** (0.3740 versus 0.5717).

The original paper shows that deeper ResNets are better - 20, 32, 44, 56 layers improve steadily. The experiment here does not come out that way.

The most plausible hypothesis is the **training budget**. The paper trains for 64,000 steps on CIFAR-10; here it is **6 epochs** on a machine without a GPU, about 450 steps. The 44-layer network has more parameters and needs more steps to converge, so at the 6-epoch mark it is still on its way - a train loss of 1.7607 is still very high, exactly the "not finished yet" sign that Deep Learning · Lesson 06 describes.

But that is still a hypothesis: proving it would require rerunning with a larger budget and seeing whether the gap reverses - something the machine here cannot handle. So the honest conclusion of section 2 has to be written precisely like this:

- ✅ **Reproduced:** plain networks degrade sharply when stacked deeper, and it is an optimization problem, not overfitting
- ✅ **Reproduced:** skip connections improve both train loss and accuracy at the same parameter budget
- ❌ **Not reproduced:** deeper ResNets are better - the hypothesis is that the training budget is too short, but it has not been verified

> ⚠️ **A small note:** every number in this lesson is **a single illustrative run** with one seed. That is enough for large gaps - 32.4 points between Plain-20 and Plain-44 cannot possibly be noise - but not enough to be sure about small differences. To be sure you would run several seeds and compare the averages, just as Deep Learning · Lesson 12 said when discussing ablations. And even then, the comparison only speaks about *this pair of architectures, at this budget, on this dataset*.

## 4. What a skip connection turns the problem into

A regular convolutional block learns a function $H(x)$. A residual block changes how the problem is posed:

$$y = F(x) + x$$

The network no longer learns $H(x)$ directly but learns **the residual** $F(x) = H(x) - x$ - what needs to be *added to* the input, not what replaces it.

```
    x ──────────────────────────┐  shortcut: goes straight, no weights
    │                           │
    ├─▶ conv 3x3 ─▶ BN ─▶ ReLU  │
    │        │                  │
    │        ▼                  ▼
    └─▶ conv 3x3 ─▶ BN ──────▶ (+) ─▶ ReLU ─▶ y
```

Posing the problem this way brings two things.

**The identity function becomes the default choice.** For a block to do nothing, the weights of $F$ only need to be pushed close to 0 - which weight decay already pulls toward. In a plain network, for a block to do nothing its convolutional layers must learn the identity function *exactly* from random weights, a much harder task. This is the direct reason for the phenomenon in section 2: a deep network only needs to "switch off" a few blocks to fall back to a shallow network, and skip connections make that switching off easy.

**The gradient has a direct path back to the start of the network.** The derivative of $y = F(x) + x$ with respect to $x$ has the form $\partial F/\partial x + 1$. That $+1$ term means that however small $\partial F/\partial x$ becomes, the gradient still gets through - exactly the mechanism Deep Learning · Lesson 08 describes with vanishing gradients, and a close relative of the forget gate in the LSTM in Lesson 10 of that chapter.

> 🔧 **Try it now:** if skip connections are that good, why not attach them to every architecture, even a network with only 5 layers?
> People do attach them, and the parameter cost is tiny - 0.1% as in the table in section 2. And in this experiment, the gap they create is **larger in the deeper network**: 4.7 points at 20 layers versus 17.4 points at 44 layers. Two depths are not yet a rule, but it matches the mechanism. In a 5-layer network the gradient does not have far to travel anyway, and the identity function is not needed either - so skip connections solve a problem that does not exist yet. This is a good example of a general principle: every technique is born to cure a specific disease, and adding it when the disease is absent only takes up space. Deep Learning · Lesson 07 ran into exactly this when it switched on all three regularization techniques at once.

## 5. Why ResNet is still everywhere

Skip connections are what make very deep networks trainable, and their consequences spread beyond the scope of this lesson:

- **Lesson 04** hooks into `layer4` of ResNet-50 - the four blocks `layer1`-`layer4` are exactly the four groups of residual blocks
- **Lesson 06** uses ResNet-18 and ResNet-50 as backbones, and measures the quality of their frozen features
- **Lessons 08 and 09** use ResNet-50 as the backbone for Faster R-CNN, RetinaNet, FCOS, Mask R-CNN and DeepLabV3 - all five
- **Lesson 10** shows that the Vision Transformer also uses skip connections inside every block, even though it drops convolution entirely

One more detail of ResNet is worth mentioning, since lesson 02 pointed out the most expensive part of VGG: ResNet **drops the fully connected block at the end entirely**, replacing it with a global average followed by a single linear layer. As a result ResNet-50 has only **25,557,032** parameters versus 138,357,544 for VGG-16 - **5.4 times fewer** despite being much deeper. (The figure of 23.6 million that appears in lessons 06 and 10 is the same network after replacing the 1,000-class classification layer with the 37 classes of Oxford Pet.)

## Lesson summary

- **Degradation with depth** is not overfitting: the 44-layer plain network has a train loss of 2.1247, close to the uniform-guess level $\ln(10) = 2.3026$ - it cannot even learn the data it has seen.
- Always read **train loss before accuracy**; it is the column that separates two diseases that need opposite cures.
- The 44-layer plain network is **32.4 points** worse than the 20-layer plain network despite having 2.4 times as many parameters, while in principle it contains the 20-layer network as a special case.
- **Skip connections improve both train loss and accuracy at the same parameter budget** (adding only ~0.1%): 4.7 points better at depth 20 and **17.4 points** at depth 44.
- This experiment does **not** reproduce the original paper's "the deeper the ResNet, the better" result; the hypothesis is that 6 epochs on a CPU is too little for the 44-layer network to converge, but the lesson could not verify that.
- $y = F(x) + x$ turns each block's job into learning **the residual**, which makes the identity function the default choice and gives the gradient a direct path thanks to the $+1$ term in the derivative.
- In this experiment, the gap created by skip connections is **larger at depth 44 than at depth 20** (17.4 points versus 4.7). Two depths and one seed are not enough to call it a rule, but it matches the mechanistic reasoning in section 4.
- ResNet drops the fully connected block at the end, so ResNet-50 has only 25,557,032 parameters versus 138,357,544 for VGG-16.

## Self-check questions

1. The train loss of the 44-layer plain network is 2.1247 on a 10-class problem. What does that number tell you, and why does it rule out a diagnosis of overfitting?
2. Why "in principle cannot" the 44-layer network be worse than the 20-layer network? Explain using the identity function.
3. Skip connections add only 0.1% more parameters. So where does their benefit come from, if not from capacity?
4. Write the derivative of $y = F(x) + x$ with respect to $x$ and point out the term that keeps the gradient alive. Relate it to the forget gate of the LSTM.
5. This lesson did not reproduce "deeper ResNets are better". State the cause, and design an experiment to verify that cause.
6. Why does the benefit of skip connections grow with depth? Base your answer on the two figures of 4.7 and 17.4 points in section 2.
7. ResNet-50 is much deeper than VGG-16 but has 5.4 times fewer parameters. Where does that difference lie?

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Deep Learning · Lesson 08** | Vanishing gradients, initialization, BatchNorm - this lesson fills in the missing skip connection part |
| **Deep Learning · Lesson 06** | Read train loss before accuracy - the very order used in section 2 |
| **Lesson 02 of this chapter** | VGG-16 and where it spends the most parameters, to understand where ResNet trims |
| **Dive into Deep Learning (d2l.ai)** | Chapter 8.6 *Residual Networks* - a full implementation of the residual block |

**Additional online sources (free):**

- [He et al. - Deep Residual Learning for Image Recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385) - the original paper; figure 1 is exactly the degradation phenomenon from section 2, measured on CIFAR-10 with a full training budget.
- [He et al. - Identity Mappings in Deep Residual Networks (ECCV 2016)](https://arxiv.org/abs/1603.05027) - the follow-up paper, analyzing why the shortcut should go straight through without any transformation.
- [Veit, Wilber, Belongie - Residual Networks Behave Like Ensembles of Relatively Shallow Networks (NeurIPS 2016)](https://arxiv.org/abs/1605.06431) - a different way of reading ResNet, worth a look once you have grasped section 4.
- [Li et al. - Visualizing the Loss Landscape of Neural Nets (NeurIPS 2018)](https://arxiv.org/abs/1712.09913) - visualizations showing that skip connections make the loss surface much smoother.
- [ResNet - torchvision documentation](https://pytorch.org/vision/stable/models/resnet.html) - the available variants and their published figures.

> **Next lesson:** [What a CNN Actually Learns](../what-cnns-actually-learn/) - the first three lessons took care of feeding images in correctly and building networks that can be trained. The next one lifts the lid to look inside: what each layer computes, which region of the image the model relies on to decide - and a test showing that the most popular explanation tool sometimes points at the wrong place.
