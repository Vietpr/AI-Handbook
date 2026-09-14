---
title: "Regularization for Deep Networks"
description: "What dropout, weight decay and data augmentation actually do, how much each is worth on a real overfitting case, and why turning on all three at once makes the model worse."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "regularization-for-deep-networks"
order: 7
pubDate: 2026-08-22
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will know what dropout, weight decay and data augmentation actually do, be able to measure how much each is worth on a real overfitting case, and understand why turning on all three at once makes the model worse.

## 1. Paying off a debt from Foundations

Foundations · Lesson 14 has a table listing ways to fight overfitting, and it says plainly that it does not teach them yet: *"You do not need to master the techniques below right away - the goal here is to know they **exist**"*. For dropout it even says *"Just remember the name at this stage"*.

That stage is now. Lesson 06 has just diagnosed a very clear case of overfitting, so we already have a patient to try the medicine on:

- **1,000 training samples** (instead of 60,000)
- A network with **two hidden layers of 1,024 neurons** - 2.9 million parameters for 1,000 samples
- Adam, 60 epochs

Baseline result: validation loss bottoms out at **0.6475** at epoch 6 and then climbs to 0.9973; best accuracy **0.8093**; final train loss 0.0487 - the model has nearly memorized the data.

Every number below is measured on that exact patient, changing only one thing at a time.

## 2. Dropout: switching off neurons so no one runs the show

**Dropout** does something that sounds slightly destructive: on every training pass, it randomly switches off a fraction $p$ of a layer's neurons - setting their outputs to 0.

```
   pass 1              pass 2              pass 3
   ●  ○  ●  ●          ●  ●  ○  ●          ○  ●  ●  ○
   ○  ●  ●  ○          ●  ○  ●  ●          ●  ●  ○  ●
   ● = active           ○ = off this pass
```

Why does this help? The network cannot rely on any particular neuron, because on the next pass that neuron may vanish. It is forced to learn features that **do not lean too heavily on each other**: a neuron is not allowed to be meaningful only when a few of its layer-mates are all present, because on the next pass they may not be. This is where dropout is often oversold - it does not make features independent of one another, it only penalizes a fragile dependence on one fixed combination. It is like a football team where anyone might miss any match: the team cannot build its game around one single player; the positions still work together, it is just that nobody is allowed to be the only link in the chain.

```python
nn.Sequential(nn.Flatten(),
              nn.Linear(784, 1024), nn.ReLU(), nn.Dropout(0.5),
              nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
              nn.Linear(1024, 10))
```

| Configuration | lowest validation loss | best accuracy | final train loss |
|---|---|---|---|
| Baseline | 0.6475 @ep6 | 0.8093 | 0.0487 |
| Dropout p = 0.2 | 0.6086 @ep12 | 0.8064 | 0.0325 |
| **Dropout p = 0.5** | **0.5943** @ep17 | **0.8110** | 0.0963 |

Three things to read off. Validation loss improves clearly (0.6475 → 0.5943). The bottom of the U **shifts to the right** - from epoch 6 to epoch 17, meaning the model tolerates longer training before it goes bad. Accuracy, on the other hand, barely changes.

Accuracy standing still while loss improves is exactly the phenomenon we met in Lesson 06: dropout cures **overconfidence**, not ranking.

> ⚠️ **A small note:** dropout is only active during training. At prediction time all neurons must be used - and that is why `model.eval()` in Lesson 05 matters so much. PyTorch also takes care of one detail for you: if 50% of the neurons are switched off during training, the total input to the next layer falls short by half, so `nn.Dropout` **scales up** the surviving neurons by $1/(1-p)$ right at training time. As a result both modes produce values on the same scale, and you do not have to adjust anything.

## 3. Weight decay: penalizing large weights

**Weight decay** has the same goal as the L2 / Ridge penalty from Machine Learning · Lesson 04 - restraining large weights - but how it is applied depends on the optimizer. The basic form adds to the loss function a penalty based on the sum of squared weights:

$$\text{total loss} = \text{cross-entropy} + \lambda \sum_j w_j^2$$

The model is forced to choose a solution with small weights, that is, a "softer" one that reacts less violently to small changes in the input. In PyTorch it is a parameter of the optimizer, not a layer:

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

| Configuration | lowest validation loss | best accuracy | final train loss |
|---|---|---|---|
| Baseline | 0.6475 | 0.8093 | 0.0487 |
| weight decay = 1e-4 | 0.6411 | 0.8065 | 0.0051 |
| weight decay = 1e-2 | **0.5947** | 0.7950 | **0.2415** |

> ⚠️ **One name, two ways of applying it.** In this experiment, `Adam(..., weight_decay=...)` applies the L2 penalty by **adding it to the gradient** before Adam's adaptive update step - that is PyTorch's default. Because Adam also normalizes the update step using the moments of the gradient, the practical effect of this approach is **not equivalent to decoupled weight decay** (shrinking the weights as a separate step). **AdamW** separates the weight-shrinking step from the gradient and keeps the decay term out of the moments - that is what Loshchilov & Hutter proposed, and today training large networks almost always uses AdamW. When reading the table above, remember that the numbers come from `Adam` + an added L2 penalty, not from AdamW.

The light setting (1e-4) changes almost nothing. The strong setting (1e-2) improves validation loss noticeably but **accuracy drops** to 0.7950, and train loss jumps to 0.2415 - a sign that the model has been held back a bit too hard and is starting to lean toward the underfitting side from Lesson 06.

This is exactly the trade-off that Machine Learning · Lesson 04 described in bias-variance language: the penalty lowers variance but raises bias, and it only pays off when the model *has* variance to reduce.

## 4. Data augmentation: making more data out of the data you have

The three sections above all restrain the model. This approach goes the other way - **widening the data**.

A shirt flipped in a mirror is still the same shirt. Shifted 2 pixels to the right, it is still the same shirt. So from 1,000 images, every epoch we can generate 1,000 slightly different images - the network never sees exactly the same picture twice.

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),      # mirror flip
    transforms.RandomCrop(28, padding=2),   # pad then crop at random → shift
    transforms.ToTensor(),
])
```

| Configuration | lowest validation loss | best accuracy | final train loss |
|---|---|---|---|
| Baseline | 0.6475 @ep6 | 0.8093 | 0.0487 |
| Augmentation | 0.6163 **@ep36** | 0.8022 | 0.3684 |

The most striking thing is the last column: train loss stops at 0.3684 instead of 0.0487. The model **can no longer memorize**, simply because the training set changes its face every epoch. And the bottom of the U moves all the way out to epoch 36 - before, it was at epoch 6.

> ⚠️ **A small note:** the transformation must **preserve the label**, and that depends on the problem. A mirrored shirt is still a shirt - valid. But mirroring a handwritten digit turns "2" into a character that does not exist, and flipping the letter "b" turns it into "d" - a completely wrong label. On FashionMNIST a horizontal flip is safe; on MNIST it is not. This is where you have to understand your data; there is no default that is right for every problem.

## 5. What if we turn on all three at once?

A natural question, and the answer is not what you might expect:

| Configuration | lowest validation loss | best accuracy | final train loss |
|---|---|---|---|
| Baseline | 0.6475 | 0.8093 | 0.0487 |
| Dropout 0.5 | 0.5943 | **0.8110** | 0.0963 |
| **All three at once** | 0.5958 | **0.7812** | **0.5551** |

Turning on all three gives the **lowest accuracy in the table** - worse even than the version with nothing. The train loss explains it: 0.5551, eleven times higher than the baseline. The model no longer overfits, but it also **cannot learn the data it has already seen**. Three doses of anti-overfitting medicine added together have pushed it all the way to the other side, into underfitting - exactly condition D from Lesson 06.

This is the most important lesson of the whole lesson, and it goes against a very common reflex: **regularization is not a case of the more the better.** Each tool is a force pulling the model toward something simpler; add enough forces together and they pull too far.

And there is a truth that needs to be said plainly, looking back over all four tables: **none of them raises accuracy meaningfully.** The best is dropout 0.5 at 0.8110 versus the baseline's 0.8093 - a gap of 0.17 points, well within the noise. What does improve consistently is **validation loss** (0.6475 → 0.594), meaning the model is less confidently wrong, and **the bottom of the U moving further out**, meaning you can train longer before things go bad.

The thing that cures the root of the problem is not in this lesson at all: the patient only has 1,000 samples. Foundations · Lesson 14 put "get more data" in the first row of its table for a reason - it is the only way to add real information, while the three techniques here only help you use that meager information more carefully.

> 🔧 **Try it now:** train loss 0.55 and validation loss 0.60, almost equal. Has the model stopped overfitting, and should you be happy?
> It has stopped overfitting, yes, but that is not necessarily good news. Two curves close together only say the model treats new and old data the same way - they do not say the model is *good*. Here a train loss of 0.55 is high (the baseline reached 0.0487), so you are most likely looking at underfitting. The procedure from Lesson 06 still holds: **look at train loss first**. A narrow gap with a high train loss is bad news, not good news.

## Lesson summary

- **Dropout** randomly switches off a fraction of neurons on every training pass, forcing the network to learn redundant features instead of depending on a few neurons; it is only active during training, so `model.eval()` is mandatory at evaluation time.
- **Weight decay** shares the goal of restraining large weights with the L2 / Ridge penalty from Machine Learning · Lesson 04, and is declared in the optimizer rather than as a layer - but how it is applied depends on the optimizer: `Adam(weight_decay=...)` adds the penalty to the gradient, while AdamW separates the weight-shrinking step from the moments.
- **Data augmentation** goes the other way: it generates valid variants of the data so the network cannot memorize - measurable as train loss stopping at 0.3684 instead of 0.0487.
- The transformation must preserve the label, and that depends on the problem: a horizontal flip is valid for clothing but ruins handwritten digits.
- On this overfitting case, all three improve **validation loss** (0.6475 → 0.594) and push the bottom of the U further out (epoch 6 → 17 or 36), but **none of them raises accuracy meaningfully**.
- Turning on all three at once gives the *worst* result (accuracy 0.7812, train loss 0.5551) - an overdose that turns into underfitting. Regularization is not a case of the more the better.
- The root cure for this case is still more data; the three techniques above only help you use scarce data more carefully.

## Self-check questions

1. Dropout randomly switches off neurons during training. Why does that act of sabotage help the model generalize better?
2. Why must `nn.Dropout` scale up the surviving neurons by $1/(1-p)$? What happens without that compensation step?
3. How does weight decay in this lesson differ from Ridge in Machine Learning · Lesson 04? Where do you declare it in PyTorch?
4. You plan to use augmentation for a traffic sign classification problem. Is a horizontal mirror flip safe? What about a 15-degree rotation?
5. Turning on dropout, weight decay and augmentation together gives lower accuracy than using none of them. Which metric do you diagnose with, and which one do you remove first?
6. All three techniques improve validation loss but not accuracy. If the model's output is used to rank applications by probability, would you consider them useful? Why?

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Foundations · Lesson 14** | Section 5 *The anti-overfitting toolbox* - the table of four techniques that this lesson measures one by one |
| **Machine Learning · Lesson 04** | Ridge, Lasso, and the bias-variance trade-off - the basis of the penalty that weight decay in section 3 reuses |
| **Dive into Deep Learning (d2l.ai)** | Chapter 3.7 *Weight Decay*; chapter 5.6 *Dropout* - both the mechanism and the $1/(1-p)$ scaling; chapter 14.1 *Image Augmentation* |

**Additional online sources (free):**

- [Srivastava et al. - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR 2014)](https://jmlr.org/papers/v15/srivastava14a.html) - the original paper, including the intuition that "the network must not depend on any single neuron".
- [`torchvision.transforms` - documentation](https://pytorch.org/vision/stable/transforms.html) - the list of available augmentations.
- [`torch.optim` - the `weight_decay` parameter](https://pytorch.org/docs/stable/optim.html) - the update rules for `Adam` and `AdamW` side by side, showing clearly which one adds the decay term to the gradient.
- [Loshchilov & Hutter - Decoupled Weight Decay Regularization (ICLR 2019)](https://arxiv.org/abs/1711.05101) - why an L2 penalty added into Adam is not equivalent to decoupled weight decay, and why AdamW separates the two steps.

> **Next lesson:** [Why Deep Networks Are Hard to Train](../why-deep-networks-are-hard-to-train/) - this lesson cured a network that learns *too well*. The next one handles the opposite and more serious case - a 20-layer network that learns nothing at all, with accuracy stuck at the level of random guessing, and three ways to get it back on its feet.
