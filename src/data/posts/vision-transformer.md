---
title: "Vision Transformer"
description: "How vision transformers turn images into patches, which CNN inductive biases they relax, and what changes in a controlled comparison."
domain: "Computer Vision"
section: "Learn"
language: "en"
translationKey: "vision-transformer"
order: 10
pubDate: 2026-08-29
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand how a transformer sees an image - cut it into patches, embed positions, then let every patch talk to every other - know which inductive bias of convolutional networks ViT drops and what it has to trade for that, and most importantly: see a very convincing comparison **lose more than half its gap** just by changing one thing that has nothing to do with the architecture.

## 1. Nine lessons built on one assumption

From Lesson 02 to Lesson 09, every architecture was a convolutional network, and all of them were built on the two assumptions about images laid out in Deep Learning · Lesson 09:

- **Locality** - nearby pixels are related, so a filter only needs to look at a small window
- **Weight sharing** - a pattern means something at any position, so the same filter slides across the whole image

These two assumptions are true of images, and precisely because they are true they are a very valuable **inductive bias**: the model does not have to learn them from data, they are built into the architecture.

But "built in" also means "cannot be dropped when not needed". With a 3×3 convolutional layer, two pixels in opposite corners of an image only influence each other after the signal passes through a great many layers - Lesson 02 computed it: VGG-16 needs all 13 convolutional layers and 5 pooling layers to reach a receptive field of 212 pixels.

**Vision Transformer** asks the opposite: what if every region of the image could talk directly to every other **right from the first layer**?

## 2. An image as a sequence of patches

A transformer works on a *sequence*. So the first step is to turn the image into a sequence, and the way it is done is surprisingly simple: **cut the image into non-overlapping squares**.

For ViT-B/16 on a 224×224 image with 16×16 patches:

$$\frac{224}{16} \times \frac{224}{16} = 14 \times 14 = \mathbf{196 \text{ patches}}$$

Each 16×16×3 patch = 768 numbers is flattened and then multiplied by a matrix to become a **patch embedding** vector. The result is 196 vectors - a sequence, exactly what the transformer needs.

```
image 224x224          cut into a 14x14 patch grid       sequence of 196 vectors
┌─────────────┐        ┌──┬──┬──┬──┬──┐
│             │        ├──┼──┼──┼──┼──┤                 [v1] [v2] ... [v196]
│    a cat    │  ───▶  ├──┼──┼──┼──┼──┤     ───▶         + positional embedding
│             │        ├──┼──┼──┼──┼──┤                  + one [CLS] token
└─────────────┘        └──┴──┴──┴──┴──┘
```

Two details have to be added, and both reveal where ViT differs sharply from a CNN:

**Positional embedding.** After cutting and flattening, the sequence of 196 vectors **no longer carries any information about which patch is where**. With convolution, position is built into the structure; with a transformer it is not, because attention treats every element of the sequence the same way. So a learned position vector is **added** to each vector. This is exactly the issue Deep Learning · Lesson 09 discussed when it said `nn.Flatten()` destroys neighbor relationships - the only difference is that ViT solves it by learning it back rather than building it in.

**The classification token `[CLS]`.** A learned vector added at the start of the sequence, not corresponding to any patch. After passing through all the layers, this vector is what gets fed into the classification layer. It plays the role of the "aggregation point", in place of the global average pooling at the end of a CNN.

Then comes attention. Deep Learning · Lesson 10 introduced the idea at the conceptual level: **every position connects directly to every position, and it can be computed in parallel**. Here it means the patch in the top-left corner and the patch in the bottom-right corner influence each other **already at the first layer** - something a CNN needs dozens of stacked layers to achieve.

The full mechanism of attention - queries, keys, values, multiple heads, and why it could replace recurrent networks - is the job of **the Generative AI chapter**, where the transformer is built out fully for language. This lesson only needs the part specific to images: cutting patches, embedding positions, and the cost of dropping an inductive bias.

## 3. What you lose by dropping the bias

This is where theory speaks fairly clearly, and it should be said before looking at the numbers.

We need to say precisely what ViT drops, because this point is very often told wrong.

**ViT still shares weights.** The same projection matrix is applied to all 196 patches; the same attention and MLP layers are applied to every token at every position. In terms of parameter savings from sharing, the two families do not differ.

What ViT drops is the **hard-wired locality** and the **translation equivariance** of convolution. A 3×3 filter is *forced* to look only at 9 adjacent pixels, and when the image shifts, the feature map shifts with it - both constraints live in the operation itself. Attention has no such constraint: every patch can look at every patch, and "patch 1 sits next to patch 2" is something the positional embedding has to **learn** rather than being given.

The theoretical consequence: **ViT needs more data to relearn what a CNN is given for free.** The original paper measured exactly this - trained on ImageNet's 1.3 million images, ViT loses to ResNet, but trained on JFT-300M it pulls ahead.

In return, with enough data, being unconstrained lets it learn relationships that a convolutional architecture cannot represent compactly.

But that is about **training from scratch**. The practical question most readers face is quite different: *"I have a pretrained model and a few hundred images - which should I pick?"* That is the question section 4 measures.

## 4. Little data: measuring transfer, not architecture

The experiment: both models are **pretrained on ImageNet**, both **fully frozen**, and only the final classification layer is trained on **370 images** of Oxford Pet (37 classes, i.e. 10 images per class), scored on **500 validation images**. One illustrative run, 2 epochs.

There is a reason to call it validation rather than test: those 500 images are used to **compare the two models and draw a conclusion**, which is exactly the selection role. This is the convention set up in Deep Learning · Lesson 12, and this lesson keeps it.

| Model | Total parameters | Trained parameters | Accuracy | Seconds/image |
|---|---|---|---|---|
| ResNet-50 (`IMAGENET1K_V2`) | 23,583,845 | 75,813 | 0.5140 | **0.056** |
| **ViT-B/16** | 85,827,109 | **28,453** | **0.8320** | 0.123 |

A gap of **31.8 points**. With 500 images, the sampling error of an accuracy number is around 2 points, so sampling noise alone can hardly explain a gap this large. But each model was trained only once, and since both models are scored on the same set of images a rigorous comparison has to be done per image - to quantify this gap with confidence you would still need several seeds.

And this is exactly where it is easy to write a wrong sentence. That wrong sentence is: *"ViT is more data-efficient than a CNN"*. The table above **does not prove** that, because the two models differ in **three** things at once - architecture, size (85.8 million versus 23.6 million parameters), and **pretraining recipe**.

The third is the easiest to check, so check it. Same ResNet-50 architecture, same data, same way of training the classification head - only change the **weight package**:

| ResNet-50 weights | **Published** ImageNet accuracy | Frozen features on Pet |
|---|---|---|
| `IMAGENET1K_V1` (original 2015 recipe) | 76.13% | **0.6920** |
| `IMAGENET1K_V2` (2021 recipe) | **80.858%** | 0.5140 |

This result deserves a slow read. The weight package that is **4.7 points worse on ImageNet** gives transfer features that are **17.8 points better**.

Strictly speaking, "only changing the weight package" involves two things that come together: a different training recipe **and** a different preprocessing - V1's `weights.transforms()` resizes the shorter side to 256 then crops 224, while V2 resizes to 232 then crops 224. Here we use exactly the preprocessing each package declares, as Lesson 01 advised, so the two factors cannot be separated.

So the correct statement is a statement about **stability**, not about cause:

> **17.8 of the 31.8 points of gap disappear just by changing ResNet-50's weight package.** A gap that can be moved this much by a change that does not touch the architecture at all cannot be attributed to the architecture.

The remaining 14.0 points - between ViT-B/16 and ResNet-50 V1 - still mix architecture with model size (85.8 million versus 25.6 million parameters in the original 1,000-class version), so they do not yet say anything about transformers versus convolution either.

> ⚠️ **A small note:** a plausible hypothesis for this: the `V2` recipe uses a lot of data augmentation and regularization to push ImageNet accuracy up, i.e. it optimizes hard for *exactly those thousand classes*, and that may make the features less general. But that is a hypothesis, not something the table above proves - proving it would require isolating each component of the training recipe. Do not flip to the opposite extreme either: in general ImageNet accuracy **does** correlate fairly strongly with transfer quality, and the survey by Kornblith et al. in Further reading shows that strong regularization is precisely where that correlation breaks. The practical lesson is therefore: **the ImageNet leaderboard is an initial indicator, not grounds for a final decision** - to decide, you have to measure on the validation set of your own problem.

## 5. Which one to choose

Pulling the whole lesson together into usable guidance, clearly separating what is backed by numbers from what is general knowledge:

| Situation | What to choose | Basis |
|---|---|---|
| Little data, using a pretrained model | **Try both**, score on validation | Table in section 4: large gap but mixed causes |
| Tight speed constraint | Lean toward CNN | ViT-B/16 is 2.2 times slower and 3.6 times heavier |
| Training from scratch, moderate data | Lean toward CNN | A built-in inductive bias helps when data is scarce |
| Training from scratch, very large data | ViT is worth trying | Published results of the original paper |
| Choosing any backbone | **Do not decide on ImageNet accuracy alone** | V1/V2 table: ImageNet ranking can run opposite to transfer quality |

The last row is the most expensive lesson of this lesson, and it ties straight back to the three-kinds-of-numbers convention of Lesson 01: **80.858% is a published number, 0.514 is a number measured on this machine, and in this case they point in opposite directions.** Two numbers about the same model that cannot substitute for each other.

> 🔧 **Try it now:** you need to choose a backbone for classifying 5 types of fabric, with 600 images. A colleague suggests taking the model at the top of the ImageNet leaderboard. How do you push back, and what process do you propose?
> Push back with exactly the V1/V2 table in section 4: the model with the higher ImageNet score gave transfer features 17.8 points worse on a different problem. The ImageNet ranking measures the ability to distinguish exactly those 1,000 classes, not the quality of features for reuse. The proposed process: pick three or four candidates from different families (a ResNet, a ViT, a lightweight model), **freeze them all** and train only the last layer - each run takes only a few minutes, as section 4 shows - then score on the validation set of the fabric problem itself. That is a cheap measurement that directly answers the question being asked. Only after picking a backbone do you consider deeper fine-tuning, following the ladder of Lesson 06.

## Lesson summary

- ViT cuts the image into non-overlapping patches - 224×224 with 16×16 patches gives **196 patches** - and then processes them as a sequence.
- Because attention treats every element of the sequence the same, ViT has to **add learned positional embeddings**; position is not built into the architecture as it is with convolution. The `[CLS]` token plays the role of the final aggregation point.
- Patches in opposite corners influence each other **already at the first layer**, something a CNN needs dozens of stacked layers to reach.
- ViT **still shares weights** (one projection matrix for every patch, the same layers for every token). What it drops is the **hard-wired locality** and **translation equivariance** of convolution - theory says that is why it needs more data when trained from scratch.
- On 370 Pet images with both frozen: ViT-B/16 reaches **0.8320** versus **0.5140** for ResNet-50 V2 - a gap of 31.8 points.
- **But more than half of that gap can be moved without touching the architecture**: same ResNet-50, just switching the weight package from V2 to V1 recovered 17.8 points - and the weight package brings along both a different training recipe and different preprocessing.
- The weight package that is **4.7 points worse on ImageNet** gives transfer features **17.8 points better**. A plausible hypothesis is that strong regularization makes the features less general, but this table does not prove that.
- The practical consequence: **do not decide on a backbone from the ImageNet leaderboard alone**; treat it as an initial indicator, then measure directly by freezing and training the last layer on your own problem.
- ViT-B/16 is 3.6 times heavier and 2.2 times slower than ResNet-50 on CPU.

## Self-check questions

1. How many patches does a 384×384 image with 16×16 patches give? How does that number affect the computational cost of attention?
2. Why does ViT need positional embeddings while a CNN does not? Relate this to the `nn.Flatten()` issue in Deep Learning · Lesson 09.
3. Name two inductive biases that a CNN has built in and ViT does not. Why is "weight sharing" **not** one of them?
4. The table in section 4 puts ViT 31.8 points ahead. List three differences between the two models that prevent us from drawing a conclusion about architecture.
5. ResNet-50 V2 beats V1 by nearly 5 points on ImageNet but is 17.8 points worse as a feature extractor. Name two factors that come with the weight package and prevent us from attributing the cause to the training recipe alone.
6. Design an experiment that separates the effect of *architecture* from the effect of *model size* in the ViT versus ResNet comparison. What do you need that this lesson does not have?
7. Why does this lesson not explain the query-key-value mechanism of attention?

## Further reading

**Foundational sources for this lesson:**

| Source | Which part to read |
|---|---|
| **Deep Learning · Lesson 10** | Attention at the conceptual level - every position connected directly, computed in parallel |
| **Deep Learning · Lesson 09** | Why `nn.Flatten()` destroys neighbor relationships, and what assumptions a CNN builds in |
| **Lessons 02 and 06 of this chapter** | Receptive field, and the ladder of fine-tuning strategies that section 5 ties into |
| **Dive into Deep Learning (d2l.ai)** | Chapter 11.8 *Transformers for Vision* - a full ViT implementation from scratch |

**Additional online sources (free):**

- [Dosovitskiy et al. - An Image is Worth 16x16 Words (ICLR 2021)](https://arxiv.org/abs/2010.11929) - the original ViT paper; figure 1 is the patch-cutting diagram of section 2, and section 4.2 is the experiment showing ViT needs large data when trained from scratch.
- [Wightman, Touvron, Jégou - ResNet strikes back (2021)](https://arxiv.org/abs/2110.00476) - the recipe behind the `V2` weights in section 4.
- [Kornblith, Shlens, Le - Do Better ImageNet Models Transfer Better? (CVPR 2019)](https://arxiv.org/abs/1805.08974) - a survey of the relationship between ImageNet accuracy and transfer quality, i.e. exactly the phenomenon in the V1/V2 table.
- [Raghu et al. - Do Vision Transformers See Like Convolutional Neural Networks? (NeurIPS 2021)](https://arxiv.org/abs/2108.08810) - compares the internal representations of the two families; read it if you want to go deeper than section 3.
- [Vision Transformer - torchvision documentation](https://pytorch.org/vision/stable/models/vision_transformer.html) - the available variants with their published numbers.

> **Next lesson:** [Representation Learning: From Fixed Labels to a Vector Space](../representation-learning-fixed-labels-to-vector-space/) - this lesson just showed how much a model's frozen features matter - and measured them by attaching a classification layer. The next lesson drops that layer altogether, uses the vectors directly, and opens up things no classification model can do: finding similar images, clustering without labels, and classifying a new set of classes without any training.
