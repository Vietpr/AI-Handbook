---
title: "What a CNN Actually Learns"
description: "Reading feature maps at each depth, implementing Grad-CAM in a dozen lines, and why a plausible-looking heatmap does not prove the model is reasoning correctly."
domain: "Computer Vision"
section: "Learn"
language: "en"
translationKey: "what-cnns-actually-learn"
order: 4
pubDate: 2026-08-26
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will be able to read feature maps at each depth and see how their sparsity grows, implement Grad-CAM yourself in a dozen or so lines, and - most importantly - know why a plausible-looking heatmap **does not** prove the model is reasoning correctly, through a test that refutes that very heatmap.

## 1. The question accuracy does not answer

Lesson 03 left us with a model that works well. But "works well" on the test set and "relies on the right thing in the image" are two different matters, and Machine Learning · Lesson 12 ran into exactly this problem in tabular form: a model can be right for the wrong reason.

With images, this has a well-known name in the trade: a model distinguishes huskies from wolves by looking at **the snow in the background**, because in the training data most wolf photos were taken on snow. Great accuracy, useless model.

This lesson opens two windows into the network. The first - feature maps - shows what the network *computes*. The second - Grad-CAM - shows which regions of the image push up the score of the chosen class. Then section 5 checks how far the second window can be trusted.

## 2. Feature maps get denser, then turn sparse

Pass an image through ResNet-50 and record the output of its four main blocks:

| Block | Output shape | Fraction of zero values | Fully silent channels |
|---|---|---|---|
| `layer1` | 256 × 56 × 56 | 39.8% | 1 / 256 |
| `layer2` | 512 × 28 × 28 | 53.9% | 0 / 512 |
| `layer3` | 1024 × 14 × 14 | 64.5% | 3 / 1024 |
| `layer4` | **2048 × 7 × 7** | **97.6%** | **1,273 / 2048** |

Two trends in opposite directions, and they are the same story.

**Space shrinks, channels swell.** From 56×56 down to 7×7 - 64 times fewer positions - while the channel count rises from 256 to 2048. The network is trading *where* for *what*: early layers know position very precisely but each channel carries little meaning; the last layer hardly distinguishes position any more but each channel corresponds to a much clearer concept.

**The deeper, the sparser.** At `layer4`, **97.6%** of the values are zero and **1,273 of the 2,048 channels are completely silent** - not activating at any position on this image.

The familiar way to tell it is: those 2,048 channels are 2,048 questions like "is there a fin?", "is there a wheel?", so most of the answers must be *no*. That picture is easy to remember, but it overstates things. Some channels do respond selectively to a particular kind of pattern and can be interpreted, but the network's representation is generally **distributed across many channels** rather than one concept per channel. And the fact that many values are zero only says the representation is *sparse on this image* - it does not yet prove that the network has "specialized well" or is working correctly.

What can be read from the table with certainty is the trend: the deeper, the higher the fraction of zero values. Interpreting what that trend means needs more evidence, and section 4 will show how hard that kind of evidence is to come by.

> ⚠️ **A small note:** the figure of 1,273 silent channels is **for this particular image**, not permanently dead channels like the dying ReLU in Deep Learning · Lesson 02. Feed in a different image and the set of silent channels will be different. To find truly dead channels you have to count over the whole dataset rather than a single image - just as lesson 02 of the previous chapter did with 1,000 images.

## 3. Grad-CAM in fifteen lines

The idea is compact: at `layer4` we have 2,048 7×7 maps. Which maps matter for the class in question? Ask the gradient. If increasing the value of channel $k$ a little makes the score of class $c$ rise a lot, then channel $k$ matters.

$$w_k = \frac{1}{7 \times 7}\sum_{i,j} \frac{\partial y_c}{\partial A^k_{ij}}, \qquad L = \text{ReLU}\Big(\sum_k w_k A^k\Big)$$

The weight $w_k$ is the average gradient over the whole map; $L$ is the weighted sum of the 2,048 maps, followed by ReLU to keep the part that *supports* that class and drop the part that works against it. Implemented with PyTorch hooks:

```python
acts, grads = {}, {}
model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))

def gradcam(x):
    model.zero_grad()
    out = model(x)
    c = int(out.argmax(1))
    out[0, c].backward()                       # gradient of the class-c score
    A, G = acts["v"][0], grads["v"][0]         # (2048, 7, 7)
    w = G.mean(dim=(1, 2))                     # per-channel weights
    cam = torch.relu((w[:, None, None] * A).sum(0))
    return c, cam / (cam.max() + 1e-8)
```

There is no library here at all - just two hooks, one `backward()` and one weighted sum. The result is a 7×7 grid for a 224×224 input image.

> ⚠️ **A small note:** it is easy to mistakenly say that each cell of the 7×7 grid "corresponds to a 32×32 pixel region". The number 32 is the **cumulative jump** (output stride) - the centers of two adjacent cells are 32 pixels apart on the original image. The **receptive field** of each cell, computed with the formula from lesson 02, is far larger than that, and the cells **overlap** heavily: each cell at `layer4` of ResNet-50 sees a region more than a hundred pixels wide. So do not read the 7×7 grid as a grid that cuts the image into 49 separate tiles. The occlusion test in section 4 still occludes along a 32×32 grid - but that is a choice of *region to intervene on*, not a claim that the region is the receptive field.

Run it on three tench images and print the grid as characters (darkening from a space to `@`):

```
image A - "tench" 0.6717         image B - "tench" 0.4509         image C - "sturgeon" 0.2512
    .:.
   .:-.
    .                                                              .+#-
 ==                                                               . =@+
+##=#**                             :=:                          .. ..
*%@=##*                            .@@@:
  .                                 ..
```

Image B is compact and clear: the model concentrates its attention on a small region in the lower half. Image A is more scattered, stretching across the whole bottom strip. Image C - the image the model gets **wrong**, calling the tench a sturgeon - concentrates its attention in the top right corner.

At this point, the usual story goes: "see, the model is looking at the fish". But that is still just *looking at a picture and telling a story*. The next section checks it with numbers.

## 4. Verifying the heatmap by occlusion

If the bright region really is what decides, then **occluding it should make the score drop sharply** - much more sharply than occluding a random region. That is an **interventional** test, and it is cheap: occlude 8 of the 49 cells (about 16% of the image area), then measure the probability of the very class the model just chose again.

Three ways of occluding to compare: the 8 **brightest** cells, 8 **random** cells (averaged over 5 draws), and the 8 **darkest** cells.

| Image | Prediction | Original probability | Occlude 8 brightest | Occlude 8 random | Occlude 8 darkest |
|---|---|---|---|---|---|
| A | tench (correct) | 0.6717 | **0.7654** ↑ | 0.6889 | 0.6936 |
| B | tench (correct) | 0.4509 | **0.3267** ↓ | 0.4334 | 0.4469 |
| C | sturgeon (wrong) | 0.2512 | **0.0103** ↓↓ | 0.2173 | 0.2856 |

**Image C is the case where Grad-CAM is entirely right.** Occluding the bright region makes the probability collapse from 0.2512 to 0.0103 - more than a twentyfold drop, while random occlusion barely changes it. That region really is what supports the prediction. Notably, this is the image the model gets **wrong**. Grad-CAM points to exactly where the model relies, and that place leads it to the wrong answer.

**Image B is right in direction, but mildly.** Occluding the bright region drops it by 0.124, random occlusion only by 0.018. A clear difference, if not a dramatic one.

**Image A refutes its own heatmap.** Occluding the "most important" region makes the model **more confident** - from 0.6717 up to 0.7654. The region Grad-CAM highlights most strongly, if it does anything, is *pulling* the prediction down. There is no reading that turns this result into "the heatmap pointed to the right place".

One third of the test images give the opposite result. That is why the next section exists.

> ⚠️ **A small note:** the occlusion test is not an absolute referee either. Occluding with zeros (which after normalization is the ImageNet mean color) creates an unnatural flat patch - itself something the model never saw during training, so the result mixes the effect of "losing information" with the effect of "adding strange noise". More rigorous tests replace the occluded region with a blurred image or with generated content that matches the background. Here we accept the cheap approach, and therefore read the results as a *relative comparison* between the three kinds of occlusion rather than by the absolute size of the drop.

## 5. A heatmap is a diagnostic tool, not evidence

Condensing the four sections into one sentence worth remembering: **Grad-CAM tells you which regions affect the score, not whether the model is "thinking" correctly.** It is a *local* explanation - for this image, this class, this model - and as the table in section 4 shows, sometimes that explanation does not even survive a simple causal test.

What it is reasonable to use it for:

**Tracing a specific error.** Image C is the textbook example: the model misnames the fish, and the heatmap plus the occlusion test tell you which region it relied on to make the mistake. That is a lead to follow up.

**Detecting a model that latches onto the background.** If across many images the bright region consistently falls outside the object - on snow, on a watermark, on the image border - that is an alarming sign, and it is valuable because it repeats across *many* images rather than one.

What it should not be used for:

**Presenting a nice heatmap as evidence that the model is trustworthy.** This is the most common use in presentations, and it is the wrong use. A heatmap that matches human intuition may just be a coincidence - image A above looks plausible yet does not survive the occlusion test.

**Drawing conclusions from one image.** For the same reason lesson 07 says "mAP on a single image cannot be used as a metric": a single observation cannot tell a pattern from chance.

> 🔧 **Try it now:** you have a model that classifies X-ray images, with accuracy 0.94. Grad-CAM on five cases all highlight exactly the lung region. What can you conclude, and what more do you need to do before using it for real?
> Very little can be concluded. Five images are far too few, and as image A shows, the bright region may not be the deciding region. What to do: (1) run the occlusion test on a large set, comparing the drop when occluding the bright region with the drop when occluding a random region - if the two are about the same, the heatmap tells you nothing; (2) separately check the cases where the model is **wrong**, because they carry more information than the correct ones; (3) look for shortcut cues in the data - text, hospital labels, scanner markings in the corner of the image, the classic case that makes X-ray models right for the wrong reason; (4) break the metrics down by patient group and by imaging device, just as Machine Learning · Lesson 12 did.

## Lesson summary

- Going deeper into the network, feature maps **shrink spatially and swell in channel count**: from 256 × 56 × 56 at `layer1` to 2048 × 7 × 7 at `layer4`.
- Sparsity grows with depth: from 39.8% zero values at `layer1` to **97.6%** at `layer4`, with 1,273 of 2,048 channels completely silent on one image. But sparsity by itself proves nothing about quality, and **do not read each channel as one concept** - representations are usually distributed across many channels.
- The 7×7 grid at `layer4` has a **jump** of 32 pixels, not a receptive field of 32 pixels; the cells see much wider regions and overlap each other.
- **Grad-CAM** can be implemented in a dozen or so lines with two hooks: take the activations and gradients at the last convolutional layer, average the gradients to get channel weights, take the weighted sum and apply ReLU.
- The occlusion test is a way to verify a heatmap by **intervention**: occlude the bright region and compare the drop with occluding a random region. It is not rigorous causal evidence, because the occluded region itself creates an input the model has never seen.
- On three test images, **two confirm and one refutes** - for image A, occluding the brightest region actually makes the probability *rise* from 0.6717 to 0.7654.
- The most convincing case (image C: 0.2512 → 0.0103) is one where the model is **wrong**: Grad-CAM points to exactly where the model relies, and that place leads to the wrong answer.
- **A plausible heatmap does not prove the model is reasoning correctly.** It is a local diagnostic tool, useful for tracing errors and detecting a model that latches onto the background, not evidence of reliability.

## Self-check questions

1. Why does the sparsity of feature maps grow with depth? Why is that a good sign rather than a sign of breakage?
2. Distinguish "1,273 silent channels on this image" from the "dead channels" of Deep Learning · Lesson 02. How would you measure to check the latter?
3. Distinguish the **cumulative jump** from the **receptive field** of a cell in the 7×7 grid. Why does confusing the two lead to misreading a heatmap?
4. In the Grad-CAM formula, why does the weight $w_k$ average the gradient over the whole 7×7 map instead of using the gradient at each position?
5. Why is there a ReLU step at the end? If you removed it, what different meaning would the map carry?
6. Image A: occluding the brightest region makes the probability rise. Give two different explanations for this phenomenon.
7. Design an occlusion test better than occluding with zeros. State the strengths and weaknesses of the approach you propose.
8. A colleague presents a slide with four nice Grad-CAM images and concludes "the model is looking at the right place". What three questions do you ask back?

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Machine Learning · Lesson 12** | The "what is the model relying on" part - the same question, on tabular data |
| **Deep Learning · Lesson 02** | Dying ReLU, to distinguish it from channels that are silent on one image |
| **Dive into Deep Learning (d2l.ai)** | Chapters 7.6 and 8.6 - ResNet's block structure, helpful for understanding why `layer4` is the place to attach a hook |

**Additional online sources (free):**

- [Selvaraju et al. - Grad-CAM (ICCV 2017)](https://arxiv.org/abs/1610.02391) - the original paper; section 3 is exactly the formula implemented in section 3 of this lesson.
- [Adebayo et al. - Sanity Checks for Saliency Maps (NeurIPS 2018)](https://arxiv.org/abs/1810.03292) - shows that several explanation methods produce plausible-looking maps even when the model's weights are randomly shuffled. Read this before trusting any heatmap.
- [Ribeiro, Singh, Guestrin - "Why Should I Trust You?" (KDD 2016)](https://arxiv.org/abs/1602.04938) - the source of the husky and snow example, along with the idea of local explanations.
- [Zeiler & Fergus - Visualizing and Understanding Convolutional Networks (ECCV 2014)](https://arxiv.org/abs/1311.2901) - section 4.2 is a systematic occlusion test, the forerunner of the test in section 4.
- [`register_forward_hook` - PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) - the hook mechanism used to grab activations from the middle of the network.

> **Next lesson:** [Image Datasets and Augmentation](../image-datasets-and-augmentation/) - the first four lessons all used models someone else had trained. The next one begins the part where you train yourself, and the first question is not which architecture but **what goes in** - how to split image datasets, which augmentations suit which problems, and which ones silently corrupt the labels.
