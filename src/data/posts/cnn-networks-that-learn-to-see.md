---
title: "CNNs: Networks That Learn to See"
description: "The two ideas behind convolutional networks, local receptive fields and shared weights, what convolution and pooling do, and why a network with ten times fewer parameters beats a fully connected one on images."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "cnn-networks-that-learn-to-see"
order: 9
pubDate: 2026-08-23
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand the two ideas behind convolutional networks - looking only locally and sharing weights - know what convolution and pooling do, and see in numbers why a network with ten times fewer parameters beats a fully connected network on images.

## 1. Eight lessons in, we are still throwing information away

Every network from Lesson 01 until now has started with exactly the same line:

```python
nn.Flatten()      # (1, 28, 28) → (784,)
```

That line unrolls the image into a row of 784 numbers. Absolute position is not lost: the pixel in row 4, column 12 always lands in slot 100, and the network is perfectly able to learn that slot 100 matters. What is lost is **neighborhood**: to `nn.Linear`, slot 100 and slot 101 - two pixels that are actually adjacent - are just two of 784 equal inputs, no closer to each other than slot 100 and slot 600. The network can relearn that relationship from data, but it has to learn it from scratch, even though it already holds true for every image.

One experiment makes this clear. Shuffle the 784 pixels into a fixed random order - applying the same shuffle to every image - and retrain from scratch. The fully connected network gets *exactly the same* result as before, because the shuffle merely renames the inputs, and every naming is equivalent to it. To that network, an image is just a bag of numbers with no shape. Run the same experiment on a CNN and the result collapses, because the thing a CNN relies on - nearby pixels are related - has been destroyed by the shuffle.

But images do have shape, and two facts about them are well worth exploiting:

- **Nearby pixels are related to each other.** An edge is something that happens between a few adjacent pixels, not between the top-left pixel and the bottom-right pixel.
- **A pattern means something at every position.** A vertical edge in the left corner is still a vertical edge when it appears in the right corner. An edge detector learned for one region can be reused for another.

A fully connected network knows neither of these facts, so it has to relearn them from data - costing parameters and examples. A **convolutional neural network (CNN)** is built with both facts already baked in.

## 2. Looking only locally, and sharing weights

In an `nn.Linear` layer, each neuron connects to **all 784** pixels. In a convolutional layer, each neuron looks at only a small square - usually 3×3 - and **the same set of 9 weights is slid across the whole image**.

```
   3x3 filter            slides across the image
   ┌───┬───┬───┐      ┌───────────────────┐
   │ w₁│ w₂│ w₃│      │ ┌───┐             │
   ├───┼───┼───┤      │ │3x3│──────▶      │   the same 9 weights
   │ w₄│ w₅│ w₆│  ──▶ │ └───┘             │   used at EVERY position
   ├───┼───┼───┤      │        ┌───┐      │
   │ w₇│ w₈│ w₉│      │        │3x3│───▶  │
   └───┴───┴───┘      └───────────────────┘
    9 weights + 1 bias                         result: a "feature map"
```

Each time the filter is placed on a 3×3 patch, we multiply pair by pair and add up - exactly the dot product from Foundations · Lesson 07 - and get one number. Sliding over the whole image gives a new grid of numbers called a **feature map**: it shows where in the image the pattern this filter is looking for appears strongly.

A layer usually has many filters, each learning a different pattern - this one catches vertical edges, that one horizontal edges, another a bright patch.

The consequence for the parameter count is staggering:

| | Parameters |
|---|---|
| `nn.Linear(784, 784)` | 615,440 |
| `nn.Conv2d(1, 16, 3×3)` - **16 filters** | **160** |

And that 160 **does not change with image size**, because the filter is only 3×3 no matter how large the image is:

| Image size | `Linear` to 256 neurons | `Conv2d(1, 16, 3×3)` |
|---|---|---|
| 28 × 28 | 200,960 | 160 |
| 64 × 64 | 1,048,832 | 160 |
| 224 × 224 | **12,845,312** | **160** |

This is why Lesson 01 said "the old fix runs out of road" for 224×224 images: the first layer alone of a fully connected network eats up nearly 13 million parameters, while a convolutional layer does comparable work with 160.

## 3. Pooling: shrink it down, keep the strongest

After a few convolutional layers, people usually insert a **pooling** layer to shrink the feature map. The most familiar kind is **2×2 max pooling**: split the map into 2×2 cells and keep the largest value in each cell.

```
   4  1 │ 0  2          max pooling 2x2
   2  3 │ 7  1     ──▶     4   7
   ─────┼─────              9   6
   9  5 │ 3  6
   1  0 │ 2  4
```

Two effects. First, the size drops by a factor of four, so the following layers become much cheaper. Second, it creates a little **positional tolerance**: if an edge shifts one pixel to the right but stays inside the same 2×2 cell, the output does not change. "A little" in the literal sense - this tolerance is local, about one pixel per 2×2 pooling layer, not the ability to withstand shifts of several pixels that many people assume. Section 5 measures exactly that.

Putting it together into a small CNN:

```python
cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28×28 → 14×14
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14×14 → 7×7
    nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
```

Notice that `nn.Flatten()` is still there - but now it sits at the **end**, after the convolutional layers have distilled the image into meaningful features. That is the key difference from the previous eight lessons, where we unrolled the image right at the start.

## 4. CNN vs. MLP: compared on FashionMNIST

Same FashionMNIST, same 10 epochs, same learning rate:

| Model | Parameters | Accuracy | Training time |
|---|---|---|---|
| MLP, 1 hidden layer of 256 | 203,530 | 0.8488 | 55 seconds |
| **CNN, 2 convolutional layers** | **20,490** | **0.8781** | 161 seconds |

The CNN uses **10 times fewer parameters** and gets **2.9 points higher** accuracy.

Compare this with Lesson 01, where adding a whole hidden layer of 256 neurons - 26 times the parameters - bought only 2.8 points. Here the same gain comes from *cutting* the parameters tenfold, simply by choosing an architecture that knows its input is an image. That is exactly what Lesson 01 promised: the big leap comes from an architecture that fits the data, not from adding parameters.

In exchange, the CNN takes **three times as long** to train despite having fewer parameters. Fewer parameters does not mean fewer computations: each filter has to slide over every position in the image, so the number of multiplications is huge even though the number of weights is tiny. This is why deep learning for images goes hand in hand with GPUs - hardware that does thousands of multiplications in parallel.

## 5. "Translation invariance" - getting it right

Many sources say CNNs are "invariant to translation". Let us check: take both trained models, shift every validation image horizontally by **3 pixels**, and measure again.

| Model | Original accuracy | After a 3-pixel shift | Lost |
|---|---|---|---|
| MLP | 0.8550 | 0.3350 | **52.0 points** |
| CNN | 0.8810 | 0.4570 | **42.4 points** |

The CNN copes better - but it collapses too. Losing 42 points is not "invariance".

The reason is worth understanding well. Convolution itself is **equivariant**: shift the image and the feature map shifts with it, rather than staying the same. Pooling provides a little tolerance, but only within the pooling cell - 2×2 forgives one pixel, not three. And the `nn.Linear` layer at the end reads the feature map by exact position, so when everything shifts by three pixels, it receives a completely different layout.

The correct statement is: **a CNN has an inductive bias suited to images; it is not invariant to translation.** If you want it to truly withstand shifts, you have to train it to - with the data augmentation from Lesson 07, namely the `RandomCrop` we used there.

> ⚠️ **A small note:** `padding=1` in `nn.Conv2d(1, 16, 3, padding=1)` is not a decorative detail. A 3×3 filter sliding over a 28×28 image without padding can only be placed at 26×26 positions - the image shrinks by 2 pixels per layer. Stack ten layers and you lose 20 pixels, meaning the 28×28 image all but disappears. Padding a ring of zeros around the image keeps its size, so you can stack as many layers as you like. The general rule for an odd $k \times k$ filter: `padding = (k-1)/2` keeps the size unchanged.

> 🔧 **Try it now:** why does sharing weights (one filter sliding across the whole image) help both with the number of parameters *and* with generalization?
> For parameters it is obvious: one set of 9 numbers instead of a separate set at every position. For generalization it is subtler: it **builds a correct assumption into the model** - that an edge in the left corner and an edge in the right corner are the same thing. A fully connected network has to learn that from data, meaning it has to see the edge at enough different positions before it learns it. The CNN gets it for free. Foundations · Lesson 13 calls this putting knowledge about the problem into the structure of the model, and it is far cheaper than letting the model figure it out from data.

## 6. This is only the doorway

This lesson stops at the level of intuition: convolution, filters, feature maps, pooling, and why they suit images. There is a great deal more left unsaid.

**Architectures.** The CNN in this lesson has 2 convolutional layers. Real networks have dozens: LeNet (1998) for handwritten digits, AlexNet (2012) which kicked off the deep learning wave in Lesson 01, VGG, and **ResNet** with skip connections - the thing Lesson 08 promised as the fix for truly deep networks.

**Problems beyond classification.** Images are not only for asking "what is this": there is also object detection (what, and where), segmentation (which object each pixel belongs to), OCR, and vision transformer models.

All of that is the content of **the Computer Vision chapter**. This lesson only needs you to grasp one thing firmly: a CNN wins not because it is bigger, but because it *knows in advance* that its input is an image.

## Lesson summary

- `nn.Flatten()` at the start of the network does not lose absolute position but does lose neighborhood: shuffle the pixels into a fixed order and retrain, and the fully connected network gives exactly the same result, while the CNN collapses.
- A CNN builds in two facts about images: nearby pixels are related (looking only locally), and a pattern means something at every position (sharing weights).
- The parameter count of a convolutional layer **does not change with image size**: `Conv2d(1, 16, 3×3)` is always 160, while `Linear` to 256 neurons goes from 200,960 (28×28 images) to 12,845,312 (224×224 images).
- Pooling shrinks the feature map and provides a little positional tolerance.
- On FashionMNIST, a CNN with 20,490 parameters reaches 0.8781, beating an MLP with 203,530 parameters (0.8488) - 10 times fewer parameters, 2.9 points better.
- Fewer parameters does not mean fewer computations: the CNN trains three times slower than the MLP, because the filters have to slide over every position.
- A CNN is **not** invariant to translation: shifting the image by 3 pixels costs it 42.4 points (the MLP loses 52.0). It has an inductive bias suited to images, and to withstand shifts it has to be trained with augmentation.

## Self-check questions

1. You shuffle the 784 pixels of every image in the same fixed random order. How is the fully connected network affected? And the CNN?
2. Why does the parameter count of a convolutional layer not depend on image size, while that of a `Linear` layer does?
3. The CNN has 10 times fewer parameters but trains three times longer. Explain this paradox.
4. How many pixels of tolerance does 2×2 max pooling provide? How does that explain the result in section 5?
5. Someone says "CNNs are translation invariant, so data augmentation is unnecessary". Based on the table in section 5, how would you argue back?
6. Sharing weights builds an assumption about the world into the model. What is that assumption, and can you imagine a kind of data for which it is **wrong**?

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chapter 7.1 *From Fully Connected Layers to Convolutions* - the same line of argument as sections 1-2, including the reasoning about equivariance; 7.2 convolution; 7.5 pooling; 8.1 LeNet |
| **An Introduction to Statistical Learning (ISLP)** | Chapter 10.3 *Convolutional Neural Networks* - filters, feature maps and pooling explained with small numeric examples |
| **Lesson 08 of this chapter** | Skip connections and ResNet - the promise recalled in section 6 |

**Additional online sources (free):**

- [`nn.Conv2d` - PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - the formula for the output size in terms of `padding`, `stride`, `dilation`.
- [CS231n - Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) - Stanford lecture notes, with animations illustrating a sliding filter.
- [LeCun et al. - Gradient-Based Learning Applied to Document Recognition (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) - the LeNet paper, the first practical CNN.
- [He et al. - Deep Residual Learning for Image Recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385) - ResNet and skip connections, the fix for deep networks that Lesson 08 promised.
- [Distill - Feature Visualization](https://distill.pub/2017/feature-visualization/) - seeing what the filters of a real CNN actually learn.

> **Next lesson:** [Sequence Data: RNNs and Where They Fall Short](../sequence-data-rnns-and-where-they-fall-short/) - a CNN exploits the *spatial* structure of images. The next lesson switches to a different kind of structure - order in time - where each input depends on what came before it.
