---
title: "Stacking Layers: Networks Learn Their Own Features"
description: "What a hidden layer actually produces, how much \"deep\" buys and where it stops, and why two narrow layers usually beat one wide layer with the same parameter budget."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "stacking-layers-networks-learn-their-own-features"
order: 3
pubDate: 2026-08-20
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand what a hidden layer actually produces, see in numbers how much "deep" buys and where it stops, know why with a comparable parameter budget two narrow layers usually learn better than one wide layer, and be able to read the story behind later layers learning on top of the features of earlier ones.

## 1. A hidden layer is a feature table the machine writes itself

Lesson 02 ended with a single neuron. Place 256 neurons side by side, all receiving the same input, and we get a **layer**:

```
              ┌──▶ neuron 1 ──▶ a₁
   784 pixels ┼──▶ neuron 2 ──▶ a₂     256 new numbers,
              ┼──▶    ⋮                each one a
              └──▶ neuron 256 ─▶ a₂₅₆  combination of all 784 pixels
```

Look carefully at the output on the right: 256 numbers. That is **a new data table**, with 256 columns, describing the same image. And each column is a weighted combination of the 784 original pixels - exactly the kind of "engineered feature" that Machine Learning · Lesson 11 had you sit down and invent by hand.

The only difference, and it is the difference of this whole chapter: **nobody wrote those 256 combinations**. They start as random numbers, and gradient descent gradually adjusts them so that the last layer classifies better. This is what is called representation learning, and this is its concrete shape.

This gives a very handy way to read a neural network:

> The last layer does exactly the job of the softmax regression in lesson 01. Every layer before it has only one task: **turn raw data into something easier to classify linearly**.

You can verify this right away. Take a trained network, cut off the last layer, and use the 256 numbers from the hidden layer as input to a new softmax regression - it will reach nearly the accuracy of the whole network. The hidden layer has already set the table.

## 2. How much deeper buys, and where it stops

The natural question: if one hidden layer sets the table well, do two layers set it better? Measured on FashionMNIST, each hidden layer with 256 neurons, 15 epochs, everything else held fixed:

| Hidden layers | Parameters | Best validation accuracy |
|---|---|---|
| 0 | 7,850 | 0.8357 |
| 1 | 203,530 | 0.8622 |
| 2 | 269,322 | 0.8666 |
| 3 | 335,114 | **0.8674** |
| 4 | 400,906 | 0.8595 |

Read this table in three beats.

**Beat one: the first hidden layer buys the most.** From 0 to 1 layer: +2.6 points. That is the step from "straight boundary" to "curved boundary" - a change in kind.

**Beat two: later layers buy less and less.** From 1 to 2: +0.4 points. From 2 to 3: +0.1 points. Still positive, but already close to the noise level between runs.

**Beat three, and this is the part worth remembering: four layers are *worse* than three.** 0.8595 versus 0.8674, despite having the most parameters in the table. More capacity and the result goes down - Foundations · Lesson 13 warned that this can happen, but here the cause is not overfitting: the training loss of the 4-layer network is not lower either. It is simply **harder to train**. The learning signal has to travel back through more layers to reach the first one, and it weakens along the way. Lesson 04 measures that weakening concretely, and lesson 08 names it and fixes it.

> ⚠️ **A small note:** do not conclude that "three layers is the best number". This table holds for *fully connected networks, on FashionMNIST, with default initialization and optimization algorithm*. Real deep networks - ResNet in lesson 09, for example - have tens to hundreds of layers and work well, but they use techniques that only lesson 08 introduces. What this table shows is: plain layer stacking hits a ceiling very early.

## 3. Wide or deep, if you can pick only one

Suppose your budget is about 200-400 thousand parameters. Do you split that budget into a few wide layers or many narrow ones?

| Architecture | Parameters | Validation accuracy |
|---|---|---|
| 1 hidden layer × 512 neurons | 407,050 | 0.8588 |
| **2 hidden layers × 256 neurons** | **269,322** | **0.8666** |
| 4 hidden layers × 158 neurons | 200,986 | 0.8561 |

Two layers win, and win with **34% fewer parameters** than one wide layer. But push on to four narrow layers and it drops again - the same phenomenon as in section 2.

The intuition behind it: each layer learns combinations *of the previous layer*. With one layer, every feature has to be built directly from raw pixels. With two layers, the second layer builds on the pieces the first layer has already made - like building a house from precast walls instead of from individual bricks. Depth allows **reuse**: a piece learned by layer 1 can take part in many different features of layer 2, so the same parameter budget can express more.

> 🔧 **Try it now:** a network has 784 inputs, one hidden layer of 256 neurons and 10 outputs. Count the parameters.
> Hidden layer: each neuron has 784 weights plus 1 intercept, times 256 neurons = $785 \times 256 = 200{,}960$. Output layer: $257 \times 10 = 2{,}570$. Total **203,530** - matching exactly the "1 hidden layer" row in the table in section 2. Notice that most of the parameters sit in the first layer, simply because the input has 784 dimensions; lesson 09 will show a way to cut this number down drastically.

## 4. What later layers see

There is a way to picture what the layers learn, and it has become a classic image of deep learning on image data:

```
   raw image         layer 1             layer 2              layer 3
  (pixels)      edges, bright blobs  corners, curves,      parts:
                                    repeated patterns    sleeves, shoe heels
      │               │                   │                    │
      └───────────────┴───────────────────┴────────────────────┘
                    each layer builds on the previous one
```

This is a qualitative description, not a law you can measure in every network. But it is true enough to be useful, and it explains why depth matters: a cat's ear is not a linear combination of pixels, but it *is* a combination of curves, and curves in turn are combinations of edges.

A very large practical consequence of this view: if the early layers learn generic things - edges, corners, patterns - then they are **less tied to the specific problem** than the later layers. An edge in a cat photo is also an edge in a photo of a shirt. This means you can take the early layers of a network trained on millions of images and reuse them for your own problem with only a few thousand images. That is the whole idea of lesson 11.

This is a matter of degree, not of yes or no: the deeper you go, the more tightly the features are tied to the original problem, and how reusable they are declines layer by layer. That is why in lesson 11 we keep the early part of a pretrained network as it is but replace the final part entirely.

> ⚠️ **A small note:** the picture of "layer 1 learns edges, layer 2 learns corners" comes mainly from **convolutional networks** on images (lesson 09), where we can look at the filters directly as images. With fully connected networks like the ones in this lesson, the learned features are much harder to interpret: each neuron in layer 1 looks at *all* 784 pixels at once, so its weights, when plotted, usually look like noise rather than forming clear edges. Do not expect to look at the weights of an `nn.Linear` and see a cat's ear.

## Lesson summary

- A hidden layer of 256 neurons turns a 784-column table into a new 256-column table, each column a weighted combination of the inputs - exactly the engineered features of Machine Learning · Lesson 11, except that the machine finds them instead of you inventing them.
- A handy reading: the last layer does the job of softmax regression, and every layer before it turns raw data into something easier to classify linearly.
- Depth buys less and less: 0 → 1 layer gains +2.6 points; 1 → 2 gains +0.4; 2 → 3 gains +0.1.
- Four layers are *worse* than three (0.8595 versus 0.8674) despite having the most parameters - not overfitting but difficulty training, the topic of lesson 04 and lesson 08.
- With the same parameter budget, two layers of 256 beat one layer of 512 with 34% fewer parameters, because later layers can reuse the pieces the earlier layer built.
- If the early layers learn generic things, they can be reused for other problems - the foundation of transfer learning in lesson 11.

## Self-check questions

1. What does a hidden layer produce, expressed in the language of "data table, rows and columns"? Exactly where does it differ from the feature engineering of the previous chapter?
2. A 4-hidden-layer network has more parameters than a 3-layer network but lower accuracy. Give two different explanations, and which numbers would you use to tell them apart?
3. Count the parameters of the network 784 → 512 → 10. Compared with the network 784 → 256 → 256 → 10, which has more parameters?
4. Why, with a comparable parameter budget, do two narrow layers usually learn better than one wide layer on this data? Give one reason that argument is not a law that always holds.
5. You cut off the last layer of a trained network and use the hidden-layer output as input to a new logistic regression. What accuracy do you expect compared with the original network, and why?
6. Someone says "if you want it better, just add layers". Based on the table in section 2, how do you argue against that?

## Further reading

**Core sources for this lesson:**

| Source | What to read |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chapter 5.1 *Multilayer Perceptrons* - how layers are stacked and the notation; chapter 5.2 *Implementation of MLPs*; chapter 5.6 discusses depth and training difficulty |
| **An Introduction to Statistical Learning (ISLP)** | Chapter 10.2 *Multilayer Neural Networks* - networks with many hidden layers, and remarks on later layers building on the features of earlier ones |
| **Foundations · Lesson 13** | Capacity: why adding parameters does not guarantee better results |

**Additional online sources (free):**

- [TensorFlow Playground](https://playground.tensorflow.org/) - add or remove layers and neurons and watch the decision boundary change shape live; this is the fastest way to get a feel for section 2.
- [Zeiler & Fergus - Visualizing and Understanding Convolutional Networks (ECCV 2014)](https://arxiv.org/abs/1311.2901) - the source of the "layer 1 learns edges, later layers learn parts" picture in section 4, with real images for each layer.
- [Distill - Feature Visualization](https://distill.pub/2017/feature-visualization/) - an interactive article showing what each neuron in a vision network responds to.
- [Nielsen - Neural Networks and Deep Learning, chapter 5](http://neuralnetworksanddeeplearning.com/chap5.html) - a free book whose chapter discusses exactly why deep networks are hard to train, leading straight into lessons 04 and 08.

> **Next lesson:** [Computational Graphs: One backward() Call for Millions of Parameters](../computational-graphs-and-backward/) - this lesson says later layers learn on top of earlier ones, but "learning" here means someone turning which knobs. The next lesson lifts the lid on that mechanism - and measures the weakening that section 2 just ran into.
