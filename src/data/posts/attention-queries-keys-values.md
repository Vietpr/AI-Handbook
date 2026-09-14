---
title: "Attention: Queries, Keys, Values"
description: "The mechanism the previous two chapters deliberately left out, computing attention by hand, and why an attention matrix must not be read as an explanation of the model's answer."
domain: "Generative AI"
section: "Learn"
language: "en"
translationKey: "attention-queries-keys-values"
order: 3
pubDate: 2026-09-01
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand the mechanism the previous two chapters deliberately left out, be able to compute an attention step by hand, and - just as important - know why **an attention matrix must not be read as an explanation** of the model's answer, with evidence measured on a real model.

## 1. A debt owed by two chapters

Deep Learning · Lesson 10 showed that a recurrent network has to walk through the steps one at a time, so the signal from the start of the sequence fades away and nothing can be computed in parallel. That lesson introduced attention at the conceptual level - *every position connects directly to every position* - and then stopped.

Computer Vision · Lesson 10 used that same mechanism for images, but it too only opened the image-specific part: cutting patches, embedding positions. The mechanism inside was left aside.

This lesson pays both debts.

## 2. Three roles for the same vector

The core idea is very similar to looking something up in a dictionary. You have a **question**, the dictionary has **entries**, and each entry has some **content**. You compare the question with the entries to know which content to take.

Attention does exactly that, but "softly": instead of picking one entry, it takes a **weighted average** of all the content.

Each input vector $x_i$ is projected into three things by three learned matrices:

$$q_i = W_Q x_i \qquad k_i = W_K x_i \qquad v_i = W_V x_i$$

- **Query** $q_i$ - "what information am I looking for"
- **Key** $k_j$ - "what kind of information do I have"
- **Value** $v_j$ - "the content I carry"

Then three steps:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} \qquad \alpha_{ij} = \text{softmax}_j(\text{score}_{ij}) \qquad y_i = \sum_j \alpha_{ij} v_j$$

Written compactly, this is the familiar formula:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

**Why divide by $\sqrt{d}$.** The dot product of two $d$-dimensional vectors grows in magnitude with $d$. Without the division, for large $d$ the scores blow up, softmax saturates into something close to one-hot, and the gradient vanishes - exactly the sigmoid illness of Deep Learning · Lesson 08. The division keeps the scores on a sensible scale.

**Why a causal mask is needed.** A language model has to guess the next token, so when processing position $i$ it may look at every position $\le i$ but **may not look** at positions $> i$ - that is, the future is blocked, the past is not. Otherwise the task becomes peeking at the answer - exactly the leakage of Foundations · Lesson 14, in structural form. The way to block it is to set the score of every pair $(i, j>i)$ to $-\infty$ before softmax, so that they receive weight 0. The grid below shows this for the Vietnamese sentence "Hà Nội là thủ đô" ("Hanoi is the capital").

```
      Hà   Nội   là   thủ   đô
Hà    ✓    ✗     ✗    ✗     ✗
Nội   ✓    ✓     ✗    ✗     ✗
là    ✓    ✓     ✓    ✗     ✗
thủ   ✓    ✓     ✓    ✓     ✗
đô    ✓    ✓     ✓    ✓     ✓
```

## 3. Many heads: many mixes in parallel

A single attention step produces only **one** way of mixing information. But a sentence holds many kinds of relationships at once - subject with verb, pronoun with what it refers to, noun with the adjective modifying it.

**Multi-head attention** runs several attention steps in parallel, each with its own $W_Q, W_K, W_V$, then concatenates the results and projects them once more. Each "head" is free to learn a different kind of relationship.

The cost does not grow, because the dimensions are split: a model with 896 dimensions and 14 heads has each head working in 64 dimensions.

> ⚠️ **A small note:** the description above is **classic multi-head attention**, where each head has its own keys and values. Many modern models use a more economical variant: Qwen2.5-0.5B has **14 query heads but only 2 key-value sets**, meaning seven query heads share one key-value set. This is called **grouped-query attention (GQA)**, and it exists to reduce memory at inference time - a topic for the AI Systems chapter. For this lesson the mechanism is unchanged; just do not be surprised when you read a real model's config and find `num_key_value_heads` smaller than `num_attention_heads`.

And the heads **really are** different. Take Qwen2.5-0.5B - **24 layers, 14 query heads** - run it on the sentence *"Hà Nội là thủ đô của Việt Nam. Thành phố này nằm ở miền Bắc."* ("Hanoi is the capital of Vietnam. This city is in the North.") and look at where the last token is looking in the first four heads of the last layer:

| Head | Three most-attended tokens |
|---|---|
| 0 | `.` 0.2629 · ` Nội` 0.1958 · `.` 0.1688 |
| **1** | **`.` 0.9919** · ` Bắc` 0.0027 · ` này` 0.0024 |
| 2 | `.` 0.2292 · ` Nội` 0.2062 · `.` 0.1516 |
| 3 | `.` 0.2680 · ` Nội` 0.1582 · `.` 0.1495 |

Head 1 puts **99.19%** of its weight on a single token, while the other three spread it out. Same layer, same input, completely different behavior. That is why models use many heads.

## 4. Where attention looks - and why not to read it as an explanation

This is where it is easy to fall into a very common trap. Because the attention matrix can be visualized, people like to point at it and say: *"the model attends to this word, so it answers like that."*

Computer Vision · Lesson 04 met exactly that trap with Grad-CAM, and the occlusion test showed that one third of the images **contradicted their own heatmap**. Here we keep that same discipline.

Look at where the last token of the sentence above attends, at three different layers:

| Layer | Four most-attended tokens |
|---|---|
| 0 | `.` 0.2367 · ` Bắc` 0.2199 · `.` 0.0967 · `H` 0.0900 |
| 12 | `H` 0.3084 · `.` 0.1225 · ` Bắc` 0.1035 · `.` 0.0826 |
| 23 | `.` 0.2796 · `.` 0.2754 · ` Nội` 0.0960 · `H` 0.0889 |

What takes most of the weight is **the periods** and **the first token of the sentence** - not the words that carry meaning. If we read this table as an explanation, we would have to conclude that the model answers based on punctuation.

And the phenomenon is even more systematic than that. Measure the share of attention going to **the first token** of the sequence, averaged over all heads, across all 24 layers:

```
layer 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
share .15 .24 .15 .71 .59 .65 .61 .73 .45 .79 .56 .78 .51 .71 .56 .59 .91 .73 .73 .63 .83 .82 .25 .15
                          ▲                                    ▲
                      jumps to 71%                        peak 91%
```

At layer 16, **91% of all attention goes to the first token**. The average across the middle layers is also around 60-80%.

That first token carries no special information - it is just the opening token. The phenomenon has a name in the field: **attention sink**. A widely accepted explanation is that softmax forces the weights to sum to 1, so when a head **does not need to take information from anywhere**, it still has to put its weight somewhere - and it picks a harmless position.

Three conclusions, and they matter more than the numbers themselves:

**High attention weight does not mean important.** Most of the weight here goes to the most meaningless place possible.

**Attention is only one piece of the computation.** The output of each layer also passes through a projection, a skip connection, an FFN block, and then 23 more layers. Looking at one attention matrix in one layer is looking at one link in a very long chain.

**It is still useful - just useful for something else.** The attention matrix lets us **directly see one information-mixing step inside the model**, which is very rare in deep learning. Use it for diagnosis, for forming hypotheses, for spotting odd behavior. Do not use it as evidence for why the model answered.

> ⚠️ **A small note:** all the numbers in sections 3 and 4 are measured on **exactly one sentence**, with **one model**. They illustrate the phenomena, they do not measure how often they occur. Attention sinks have been documented by many studies across many different models - listed in Further reading - but the specific figure of 91% at layer 16 belongs to this run alone.

## 5. Why attention could replace RNNs

Back to the problem that Deep Learning · Lesson 10 left open.

**Short paths.** In a recurrent network, the signal from token 1 to token 500 has to pass through 499 steps, each one multiplying by another matrix - the textbook recipe for vanishing gradients. With attention, any two positions are connected **directly** through one computation: the distance is always 1.

**Parallelism.** A recurrent network has to finish step $t$ before it can compute step $t+1$. Attention computes $QK^\top$ for **all** pairs of positions in a single matrix multiplication. This is the real reason the transformer won: not that it is smarter, but that it **can be trained on parallel hardware**, so people could train much larger models on much more data.

**The cost.** The matrix $QK^\top$ has size $n \times n$ where $n$ is the sequence length, so compute and memory grow **quadratically** with length. Doubling the context length quadruples the cost. This is why context windows used to be small numbers, and it is a problem a great deal of research tries to work around - but that is a topic for the AI Systems chapter.

> 🔧 **Try it now:** a model has a context of 4,096 tokens. You want to raise it to 16,384. By how many times does the cost of the attention part grow?
> **16 times**, not 4. The number of position pairs is $n^2$, so increasing $n$ by 4 times increases the number of pairs by $4^2 = 16$ times, which drives both the time and the memory for the weight matrix. This is why long-context models are not simply "the same architecture with a bigger number" - they need specific techniques to get around that quadratic wall.

## Lesson summary

- Attention projects each input vector into three roles: **query** (what it needs), **key** (what it has), **value** (the content), then takes a weighted average of the values.
- Dividing by $\sqrt{d}$ keeps the dot product from growing with the dimension, avoiding softmax saturation and vanishing gradients.
- The **causal mask** stops each position from looking **ahead** - position $i$ may only look at positions $\le i$; without it the model peeks at the answer.
- **Multiple heads** let the model learn many kinds of relationships in parallel, and they really are different: in the same layer of Qwen2.5-0.5B, one head puts **99.19%** on a single token while three others spread their weight out.
- **Do not read an attention matrix as an explanation.** On the test sentence, most of the weight goes to **the periods** and **the first token of the sentence**, not to words that carry meaning.
- **Attention sink**: the share of attention going to the first token reaches **91% at layer 16**, and 60-80% in most of the middle layers - a way for an attention head to "take nothing" when softmax forces the sum to be 1.
- Attention beat recurrent networks thanks to **paths that are always length 1** and **being parallelizable**, not by being smarter.
- The price is a cost that is **quadratic in sequence length**: doubling the context quadruples the cost.

## Self-check questions

1. Explain the roles of query, key and value using the dictionary lookup example. Where does attention differ from a real lookup?
2. Why must we divide by $\sqrt{d}$? What happens if we drop it?
3. What does the causal mask block, and why does training become meaningless without it?
4. In the same layer, one head puts 0.9919 on a single token while the other heads spread their weight out. Which design choice does that justify?
5. Qwen2.5-0.5B has 14 query heads but only 2 key-value sets. What is that technique called, and what does it save?
6. Most of the attention goes to the periods and the first token of the sentence. Why does that refute the reading "the model attends to X so it answers Y"?
7. What is an attention sink, and which explanation fits the fact that softmax forces the weights to sum to 1?
8. State two reasons attention could replace recurrent networks, and which one matters more in practice.
9. The context grows from 2,048 to 8,192 tokens. By how many times does the cost of the attention part grow? Explain.

## Further reading

**Foundational sources for this lesson:**

| Source | Which part to read |
|---|---|
| **Deep Learning · Lesson 10** | Why recurrent networks run out of breath, and attention introduced at the conceptual level |
| **Computer Vision · Lesson 10** | Attention applied to images: cutting patches, embedding positions |
| **Computer Vision · Lesson 04** | Grad-CAM and the occlusion test - the same discipline of "a visualization tool is not an explanation" |
| **Dive into Deep Learning (d2l.ai)** | Chapters 11.1-11.5 *Attention Mechanisms and Transformers* |

**Additional online sources (free):**

- [Vaswani et al. - Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762) - the original paper; section 3.2 is exactly the formula in section 2.
- [Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - the most widely cited visual explanation.
- [Jain & Wallace - Attention is not Explanation (NAACL 2019)](https://arxiv.org/abs/1902.10186) - the academic basis for the warning in section 4.
- [Xiao et al. - Efficient Streaming Language Models with Attention Sinks (ICLR 2024)](https://arxiv.org/abs/2309.17453) - names and analyzes the phenomenon measured in section 4.
- [Bertviz](https://github.com/jessevig/bertviz) - an interactive tool for viewing attention matrices, handy for trying it on your own sentences.

> **Next lesson:** [The Transformer from Scratch](../transformer-from-scratch/) - attention is the most important piece but not the only one. The next lesson assembles a complete Transformer - position information, normalization, the FFN block, skip connections - then **pulls out each brick** to measure how badly things break without it. The results include one very surprising finding.
