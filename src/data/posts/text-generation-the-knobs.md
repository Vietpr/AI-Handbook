---
title: "Text Generation: The Knobs"
description: "The decoding step between a probability distribution and actual text, what temperature, top-k and top-p do to that distribution, and choosing settings from measurements rather than feel."
domain: "Generative AI"
section: "Learn"
language: "en"
translationKey: "text-generation-the-knobs"
order: 5
pubDate: 2026-09-02
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will know that there is one more decision step between a probability distribution and actual text, understand exactly what temperature, top-k and top-p do to the distribution, and be able to choose a configuration that suits the task at hand - based on measurements, not feel.

## 1. The model does not generate text

Lesson 01 ended with an observation: the model returns a **probability distribution** over the entire vocabulary for the next token. That is all it does.

Getting from that distribution to one specific token is **a separate step, outside the model**. That step has a name: the **decoding strategy**. Same model, same prompt, change the decoding strategy and you get completely different text.

Lesson 01 also showed why this step matters unevenly: the distribution can be **very sharp** (0.9989 for a single character) or **very flat** (0.4663 for the top candidate). Where it is sharp, any way of choosing gives the same result. Where it is flat, the way you choose decides where the text goes.

This lesson uses Qwen2.5-0.5B-Instruct with the fixed prompt *"Viết một câu mở đầu cho bài giới thiệu về học máy."* ("Write an opening sentence for an introduction to machine learning."), and each configuration is run with **three different seeds**.

## 2. Greedy: always take the highest

The simplest approach: at each step, take the token with the highest probability.

```python
out = model.generate(**enc, max_new_tokens=45, do_sample=False)
```

Results with three different seeds (roughly: "To understand machine learning, we need to understand that it is not just a tool..."):

```
seed 0: "Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó không chỉ là một công cụ…"
seed 1: "Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó không chỉ là một công cụ…"
seed 2: "Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó không chỉ là một công cụ…"
```

The three outputs are **identical character for character**. That is the defining property of greedy decoding: it is **deterministic**. There is no random draw, so the seed has no effect.

That determinism is a real strength in many situations: data extraction, classification, translation, or anything where rerunning must give exactly the same result as before. Lesson 08 uses greedy decoding for all of its prompt experiments for precisely this reason.

The weakness: it picks the best token **at each step**, not the best sequence. A token with high probability right now can lead into a dead end. And for tasks that need variety - writing ad copy, generating ideas, producing several versions - a single answer is useless.

## 3. Temperature: reshaping the distribution before drawing

To get variety you have to **draw randomly** from the distribution instead of taking the maximum. **Temperature** is the knob that adjusts how sharp the distribution is before drawing:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T < 1$ makes the distribution **sharper** - strong candidates get stronger
- $T = 1$ keeps the distribution exactly as the model produced it
- $T > 1$ makes the distribution **flatter** - weak candidates are lifted up

Numbers say it more clearly than words. For a single generation step, measure **what share of the total probability the top five candidates hold**:

| Temperature | Probability of top candidate | Sum of top 5 candidates |
|---|---|---|
| 0.3 | 0.9971 | **0.9994** |
| 0.8 | 0.6311 | 0.7799 |
| 1.5 | 0.1142 | 0.2147 |
| **2.5** | **0.0056** | **0.0147** |

Read the last row carefully. At $T = 2.5$, **the five best candidates hold only 1.47% of the probability** - meaning **the remaining 98.5% sits in the long tail** of hundreds of thousands of tokens that the model itself considers nearly impossible. Draw from that distribution and you will almost certainly land in the tail.

And the output confirms exactly that:

| $T$ | Sample |
|---|---|
| 0.3 | *"Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó là một lĩnh vực nghiên cứu và ứng dụng tiên tiến…"* |
| 0.8 | *"Trong thế giới hiện đại, học máy đang trở thành một công cụ quan trọng để phát triển các ứng dụng…"* |
| 1.5 | *"Bóng đeo tay trong số tất cả những vật dụng quen thuộc của bạn không có khả năng cung cấp giá trị cao nhất…"* |
| **2.5** | *"Bật ngõ vào tương lai**ment** hiện hành: Mở ra một khía cạnh học trò mà có thể dạy người khác…"* |

At 0.3 the text is on topic ("...it is an advanced field of research and application") and nearly repeats the greedy version. At 0.8 it is still on topic ("In the modern world, machine learning is becoming an important tool...") but the opening is already different. At 1.5 the model **goes completely off topic** - "bóng đeo tay" (a "wearable ball") has nothing to do with machine learning. At 2.5 the language starts to break down: the word `laiment` is Vietnamese with an English suffix stuck on, a clear sign of drawing a token the model would almost never choose.

## 4. Top-k and top-p: cutting the tail before drawing

The table in section 3 points to the real problem with high temperature: **the tail**. Temperature lifts every candidate evenly, including tokens that do not fit at all.

Two techniques solve this by **removing the tail before drawing**:

**Top-k** - keep only the $k$ highest candidates, discard the rest, renormalize and draw. Simple, but a fixed $k$ does not suit every situation: where the distribution is sharp (Lesson 01: 0.9989 for a single character), $k=10$ still gives 9 absurd candidates a chance; where the model is genuinely undecided, $k=10$ may cut off reasonable candidates.

**Top-p** (nucleus sampling) - keep the smallest group of candidates whose **total probability reaches $p$**, discard the rest. That group **adapts on its own**: when the distribution is sharp it contains only one or two tokens, when the distribution is flat it widens. This is more sensible than top-k, and it is often enabled by default - but **the default value differs from place to place**, so do not assume it.

Output on the same prompt:

| Configuration | Sample |
|---|---|
| top-k 10 | *"Trong thế giới hiện đại, học máy đang trở thành một công cụ quan trọng để phát triển…"* |
| top-p 0.5 | *"Trong thế giới hiện đại, học máy không chỉ là một công cụ giúp chúng ta xử lý thông tin…"* |
| top-p 0.9 | *"Bật tinh thần học máy - khởi chạy hệ thống tích lũy thông tin, đưa ra khả năng xử lý tương lai"* |

The first two configurations give readable, on-topic text. Top-p 0.9 at temperature 1.0 has already started to get strange - *"Bật tinh thần học máy"* (literally "switch on the machine learning spirit") is a phrase nobody would write. It shows that top-p is not magic: keeping 90% of the probability in a distribution that is already flat still leaves many bad candidates.

**Seven of the eight sampling configurations gave three different outputs for the three seeds.** Only greedy gave three identical ones. That is the central trade-off of this lesson: **deterministic or diverse - pick one.**

> ⚠️ **A small note:** temperature, top-k and top-p **can be applied at the same time**, and a library's defaults are often not what you think. In `transformers`, if you set `temperature` but forget `do_sample=True`, the parameter is **ignored entirely** and you are still running greedy. Many APIs also preset a `top_p` other than 1 even if you do not declare one. Before concluding that "temperature has no effect", check whether the configuration you think you are running is actually the one running - this is a silent bug of the same family as the image preprocessing issue in Computer Vision · Lesson 01.

## 5. Choose the configuration by task, not by habit

And what "default" means in practice is worth checking at the source. The very Qwen2.5-0.5B-Instruct checkpoint used throughout this chapter ships with a `generation_config`:

```
do_sample = True · temperature = 0.7 · top_p = 0.8 · top_k = 20
```

So if you call `generate()` without declaring anything, you are **not** running greedy and you are **not** running top-p 0.9 either - you are running exactly the set of values above. That is why every experiment in sections 2 and 3 passes its parameters explicitly.

The whole lesson condensed into a table of **starting points** - not a universal standard, because the sensible numbers depend on the model, the task and how you grade:

| Task | Configuration | Reason |
|---|---|---|
| Extraction, classification, tool calling | **greedy** | Reruns must give exactly the same result; variety is harmful |
| Document question answering, summarization | greedy or low $T$ (0.2-0.4) | Staying close to the source matters more than nice wording |
| Conversational assistant | $T \approx 0.7$-0.8 with top-p 0.9 | Balance between natural and stable |
| Creative writing, generating many options | higher $T$, top-p 0.9-0.95 | Variety is the goal |
| Need many options, then choose | draw many samples, then **filter with your own criteria** | Do not count on a single draw hitting the mark |

The last row deserves a comment. When you need high quality, a more effective approach than tuning temperature is to **generate many samples and choose** using a criterion outside the model - format checks, running the code, checking against constraints. Computer Vision · Lesson 12 used exactly this approach: the OCR model misread, but three arithmetic constraints on the invoice caught the error.

> 🔧 **Try it now:** your system extracts dates from text and runs at `temperature=0.7`. A colleague reports that the same document gave two different results on two runs. How do you fix it, and why is this a design bug rather than a model bug?
> Switch to `do_sample=False`. Extracting a date is a task with **exactly one correct answer**, so variety is pure harm - you are paying for a feature that breaks your own task. It is a design bug because the model is doing exactly what it was told: `temperature=0.7` is the instruction "draw randomly from the reshaped distribution", and it complies. After switching to greedy, the output becomes deterministic - and you should also add a format check with a regular expression, because greedy only guarantees *repeatable*, not *correct*.

## Lesson summary

- The model only returns a **probability distribution**; which token to pick is **a separate step outside the model**, called the decoding strategy.
- **Greedy is deterministic**: three different seeds give three outputs **identical character for character**. It suits extraction, classification and tool calling.
- **Temperature** divides the logits before the softmax. Measured on one generation step: the top five candidates hold **99.94%** of the probability at $T=0.3$ but only **1.47%** at $T=2.5$ - that is, 98.5% piles into the tail.
- The output matches those numbers exactly: $T=1.5$ drifts off topic to "bóng đeo tay" (a "wearable ball"), $T=2.5$ breaks the language into `laiment`.
- **Top-k** keeps a fixed $k$ candidates; **top-p** keeps the smallest group whose total probability reaches $p$, so it **adapts to how sharp the distribution is** - more sensible than top-k.
- **Every checkpoint has its own defaults**: Qwen2.5-0.5B-Instruct ships with `temperature=0.7`, `top_p=0.8`, `top_k=20`, `do_sample=True`. Calling `generate()` with no parameters runs exactly that set, not greedy.
- Top-p is not magic: top-p 0.9 at temperature 1.0 still produces *"Bật tinh thần học máy"*.
- **Seven of the eight sampling configurations gave three different outputs**; only greedy was repeatable. Deterministic or diverse - pick one.
- A common trap: set `temperature` but forget `do_sample=True`, and the parameter is **ignored** while you are still running greedy.
- When you need high quality, **generating many samples and filtering with criteria outside the model** is more effective than tuning temperature.

## Self-check questions

1. Why do we say "the model does not generate text"? Which step actually produces the token?
2. Greedy gives three identical outputs for three seeds. Explain why, and name two tasks where that property is what you want.
3. At $T = 2.5$, the top five candidates hold only 1.47% of the probability. What does that number predict about the output, and does the actual output match?
4. Compare top-k with top-p. Why is top-p more sensible when the sharpness of the distribution changes with context?
5. Lesson 01 measured a distribution of 0.9989 in one place and 0.4663 in another. How does that affect what the knobs do?
6. You set `temperature=1.2` but the output is still exactly the same on every run. What is the most likely cause?
7. For a document question answering system, why should you use a low temperature rather than a high one?
8. Why is "generate many samples, then filter" usually more effective than tuning temperature when you need quality? Give an example of a filtering criterion for a specific task.

## Further reading

**Foundational sources for this lesson:**

| Source | Which part to read |
|---|---|
| **Lesson 01 of this chapter** | The next-token distribution, and how it is sharp or flat depending on context |
| **Machine Learning · Lesson 03** | Softmax and how temperature reshapes it |
| **Computer Vision · Lesson 12** | Using constraints outside the model to catch output errors |

**Additional online sources (free):**

- [Holtzman et al. - The Curious Case of Neural Text Degeneration (ICLR 2020)](https://arxiv.org/abs/1904.09751) - the paper that introduced top-p, and explains why taking the maximum produces repetitive text.
- [Generation strategies - Hugging Face documentation](https://huggingface.co/docs/transformers/generation_strategies) - every parameter of `generate`, including the `do_sample` trap.
- [Fan, Lewis, Dauphin - Hierarchical Neural Story Generation (ACL 2018)](https://arxiv.org/abs/1805.04833) - the source of top-k sampling.
- [How to generate text - Hugging Face blog](https://huggingface.co/blog/how-to-generate) - compares greedy, beam search, top-k and top-p with runnable examples.

> **Next lesson:** [Model Scale: What We Know and What We Don't](../model-scale-what-we-know-and-what-we-dont/) - the first five lessons all used a 0.5-billion-parameter model, and it did reasonably well. The next lesson measures **where it breaks** - and then tackles a hard question head on: from observing a small model, what can we infer about large models, and what **can't** we?
