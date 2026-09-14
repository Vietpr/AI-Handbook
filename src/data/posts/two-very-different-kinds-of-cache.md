---
title: "Two Very Different Kinds of Cache"
description: "A prefix cache reuses computation and does not change the answer, a semantic cache reuses old answers and can be wrong; measuring the first's benefit and why the second cannot be made safe with a cosine threshold alone."
domain: "AI Systems"
section: "Learn"
language: "en"
translationKey: "two-very-different-kinds-of-cache"
order: 9
pubDate: 2026-09-10
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will be able to tell apart two things that are both called a "cache" in language model systems - one reuses an **equivalent computation** of the beginning of the prompt and in principle does not change the answer, the other reuses **an old answer** and can return a wrong one - measure the benefit of the first, and see in numbers why the second cannot be made safe with a cosine threshold alone.

Lesson 08 finished building the server. The natural next question: can we cut down its work? Many requests share parts - the same long system prompt, the same question asked again. Recomputing from scratch every time sounds wasteful.

"Add a cache" is the familiar answer. But in language model systems that phrase refers to **two completely different things**, and confusing them means sliding from "performance optimization" to "returning wrong answers".

## 1. Two things with the same name

| | **Prefix cache** | **Semantic cache** |
|---|---|---|
| What it reuses | The **KV computation** of an identical run of leading tokens | The **answer** to an old question |
| Hit condition | The leading token sequence **matches exactly**, token by token | The new question is **close in meaning** to an old one, by cosine |
| Does the model run | **Yes** - it only skips the part of prompt prefill already computed | **No** - it returns the old answer directly |
| Can it change the answer | In principle no - the computation is equivalent; the sequential test in section 3 gave 8/8 identical | **Yes** - that is exactly what it does |
| What it saves | The prompt prefill phase, i.e. TTFT | The entire model call |

Lesson 02 laid the foundation for the first column: the key and value of a token depend only on the tokens before it. If two requests share the same first 2,800 tokens, then the KV of those 2,800 tokens is **identical** - computing it once and using it for both is entirely exact.

The second column has no such mathematical foundation. It rests on an assumption: questions close in meaning have the same answer. Section 4 measures how well that assumption holds up.

## 2. Prefix cache: measuring

A very common situation: an internal assistant has a **long system prompt** - here a 40-article set of regulations, **2,811 tokens** - and every user question comes after exactly that system prompt. Each full prompt is **2,843 tokens** long, of which only the last 32 or so tokens are the question, differing between requests.

We measure the **prompt prefill time** of 24 requests sent **sequentially** (each request generates only 1 token, so that time is exactly the prefill phase), on Qwen2.5-7B-Instruct-AWQ, one RTX 3060, with vLLM's prefix cache turned off and then on:

| | First request | Next 23 requests (median) | Range |
|---|---|---|---|
| Prefix cache **off** | 1.535 s | 1.548 s | 1.539 - 1.563 s |
| Prefix cache **on** | **1.544 s** | **0.0351 s** | 0.033 - 0.055 s |

With it off, **every** request pays the full 1.5 seconds to prefill 2,843 tokens again - including the 2,811 system prompt tokens it prefilled in exactly the same way a second earlier.

With it on, the first request still pays in full - there is nothing in the cache to use. From the second request on, the system prompt part is taken from the cache, and prefill time drops to **0.035 seconds**: **44 times faster**. The engine reports it itself: out of **73,965** tokens to prefill, **70,432** tokens hit the cache - **95.2%**.

The prefix cache only cuts the **prompt prefill phase**. The token generation phase costs exactly as much as before, because lesson 01 showed that every generated token must run through the whole model no matter where its context came from. So the benefit is largest when the prompt is long and the answer short; when the answer runs to hundreds of tokens, the 1.5 seconds saved is only a small part of E2E.

> ⚠️ **My first measurement spoiled itself.** In the first run, the "first request" with the cache on also took only 0.035 seconds - because the script's warm-up step used **the same prompt-building function**, and so the same system prompt, which loaded it into the cache before measuring. The measurement in the table above warms up with a prompt that does **not** contain the system prompt. The general lesson for any cache measurement: **you must prove that the miss is a real miss**, otherwise you are measuring two hits and calling one of them a miss.

## 3. Does the prefix cache change the answer?

To state it precisely: the prefix cache **does not reuse answers**; it only reuses the **KV computation** of the prefix. With the same state and deterministic decoding, the result must be equivalent to recomputing.

The check: 8 questions, deterministic decoding (`temperature = 0`), each sent on its own, run once with the cache off and once with it on.

| | Result |
|---|---|
| Answers **identical character for character** between off and on | **8 / 8** |

And when **sampling** (`temperature = 0.8`), sending the same prompt three times:

| | Distinct versions in 3 calls |
|---|---|
| Cache off | 2 |
| Cache on | 2 |

Two calls with the same prompt still produce different sentences - **regardless of the cache**. The randomness comes from the sampling step, not from the cache. One more detail worth noticing: the three answers in the two runs **repeat exactly, in the same order**, because both runs initialize the engine with the same random seed. If you want two runs to produce different sentences, the seeds must differ.

One limitation of this check: 8 questions, sent **one at a time**. When many requests run together in a batch, the order in which floating-point numbers are added can change slightly, and where two tokens score almost equally, that small difference can change the chosen token. This lesson does not measure that case - lesson 12 does, and batched execution changed 3 out of 32 answers - so 8/8 must not be read as "never different".

## 4. Semantic cache: measuring

Now the other one. The idea: store the question-answer pairs we already have. When a new question arrives, encode it with an embedding, find the nearest old question, and if the cosine exceeds a threshold, **return the old answer directly** without calling the model.

To measure it, I built 10 base questions - the kind of question a business assistant often gets - and for each one:

- **2 paraphrases**: the same question, different words. The cache **must** hit.
- **2 near-duplicates**: almost the same words, but with **one key condition changed** - adult to child, over 500 million to under 500 million, this year to last year. If the cache hits, it **returns the wrong answer**.
- Plus 5 completely off-topic questions.

The embedding model is `multilingual-e5-small`, the same model as in Generative AI · Lesson 10.

| Question type | Cosine with the base question |
|---|---|
| Paraphrase - must hit | **0.9221 - 0.9894** |
| Near-duplicate, different condition - a hit is wrong | **0.8855 - 0.9956** |
| Off-topic - highest against any base question | 0.7511 - 0.8423 |

The first two ranges **overlap almost completely**. The highest-scoring near-duplicates:

| Cosine | Near-duplicate | Base question |
|---|---|---|
| **0.9956** | Chuyển khoản **dưới** 500 triệu có cần xác thực thêm không? ("Do transfers *under* 500 million need extra verification?") | … **trên** ("over") 500 triệu … |
| **0.9951** | Hạn chót nộp báo cáo thuế **năm ngoái** là ngày nào? ("When was the tax filing deadline *last year*?") | … **năm nay** ("this year") … |
| **0.9910** | Văn phòng Hà Nội **đóng** cửa lúc mấy giờ? ("What time does the Hà Nội office *close*?") | … **mở** ("open") cửa … |

And the **lowest**-scoring paraphrase is *"Vé người lớn giá bao nhiêu?"* ("How much is an adult ticket?") - **0.9221** against the base question *"Giá vé vào cổng cho người lớn là bao nhiêu?"* ("What is the admission price for adults?").

In other words: a question with the **opposite meaning** - "under" in place of "over" - is judged by the embedding model to be **closer to the base question** than a paraphrase with the correct meaning. And all 20 near-duplicates have their base question as their nearest neighbor in the cache.

Sweeping the threshold:

| Cosine threshold | Paraphrases hit correctly | Near-duplicates given the **wrong answer** | Off-topic questions hit |
|---|---|---|---|
| 0.80 | 100% | **100%** | 40% |
| 0.85 | 100% | **100%** | 0% |
| 0.90 | 100% | **95%** | 0% |
| 0.92 | 100% | **90%** | 0% |
| 0.94 | 90% | **65%** | 0% |
| 0.96 | 55% | **35%** | 0% |

**On this test set, no threshold cleanly separates the two groups.** At 0.92, every paraphrase hits - but **9 out of 10** changed-condition questions hit too, meaning they receive the answer to a different question. Raise it to 0.96 and you already miss nearly half the paraphrases while still answering **a third** wrongly. Even at 0.98, the three changed-condition questions at the top of the table above still pass the threshold.

Generative AI · Lesson 10 showed that a cosine threshold does not carry over from one dataset to another. Here the situation is worse: on **the same set**, the questions that should be rejected are **closer** than the questions that should be accepted. No threshold can separate two groups when they overlap in the wrong direction.

> ⚠️ **The test set is small, and deliberately hard.** 10 base questions, 20 paraphrases, 20 near-duplicates - written by me, and I wrote the near-duplicates **so that** they would be hard. This table does not measure the wrong-answer rate on the real traffic of any system. It proves something narrower and firmer: there **exist** questions with a different meaning that cosine scores higher than questions with the same meaning, and they are not rare - all it takes is changing one word "trên" (over) into "dưới" (under).

## 5. Why the embedding does not see the word "under"

A sentence embedding compresses the whole sentence into one fixed vector. It is good at capturing the **topic** and **shape** of the sentence: this question is about transfers, about a money threshold, about verification. Changing "over" into "under" changes only **one** token out of about fifteen; the topic, the structure and nearly all the vocabulary stay the same, so the vector barely moves.

But for the person asking, that word "under" is **the whole** question. This is the fundamental mismatch between what an embedding measures - *are the two sentences about the same thing* - and what an answer cache needs - *do the two sentences have the same answer*. Two questions can be about the same thing and have opposite answers.

Generative AI · Lesson 09 has an example from the same family: the model returns the correct **current** figure for a question about **2019**. Here it is a cache returning the answer for **this year** to a question about **last year**, with cosine 0.9951. A time-reference error does not need the model to make anything up; a cache is enough.

## 6. The cache key is not just cosine

If you still want to use an answer cache, the design question is not "what threshold", but **"in what respects must two requests be identical for the answer to be shared"**. An answer is generated under many conditions, and each of those conditions must be part of the key:

| Key component | Why |
|---|---|
| Model version | Change the model and the answer changes |
| System prompt version | Change the rules and the answer changes |
| Document store version | When documents are updated, the old answer becomes wrong - lesson 10 |
| **The asker's permissions** | Person A may see documents that person B may not; sharing the answer is a **data leak** between the two |
| Time, if the answer depends on time | "This year", "today", "current price" expire |
| Tool state | The answer is based on the lookup results at that moment |

The fourth component is the most dangerous and the easiest to forget. An answer cache shared by everyone, keyed only on the question text, will hand a person without permission exactly the answer that a person with permission just received.

So when is an answer cache acceptable? When the answer **has been reviewed by a person** and **does not depend** on the asker, the time or private documents - a set of frequently asked questions, for example. And even then, start with **exact match** after normalizing the text (removing extra whitespace, ignoring case) before thinking about cosine. Exact match misses many paraphrases, but it never returns the answer for "under 500 million" to the question "over 500 million".

> 🔧 **Try it now:** your team wants to add a cache to cut model-call costs for a bank's customer service assistant. What do you propose?
> Split it into two tiers. **Tier one**, turn on the prefix cache - it should almost always be on: it does not reuse answers, only computation, and the bank assistant's long system prompt is exactly the case that benefits most, as section 2 measured 44 times in the prefill phase. **Tier two**, the answer cache: only for questions that **do not depend on the asker** - opening hours, how to reset a password - with reviewed answers, a key that includes the document version and an expiry time, starting with exact match. **Do not** use it for anything involving accounts, limits or amounts: section 4 shows that "transfers under 500 million" and "over 500 million" are 0.0044 cosine apart.

## Lesson summary

- "Cache" in language model systems is a shared name for **two completely different things**: the prefix cache reuses the **KV computation** of an exactly matching run of leading tokens; the semantic cache reuses **an old answer** for a question close in meaning.
- **Prefix cache**, measured on a 2,811-token system prompt: prompt prefill goes **from 1.548 s down to 0.035 s** - **44 times** faster - for every request after the first; **95.2%** of tokens hit the cache. It only cuts the prompt prefill phase, not the token generation phase.
- A cache measurement must prove **the miss is a real miss**: my first measurement accidentally warmed the cache in the warm-up step.
- The prefix cache **does not reuse answers**, only the equivalent computation of the prefix: a sequential, deterministic-decoding test gave **8/8** identical answers with the cache off and on. Sampling still produces different sentences between calls - regardless of the cache. The check sent one question at a time; it must not be read as "never different" when running in batches.
- **Semantic cache**: paraphrases have cosine **0.9221 - 0.9894**, questions with a key condition changed have **0.8855 - 0.9956**. The opposite-meaning question "under 500 million" reaches **0.9956** - higher than every paraphrase.
- **On this deliberately hard test set, no cosine threshold cleanly separates** paraphrases from changed-condition questions: at 0.92 all paraphrases hit but **90%** of changed-condition questions get the wrong answer; at 0.96 nearly half the paraphrases are missed and **35%** still get the wrong answer.
- The reason: an embedding measures *whether two sentences are about the same thing*, while an answer cache needs *whether two sentences have the same answer*. Changing one word can change the answer while barely changing the vector.
- The key of an answer cache must include the model version, system prompt, document store, **the asker's permissions**, time and tool state. Leaving out the asker's permissions is a data leak. Use it only for reviewed answers that do not depend on the asker, and start with exact match.

## Self-check questions

1. Why is sharing the KV of an identical prefix **exact**, while sharing the answer between two questions close in meaning is not? Use what you learned in lesson 02.
2. The prefix cache cuts 1.5 seconds from prefill. For a 500-token answer at the speed in lesson 08, what fraction of E2E does that saving represent? When is a prefix cache most worthwhile?
3. The first measurement gave the "first request" only 0.035 seconds with the cache on. Explain why that is a sign of a broken measurement, and how to prove that a miss is a real miss.
4. Deterministic decoding gave 8/8 identical answers with the cache on and off. Why does the lesson still not allow the conclusion "the prefix cache never changes the answer"?
5. Why does the question "transfers **under** 500 million" have a higher cosine with the base question than a paraphrase with the correct meaning?
6. Can you choose a cosine threshold for the semantic cache on this lesson's test set? Argue using the threshold sweep table.
7. Why must "the asker's permissions" be part of the answer cache key? Describe a concrete data leak scenario if it is missing.
8. Design a caching strategy for an HR policy lookup assistant whose questions depend on the employee's rank. State clearly which tier uses which kind of cache, and what the key contains.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Lesson 02 of this chapter** | Why the KV of a prefix depends only on the tokens before it |
| **Lesson 01 of this chapter** | The prompt prefill phase and the token generation phase - the prefix cache only touches the first |
| **Generative AI · Lesson 10** | Sentence embeddings, and why a cosine threshold does not carry over from one dataset to another |
| **Generative AI · Lesson 09** | Time-reference errors - the same family as returning the "this year" answer to a "last year" question |

**Going further:**

- [vLLM - Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) - the mechanism that hashes each KV block to recognize a repeated prefix.
- [Zheng et al. - SGLang: Efficient Execution of Structured Language Model Programs (NeurIPS 2024)](https://arxiv.org/abs/2312.07104) - RadixAttention, organizing the prefix cache as a tree for many requests sharing different leading segments.
- [Reimers & Gurevych - Sentence-BERT (EMNLP 2019)](https://arxiv.org/abs/1908.10084) - the basis of sentence embeddings, and why they are trained to measure similarity of meaning, not of conditions.

**Source code for this lesson:** [`code/b09_tiento.py`](../../code/ai-systems/b09_tiento.py) - the prefix cache, on and off; [`code/b09_ngunghia.py`](../../code/ai-systems/b09_ngunghia.py) - the semantic cache test set and threshold sweep.

> **Next lesson:** [Vector Stores in Production](../vector-stores-in-production/) - the semantic cache just showed that embeddings can fool us at the level of single sentences. The next lesson looks at embeddings at the level of the whole store: when you need an approximate index, what adding and deleting documents does to the index, and what happens when you change the embedding model and forget a step.
