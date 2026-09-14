---
title: "Continuous Batching and Paged KV Cache"
description: "Separating the two ideas usually lumped together as \"why vLLM is fast\", designing a comparison that does not credit the whole gap to one idea, and why short requests end up waiting for long ones."
domain: "AI Systems"
section: "Learn"
language: "en"
translationKey: "continuous-batching-and-paged-kv-memory"
order: 6
pubDate: 2026-09-08
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will be able to tell apart the two ideas usually lumped together as "why vLLM is fast" - **continuous batching** and **paged KV cache** - design a comparison that does not credit the whole gap to a single idea, and see in numbers why someone asking a short question ends up waiting for someone asking a long one.

Lesson 05 ended with an uncomfortable number: in that lesson's static batch - which returns results for the whole batch at once - short requests were slowed **7 times** because they had to wait for the longest request in the same batch, while their slots sat idle until the batch ended. The fix sounds obvious: **whoever finishes leaves, and a free slot gets a new request right away**.

This lesson measures that fix through vLLM - and watches out for a trap: vLLM differs from the hand-written static batch in **many** places, not only in how it schedules.

## 1. Two ideas, solving two different problems

**Continuous batching** (also called per-step scheduling) solves a **time** problem. Instead of deciding the batch's membership once and running to the end, the scheduler decides again **after every decode step**: requests that just finished are pulled out, and waiting requests are placed into the free slots.

```
Static batching - 4-slot batch, 6 requests of different lengths (each cell is a decode step):

slot 1 A A A A A A A A          E E E E
slot 2 B B . . . . . .          F F F F F F
slot 3 C C C . . . . .          (empty)
slot 4 D D D D . . . .          (empty)
       └── batch 1: wait for A ┘ └ batch 2 ┘       "." = idle cell

Continuous batching - the same 6 requests:

slot 1 A A A A A A A A
slot 2 B B E E E E
slot 3 C C C F F F F F F
slot 4 D D D D
       E enters as soon as B finishes, F as soon as C finishes - no cell runs idle
```

**Paged KV cache** (the idea behind PagedAttention) solves a **memory** problem. The naive approach reserves for each request a contiguous KV cache region large enough for the **maximum** length it could reach. If a request uses only 300 tokens but gets space for 2,048 tokens, 85% of that region sits empty. Paging splits the KV cache into **small fixed-size blocks**, allocated gradually as the request actually grows - just like an operating system allocates memory by pages.

The two ideas complement each other - continuous scheduling needs flexible memory to fit new requests into free slots - but they **are two separate things**, and a system can have one without the other.

## 2. Designing the comparison so you do not fool yourself

The easiest comparison is: take the hand-written static batch from lesson 05, take vLLM, run the same mixed-length load, and credit the gap to continuous batching.

That is wrong. vLLM differs from the hand-written static batch in a whole range of places: more optimized kernels, pre-captured computational graphs (*CUDA graphs*) for common sizes, no return to Python after each step, chunked prefill, plus continuous scheduling and paged memory. A single comparison only tells you how much better **the whole machinery** is, not which part comes from where.

So the experiment has **four cells**:

| | Load A: all requests **the same length** | Load B: **mixed** lengths |
|---|---|---|
| **Hand-written static batch, 16 slots** | cell 1 | cell 2 |
| **vLLM, limited to exactly 16 slots** | cell 3 | cell 4 |

Both loads have **64 requests**, 128-token prompts, and **the same total of 11,776 tokens to generate**: load A is 64 requests × 184 tokens, load B is 16 requests each of 32, 64, 128 and 512 tokens. Model and card as in lesson 05: Qwen2.5-1.5B-Instruct, bfloat16, RTX 3060.

The logic of the separation is as follows. In **load A**, there is no "slowest member" - all requests are equally long, so continuous scheduling has no free slots to fill. The gap between cell 3 and cell 1 is therefore the distance between the two machines **when there is no free slot to reuse yet** - including kernels, prebuilt graphs, and also the parts of the scheduler that still do work when all requests are equally long, such as how requests are admitted or how prefill is chunked. In **load B**, mixed lengths create free slots. Dividing the ratio in load B by the ratio in load A gives **the extra gain tied to lengths becoming mixed** - exactly the part that continuous batching is designed to win.

That is an estimate, not a clean separation: the parts of a machine do not multiply together that neatly. But it is far better than crediting everything to one idea.

## 3. Throughput

| | Load A - same length | Load B - mixed | Loss when mixed |
|---|---|---|---|
| **Static batch, 16 slots** | 861.2 tok/s | 310.9 tok/s | **−64%** |
| **vLLM, 16 slots** | 1,262.5 tok/s | 926.8 tok/s | −27% |
| vLLM, 256 slots | 3,707.1 tok/s | 1,731.9 tok/s | −53% |

The static batch loses **64%** of its throughput when the load is mixed - consistent with the calculation: each batch of 16 requests contains all four length types, so it has to run 512 steps, while on average only 184 are needed, meaning **64.1%** of cells run idle.

Now the separation:

| | Ratio vLLM 16 slots ÷ static batch |
|---|---|
| Load A - when there are **no** free slots to reuse yet | 1,262.5 ÷ 861.2 = **1.47 times** |
| Load B - **the whole machinery** | 926.8 ÷ 310.9 = **2.98 times** |
| Extra gain **tied to** mixed lengths | 2.98 ÷ 1.47 ≈ **2.0 times** |

Read it correctly: at **the same 16 slots**, the vLLM machinery is already about 1.5 times faster than the hand-written batch **when there are no free slots to reuse**. When lengths are mixed, the gap widens by roughly **a factor of two** - consistent with continuous batching reusing free slots, but this division is not an exact causal separation. If you ran only cells 2 and 4 and wrote "continuous batching increased throughput nearly 3 times", a third of that sentence would belong to other things.

The third row shows the effect of **adding slots**: going from 16 to 256 slots multiplies load A's throughput by 2.94 - exactly the lesson of lesson 05, a larger batch means each read of the weights serves more people.

## 4. Per-request latency

Throughput is a system number. Users feel **their own** latency. Measuring load B again, this time recording when each request got its first token and when it finished; all 64 requests arrive at $t = 0$:

| Request length | Static batch, 16 slots | vLLM, 16 slots | vLLM, 256 slots |
|---|---|---|---|
| | TTFT / **E2E** (median, seconds) | TTFT / **E2E** | TTFT / **E2E** |
| 32 tokens | 16.38 / **27.83** | 1.88 / **2.26** | 0.37 / **0.83** |
| 64 tokens | 16.38 / **27.83** | 2.10 / **2.91** | 0.37 / **1.32** |
| 128 tokens | 16.38 / **27.83** | 2.10 / **3.72** | 0.37 / **2.27** |
| 512 tokens | 16.38 / **27.83** | 2.10 / **8.51** | 0.37 / **7.08** |

The static batch column has **the same number on every row**. It is not a copying error: this lesson's hand-written static batch returns results when **the whole batch** is done, and every batch contains a 512-token request. Someone asking for 32 tokens waits **27.8 seconds** - exactly the same as someone asking for 512 tokens.

The two vLLM columns are entirely different: **each person's latency is proportional to their own length**. A 32-token request finishes after **2.26 seconds** with 16 slots and **0.83 seconds** with 256 slots - **12 times** and **34 times** faster than the static batch.

> ⚠️ **Two tables, two runs - do not mix the numbers.** The table in section 3 and this table come from **two separate runs**. Between the two runs, the static batch's throughput on load B differs by about **17%** (310.9 versus 257.3 tok/s), while vLLM's differs by only 3-4%. The static batch returns to Python after every decode step, so it is sensitive to the shared server having another process occupying nearly all the CPU (lesson 05 ran into exactly this). That is why the separation in section 3 uses numbers from **a single** run only, and this table is used only to read the **shape** of latency by length.

## 5. Why 256 slots "lose" more than 16 slots on a mixed load

The last row of the table in section 3 has something odd: vLLM with 256 slots loses **53%** of its throughput on a mixed load, much more than the 27% of 16 slots. More slots but worse?

The 256-slot column in section 4 explains it. With 256 slots, all 64 requests are admitted **right from the start**. The short requests finish quickly, and then… there is nobody left to put into the free slots. For the last 384 steps - **3/4 of the decode steps** of the whole load - only the 16 requests of 512 tokens are running, so the system falls back to **batch 16**. The whole load finishes after **7.08 seconds**, and those 7.08 seconds are exactly the time a 512-token request needs to go all the way on its own.

This is a limit no scheduler can overcome: **a finite block of requests arriving at the same time cannot finish earlier than its longest request.** Continuous batching can only fill free slots when **someone is still waiting** to fill them. With 16 slots and 64 requests, someone is always waiting, so the benefit shows in full; with 256 slots the queue runs dry right at the start.

Real users do not arrive as one block and stop - they arrive continuously, as lesson 07 will simulate. Under continuous load near high utilization, new requests keep coming, so a slot that just freed up usually has a request waiting to fill it; under low load free slots simply stay free, and an idle GPU is normal. But this measurement is an important reminder about how to **read benchmarks**: a block of requests submitted at once measures **the time to finish the whole block**, and that time is bounded below by the longest request, no matter how good the machinery is.

## 6. Paged KV cache

This lesson **cannot measure separately** the effect of paged memory: vLLM has no switch to turn it off, so there is no control cell. This section is **arithmetic** from the numbers vLLM reports about itself - the third source that lesson 02 said to read separately.

With Qwen2.5-1.5B and a memory utilization of 0.8, vLLM reports room for **219,328 tokens** of KV cache, and computes on its own "maximum **107.09** concurrent requests if each request uses the full 2,048 tokens".

That figure of 107 is exactly the capacity of **reserving by maximum length**: if each request has space held for 2,048 tokens regardless of how much it uses, then 219,328 ÷ 2,048 ≈ 107 requests, no more. In this lesson's load, however, a request averages $128 + 184 = 312$ tokens. If memory were allocated only for the part actually used - in small blocks - the same space would hold about 219,328 ÷ 312 ≈ **700** requests, roughly **6.5 times** as many.

That is a **theoretical comparison against reserving by maximum length**, computed on one specific load - not the benefit of paged memory in general. A non-paged allocator can still allocate dynamically by actual length, and then the gap is much smaller. What paged memory neatly solves is **allocating in blocks as needed without requiring a contiguous region**, so it avoids fragmentation when requests of different lengths keep coming and going. Every KV cache slot held but unused is a user not being served - but this lesson cannot measure how much of that vLLM actually recovers.

The formula from lesson 02 also checks the self-reported figure: 219,328 tokens × 28 KiB/token ≈ **5.86 GiB**, matching what remains after subtracting the 2.89 GiB of weights and the overhead items from the budget of 0.8 × 11.63 GiB.

> 🔧 **Try it now:** a blog post says "switching to vLLM gave us 3 times the throughput thanks to continuous batching". How do you read that sentence?
> Ask two questions. **One:** did they run a load where all requests have the same length? If not, the factor of 3 includes the parts coming from kernels, prebuilt computational graphs, and not returning to Python - in this lesson, with no free slots to reuse, that gap was already 1.5 times. **Two:** did they keep the number of slots the same? Adding slots by itself multiplied throughput by nearly 3 on a same-length load. Their number may be correct; the sentence crediting it to **one** idea is almost certainly wrong.

## Lesson summary

- **Continuous batching** solves the time problem: the scheduler decides the batch's membership again **after every decode step**, pulling out finished requests and putting waiting ones in. **Paged KV cache** solves the memory problem: it allocates the KV cache in small blocks as needed, instead of reserving for the maximum length. The two ideas complement each other but are separate things.
- Comparing vLLM with a hand-written static batch on a single load does **not** isolate the effect of scheduling, because vLLM also differs in kernels, prebuilt graphs, and not returning to Python.
- The four-cell design: {static batch, vLLM with the same 16 slots} × {same length, mixed}, with the same 11,776 tokens to generate. Same-length load - with no free slots to reuse yet: vLLM is **1.47 times** faster. Mixed load: **2.98 times**. The extra gain tied to mixed lengths: **about 2 times** - consistent with continuous batching, not an exact causal separation.
- The static batch loses **64%** of its throughput on the mixed load, matching the 64.1% idle cells computed in advance; vLLM with the same 16 slots loses only 27%.
- Per-request latency: in this lesson's static batch - which returns the whole batch at the end - **every** request, including 32-token ones, waits **27.8 s**. In vLLM, latency is proportional to each request's own length: a 32-token request finishes after **2.26 s** (16 slots) and **0.83 s** (256 slots).
- A finite block of requests arriving at once **cannot finish earlier than its longest request** - the reason 256 slots "lose" 53% on the mixed load. Continuous batching only fills free slots while someone is still waiting.
- Paged memory cannot be measured separately (there is no off switch). Arithmetic from vLLM's self-reported capacity: reserving for 2,048 tokens holds only **107** requests; allocating just what is used on this load holds about **700** - roughly 6.5 times, but that is **a comparison against reserving by maximum length**, not the benefit of paged memory in general; a dynamic non-paged allocator can also narrow that gap.
- The two tables in the lesson are two runs; the static batch differs by 17% between them because it is sensitive to a busy CPU. Do not mix numbers across runs.

## Self-check questions

1. Distinguish continuous batching from paged KV cache: what problem does each solve, and can a system have one without the other?
2. Why is it wrong to compare vLLM with a hand-written static batch on a mixed load and credit the whole gap to continuous batching?
3. Explain the logic of the four-cell design. Which benefit of continuous batching does a **same-length** load remove? Why does it still **not** cleanly isolate everything outside scheduling?
4. The division 2.98 ÷ 1.47 gives an **estimate**, not an exact separation. Why?
5. In the static batch, a 32-token request and a 512-token request have the same E2E. Explain this using the static batch's result-return mechanism.
6. vLLM with 256 slots loses more throughput than with 16 slots on a mixed load. Explain, and say what would change if requests arrived continuously instead of as one block.
7. A model has a KV cache that holds 100,000 tokens, a maximum length of 4,096, and an actual average length of 500. Compute the capacity when reserving by maximum length and when allocating just what is used. Why is the second number a ceiling rather than a number actually achieved?
8. You run the same benchmark twice and see the static baseline differ by 17% while vLLM differs by only 3%. What does that tell you about how you should present the results?

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Lesson 05 of this chapter** | The static batch, the slowest member, and why a large batch raises throughput |
| **Lesson 02 of this chapter** | The KV cache formula, and why the engine's self-reported numbers must be read separately |

**Going further:**

- [Yu et al. - Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) - the paper proposing per-step scheduling of decode.
- [Kwon et al. - Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180) - paged KV cache, and the measurement of reservation waste that section 6 could only compute by arithmetic.
- [Agrawal et al. - Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (OSDI 2024)](https://arxiv.org/abs/2403.02310) - chunked prefill, one of the things "outside scheduling" mentioned in section 2.

**Source code for this lesson:** [`code/b06_tinh.py`](../../code/ai-systems/b06_tinh.py) and [`code/b06_vllm.py`](../../code/ai-systems/b06_vllm.py) - the four throughput cells; [`code/b06_tre.py`](../../code/ai-systems/b06_tre.py) - per-request latency by length.

> **Next lesson:** [Measuring Serving Load Correctly](../measuring-serving-load-correctly/) - every measurement so far has submitted a whole block of requests at once - and section 5 just showed that this approach has its own limit. The next lesson has simulated users arrive spread out like real people, and shows that the most common way of simulating load reports latency nearly eight times better than reality.
