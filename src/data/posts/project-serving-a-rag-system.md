---
title: "End-to-End Project: Deploying a RAG System"
description: "Deploying a retrieval-augmented QA system for multiple users, measuring each stage, and reading results that include the metrics that got worse."
domain: "AI Systems"
section: "Learn"
language: "en"
translationKey: "project-serving-a-rag-system"
order: 12
pubDate: 2026-09-11
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will be able to take a retrieval-augmented question answering system from a sequential script to a service for many users, tell apart **a bundled platform switch** from pairs that **change one variable**, and read a results table that includes the cells that **got worse** - the thing a "before and after" table hides.

Generative AI · Lesson 12 built a question answering system over this very series: split into passages, encode, find the three nearest passages, and hand them to a 0.5B model to write an answer with citations. It runs on a personal computer and answers one question at a time.

This lesson takes exactly that system through what this chapter has measured - the quantized 7B model from lesson 04, the continuous batching from lesson 06, the prefix cache from lesson 09, the streaming from lesson 08, and a load limit derived from lesson 07. **A → B is a bundled platform switch; from B onward, the comparison pairs keep the same serving platform and change one component or load mode at a time to read the trade-offs.**

## 1. Load and metrics

**The load** consists of 32 questions:

- **20 labeled questions** from Generative AI · Lesson 12 - for each one we know in advance which article holds the answer.
- **8 new questions** about this chapter itself - the prompt prefill phase, the KV cache, static batching, closed-loop and open-loop load…
- **4 questions thought to be outside the store** - koi fish in a cement pond, a recipe for beef phở, today's gold price, the 2018 World Cup winner. The only correct answer is *"Không tìm thấy trong tài liệu."* ("Not found in the documents."). Section 2 will show that the koi question - even though the store still has **no** correct grounds to answer it - has **leaked into the store**: the question itself and a wrong answer are present in the documents. So the lesson reports two groups separately: **3 unanswerable questions, clean**, and **1 unanswerable question, hard, leaked**.

**The prompt** consists of a fixed system prompt - four rules plus two examples, one with citations and one refusal - then the three numbered retrieved passages and the question. The system prompt is **323 tokens** long; the whole prompt averages **1,127 tokens**.

**Speed metrics**, using exactly the names from lesson 01 and lesson 08: throughput, the moment the user **sees the first word**, and E2E, each at p50 and p95.

**Quality metrics**, scored automatically:

| Metric | On which question group |
|---|---|
| Has citations | 28 in-store questions |
| Valid citations - every source number is within [1], [2], [3] | 28 in-store questions |
| Correct refusal | 3 unanswerable questions, clean; the koi question - hard, leaked - reported separately, see section 2 |
| Wrong refusal - saying "not found" when the documents have it | 28 in-store questions |

These are **proxy metrics**. They catch the errors that Generative AI · Lesson 12 ran into - citing source numbers that do not exist, not knowing how to refuse - but they do **not** score whether an answer is correct. Section 5 will show that they also miss a kind of error nobody intended to check.

## 2. Stage 0: the retrieval layer, measured once for all stages

Following the principle from Generative AI · Lesson 12 - **measure the two layers separately** - the retrieval layer runs **once** before all stages, and every stage uses exactly the same three passages for each question. That way every difference between stages in section 3 comes from the generation layer, not mixed up with retrieval.

| | Value |
|---|---|
| Store | 1,953 passages - the whole series at run time |
| Encoding the whole store with e5-small on the GPU | **22.4 s** |
| One retrieval - encoding the question and a full scan - p50 / p95 | **4.07 / 21.48 ms** |
| Article hit@3 on 28 in-store questions | **26 / 28** |
| Mean cosine of the top passage - in-store questions | 0.8776 |
| Mean cosine of the top passage - out-of-store questions | **0.8374** |

**Retrieval is cheap.** Four milliseconds, compared with an E2E on the order of seconds in every stage below: under 1% of the time. Optimizing retrieval latency in this system is optimizing in the wrong place - just as lesson 10 measured, a full scan is enough at the scale of a few thousand passages.

**The retrieval score carries a signal, but not enough to stand on its own.** The two averages - 0.8374 versus 0.8776 - are close, but an average cannot tell you whether the two groups can be separated. Looking at each question:

| | Top passage score |
|---|---|
| 28 in-store questions | lowest **0.8471**, highest 0.9152 |
| 3 unanswerable questions, clean | 0.7935 · 0.8236 · 0.8251 - highest **0.8251** |
| The koi question - unanswerable, but the question has leaked into the store | **0.9075** |

The koi question scores 0.9075 - higher than almost all in-store questions - and its top retrieved passage is… Generative AI · Lesson 12, the lesson that used **exactly that question** as an example, along with the verbatim made-up answer of the 0.5B model at the time. The store contains no correct grounds about raising koi, but it **contains the question itself**, and a wrong answer too. So the label *"no answer in the store"* **is still correct** - the reference answer is still *"Không tìm thấy trong tài liệu."* - but this case is no longer a **clean** unanswerable question: the question and a wrong answer have **leaked into the store** (*leakage*). It has become a **hard negative**, a kind that real systems meet more often than we think: documents about exactly that topic, even mentioning exactly that question, that do not answer it. The retriever does its job of measuring similarity correctly when it pulls the passage mentioning that exact question to the top; but **a high similarity score does not prove the passage contains the grounds for a correct answer**.

The trap does not stop at the retrieval layer. In stage A, the 0.5B model answers the koi question with exactly the sentence *"Chương này nói về cách nuôi cá koi trong hồ xi măng."* ("This chapter is about how to raise koi in a cement pond.") - the verbatim made-up answer that the previous chapter quoted as an example of an error. It copies its own error back out of the documents.

So the lesson reports **two groups separately**. For the **3 clean unanswerable questions**, the retrieval score **cleanly separates** them from the in-store questions on this test set: the highest clean question is 0.8251, the lowest in-store question 0.8471 - a threshold between those two numbers keeps all 28/28 and rejects all 3/3. Only 3 negative questions is not enough to **choose** a threshold for a real system, but it is enough to see that the retrieval score carries a signal for the **easy** kind of question. The koi question, meanwhile, scores **0.9075** - passing any threshold that still keeps the in-store questions - even though the store has no correct grounds. That is the real lesson of this section: **a similarity threshold can filter out easy off-topic questions, but it cannot solve the "is this answerable" problem when the store contains passages very similar to the question whose content is distracting or wrong.** Refusal can be placed in many places - a threshold calibrated on the retrieval score, a reranker, a separate step asking "does this passage answer the question", or rules based on metadata. The system in this lesson places it in the generation layer - through the prompt - and that is **a design choice**, not a requirement.

## 3. Six stages, from script to service

| Stage | Change from the previous stage |
|---|---|
| **A** | Baseline: exactly the system from the Generative AI chapter - Qwen2.5-0.5B, `transformers`, **one question at a time** |
| **B** | Switch to **Qwen2.5-7B AWQ, Marlin kernel**, on a vLLM server - lesson 04, lesson 08. Still one question at a time |
| **C** | Let requests **arrive concurrently**, open-loop load at 2 requests/second - the continuous batching from lesson 06 has work to do |
| **D** | Turn on the **prefix cache** - lesson 09 |
| **E** | Answer **with streaming** - lesson 08. Same run as D; only what the user sees changes |
| **F** | **Overload**: 12 requests/second. **F₀** has no limit; **F** limits in-flight requests to 20 and rejects the excess immediately |

Every stage from B onward runs on one RTX 3060. The model generates at most 200 tokens per question, with deterministic decoding.

> ⚠️ **A → B is not a one-variable change.** A uses plain Qwen2.5-0.5B through `transformers`; B uses Qwen2.5-7B-AWQ, the Marlin kernel, on a vLLM server. I bundled this step because the lesson wants to compare two *real* deployment configurations, and because the 7B bf16 version cannot run on a 12 GB card, so there is no way to change only the model size while keeping everything else. But that is **a design choice**, not an insurmountable constraint: it is entirely possible to run an extra 0.5B stage on vLLM to separate the engine from the model. The lesson did not do that, so this remains a weakness of the measurement rather than something already justified. So A → B only tells you the difference between **two complete deployment configurations**; it does not isolate the individual effect of parameter count, quantization, compute kernel or serving engine. From B onward, the pairs B → C, C → D, D → E and F₀ → F keep the same 7B-AWQ/Marlin on vLLM platform and change the factor named at each stage.

**Speed:**

| Stage | Throughput (req/s) | User **sees first word** p50 / p95 (s) | E2E p50 / p95 (s) | Rejected |
|---|---|---|---|---|
| A | 0.82 | 0.73 / 3.47 | 0.73 / 3.47 | - |
| B | 0.64 | 1.42 / 3.35 | 1.42 / 3.35 | - |
| C | 1.41 | **29.10 / 66.37** | **29.10 / 66.37** | - |
| D | 1.63 | 1.39 / 8.84 | 1.39 / 8.84 | - |
| E | 1.63 | **0.06** / 1.40 | 1.39 / 8.84 | - |
| F₀ | 9.33 | 0.56 / 1.51 | 9.31 / **30.31** | 0 |
| F | 8.24 | 0.11 / 0.15 | 1.84 / **5.08** | **153 / 545 (28%)** |

From A to D, the answer is sent as one block when it is done, so the user sees the first word **at the same moment** as E2E. From E onward there is streaming, so the first word appears when the first token does. The two F rows run at a completely different load (12 versus 2 requests/second); they are a pair to compare with each other, not with the rows above.

**Quality:**

| Stage | Has citations | Valid | Correct refusal (3 clean) | Hard negative: koi | Wrong refusal | Answers with Chinese characters |
|---|---|---|---|---|---|---|
| A - 0.5B/`transformers` | **5 / 28** | 4 / 28 | 3 / 3 | ✗ fell into the trap | 0 / 28 | 0 / 32 |
| B - 7B-AWQ/Marlin/vLLM, one at a time | **27 / 28** | 27 / 28 | 3 / 3 | ✓ refused | 1 / 28 | **4 / 32** |
| C, D, E, F₀, F | 26 / 28 | 26 / 28 | 3 / 3 | ✓ refused | 1 / 28 | 4 / 32 |

On the koi hard negative, stage A **falls into the trap** - copying the wrong answer found in the store - while stages B through F all refuse correctly.

**Memory** has no row in the table, and that is deliberate: stage A is measured by the amount PyTorch allocates (peak **0.99 GiB**), while the server stages are measured by the number the driver reports (about **11 GB**), because `gpu_memory_utilization=0.85` sets a **maximum budget of 85% of VRAM** for the model executor: vLLM does a trial run to measure the non-KV items - weights, peak activations, CUDA graphs and memory outside Torch - then takes the rest of the budget as the size of the KV memory, and reserves that space whether or not it gets used. The two numbers measure two different things - lesson 02 warned not to put them side by side.

## 4. Reading the table - including the cells that got worse

**A → B: the bundled platform switch raises citations, and three cells get worse.** Answers with citations rise from **5 to 27** out of 28. Refusal, though, has to be read by the two groups. On the **3 clean unanswerable questions**, both A and B refuse all 3/3 - **no improvement**. The difference is in the **hard negative**: configuration A falls into the trap, copying the wrong answer found in the store; configuration B refuses correctly. Before the two groups were separated, the table said "3/4 up to 4/4" - a number mixing two completely different kinds of question. Separated, the conclusion is more precise: B **holds firm on a question with distracting grounds** where A does not - on exactly one case, so it is only a sign. Citations measure **following instructions**, not whether the answer is correct; section 5 will show that an answer with valid citations can still be broken. But at the same time, configuration B is slower per question (E2E p50 **0.73 → 1.42 s**), has lower sequential throughput (**0.82 → 0.64** req/s), and gains one **wrong refusal** - saying "not found" even though the documents have it. This is the difference between two deployment configurations; this measurement does not allow attributing any single cell to parameter count, quantization, Marlin or vLLM.

**B → C: adding concurrency without enough capacity - the system collapses on latency.** With requests arriving at 2 per second, E2E p50 jumps from **1.42 to 29.10 seconds**, and p95 to **66 seconds**. Throughput does rise (0.64 → 1.41) but cannot keep up with the arrival rate, so the queue swells - exactly the phenomenon from lesson 07. This is the biggest worsened cell in the table.

Why is it overloaded at only 2 requests per second, when lesson 07 measured the server handling about 9? Because the prompts here are **1,127 tokens** long, while in lesson 07 they were only a few dozen. In stage B, prefilling one prompt takes 0.632 seconds, a prefill capacity of about **1,780 tokens per second**. The actual load arrives at **1.67 requests per second** (100 requests in 60 seconds), needing 1.67 × 1,127 ≈ **1,880 prefill tokens per second** - more than that capacity. And lesson 01 showed that the prompt prefill phase is **compute**-bound, so batching rescues the generation phase but does little for the prefill phase. This is a **rough capacity calculation**: the 1,780 capacity was measured running one question at a time, while under concurrency the engine batches and chunks prefills, the generation phase shares the GPU, and the scheduler has its own effects. It is **very consistent** with C becoming unstable, but it is not proof that it is the only cause.

**C → D: the prefix cache moves the capacity edge entirely.** With the cache on, under the same load: E2E p50 goes from **29.10 down to 1.39 seconds** - a 21-fold reduction. The 323-token system prompt, **28.7%** of each prompt, is now **reused** whenever its KV blocks are still in the cache - no need to prefill it every time. Prefill demand drops to about 1.67 × (1,127 − 323) ≈ **1,340 tokens per second** - below the rough capacity figure above. By the same calculation, this is consistent with that thirty percent reduction pushing the system from the unstable side to the stable side of the same edge.

Lesson 09 measured the prefix cache making TTFT 44 times faster - a nice number that sounds like a minor optimization. Here it is tied to something bigger: **whether the system holds up at this load at all**. The calculation in the paragraph above is a rough estimate, so what can be stated firmly is: turning on the prefix cache is the **only** change between C and D, and it turns an unstable system into a stable one under exactly this load; the capacity calculation gives an explanation in the right direction and of the right magnitude.

And there is still a cell that is not pretty: D has a **p95 of 8.84 seconds**, six times the p50. The system holds up but stands close to the edge, and whenever requests bunch up, the latency tail grows right away.

**D → E: streaming - only changes what the user sees.** In the same run, the user sees the first word after **0.06 seconds** instead of **1.39**, and at p95 after **1.40** instead of **8.84**. E2E does not change by a single digit, just as in lesson 08: the machine does not get faster, the user starts receiving value sooner.

**F₀ → F: limiting load - trading rejections for latency.** At 12 requests per second, with no limit every request is accepted and E2E p95 is **30.3 seconds**. With a limit of 20 in-flight requests, E2E p95 falls to **5.1 seconds**. The number 20 is a **starting point**, taken from Little's law in lesson 07: about 9 requests per second times about 2 seconds per request gives about 18 requests in the system **on average** around that operating point. Little's law does not say which limit **guarantees** a latency - it only gives the average in a stable system. So the right approach is to pick a starting point like that, measure with open-loop load how p95 and the rejection rate behave, then adjust. In exchange, **153 out of 545 requests (28%) are rejected immediately**, and throughput drops 12%.

That is a real trade-off, not a free improvement. But it is a trade-off **in the right direction**: those rejected get an "overloaded, try again later" answer within a few milliseconds, instead of everyone waiting half a minute together. And it protects the server from the kind of crash dissected in lesson 11 - when the number of concurrently running requests reached 193, the engine died.

## 5. Quality: what the speed table cannot see

**Batched execution changes answers, even with deterministic decoding.** Comparing each answer across stages:

| Stage pair | Answers identical character for character |
|---|---|
| B (one at a time) - C (batched) | 30 / 32 |
| B - D | 29 / 32 |
| C - D | 31 / 32 |
| D - F₀ | 31 / 32 |

The three answers that differ between B and D are all **identical at the beginning** and then veer off in another direction **partway through** - at characters 68, 181 and 278. Lesson 09 warned about this limitation without measuring it: when many requests run together in a batch, the order in which floating-point numbers are added changes, and where two tokens score almost equally, that small difference changes the chosen token - and the rest follows the new path. Now there is a number: about **one answer in ten**. The practical consequence: **do not use "deterministic decoding" as a reason to believe two runs on a server will produce the same text**, and do not compare the quality of two configurations by comparing them word for word.

**Configuration B-F switches to Chinese mid-sentence.** The 7B-AWQ/Marlin configuration on vLLM answers with Chinese characters in **4 out of 32** questions - always the same four, in every stage from B to F. Configuration A with 0.5B/`transformers` does so in **none**. The answer about shared key-value shows how it happens:

> *"…Cụ thể, với 14 đầu truy vấn nhưng chỉ 2 bộ khóa-giá* **值，这里应该是**…*"*

In the middle of writing Vietnamese ("Specifically, with 14 query heads but only 2 key-value sets"), the syllable "trị" is replaced by the character **值** - the very Chinese character that the Vietnamese word "giá trị" (value) is borrowed from - and the model carries on in Chinese.

This error is stable across every stage B-F, so the serving changes after B - concurrency, the prefix cache, streaming and the load limit - neither create it nor cure it. But because A → B is a bundled change, the data only ties the error to **the whole generation stack of configuration B**, not enough to attribute it to model size, quantization, the Marlin kernel or vLLM individually. And **the citation criteria cannot detect language errors**: they only check citations, and those answers still have valid citations. This error has to be tracked with a separate measurement - which is exactly the "Answers with Chinese characters" column in the table in section 3. Exactly the lesson of Generative AI · Lesson 12: **a rule only catches exactly what it checks**. To catch this error you need a dedicated rule - for example, counting characters outside the Vietnamese alphabet - and a real system should have that rule before handing it to users.

## 6. What is still missing before real users

The table in section 3 shows that the system runs, and runs fast. It is not yet a service. The missing jobs, each tied to a lesson that measured it:

| Job | Why | Lesson |
|---|---|---|
| Check the output language | 4/32 answers switch to Chinese and the scorer does not see it | section 5 |
| Key the cache by the asker's permissions, if there is an answer cache | Sharing an answer between two people with different permissions is a data leak | 09 |
| Record all fields for each request, **including the retrieval score** | Without it, an embedding model change error leaves no signature | 10, 11 |
| A labeled question set, rerun every time the model, prompt or store changes | The quality table in section 3 is only worth something if it can be measured again | 10 |
| Capacity planning with **open-loop load**, with **the real prompt lengths** | Stage C was overloaded at 2 req/s while lesson 07 measured 9 - because the prompts are dozens of times longer | 07 |
| A limit on in-flight requests and a polite rejection message | Stage F: trading 28% rejections for a six-fold lower p95, and avoiding the crash from lesson 11 | 07, 11 |
| Leave spare memory outside the engine's budget | The 0.9 level killed the engine four times in this chapter | 04, 09, 11 |

None of these jobs make the model smarter. All of them are about making a model that is already smart enough **able to serve real people** without answering in the wrong language, leaking data, crashing when busy, or breaking without anyone knowing.

> 🔧 **Try it now:** your team lead wants to open this system to 500 employees, "because lesson 07 measured the server handling 9 requests per second". How do you respond?
> The number 9 was measured with prompts of a few dozen tokens; this system's prompts are **1,127** tokens long, and stage C shows it becomes overloaded at under 2 requests per second without the prefix cache. So first you must re-measure capacity with **open-loop load using this system's real distribution of prompt and answer lengths**. Then estimate the real load: 500 people, each asking a few questions a day, concentrated in office hours - that gives a requests-per-second figure at peak hours, which can be compared against the measured capacity. And whatever number comes out, you should still turn on the prefix cache and set an in-flight request limit as in stages D and F - under this lesson's load, those are the two changes that go hand in hand with the system holding up at the edge.

## Lesson summary

- Put the RAG system from the Generative AI chapter through **six stages**, on the same question set and the same three retrieved passages for each question. A → B is **a bundled platform switch**; from B onward, the comparison pairs keep the same serving platform and change the factor named at each stage.
- **Stage 0:** retrieval takes **4.07 ms**, under 1% of E2E; article hit@3 **26/28**. On the 3 clean unanswerable questions, the retrieval score cleanly separates them from the 28 in-store questions (0.8251 versus 0.8471) - 3 negatives are not enough to choose a threshold. The koi question still has no answer in the store, but the question and a wrong answer **have leaked into the store**: it scores **0.9075**, passing every threshold that still keeps the in-store questions. **A similarity threshold can filter out easy off-topic questions, but cannot solve the "is this answerable" problem.** Where to place refusal is a design choice.
- **Platform switch A → B** - from 0.5B/`transformers` to 7B-AWQ/Marlin/vLLM: answers with citations rise **from 5 to 27** out of 28; refusal is **unchanged** on the 3 clean questions (3/3 for both), but B holds firm on the koi hard negative where A falls into the trap - one case, so only a sign. In exchange, each question is slower (0.73 → 1.42 s) and there is one wrong refusal. This is a difference between two configurations, not a measurement that isolates the effect of model size; the metrics also do not score whether answers are right or wrong.
- **Adding concurrency without enough capacity** (B → C): E2E p50 rises to **29.1 s**. A rough calculation - needing about 1,880 prefill tokens per second, while one question at a time prefills about 1,780 - is consistent with the system becoming unstable.
- **The prefix cache** (C → D) - the only change between the two stages: the 323-token system prompt, 28.7% of the prompt, is reused while it is still in the cache; E2E p50 drops **21 times** (29.1 → 1.39 s) and the system goes from unstable to stable. The rough capacity calculation gives an explanation in the right direction, not proof of a single cause.
- **Streaming** (D → E): the first word appears after **0.06 s** instead of 1.39 s; E2E is unchanged.
- **Load limiting** (F₀ → F): at 12 req/s, E2E p95 goes from **30.3 down to 5.1 s**, in exchange for **28%** of requests being rejected immediately and throughput dropping 12%. The limit taken from Little's law is only a starting point; it has to be re-measured and adjusted.
- **Batched execution changes answers** even with deterministic decoding: B and D are identical on **29/32**; three answers veer off partway through.
- **Configuration B-F using 7B-AWQ/Marlin on vLLM switches to Chinese** in **4/32** answers, stable across all stages; configuration A does not. A → B is a bundled change, so the error cannot yet be attributed to a single component. The citation scorer does not catch it - a rule only catches exactly what it checks.
- Still missing before real users: output language checks, a cache keyed by permissions, logs with all fields, a question set that can be re-measured, capacity planning with open-loop load and real prompt lengths, load limiting, and spare memory.

## Self-check questions

1. Why is the retrieval layer run once and shared by every stage, instead of being rerun at each stage?
2. A → B clearly raises the citation metric but has three cells that get worse. Name all three and list the components that changed at the same time. Why must these differences not be attributed to model size alone? What do those two quality metrics still miss, and what additional test is needed before saying one configuration is "better"?
3. Stage C is overloaded at under 2 requests per second, while lesson 07 measured the same server handling about 9. Explain with numbers.
4. The system prompt is only 28.7% of the prompt, yet turning on the prefix cache cuts E2E p50 by 21 times. Why does a small change cause such a large effect? Relate it to lesson 07.
5. In stage E, why does E2E stay the same while the time to see the first word drops by more than 20 times?
6. How was the number 20 chosen as a starting point? If each request took 4 seconds on average instead of 2, how would the starting point change - and why, however it is chosen, must it still be re-measured with open-loop load?
7. B and D give different answers on 3 out of 32 questions despite the same deterministic decoding. Explain the mechanism, and state one consequence for comparing the quality of two configurations.
8. The scorer gives B a citation score of 27/28, but 4 answers switch to Chinese. Design a check rule that catches this error, and describe one case where that rule gives a false alarm.
9. You must do capacity planning for this system to serve 2,000 people. List the measurements needed, in order, and say what question each measurement answers.
10. The koi question has no answer in the store, but the question and a wrong answer have leaked into the store. Why is its "unanswerable" label still correct? State what it teaches at the retrieval layer and at the generation layer, and propose a check step that detects this kind of leakage **before** scoring.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Generative AI · Lesson 12** | The original question answering system, and the principle of measuring the two layers separately |
| **Lesson 01 of this chapter** | Why the prompt prefill phase is compute-bound - the reason stage C is overloaded |
| **Lesson 07 of this chapter** | Open-loop load, Little's law, and the stability edge |
| **Lesson 09 of this chapter** | The prefix cache, and the "batched execution can change tokens" limitation |
| **Lesson 11 of this chapter** | The fields that must be recorded, and the crash that stage F guards against |

**Going further:**

- [Lewis et al. - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)](https://arxiv.org/abs/2005.11401) - the original paper on the retrieval-augmented question answering architecture.
- [He - Defeating Nondeterminism in LLM Inference (Thinking Machines, 2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - why batched execution makes deterministic decoding produce different text, the phenomenon in section 5.
- [Beyer et al. - Site Reliability Engineering, chapter "Handling Overload"](https://sre.google/sre-book/handling-overload/) - rejecting early under overload, the basis of stage F.

**Source code for this lesson:** [`code/b12_chung.py`](../../code/ai-systems/b12_chung.py) - load, prompts, scorer; [`code/b12_chuanbi.py`](../../code/ai-systems/b12_chuanbi.py) - stage 0, the retrieval layer; [`code/b12_A.py`](../../code/ai-systems/b12_A.py) - stage A; [`code/b12_client.py`](../../code/ai-systems/b12_client.py) - the client for stages B through F; [`code/b12_tinh_lai.py`](../../code/ai-systems/b12_tinh_lai.py) - recomputes the refusal metric on the 3 clean questions and reports the hard negative separately, from saved results, without rerunning the model.

## AI Systems chapter wrap-up

This chapter began with a subtraction and ends with a service. Every link in between has a measured number:

- **One call is two phases** - prompt prefill at 6,602 tokens/s, token generation at 58 tokens/s on the same card, because the generation phase uses only **0.65%** of the compute capacity (lesson 01).
- **The KV cache can be computed in advance** - the formula gives 28,672 bytes per token, measured at exactly 28,672; and the logits are **10.6 times** larger (lesson 02).
- **The model does not fit** - 15.23 GB on a 12.49 GB card; and budgeting with "weights plus KV cache" falls short by **59%** (lesson 03).
- **Making it fit** - AWQ at 5.2 GiB; switching to the right compute kernel is another **10.4 times** faster; the memory limit appears exactly where the division predicts (lesson 04).
- **Serving many people** - throughput rises 58 times from batch 1 to 64 almost for free (lesson 05); short requests no longer wait 27.8 seconds behind long ones (lesson 06).
- **Measuring correctly** - closed-loop load reports a p95 of 3.6 seconds, open-loop load gives **27** (lesson 07); streaming does not change the machine's TTFT, only when the user sees text (lesson 08).
- **Around the model** - the prefix cache is 44 times faster without reusing answers, the semantic cache wrongly answers the "under 500 million" question at cosine **0.9956** (lesson 09); changing the embedding model falls from 0.964 to **0.036** without a single error line (lesson 10).
- **Knowing when it breaks** - a crash diagnosed from the server's log and one multiplication: **193 × 152,064 × 4 bytes**, matching exactly the temporary tensor the stack points to (lesson 11).
- **Putting it together** - and seeing that the prefix cache, which sounds like a minor optimization, is the only change that turns the system from unstable to stable under exactly this lesson's load.

If you take only one thing from this chapter, let it be this: **every number in a serving system can be partly computed in advance, and the rest has to be measured under your own load.** Formulas tell you whether the system fits, whether it is fast, where it breaks. Measurement - with the right prompt lengths, the right way users arrive, the right hardware - tells you the real number.
