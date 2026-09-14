---
title: "Observability for Serving Systems"
description: "What to record for every request so you can diagnose later, reading the metrics the serving engine exposes, the signature each kind of incident leaves, and a post-mortem of a server crash from its own logs."
domain: "AI Systems"
section: "Learn"
language: "en"
translationKey: "observability"
order: 11
pubDate: 2026-09-11
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will know what to record for each request so you can diagnose problems later, be able to read the quantities the serving engine measures itself, recognize the **signature** each kind of incident leaves on those quantities - from real measurements rather than a memorized lookup table - and be able to dissect a server crash from its own logs.

Lesson 10 ended with an error that left no log line: the embedding model was changed, quality fell to the level of random guessing, and among what that test recorded, the direct sign was a shift in the retrieval score distribution. Earlier lessons also ran into all kinds of other failures: the queue swelling in lesson 07, the engine crashing from running out of memory in lesson 04 and lesson 09.

This lesson gathers all of that into one practical question: **when users complain, where do you look to find out what is broken?**

## 1. What to record for each request

The principle: every recorded field must answer a specific diagnostic question. A field that answers no question is not needed; a question that no field answers will turn into a session of guesswork.

| Field | Answers the question |
|---|---|
| Request ID, arrival time | Which request, when - links the logs of the different layers together |
| **Queue wait time** | Is the system overloaded - lesson 07 |
| **Prompt prefill time** | Is the prompt too long - lesson 01 |
| **Token generation time**, TPOT | Is the generation phase abnormally slow - lesson 05 |
| TTFT, E2E | What the user sees - lesson 08 |
| **Input tokens, output tokens** | Cost, and whether this request is unusual |
| **Stop reason** (`stop` or `length`) | Did the answer finish naturally or get cut off |
| Retrieved passages, **score of the top passage** | Is the retrieval layer bringing the right documents - lesson 10, and Generative AI · Lesson 12 |
| Cache hit or not, which kind | Did the answer have to be computed fresh - lesson 09 |
| Model, prompt, store, configuration versions | Did two requests run on the same system |
| Error, if any | Which layer failed |

The "score of the top passage" row is the one most often left out and the most expensive to be missing: among what that test recorded, it is the direct sign of the embedding model change error in lesson 10. A system that also records **the embedding model version** and **the index version** for each request would see that error even sooner, right at the moment of the change.

## 2. What the engine measures itself

The vLLM server from lesson 08 exposes two kinds of quantities out of the box at the `/metrics` endpoint.

**Per-request cumulative quantities** - each completed request contributes one value; read before and after a time window and take the difference to get the average over that window. The list measured on this chapter's server:

`time_to_first_token_seconds` · `request_queue_time_seconds` · `request_prefill_time_seconds` · `request_decode_time_seconds` · `inter_token_latency_seconds` · `request_time_per_output_token_seconds` · `e2e_request_latency_seconds` · `request_inference_time_seconds` · `request_prompt_tokens` · `request_generation_tokens` · `request_params_max_tokens` · `request_prefill_kv_computed_tokens` · `iteration_tokens_total` · …

**Instantaneous quantities** - the state of the engine **at the moment you ask**: `num_requests_running` (number of requests running), `num_requests_waiting` (number of requests waiting), `kv_cache_usage_perc` (percentage of the KV cache in use). These must be **sampled continuously** - asking once after everything has calmed down shows nothing.

Note that the first kind has **already split** a request's time into segments: waiting, prefill, generation - exactly the breakdown formula from lesson 01, now measured for each request by the engine. And **TTFT is not a separate segment to add on**: it counts from when the request arrives, so it already contains both the waiting segment and the prefill segment. In the overload scenario of section 3, 0.472 s of waiting plus 0.288 s of prefill is 0.760 s - contained within the TTFT of 0.853 s; the rest is input processing and the first token generation step. Adding TTFT to the waiting time counts it twice.

## 3. Four scenarios, four signatures

Instead of presenting a lookup table of "which symptom means what is broken" and asking you to trust it, this lesson **creates** four scenarios on the same server - Qwen2.5-7B-Instruct-AWQ, one RTX 3060 - and reads which quantities each scenario leaves its signature on:

- **Normal**: requests arrive at 2 per second, short prompts, generating 128 tokens.
- **Overload**: as above but 14 per second - beyond the serving capacity measured in lesson 07.
- **Long prompt**: 1 per second, a **2,905**-token prompt, generating 64 tokens.
- **Long output**: 1 per second, short prompts, generating 1,024 tokens.

Averages over all requests in each scenario, read from `/metrics`; the last three rows are the maximum values when sampling every 0.5 seconds. Memory utilization is 0.85 - the reason is in section 5.

| | Normal | Overload | Long prompt | Long output |
|---|---|---|---|---|
| Tokens in / out (average) | 44 / 128 | 44 / 128 | **2,905** / 64 | 44 / **1,024** |
| Queue wait (s) | 0.000 | **0.472** | 0.000 | 0.000 |
| Prompt prefill (s) | 0.036 | 0.288 | 0.082 | 0.062 |
| TTFT (s) | 0.051 | **0.853** | 0.120 | 0.083 |
| Between two tokens (s) | 0.016 | **0.147** | 0.017 | 0.026 |
| Token generation (s) | 2.02 | **18.61** | 1.06 | **26.54** |
| E2E (s) | 2.07 | 19.46 | 1.18 | **26.62** |
| Running - maximum | 12 | **256** | 4 | 33 |
| Waiting - maximum | 0 | **44** | 0 | 0 |
| KV cache - maximum | 1.4% | 28.0% | 4.1% | 26.1% |

**Overload** is the **only** scenario with a queue wait time above 0 and requests left waiting. The running count stops at exactly **256** - the engine's default slot limit - while the KV cache is only 28% used. So the system has run out of **slots**, not out of **memory**. And because 256 requests run together in one batch, the time between two tokens grows **9.2 times** - exactly the region where lesson 05 showed the generation phase leaving the regime governed only by bandwidth. Prompt prefill time also grows 8 times even though the prompts are still short, because prefill steps have to squeeze in between a huge token generation batch.

**Long output** has **the worst E2E in the table** - 26.6 seconds - while TTFT and the queue are both normal. The most direct signature of the workload is **1,024 output tokens**; the other system quantities also change as a consequence of having to generate for longer: the time between two tokens is 0.026 instead of 0.016 seconds, the maximum running count is 33 instead of 12, and the KV cache rises to 26.1% instead of 1.4%. If you look only at E2E, you would think the system is slow; in fact the system is doing exactly what it was asked, the job is just long.

**Long prompt** has its clearest signature in **input tokens: 2,905**. The timings barely move - prompt prefill is only 0.082 seconds - and that is a **trap of this very measurement**: every request in that scenario uses **the same** prompt, so the server's prefix cache (lesson 09) swallows almost all of the prefill. Lesson 09 measured a prompt of about 2,843 tokens taking **1.55 seconds** without the cache. So with **different** long prompts - the more common case - TTFT will be much higher than in this table. The general lesson: **timings are a context-dependent signature; token counts are a stable signature.** Record both.

> ⚠️ **The second run of this table broke on the long prompt scenario itself.** I built the prompt by repeating one clause 120 times without counting tokens; the result was longer than the server's 4,096-token limit, and the server returned code **400** - exactly the input limiting feature from lesson 08. But my client treated the first error as fatal, so the long output scenario that came after did not run either. Two fixes: build prompts **by the token count measured with the tokenizer**, and have the client **count errors per request**. A load testing tool that dies because of one failed request cannot measure at exactly the moment the system is most worth measuring.

## 4. From symptom to quantity

| Users complain | Look at first | Signature measured in this chapter | Hypothesis to check |
|---|---|---|---|
| "It takes ages for the first word to appear" | **Queue wait**, then **input tokens** | Overload: 0.47 s wait. Long prompt: 2,905 input tokens | Overload → limit load or add machines (lesson 07). Long prompt → shorten it, prefix cache (lesson 09) |
| "The text appears slowly" | **Between two tokens**, **running count** | Overload: 9.2 times higher at 256 running | Batch too large (lesson 05) |
| "Answers take long" but text appears immediately | **Output tokens**, stop reason | Long output: 1,024 tokens, normal TTFT | Long answers - check `max_tokens` and the prompt |
| "It hangs when lots of people use it" | **Waiting**, queue wait over time | Overload: waiting rises to 44 | Capacity exceeded (lesson 07) |
| "The server threw 500 and then vanished" | Log: **running**, **KV**, **bytes requested** | Section 5: 193 running, KV 15.9%, 112 MiB requested | Type one memory limit (lesson 04) |
| "Off-topic answers", no errors | **Top passage retrieval score** | Lesson 10: dropped from 0.878 to 0.120 | Retrieval pipeline, index, corpus or question distribution changed or incompatible; lesson 10 illustrates it with an encoder version mismatch |
| "Answers mixed with Chinese" | Check characters outside the alphabet | Lesson 12: 4 out of 32 answers | Output content error - check the model, prompt, decoding and add a dedicated rule |

The last two rows deserve the most attention: **no timing quantity of the engine catches them**. The serving engine runs perfectly in both cases. Only fields recorded by the **application** itself - retrieval scores, output checks - can see them. That is why the table in section 1 has rows that `/metrics` will never provide.

## 5. Dissecting a real crash

The first run of the section 3 experiment used the default memory utilization of 0.9. The "normal" scenario ran to completion. In the "overload" scenario, after about 25 seconds the server returned error 500 for every request, and then the process disappeared.

There is nothing to guess. The server's log recorded everything, and we read it like a medical record.

**What happened before the crash** - the engine's periodic statistics lines, 10 seconds apart:

| Time | Running | Waiting | KV cache in use |
|---|---|---|---|
| 15:53:12 | 3 | 0 | 0.5% |
| 15:53:22 | 46 | 0 | 3.6% |
| 15:53:32 | 144 | 0 | 11.4% |
| 15:53:37 - crash | **193** | 0 | **15.9%** |

**Last words:**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
GPU 0 has a total capacity of 11.63 GiB of which 84.88 MiB is free.
```

Read three clues at once.

**It is not a lack of KV cache.** At the moment of death, the KV cache was only **15.9%** used. That alone is enough to remove the KV cache from the list of suspects.

**It died while requesting 112 MiB.** That specific number is the best clue. Try a multiplication: **193** running requests × **152,064** tokens in the vocabulary × **4** bytes:

$$193 \times 152{,}064 \times 4 = 117{,}393{,}408 \text{ bytes} = 111.95 \text{ MiB}$$

It matches. And it is not a coincidental match: the log kept the full **stack** of the error, pointing to exactly where the allocation happened:

```
vllm/v1/sample/sampler.py, in apply_penalties
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
```

The engine died **inside the sampler**, at the step that applies the frequency penalty to the logits. The multiplication `frequency_penalties.unsqueeze(1) * output_bin_counts` creates a **temporary tensor the same size as the logits** - 193 rows, each row with one cell for every token in the vocabulary, 32-bit floats - before subtracting it from the logits. That temporary tensor is exactly the 112 MiB the engine could not get. It belongs to precisely the "temporary activations" item in the breakdown from lesson 03, the same family as the logits that lesson 02 measured at more than ten times the per-token KV cache: something proportional to **the number of running requests × the vocabulary size**, not to the context.

**So this is the type one limit from lesson 04**: not the engine running out of slots to queue requests, but the activations of a large batch not fitting in the memory outside the budget the engine reserves for itself. The larger the batch, the larger that tensor - with 193 requests it went over the edge.

The fix follows directly from the diagnosis: **leave more memory outside the budget** - lower the memory utilization - or **limit the number of requests running at once**, which limits the size of those logits-sized tensors. The two reruns used the first approach, at **0.85**: the overload scenario **ran to completion both times**. The running count reached **256** - the engine's slot limit - without crashing; the excess waited, up to **44** requests; KV peaked at **28%**. Under the same load, this time the system **queued** instead of **dying** - meaning the type one limit had been pushed far away, leaving only the type two limit from lesson 04.

And note that the entire diagnosis came from **the server's own log** - the running request counter, KV utilization, the byte count and the stack in the error message - plus one multiplication. No extra tooling was needed. The only requirement is that **the log is kept**: if the log dies along with a dead server, you have to reproduce the incident to find out what happened.

## 6. A minimal dashboard

A dashboard only deserves to exist if every panel on it leads to an action. Here is a minimal set, each panel tied to a phenomenon this chapter has measured:

| Dashboard panel | Alert when | Why - which lesson |
|---|---|---|
| TTFT, TPOT, E2E - **p50 and p95** | p95 exceeds the acceptable level | The average hides the worst-off users - lesson 07 |
| Queue wait, waiting count | Above 0 for a sustained period | The unique signature of overload - section 3 |
| Running count | Hits the slot limit, or nears the level that once crashed the machine | 256 is the limit; 193 once crashed at 0.9 - section 3, section 5 |
| KV cache in use | Approaches 100% | Past the KV limit, throughput collapses by 64% - lesson 04 |
| Unfinished requests over time | Rises across several consecutive windows | The system is unstable - lesson 07 |
| Error rate by code (4xx, 5xx) | 5xx above 0; 4xx rising abnormally | 400 is the input limit doing its job; 500 is a dead machine - lesson 08, section 5 |
| Top passage retrieval score distribution | Sudden drop | A signal of an incident in retrieval, the index or the corpus that may leave no log line - lesson 10 |
| Rate of outputs missing citations, in the wrong language | Rises | A rule only catches what it checks - lesson 12 |

Every alert threshold should be set **relative to a normal day of that same system**, not relative to numbers in a book: the table in section 3 shows that on the same server, the "normal" signature already depends on the prompt length and answer length of the load.

> 🔧 **Try it now:** since 9 a.m., E2E p95 has risen from 3 to 25 seconds. TTFT p95 is still 0.1 seconds, queue wait is 0, the running count is normal. What do you suspect and what do you check?
> A normal queue and TTFT make **queue-dominated** overload less likely, but they are not enough to rule out every cause: the prefix cache can hide the cost of long prompts, and contention in the generation phase can make E2E and TPOT worse before the queue becomes visible. Check three branches next:
>
> - **Output tokens rising** → suspect a long-output workload; look at the stop reason, the prompt version and the model.
> - **TPOT rising sharply** while output tokens are unchanged → suspect resource contention in the generation phase; compare against the running count and concurrent load.
> - **Input tokens rising** → suspect longer prompts or input workload, even if the prefix cache is hiding part of the prompt prefill latency.
>
> Only if output tokens have shot up and many answers stop by hitting `max_tokens` can you conclude that the system has to generate longer outputs, and address the cause in the prompt, the kind of questions or the model before thinking about adding GPUs.

## Lesson summary

- Every field recorded for a request must answer a diagnostic question: queue wait, prompt prefill, token generation, TTFT, E2E, input and output token counts, stop reason, **retrieval score**, cache hit, versions, errors.
- The serving engine measures two kinds of quantities itself: **per-request cumulative** quantities (read the difference before and after) and **instantaneous** quantities - running count, waiting count, KV in use - which must be sampled continuously.
- Four scenarios on the same server leave four different signatures. **Overload**: the only one with a queue (0.47 s wait, 44 waiting), running count hits the **256** limit while KV is only at 28%, time between two tokens grows **9.2 times**. **Long output**: the worst E2E (26.6 s) with normal TTFT and queue; the most direct signature of the workload is **1,024 output tokens**, while TPOT, running count and KV rise from having to generate for longer. **Long prompt**: the direct signature is **2,905 input tokens**.
- Timings are a context-dependent signature: a repeated long prompt is swallowed by the prefix cache (0.082 s versus 1.55 s without the cache). Token counts are a stable signature. Record both.
- A crash can be diagnosed from **the server's log and one multiplication**: 193 running, KV only at 15.9%, 112 MiB requested = 193 × 152,064 × 4 bytes, and the stack pointing to the penalty step in the sampler - a **logits-sized temporary tensor**, i.e. the type one limit from lesson 04. Lowering memory utilization to 0.85 makes the same load queue instead of die.
- Two groups of failures that **no engine quantity catches**: incompatible retrieval, index or corpus; and output content errors such as the wrong language. Only fields recorded by the application itself can see them.
- A load testing tool must **count errors per request**; and test prompts must be built **by token count**, not by number of repetitions.
- A minimal dashboard: every panel leads to an action, thresholds set relative to a normal day of the same system.

## Self-check questions

1. Pick three fields from the table in section 1 and, for each one, say which kind of incident could not be diagnosed without it.
2. Why must instantaneous quantities such as the waiting count be sampled continuously, while cumulative quantities only need to be read twice?
3. In the overload scenario, the KV cache is only 28% used, yet the system already has requests waiting. Explain.
4. The long output scenario has the worst E2E in the table. Why must you not conclude "the system is slow"?
5. The long prompt scenario spends only 0.082 s in the prefill phase. Why is that number low, and what might the corresponding number be in a real system?
6. Reread section 5: list the clues in the log - running count, KV utilization, bytes requested, the stack - and the calculation that confirms the culprit. Why is matching the byte count alone not enough to name the culprit?
7. Name two groups of failures that the serving engine's `/metrics` can never see - one on the retrieval side, one in the output content - and which application-recorded field would catch each.
8. Design three alerts for the system in lesson 12, each stating the quantity, what the threshold is relative to, and the action to take when it fires.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Lesson 01 of this chapter** | The breakdown formula E2E ≈ queue + prefill + decode - the engine measures each segment for you; and why TTFT already includes the wait |
| **Lesson 04 of this chapter** | The two types of memory limit - the basis of section 5 |
| **Lesson 07 of this chapter** | What overload looks like from the outside |
| **Lesson 10 of this chapter** | Errors that leave no log line |

**Going further:**

- [vLLM - Production Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html) - the meaning of each quantity in section 2.
- [Beyer et al. - Site Reliability Engineering, chapter "Monitoring Distributed Systems"](https://sre.google/sre-book/monitoring-distributed-systems/) - the four golden signals: latency, traffic, errors, saturation.
- [OpenTelemetry - Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - consistent naming for the fields in section 1.

**Source code for this lesson:** [`code/b11_dauvet.py`](../../code/ai-systems/b11_dauvet.py) - the four scenarios, reading `/metrics` and sampling the instantaneous quantities.

> **Next lesson:** [End-to-End Project: Deploying a RAG System](../project-serving-a-rag-system/) - eleven lessons have built all the pieces. The final lesson takes the question answering system from the Generative AI chapter itself, puts it through six stages - from a sequential script to a server with quantization, batching, caching, streaming and limits - measures each stage under the same load, and the final table must show the cells that got worse too.
