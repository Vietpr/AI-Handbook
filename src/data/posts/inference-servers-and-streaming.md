---
title: "Inference Servers and Streaming"
description: "What an inference server adds around the model, why it takes a minute to become ready, and measurements showing that streaming does not make the model any faster, it only changes when users start receiving output."
domain: "AI Systems"
section: "Learn"
language: "en"
translationKey: "inference-servers-and-streaming"
order: 8
pubDate: 2026-09-09
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will know what an inference server adds around the model, why it takes a whole minute to become ready, and be able to show with numbers that **streaming does not make the model one bit faster** - it only changes the moment users start receiving value.

Up to lesson 06, every measurement called the model from inside a script. Real users do not do that. They send an HTTP request, and between them and the model sits an **inference server**: it receives requests, queues them, batches them, calls the model, and returns results.

This lesson sets up such a server on the very 7B model that lesson 04 squeezed onto one card, then measures something intuition often gets wrong.

## 1. From script to service

One command is enough to get a server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin \
    --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8011
```

This server exposes three endpoints that this lesson and later ones use:

| Endpoint | What it is for |
|---|---|
| `POST /v1/chat/completions` | Receives requests in the common API format, with or without streaming |
| `GET /health` | Answers a single question: is the server ready to take work yet |
| `GET /metrics` | Exports quantities measured from **inside** the engine - lesson 11 builds a whole lesson on this endpoint |

**Startup is not instant.** From typing the command to `/health` answering OK takes **90 seconds**. Of that, loading the weights takes only **23.4 seconds**; the rest is the engine profiling memory, reserving the KV cache region, and preparing kernels for common batch sizes. The practical consequence: **do not send traffic to a server that has just started** simply because the process is running. A live process does not mean ready. The orchestrator - whether a simple script or Kubernetes - must query `/health` and wait.

At startup, the server reports that it has room for **77,744 tokens** of KV cache, or about **18.98 concurrent requests** if each request uses the full 4,096-token context. This is the third source of numbers that lesson 02 said to read separately: *capacity reserved*, not *amount in use*. It also differs by a few percent from the 76,208 tokens when running the same model in script mode in lesson 04, because the two ways of running reserve memory for different overhead items. Do not read that small difference as a finding.

## 2. Three quantities, and two response modes

A request can be answered in two ways.

**Without streaming:** the server generates the entire answer and then sends it as a single block.

**Streaming**: the server sends each piece as soon as it is generated.

It is very easy to say "streaming lowers TTFT". That statement is wrong, and wrong in a way that breaks diagnosis. You have to separate three quantities:

```
                         model generates token 1     model generates last token
                                  │                            │
request arrives ─────────────────►●────────────────────────────●
                                  │                            │
                  engine TTFT ────┘                            │
                                                               │
NO STREAMING: user sees ... nothing yet ... nothing ... ...    ● ← sees everything at once
STREAMING:    user sees           ● token 1 ● 2 ● 3 ● ...      ● ← done
                                  ▲
                    moment the first output is seen
```

| Quantity | Measured by | Meaning |
|---|---|---|
| **Engine TTFT** | The server, via `/metrics` | From when the request **reaches the server** to when the model **generates** the first token - including queue waiting |
| **Time to first visible output** | The client | When the user **receives** the first piece of text |
| **E2E** | Both | When the full answer is available |

In non-streaming mode, the first token is still generated at exactly that moment. It just sits waiting in the server until the whole answer is done.

## 3. Measuring: with and without streaming

Twenty **sequential** requests - no request competes with another for slots - each generating exactly 256 tokens, run once in streaming mode and once in non-streaming mode. Engine TTFT is read from `/metrics` before and after each run, taking the difference.

| | First visible output (client) | E2E (client) | **Engine TTFT** | E2E (engine) | Queue wait (engine) |
|---|---|---|---|---|---|
| **Streaming** | **0.026 s** | 3.852 s | **0.0284 s** | 3.847 s | 0.0 s |
| **Non-streaming** | **3.850 s** | 3.850 s | **0.0281 s** | 3.850 s | 0.0 s |

Model: Qwen2.5-7B-Instruct-AWQ, Marlin kernels, one RTX 3060.

## 4. Streaming does not make the model faster

**Engine TTFT does not change: 0.0284 versus 0.0281 seconds.** Streaming **does not make the model generate the first token any faster**. It does not touch the model.

**Time to first visible output changes by nearly 150 times: 0.026 versus 3.850 seconds.** In non-streaming mode, the user stares at a blank screen for almost four seconds - not because the model is slow, but because the first token was already generated at 0.028 seconds and **held back** until the remaining 255 tokens were done.

**E2E does not change: 3.852 versus 3.850 seconds.** The machine gets neither faster nor slower. In this measurement, the extra cost of sending many small pieces is too small to detect.

So the correct statement is: **streaming does not make the model faster; it changes the moment users start receiving value.** For a 256-token answer, that is the difference between "the assistant responds immediately" and "the assistant hangs for four seconds".

> ⚠️ **Two clocks, do not compare at the millisecond level.** In the streaming row, the client sees the first text at 0.026 s, while the engine reports a TTFT of 0.0284 s - the client is 2 ms *earlier* than the engine, which sounds absurd. Nothing is absurd: the two numbers are measured from **two different starting points**, one is a median and the other a mean, on two different clocks. This is exactly the "three sources of numbers" lesson from lesson 02. When comparing two sources, compare at the level of **order of magnitude**; a difference of a few milliseconds between two sources is definitional noise, not a finding.

## 5. When streaming does not help

The result above is easily stretched into "always turn streaming on". There are three cases where it does not help at all.

**When TTFT itself is already large.** Streaming only cuts the wait **after** the first token. If the first token arrives late - because of a long prompt as in lesson 01, because the whole batch has to be prefilled at once as in lesson 05, or because the request is sitting in a queue - the user still stares at a blank screen for just as long. The table above has zero queueing and short prompts, so TTFT is only 0.028 s; lesson 07 shows how that number swells under load.

**When the reader is a program.** If the output is fed into a JSON parser, or is a tool call, the program has to wait for the complete answer before it can do anything. Half a JSON object is useless. The benefit of streaming is a benefit to **human perception**; machines do not perceive.

**When infrastructure in between buffers it.** A proxy or load balancer can **buffer** the small pieces before forwarding them, wiping out the benefit above without the server ever knowing. The only way to find out is to **measure from a real client**, over the same network path users take, not directly on the server as this lesson does.

Streaming also has an operational cost. Once the server has sent the first piece, it has already returned a successful HTTP status code - so if an error occurs midway, there is no longer a way to report it through a status code; it has to be reported inside the data stream itself. And if the user closes the tab midway, the server should stop generating for that request, otherwise it keeps burning GPU on an answer nobody reads. This lesson does not measure these two things, but they are two things you should **check on your own server** before deploying.

> 🔧 **Try it now:** users complain that the assistant "always stalls for about a second before the text starts flowing", even though streaming is on. What do you measure?
> Separate exactly the three quantities from section 2. Read **engine TTFT** from `/metrics`: if it is also about one second, the problem lies in the model or the queue - a prompt that is too long, a batch that is too large, or an overloaded server - and streaming could never touch that. If engine TTFT is only a few tens of milliseconds while the **client** sees the first text after one second, that second is lost **in between**: a proxy buffering data, or a slow network. A number measured at both ends tells you where to fix things; a number measured at one end only tells you that users are unhappy.

## 6. What a server has to handle besides the model

This lesson uses an off-the-shelf server, but it is worth listing what it does for you - because you would have to handle it if you wrote your own, and configure it correctly if you use an existing one:

- **Ready is different from alive.** `/health` must answer "not yet" until the model has finished loading - 90 seconds here.
- **Input limits.** `--max-model-len 4096` rejects longer requests outright, instead of letting them exhaust memory for everyone else.
- **Queueing and batching.** Everything from lessons 05 and 06 happens inside, invisible to the caller.
- **Measuring from inside.** `/metrics` exports TTFT, time between tokens, E2E, queue waiting time, prefill time and decode time, and input and output token counts. Lesson 11 uses exactly this list.

## Lesson summary

- An inference server adds around the model: an HTTP API, a queue, batching, readiness checks via `/health`, and internal measurement via `/metrics`.
- The 7B AWQ server here takes **90 seconds** to become ready, even though loading the weights takes only **23.4 seconds**. A live process does not mean ready - the orchestrator must wait for `/health`.
- You have to separate **three** quantities: engine TTFT, the moment the client sees the first output, and E2E.
- Measured on 20 sequential requests, 256 tokens each: **engine TTFT is nearly unchanged** (0.0284 s streaming, 0.0281 s non-streaming); **the time until the first text is seen changes by nearly 150 times** (0.026 s versus 3.850 s); **E2E is unchanged** (3.852 versus 3.850 s).
- The correct statement: **streaming does not make the model faster; it changes the moment users start receiving value.**
- The two measurement sources (client and engine) should only be compared at the order-of-magnitude level; a 2 ms difference between them is definitional noise.
- Streaming does **not** help when TTFT is already large, when the reader is a program, or when infrastructure in between buffers the data. And it changes how errors are reported midway, along with requiring generation to stop when the user leaves.

## Self-check questions

1. Why is "the server process is running" not enough to start sending traffic? Give the specific numbers from the lesson.
2. Explain why engine TTFT is almost the same in both modes, while the moment the user sees the first text differs by nearly 150 times.
3. In non-streaming mode, when is the first token generated, and where is it until the user receives it?
4. The client sees the first text at 0.026 s while the engine reports a TTFT of 0.0284 s. Why must you not conclude that the client is "faster" than the engine?
5. Name three cases where turning on streaming does not improve the experience, and explain each using a quantity from section 2.
6. A colleague measures that streaming works well when calling the server directly, but real users still see text appearing in large chunks. What do you suspect, and where do you measure to confirm it?
7. Why does error reporting become harder once streaming has started?
8. A system generates JSON for another program to read. Should streaming be turned on? Argue using the quantities in the lesson.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Lesson 01 of this chapter** | The five quantities, the formula E2E ≈ TTFT + (N−1)·TPOT, and the convention that TTFT counts from when the request arrives |
| **Lesson 02 of this chapter** | The three sources of numbers, and why the engine's numbers cannot be compared directly with yours |
| **Lesson 05 of this chapter** | Why static batching makes TTFT grow linearly with batch |

**Going further:**

- [vLLM - OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - the command-line parameters used in section 1.
- [vLLM - Production Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html) - the meaning of each quantity in `/metrics`, the basis for lesson 11.
- [MDN - Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) - the streaming mechanism this kind of API uses.

**Source code for this lesson:** [`code/b08_luong.py`](../../code/ai-systems/b08_luong.py) - measures the three quantities in both modes, reading engine TTFT from `/metrics`.

> **Next lesson:** [Two Very Different Kinds of Cache](../two-very-different-kinds-of-cache/) - the server is running and measurable. The next lesson tries to cut down its work with caching - and shows that "cache" in language model systems is a shared name for two entirely different things: one does not reuse answers but only reuses computation, while the other replaces the answer outright with an old one.
