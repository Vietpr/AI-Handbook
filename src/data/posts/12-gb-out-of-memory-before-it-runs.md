---
title: "12 GB: Out of Memory Before It Even Runs"
description: "Estimating whether a model will fit on your GPU before downloading it, then validating the estimate against a real out-of-memory error."
domain: "AI Systems"
section: "Learn"
language: "en"
translationKey: "12-gb-out-of-memory-before-it-runs"
order: 3
pubDate: 2026-09-07
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will be able to work out **before downloading** whether a model fits your GPU, read and understand every line of a real GPU out-of-memory error, distinguish **three ways of counting memory** that give three different numbers, and know why the sum "weights plus KV cache" always comes up short.

The two previous lessons measured on Qwen2.5-1.5B because it runs. But the model this chapter is really aiming at is the **7B** - big enough to do real work, and big enough not to fit.

This is the lesson where everything breaks. That is deliberate: you have to see it break before lesson 04 fixes it.

## 1. A two-line calculation, done before downloading 15 GB

Do not look up the parameter count on the model page and multiply in your head. The checkpoint declares its own size, in the file `model.safetensors.index.json`:

```
"total_size": 15231233024
```

**15,231,233,024 bytes = 15.23 GB.** That is all the weights, in the data type the model was published in - **bfloat16**, that is 2 bytes per parameter.

And the GPU:

```python
torch.cuda.get_device_properties(0).total_memory   # 12,490,457,088 bytes
```

**12.49 GB.**

$$15.23 - 12.49 = \mathbf{2.74 \text{ GB short}}$$

No KV cache yet. No activations yet. Not a single user yet. The weights alone are 2.74 GB too much.

> ⚠️ **Units are the first trap, and they throw off the calculation from the very first step.** This GPU is sold labeled "12 GB". `nvidia-smi` reports **12,288 MiB**, which is 12 binary GiB, which is **12.88 decimal GB**. And PyTorch reports **12.49 GB**. Three numbers for the same GPU. The difference between the last two - about **379 MiB** - is what the driver and firmware hold back, which CUDA never touches. Rule: **always convert everything to bytes before adding or subtracting**, and take the number PyTorch reports as the ceiling, not the number printed on the box.

## 2. Try loading it: what the error looks like

The calculation says it does not fit. But a line of arithmetic does not teach you to recognize the error at 2 a.m. So load it for real:

```python
m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", dtype=torch.bfloat16).to("cuda")
```

And here is exactly what comes back:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB.
GPU 0 has a total capacity of 11.63 GiB of which 30.88 MiB is free.
Including non-PyTorch memory, this process has 11.54 GiB memory in use.
Of the allocated memory 11.34 GiB is allocated by PyTorch, and 105.47 MiB
is reserved by PyTorch but unallocated. If reserved but unallocated memory
is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid
fragmentation.
```

This message is worth reading line by line, because **it lists by itself exactly the four memory items** that section 4 will rebuild.

| Piece of the message | Meaning |
|---|---|
| `Tried to allocate 130.00 MiB` | It died when requesting **one** more 130 MiB tensor. Not because it was short 2.74 GB all at once - but because it asked for one more small piece and none was left |
| `total capacity of 11.63 GiB` | The GPU's real ceiling, in GiB - exactly the 12.49 GB in section 1 |
| `11.34 GiB is allocated by PyTorch` | Live tensors: the part of the weights already loaded, about **80%** |
| `105.47 MiB is reserved by PyTorch but unallocated` | Reserved by the allocator but not yet used - **item 4** |
| `Including non-PyTorch memory, this process has 11.54 GiB` | 11.54 − 11.34 = **0.20 GiB** not belonging to PyTorch: the CUDA context and the driver's share |
| `30.88 MiB is free` | Exactly that much is left |

Three things are worth taking away right now.

**It does not die right at the start.** It loads 11.34 of the 14.18 GiB of weights - about **80%** - before it breaks. That means you sit waiting for a few dozen seconds, the progress bar nearly finishes, and only then does it crash. This is why the two-line calculation in section 1 is worth it: it takes two minutes and avoids a whole trial-and-error loop.

**The number in the message is what it just requested, not how much it is short.** "Tried to allocate 130.00 MiB" easily leads a newcomer to think they are only 130 MiB short and go reduce the batch a little. In reality it is 2.74 GB short.

**The advice at the end of the message is only right in one case.** `expandable_segments:True` fixes **fragmentation** - the situation where total free memory is enough but no contiguous block is large enough. Here the reserved-but-unused part is only **105.47 MiB** while we are 2.74 GB short, so fragmentation is **not** the cause, and turning on that flag saves nothing. The message itself says *"if reserved but unallocated memory is large"* - that condition is not met. Read the message carefully before following it.

## 3. Three ways of counting, three numbers

Before dissecting, we must agree on **what we measure with**. There are three ways to count GPU memory, and they are **never equal**.

| Counting method | What question it answers |
|---|---|
| `torch.cuda.memory_allocated()` | How much the live tensors take |
| `torch.cuda.memory_reserved()` | How much PyTorch's allocator has requested from the driver |
| `nvidia-smi` for the process | How much the driver sees this process taking |

Measuring all three at four moments, on Qwen2.5-1.5B / RTX 3060:

| Moment | Allocated (MiB) | Reserved (MiB) | Driver reports (MiB) |
|---|---|---|---|
| After initializing CUDA, **nothing allocated yet** | 0.0 | 0.0 | **102** |
| After loading weights | 2,945.3 | 3,134.0 | 3,236 |
| After a 4,096-token forward pass | 4,254.4 | 4,722.0 | **4,856** |
| After freeing and cleaning up | 2,954.4 | 3,156.0 | 3,290 |

The first row is the most surprising: **102 MiB disappear before you allocate a single byte.** That is the **CUDA context** - the driver loads compute kernels, builds tables, allocates workspace. No model memory formula contains this item, and it does not scale with the model.

The third row is the one to remember. At the same moment, the three counting methods give **4,254 · 4,722 · 4,856**. The gap between the smallest and the largest is **602 MiB** - as much as an entire small model.

The last row adds an unpleasant detail: after deleting the results and calling cleanup, memory **does not return to its initial level**. Allocated stays at 2,954 instead of 2,945, the driver stays at 3,290 instead of 3,236. Freeing in CUDA is not a clean inverse of allocating.

## 4. What takes up the space

Now dissect the third row of the table above - the 4,096-token forward pass - using the measured numbers:

| Item | Measured (MiB) | Source |
|---|---|---|
| Weights | **2,944.4** | Sum of parameter sizes |
| Temporary activations - **logits** | **1,187.0** | Size of the returned tensor |
| KV cache | **112.0** | Lesson 02, formula already verified |
| Other miscellaneous tensors | ~11.0 | Remainder |
| **Total live tensors** | **4,254.4** | `memory_allocated` |
| Reserved by the allocator but unused | **+467.6** | `memory_reserved` minus the above |
| CUDA context and non-PyTorch share | **+134** | `nvidia-smi` minus the above |
| **Driver sees the process taking** | **4,856** | `nvidia-smi` |

Rewritten as four items, in the order they accumulate:

```
   weights                                2,944.4 MiB
 + KV cache                                 112.0 MiB
 + activations and temporary workspace    1,198.0 MiB   ← logits take almost all of it
 + runtime and allocator overhead           601.6 MiB   ← includes the CUDA context,
 ───────────────────────────────────────────────────      reserved-but-unused memory,
 = GPU memory actually consumed           4,856.0 MiB      and fragmentation
```

> ⚠️ **Fragmentation is not a separate item to add.** It is very easy to list five boxes and add all five - and double count. Fragmentation is a phenomenon *inside* the fourth item: it makes the "reserved but unused" part swell, rather than taking up some other region. The error message in section 2 measures exactly that item and calls it `reserved but unallocated`.

And note that **activations during inference are not like those during training**. Training has to keep the activations of every layer for backpropagation, so they live for the whole step. Inference does not: the activations of layer $i$ can be thrown away as soon as layer $i+1$ has used them. What remains large is the **logits**, which lesson 02 measured - **296.8 KiB per token position**, 10.6 times the KV cache.

## 5. Why the familiar sum always comes up short

The most common way to budget is to add two items:

$$\text{weights} + \text{KV cache} = 2{,}944.4 + 112.0 = 3{,}056.4 \text{ MiB}$$

But the driver reports **4,856 MiB**. Short by **1,800 MiB, that is 59%**.

The consequence is very concrete. Suppose you have a 12.49 GB GPU, the 1.5B model, and you plan with the two-item sum: "weights 2.9 GB, leaving 9.5 GB for the KV cache, at 28 KiB per token that is 348,000 tokens, divided by a 4,096 context is **85 people at once**." That number is not off by a few percent - it misses the logits item entirely, which scales with context *and* with batch.

So the principle is: **do not budget memory with a formula and deploy straight away. Budget with a formula to rule out early the options that certainly will not fit - like the 7B model in section 1 - then measure for real to decide.** The formula answers "does it fit"; only measurement answers "how much fits".

> 🔧 **Try it now:** your system is running fine with 8 concurrent users. You raise the maximum context from 4,096 to 8,192 tokens, thinking it only costs a little more KV cache. It crashes out of memory. Why?
> Because you only added the KV cache. For the 1.5B model, doubling the context grows each person's KV cache from 112 to 224 MiB - an extra 896 MiB for 8 people, which sounds bearable. But if the system builds logits for all positions during prefill, that item doubles too, from 1,187 to **2,374 MiB for one prefill**. That is where it crashes. Lesson 06 will show that dedicated serving engines only build logits for the last position, and that changes this calculation completely.

## 6. Why we have not swept the memory boundary yet

A natural question at this point: would it not be neater to sweep `batch × length` until it breaks and draw the out-of-memory boundary?

Yes - but **it cannot be done here**, and the reason is worth stating.

This whole lesson just proved the 7B model **cannot be loaded** onto the GPU. If it cannot be loaded there is nothing to sweep: every batch and length configuration breaks at the same place, before it gets to run. Sweeping on the 1.5B model is possible, but it answers a question about a different model from the one this chapter is aiming at.

So the right order is: **lesson 03 proves it does not fit · lesson 04 makes it fit · only then can we sweep the memory boundary of that same 7B model.** That boundary is in lesson 04, once there is a 7B model that really runs.

## Lesson summary

- Weight size is read directly from the checkpoint (`model.safetensors.index.json` → `total_size`), not inferred from the parameter count: Qwen2.5-7B is **15.23 GB**, the RTX 3060 GPU is **12.49 GB**, **2.74 GB short** before there is any user at all.
- Units are a trap: the same GPU gives **12,288 MiB** according to `nvidia-smi` and **12.49 GB** according to PyTorch. Convert everything to bytes, and take the PyTorch number as the ceiling.
- A real error can be read: it loads up to **80%** of the weights before breaking when requesting another **130 MiB**. The number in the message is the amount **just requested**, not the amount **still missing**.
- The `expandable_segments:True` advice at the end of the message only fixes **fragmentation**. Here the reserved-but-unused part is only **105.47 MiB** while we are 2.74 GB short, so it saves nothing - read the condition carefully before following it.
- **Three ways of counting give three numbers:** at the same moment, `memory_allocated` 4,254 · `memory_reserved` 4,722 · `nvidia-smi` **4,856** MiB. A 602 MiB gap between the smallest and the largest.
- **The CUDA context costs 102 MiB before a single byte is allocated**, and it does not scale with the model.
- Four memory items: **weights 2,944.4 + KV cache 112.0 + temporary activations 1,198.0 + runtime and allocator 601.6 = 4,856 MiB**. Fragmentation is **not** a fifth item - it lives inside the fourth, making the reserved-but-unused part swell.
- Budgeting with "weights + KV cache" gives **3,056 MiB** while reality is **4,856 MiB** - **59% short**. The formula answers *"does it fit"*; only measurement answers *"how much fits"*.
- This lesson deliberately **does not sweep** `batch × length`: if the model cannot be loaded, there is nothing to sweep. That boundary is in lesson 04, once the model fits.

## Self-check questions

1. Why should you read `total_size` in the checkpoint instead of multiplying the parameter count by the number of bytes?
2. A GPU says "24 GB" on the box. Name three different numbers you might see for that same GPU, and which one should be used as the ceiling when planning.
3. The error message says `Tried to allocate 130.00 MiB`. Why does reducing the batch a little usually not rescue the situation in section 2?
4. When is `expandable_segments:True` actually useful? Use the numbers in the error message itself to explain why it is useless here.
5. Distinguish `memory_allocated`, `memory_reserved` and the number `nvidia-smi` reports. Which one must you use to know how much room is left for another process on the same GPU?
6. Why are activations during **inference** much lighter than during **training**, and why are logits the exception?
7. Explain why fragmentation should not be listed as a separate item when adding up memory.
8. You have a 24 GB GPU, a 13B model at 16-bit, an 8,192 context, a 128,000 vocabulary. Estimate the four memory items for **one** request, then say clearly which number in your estimate is the least certain and why.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Lesson 02 of this chapter** | The KV cache formula, and the measurement showing logits are 10.6 times larger |
| **Lesson 01 of this chapter** | The two phases, and why prefill is where logits balloon |

**Going further:**

- [PyTorch - CUDA semantics: Memory management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) - why `memory_reserved` is larger than `memory_allocated`, and what `PYTORCH_ALLOC_CONF` does.
- [Kwon et al. - Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180) - measures the waste from allocating the KV cache as contiguous blocks; lesson 06 will actually run this system.
- [Korthikanti et al. - Reducing Activation Recomputation in Large Transformer Models (MLSys 2023)](https://arxiv.org/abs/2205.05198) - an anatomy of activations, and why the numbers during training differ completely from inference.

**Source code for this lesson:** [`code/b03_bonho.py`](../../code/ai-systems/b03_bonho.py) - the three counting methods and the four-item anatomy; [`code/b03_oom.py`](../../code/ai-systems/b03_oom.py) - loads the 7B model onto a 12 GB GPU to capture the real error message.

> **Next lesson:** [When a Model Doesn't Fit on Your GPU: Three Options](../three-ways-out-when-a-model-doesnt-fit/) - being 2.74 GB short leaves three ways out - split the model across GPUs, offload part of it to the CPU, or shrink the model itself. The next lesson measures all three on this same hardware, explains which option the goal of running on **one** GPU leads to, and only once the 7B model fits does it sweep the memory boundary this lesson still owes.
