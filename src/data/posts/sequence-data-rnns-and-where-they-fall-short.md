---
title: "Sequence Data: RNNs and Where They Fall Short"
description: "Measuring the two main failure modes of recurrent networks and seeing why they lead naturally to attention."
domain: "Deep Learning"
section: "Learn"
language: "en"
translationKey: "sequence-data-rnns-and-where-they-fall-short"
order: 10
pubDate: 2026-08-23
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand why ordered data needs a different architecture, know how a recurrent network processes a sequence, and see in numbers the two places it falls short - the places that lead straight to attention.

## 1. A different kind of structure

Lesson 09 exploited **spatial** structure: nearby pixels are related. There is another kind of structure just as worth exploiting - **order**.

- *"Hôm nay trời không đẹp"* ("the weather is not nice today") and *"Hôm nay trời đẹp không"* ("is the weather nice today?") use exactly the same five words, with completely different meanings.
- A stock price today depends on the sequence of previous days, not on an unordered set of prices.
- A spoken sentence has variable length: three words or thirty words are both valid samples.

The networks of the previous nine lessons handle this kind of data very clumsily, for two reasons. They need **fixed-length** input - `nn.Linear(784, 256)` accepts exactly 784 numbers, so sequences of different lengths must be padded or truncated to a preset cutoff, and that cutoff both wastes space on short sentences and chops off long ones. And they have **no memory**: each input is processed independently, carrying nothing over from the previous input, so if you want them to know "this word comes after that word" you have to build the position information into the input yourself. It can be done, but it means doing the model's job for it - exactly the vicious circle Lesson 01 pointed out.

## 2. RNNs: a loop with memory

A **recurrent neural network (RNN)** solves both with one idea: process the sequence **one step at a time**, and carry a **hidden state** from one step to the next.

```
   x₁        x₂        x₃              xₜ
   │         │         │               │
   ▼         ▼         ▼               ▼
  ┌───┐ h₁  ┌───┐ h₂  ┌───┐ h₃      ┌───┐
  │ f │────▶│ f │────▶│ f │──▶ … ──▶│ f │──▶ prediction
  └───┘     └───┘     └───┘         └───┘
   the SAME set of weights reused at every step
```

At each step, the network takes a new input $x_t$ together with the old state $h_{t-1}$, and produces a new state:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

The state $h_t$ is a summary of everything read up to step $t$. Once the whole sequence has been read, we take the final state and feed it into a classification layer.

Notice the detail that **the same set of weights is reused at every step** - exactly the weight-sharing idea of the CNN in Lesson 09, except that it slides through *time* instead of through *space*. As a result, sequences of any length can be processed with a constant number of parameters.

To try an RNN without switching datasets, we read a 28×28 FashionMNIST image as **a sequence of 28 rows**, each row a vector of 28 numbers:

```python
class RowRNN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.rnn = nn.RNN(28, hidden, batch_first=True)   # each step reads 1 row
        self.fc  = nn.Linear(hidden, 10)
    def forward(self, x):
        out, _ = self.rnn(x.squeeze(1))    # (batch, 28 steps, hidden)
        return self.fc(out[:, -1])         # take the state at the LAST step
```

| Model | Parameters | Accuracy | Time (10 epochs) |
|---|---|---|---|
| **RNN** | 21,514 | 0.8289 | 93 seconds |
| **LSTM** | 82,186 | **0.8687** | 172 seconds |

The RNN reaches 0.8289 with only 21,514 parameters - reading the image row by row, with no idea what an image is. But notice that the second row beat it by nearly 4 points, and its name deserves a section of its own.

## 3. Why LSTM exists

Look back at the formula in section 2: the state $h_t$ is computed from $h_{t-1}$, while $h_{t-1}$ is computed from $h_{t-2}$, and so on. For a signal from step 1 to influence the prediction at step 28, the gradient has to travel back through **28 consecutive multiplications**.

That is exactly the situation of Lesson 08, except that the depth here is *sequence length* rather than the number of layers. Same disease: multiply many factors below 1 together and the gradient vanishes, and the RNN **forgets the beginning of the sequence**.

**LSTM (long short-term memory)** fixes this by adding a separate path for memory to travel along, together with three learned "gates" that decide which information to keep, which to drop, and which to output:

```
   memory state ────────────────────────────▶  a nearly straight path,
                    ↑ forget gate ↑ input gate  gradients shrink little
```

The core idea is to change the *kind* of multiplication the memory is subjected to. The memory state is still multiplied - the actual formula is $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, so it is still multiplied by the forget gate $f_t$ at every step. The difference is that $f_t$ is a number between 0 and 1 that the network **learns for itself** at each step, rather than a fixed weight matrix multiplied by the derivative of the activation function as in an RNN. When the network sees a need to remember, it learns to push $f_t$ close to 1, and the chain of multiplications becomes almost a running sum - gradients flowing back through it shrink very little. In other words, LSTM does not eliminate multiplication; it gives the network a knob to decide for itself when that multiplication is allowed to fade the memory. This is a close relative of the ResNet skip connections mentioned in Lessons 08 and 09. The table in section 2 shows it pays off: 0.8687 versus 0.8289, in exchange for nearly four times the parameters. **GRU** is a leaner variant with two gates instead of three.

## 4. Where both of them fall short

A 28-step sequence is short. Let us stretch it: read the same image but **one pixel at a time** - a sequence of 784 steps instead of 28, exactly the same amount of information, just 28 times longer.

| Sequence length | RNN | LSTM |
|---|---|---|
| 28 steps (reading by row) | 0.8289 | 0.8687 |
| **784 steps (reading pixel by pixel)** | **0.1138** | **0.1004** |

Both **collapse completely** to the random-guessing level of 0.1. LSTM cannot save it - it is even slightly lower than the RNN.

And there is a second number just as worth noting: the LSTM at 784 steps takes **3,825 seconds for 3 epochs** - 64 minutes. The reason lies in the nature of the architecture: step $t$ needs the state from step $t-1$, so **the 784 steps must run sequentially and cannot be computed in parallel**. While the CNN in Lesson 09 computes every position at once, the RNN is forced to wait in line.

Two problems, and they are two different problems:

- **Long sequences are remembered poorly.** Information has to pass through each step one by one, and the path between two positions 700 steps apart is 700 transformations.
- **Long sequences train slowly.** The computation is sequential and cannot take advantage of parallel hardware.

> 🔧 **Try it now:** the same 28×28 image, read as 28 rows, gives the RNN 0.8289; read as 784 pixels, it drops to 0.1138. The amount of information is identical - so what was lost?
> Not information, but **distance**. When reading by row, two vertically adjacent pixels are exactly 1 step apart in the sequence. When reading pixel by pixel, they are 28 steps apart, and pixels in opposite corners are 783 steps apart. The gradient has to travel back through that many multiplications to connect them. The longer the sequence, the further apart the same relationship in the data is stretched, and that is what an RNN cannot tolerate.

> ⚠️ **A small note:** the 784-step experiment ran for only 3 epochs, precisely because it is so slow. So the 0.10 figure reflects both the difficulty of remembering and insufficient training - and the two cannot be separated here. But that inseparability is exactly what is worth remembering: with long sequences, "hard to learn" and "too slow to learn in time" are bound together into a single problem.

## 5. Attention: letting every position talk directly

The fix for both problems comes from a simple question: why does information have to travel *through each step*? If step 700 needs to know what step 1 said, why not let it **look directly** at step 1?

That is **attention**. Instead of a hidden state passed from hand to hand through each step, every position in the sequence gets to look at *every* other position at once, and learns where to pay attention.

```
  RNN:  x₁ → x₂ → x₃ → … → x₇₈₄     the path from x₁ to x₇₈₄ is 783 steps

  Attention:  x₁ ←──────────────→ x₇₈₄   direct link, a path of 1 step
              every pair of positions is linked directly
```

This cures both diseases at once. The path between any two positions is just **one step**, so there is no more compounding of 700 factors. And because there is no sequential dependency, all positions **can be computed in parallel** - exactly what GPUs do best.

The architecture built entirely on that idea is called the **Transformer**, and it is the foundation of the language models you use every day. That is the content of **the Generative AI chapter**, not this lesson.

What this lesson needs to leave you with is the story of *why* attention came about: not because someone pulled a clever idea out of thin air, but because the two numbers in section 4 - 0.1138 and 3,825 seconds - describe a wall that recurrent networks could not get past.

## Lesson summary

- Ordered data needs its own architecture: a fully connected network demands fixed-length input and carries nothing over from one sample to the next.
- An RNN processes a sequence step by step, carrying a hidden state that summarizes everything read so far, and shares one set of weights across all steps - similar to weight sharing in a CNN, but through time.
- Reading FashionMNIST images as sequences of 28 rows: the RNN reaches 0.8289 with 21,514 parameters; the LSTM reaches 0.8687 with 82,186 parameters.
- LSTM exists because the gradient has to travel back through every step - the same vanishing gradient disease as in Lesson 08, with sequence length playing the role of depth. It routes memory through a learned forget gate instead of a fixed weight matrix, so the network decides for itself when to keep the memory intact.
- Stretch the sequence to 784 steps and **both the RNN and the LSTM collapse to random guessing** (0.1138 and 0.1004), and the LSTM takes 64 minutes for 3 epochs because it has to run sequentially.
- Two different problems: long sequences are remembered poorly, and long sequences train slowly because they cannot be parallelized.
- **Attention** fixes both by linking every pair of positions directly - the path shrinks to one step, and computation can run in parallel. The architecture built on it is the Transformer, the content of the Generative AI chapter.

## Self-check questions

1. Give two reasons a fully connected network handles a variable-length Vietnamese sentence very clumsily, and the price paid for each makeshift fix.
2. An RNN shares weights across all steps. Which property of the CNN in Lesson 09 does this correspond to, and what benefit do both buy?
3. Why does the vanishing gradient problem from Lesson 08 come back with RNNs, even though an RNN has only one layer?
4. In an RNN, the memory is multiplied by a fixed weight matrix at every step; in an LSTM it is multiplied by the forget gate $f_t$. Why does that change help the gradient survive long sequences? Which architecture in Lessons 08-09 uses a related trick?
5. At 784 steps, the LSTM takes 64 minutes for 3 epochs while the CNN in Lesson 09 takes only 161 seconds for 10 epochs. What architectural difference causes this gap?
6. Attention fixes two problems of RNNs. Name the two problems and how attention handles each one.

## Further reading

**Foundational sources for this lesson:**

| Source | What to read |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chapter 9 *Recurrent Neural Networks* - hidden state, weight sharing through time, and section 9.7 on backpropagation through time; chapter 10.1 *Long Short-Term Memory* - the three LSTM gates; chapter 11.1 introduction to attention |
| **Lesson 08 of this chapter** | Vanishing gradient - the same disease, with sequence length in place of the number of layers |

**Additional online sources (free):**

- [Christopher Olah - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - the most widely cited explanation of LSTM, with very clear diagrams of each gate.
- [Karpathy - The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) - RNNs generating text, with real examples.
- [`nn.RNN` / `nn.LSTM` - PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) - pay attention to the `batch_first` parameter and the shape of the output.
- [Hochreiter & Schmidhuber - Long Short-Term Memory (Neural Computation, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) - the original LSTM paper.
- [Vaswani et al. - Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762) - the Transformer paper; its title is the answer to section 5.

> **Next lesson:** [Transfer Learning: Standing on the Shoulders of a Trained Model](../transfer-learning-standing-on-a-trained-model/) - every experiment in this chapter so far has had 60,000 training images - a luxury. The next lesson handles a far more realistic situation: you have only a few thousand images, and a way to borrow what another model has learned from millions of images.
