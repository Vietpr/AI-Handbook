---
title: "What Is Jev? When AI Stops Just Talking and Starts Making Decisions"
description: "Jev is TypeSafe AI's first System One Model, designed to return structured decisions that software can use directly."
section: "Blog"
type: "Concept"
language: "en"
translationKey: "jev-system-one-model"
pubDate: 2026-09-22
featured: true
draft: false
---

A few years ago, mentioning artificial intelligence would make many people think immediately of ChatGPT, Gemini, or Claude - tools that can write emails, program, summarize documents, and answer questions in a remarkably human way.

Recently, however, the AI community has begun talking about a new name: **Jev**.

Jev was not created to be another chatbot. It does not try to write long passages or act as a conversational assistant. Instead, it is designed for a different job:

> **Make fast, structured decisions that other software can use directly.**

That could make Jev an important piece of AI's next phase: a shift from **AI that can answer** to **AI that can participate in operating systems**.

> **Note:** At the time of publication, Jev is still in early access. The technical descriptions and performance figures in this article come from TypeSafe AI's own materials and are not independent evaluations across every kind of workload.

## 1. The problem with today's AI: intelligent, but not always suited to controlling systems

To understand why Jev exists, it helps to look at how generative AI is usually used today.

Large language models are excellent at processing language. For example, you might provide this request:

> This customer is complaining about a delayed delivery. Analyze how serious the situation is.

A chatbot might respond:

> This case may need priority handling because the customer is having a negative experience...

That answer is reasonable for a person. For a software system, however, the paragraph is not an ideal output. Software usually needs a more explicit decision:

```json
{
  "priority": "high",
  "needs_human_support": true,
  "confidence": 0.92
}
```

The system can act on this result immediately:

- Move the ticket into the priority queue.
- Hand the conversation to a support agent.
- Act automatically only when confidence exceeds a defined threshold.

Computers work better with structured outputs like this than with paragraphs that must be read, parsed, and validated. This is the gap Jev is trying to address.

## 2. How is Jev different from a chatbot?

One way to understand the difference is to compare two roles.

### A chatbot is like an expert adviser

You ask:

> How should I handle this customer email?

A chatbot can analyze the context, explain the causes, and suggest several options. That flexibility is valuable when a human will read the final result.

The same flexibility also means the model may phrase its answer differently from one run to the next. If its output goes directly into code, the system must still verify that the response follows the expected schema, uses valid data types, and stays within the allowed choices.

### Jev is like a decision layer inside software

Imagine an e-commerce platform processing millions of events each day. Its systems continually need to decide:

- Which orders may be fraudulent?
- Which customers need priority support?
- Which cases should be escalated for human review?
- Which products fit a particular campaign?

The platform does not need a long analysis for every event. It needs values that code can branch on:

```json
{
  "fraud_probability": 0.87,
  "requires_review": true,
  "confidence": 0.95
}
```

Fast, explicit, and easy to compose into a workflow - that is the kind of problem Jev targets.

### But modern LLMs can return JSON too, right?

Yes. Many LLM APIs now support JSON mode, structured outputs, or function calling.

TypeSafe AI's argument is not merely about the **appearance of JSON**. According to its documentation, Jev is built around the decision problem itself: it receives state and typed questions, then returns typed values, probability distributions, and confidence. It does not first generate a passage of text and then force that passage into a schema.

In other words, structured output is an added capability for many LLMs. For Jev, structured decisions are the core interface.

## 3. What is a System One Model?

TypeSafe AI calls Jev its first **System One Model**. The name is inspired by Daniel Kahneman's distinction between two modes of thinking in *Thinking, Fast and Slow*.

### System 2: slow, multi-step thinking

Writing an article, analyzing a difficult problem, or building a business plan often requires us to:

1. Gather information.
2. Reason through several steps.
3. Compare alternatives.
4. Explain a conclusion.

Powerful reasoning models and chatbots are particularly useful for this kind of work. The tradeoff is that the process can require more time and compute.

### System 1: fast judgment

When you see a red traffic light, you do not need to reanalyze every traffic rule. Your brain reaches a decision almost immediately: **brake**.

Jev is designed around a similar pattern:

**Receive state → evaluate questions → return typed decisions.**

TypeSafe AI's documentation currently describes three main question types:

- **Choice:** choose one option from a list.
- **Score:** score the state against a rubric.
- **Noul:** estimate how true a statement is from 0 to 1.

Several questions can be evaluated in parallel against the same state. Instead of asking the model to invent the entire business process, a developer decomposes the problem into small judgments and combines their results in code.

> “System One Model” is currently TypeSafe AI's name for this model class, not an industry-wide standard.

## 4. Why do speed and cost matter?

Speed may not matter much when you ask AI to write one email. It matters greatly when AI sits inside a software loop that runs millions of times.

Imagine a company handling:

- 10 million customer requests per day.
- Millions of automated decisions.
- Thousands of agents and workflows running continuously.

If every small decision calls a large reasoning model, latency and cost can grow quickly. A model specialized for short judgments, parallel evaluation, and non-generative output may be a better fit for this layer.

In workflow evaluations published by TypeSafe AI, the company reports that Jev reached up to **193.6 times faster** and **444.6 times cheaper** on some tested workloads. TypeSafe AI also notes that these figures may represent the high end of real-world gains and that evaluations created by its own team may contain bias.

The responsible interpretation is not “Jev is always hundreds of times faster than every LLM.” It is this:

> For suitably shaped problems, a model specialized for decisions may be substantially more efficient than using a text-generation model at every step.

## 5. Where could Jev be used?

### Customer support

An LLM can still handle the natural-language conversation. Before or after a response, Jev could assess:

- Is this a routine question?
- Is the customer angry?
- Are there signs that the customer may leave?
- Should the conversation be escalated to a person?

The surrounding code can choose an action based on the probabilities and confidence.

### E-commerce

When a customer places an order, a system may need to assess:

- Does the transaction look unusual?
- Should it require additional verification?
- Is this customer a good fit for a particular voucher?
- Which cases need manual review?

Jev could act as an evaluation layer while the company's final business rules remain in ordinary code.

### AI agents

This may be one of the more interesting applications. An agent continually needs to decide:

- What should happen next?
- Should another tool be called?
- Is the current result sufficient?
- When should the agent stop or hand control to a person?

Calling a large LLM on every loop can be both slow and expensive. A fast decision layer could handle routing, scoring, or guardrails, while the larger model is reserved for tasks that genuinely require deep reasoning or content generation.

## 6. Will Jev replace ChatGPT?

The short answer is **no**.

Jev and chat-oriented models solve different classes of problems:

| Dimension | Chatbot or generative LLM | Jev |
| --- | --- | --- |
| Primary goal | Conversation, content, reasoning | Decisions inside software |
| Output | Text or token sequences | Typed values, probabilities, confidence |
| Main consumer | Usually a person | Usually code and workflows |
| Strength | Flexible expression and open-ended tasks | Fast, structured, easy to branch on |
| Examples | Email, code, analysis | Classification, scoring, routing, guardrails |

The future may not be one model doing everything. A practical AI system could include:

- A large model for deep reasoning and content generation.
- A fast model for classification and decisions.
- A specialist model for images.
- A retrieval system for finding information.
- Traditional code for applying rules and controlling actions.

As in a company, the chief executive does not perform every job. Different components are suited to different kinds of work.

## 7. Structured does not mean always correct

This distinction is easy to miss.

When TypeSafe AI says that Jev does not generate free-form strings or make type errors, it means the output is constrained to a predefined schema. If a question requires a boolean, a score, or one option from a list, the model will not suddenly return a poem or an unknown field.

Jev can still **make the wrong decision**.

That is why probabilities and confidence matter. Software can define thresholds such as:

```text
confidence >= 0.90  → act automatically
confidence >= 0.70  → request more data
confidence < 0.70   → escalate to a person
```

The schema tells the system what shape the output has. Confidence helps the system decide how much to trust it. Neither replaces evaluation on the organization's real data.

## 8. The broader view: AI is entering a new phase

Jev reflects a broader trend in AI.

The first phase focused on this question:

> How intelligent an answer can AI produce?

As agents and automation become more common, the next question is:

> Can AI make decisions quickly, consistently, and reliably enough to participate in operating millions of tasks?

Jev is one attempt to solve that problem by changing the interface between models and software. Instead of returning only words, it returns decisions that code can inspect, combine, and act upon.

Jev may not be the final answer. The product is still new, most benchmarks currently come from its own developers, and real-world effectiveness will depend heavily on the workload.

But the direction Jev represents is worth watching:

> **The future of AI is not only about speaking better. It is also about making more useful decisions inside software.**

## References

- [Introducing System One Models & Jev - TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Introduction - TypeSafe AI Documentation](https://docs.typesafe.ai/introduction)
- [TypeSafe AI](https://typesafe.ai/)
