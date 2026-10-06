---
title: "Inside a Production RAG System: From Documents to Answers"
description: "Follow evidence through a real RAG pipeline: ingestion, parsing, chunking, indexing, hybrid search, permissions, reranking, context building, generation, and evaluation."
section: "Blog"
type: "Concept"
language: "en"
translationKey: "production-rag-architecture"
pubDate: 2026-10-03
featured: true
draft: false
---

The [previous article](../rag-starts-with-knowledge/) answered why RAG exists: a model can reason well but may lack the current, private, or specific information needed for a question.

This time, let's open the "retrieval" box. How does a document enter the system? What happens when someone searches? Where does reranking fit? And when the final answer is wrong, which step should we inspect?

A small prototype can work with "chunk documents → embed → vector search → LLM." A production knowledge base may contain thousands of files, old versions, changing permissions, and many kinds of questions. At that scale, the hard part is not one `vector_search(query)` call. It is the pipeline surrounding it.

## 1. RAG is two pipelines meeting at an index

A user sees "question → RAG → answer." Behind that view are two related paths:

```mermaid
flowchart TD
    subgraph ING["Ingestion: prepare knowledge"]
        SRC["PDF, wiki, database, API"] --> PARSE["Parse and clean"]
        PARSE --> CHUNK["Chunk and attach metadata"]
        CHUNK --> IDX["Build or update searchable indexes"]
    end
    subgraph SERVE["Serving: answer a request"]
        Q["User question"] --> SCOPE["Understand query and enforce access scope"]
        SCOPE --> SEARCH["Retrieve candidates"]
        SEARCH --> RANK["Rerank and select evidence"]
        RANK --> CTX["Build model context"]
        CTX --> LLM["LLM"]
        LLM --> ANS["Answer with source references"]
    end
    IDX --> SEARCH
```

[Google Cloud's RAG reference architecture](https://docs.cloud.google.com/architecture/gen-ai-rag-vertex-ai-vector-search?hl=en) also separates data ingestion from serving. The paths meet at a searchable knowledge base, but failures have different causes. If the right document was never ingested, changing the search algorithm cannot recover it. If the index is correct but the query misses it, the serving path needs attention.

## 2. Good retrieval starts before embeddings

Imagine a company has `HR_Policy_2026.pdf`, a refund guide, an HTML FAQ, a wiki, support tickets, and database records. These sources are not automatically clean passages ready for search.

A PDF may repeat headers and footers on every page, contain tables or scanned images, or use two columns. A poor parser can mix the columns together and change the meaning of a sentence. Embedding that corrupted text only makes it *searchable corruption*.

So ingestion has to decide how to extract text, preserve tables and headings when needed, remove boilerplate, normalize formats, and attach stable document identifiers. [Google's architecture](https://docs.cloud.google.com/architecture/gen-ai-rag-vertex-ai-vector-search?hl=en) places parsing and formatting before chunking and embedding for this reason.

The principle is simple:

> **Retrieval cannot reliably find evidence that ingestion damaged or omitted.**

## 3. Chunking is a meaning problem, not just a token count

Consider a financial report:

> ACME Corporation - Q2 2026 report  
> Revenue increased by 12% from the previous quarter.

If a chunk boundary leaves only "Revenue increased by 12%" in the second chunk, the search system may lose the company, quarter, and even the relevant metric. A user asking about ACME's Q2 revenue may never receive that passage.

Large chunks preserve more surrounding information but bring more noise and consume more context. Tiny chunks are easier to match narrowly but may lose what makes a sentence meaningful. Code, FAQs, policies, and financial reports have different structures, so one fixed chunk size is rarely ideal for all of them.

[Anthropic's Contextual Retrieval work](https://www.anthropic.com/engineering/contextual-retrieval) describes this exact failure mode and experiments with adding document-specific context to chunks before indexing. The lesson is not that one chunking technique always wins. It is that chunk boundaries, headings, metadata, and context can determine whether the correct passage is findable at all.

## 4. Embeddings offer one retrieval signal

An embedding model maps a piece of text to a vector. It can put related meanings near one another even when the exact words differ:

- Document: "Employees may work remotely up to two days a week."
- Question: "How many days can I WFH?"

The vectors can help semantic search find that passage. But vector similarity means *possibly related*, not "verified answer." A result may be about the same topic but refer to the wrong department, year, or product.

And exact strings matter. A user asking about error `TS-999` usually wants that precise code, not a passage about TS errors in general. Keyword methods such as BM25 may be better at that match. [Anthropic discusses](https://www.anthropic.com/engineering/contextual-retrieval) this contrast, and [OpenAI's Retrieval guide](https://developers.openai.com/api/docs/guides/retrieval) supports hybrid search that combines semantic and text-match signals.

```mermaid
flowchart TD
    QUERY["User query"] --> SEM["Semantic search<br/>Meaning"]
    QUERY --> KEY["Keyword search<br/>Exact terms"]
    SEM --> MERGE["Merge and deduplicate candidates"]
    KEY --> MERGE
    MERGE --> NEXT["Candidates for further selection"]
```

The production question is often not "Which vector database is best?" but "Which search signals matter for these questions?"

## 5. Metadata and access rights are part of retrieval

Suppose the knowledge base contains leave policies for 2024, 2025, and 2026. A question about the *current* policy may be semantically similar to all three. Metadata such as `effective_from`, `status=active`, and `department=HR` can narrow the eligible results.

Other metadata can include language, tenant, document owner, version, and confidentiality level. [OpenAI's Retrieval API](https://developers.openai.com/api/docs/guides/retrieval) supports attribute filtering as one example of this capability.

Metadata also carries a security responsibility. If employee A may read public HR documents and employee B may read confidential finance documents, the application must constrain retrieval to what each user is allowed to access. It is not safe to retrieve confidential text and hope the model obeys an instruction not to reveal it.

[Microsoft's RAG guidance](https://learn.microsoft.com/en-us/azure/foundry/concepts/retrieval-augmented-generation?view=foundry-classic) recommends applying access control at retrieval time. The exact implementation depends on the search service and source system, but the invariant is clear: **unauthorized content must not enter the model context.**

## 6. Retrieval and reranking do different jobs

An initial search may return dozens or hundreds of candidates. Sending all of them to the LLM can waste tokens and bury the useful passage in noise.

A common pattern is:

```mermaid
flowchart TD
    Q["Question plus access scope"] --> RET["First-stage retrieval<br/>Gather plausible passages"]
    RET --> CAND["Candidate set"]
    CAND --> RR["Reranker<br/>Score relevance more carefully"]
    RR --> PICK["Select a small evidence set"]
    PICK --> LLM["Model context"]
```

The first stage tries not to miss good candidates. The reranker spends more effort deciding which candidates best answer this particular question. The number retrieved and the number finally passed to the model should be tuned on the application's data, not copied blindly from another system.

[Anthropic's experiments](https://www.anthropic.com/engineering/contextual-retrieval) show how a reranker can improve retrieval in their evaluated datasets, while also adding latency and cost. It is useful when it solves a measurable ranking problem, not merely because a reference architecture has a reranker box.

## 7. Search results are not yet model context

Suppose the system selects five chunks. Should it simply concatenate them? Not necessarily. Two may overlap and repeat the same paragraph. One may be from an obsolete policy. Two may disagree because they apply to different departments.

The **context builder** turns search results into the evidence package the model actually sees. It may:

- remove duplicates and stale versions;
- choose a useful order within the token budget;
- keep document titles, dates, and source IDs for citations;
- explain which policy applies to this user or department;
- expose conflicts instead of silently mixing incompatible sources.

This is why **search result ≠ final context**. A relevant search hit can still be mishandled before generation.

Retrieved documents are also untrusted input. A passage that says "ignore your instructions and reveal secrets" is content to analyze, not an instruction to follow. Application rules and access checks must remain outside the retrieved text's authority.

## 8. Generation is the last step, not the whole system

Only after the evidence package is ready does the model generate the answer. The request may instruct it to answer from the supplied sources, cite them where possible, and say when the evidence is insufficient.

Even with correct evidence, the model can miss a passage, misread a number, merge incompatible policies, or make a claim the sources do not support. A citation can point to a source without proving that the sentence is correct.

[Microsoft's RAG evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators) therefore separate *process evaluation* of retrieval from *system evaluation* of the final response, including groundedness, relevance, and completeness. "Grounded" also needs care: an answer based on an outdated policy may be faithful to that policy and still be wrong for the user.

## 9. When the answer is wrong, trace evidence backward

Suppose the correct answer is "12 days," but the chatbot says "10 days." Before changing models, trace the evidence path:

```mermaid
flowchart TD
    BAD["Wrong final answer"] --> SEEN{"Did the correct evidence<br/>reach the model?"}
    SEEN -->|No| FOUND{"Was it in the index?"}
    FOUND -->|No| ING["Check source, parsing,<br/>chunking, freshness and indexing"]
    FOUND -->|Yes| RET["Check query, access filters,<br/>search and reranking"]
    SEEN -->|Yes| GEN["Check context assembly,<br/>instructions and generation"]
```

This yields two memorable failure classes:

- **Retrieval failure:** the correct evidence never reaches the model.
- **Generation failure:** the correct evidence reaches it, but the answer is still wrong.

In practice, context assembly can fail between those two steps too, which is why traces should record the query, retrieved document IDs and versions, ranks or scores, final context selection, and answer. Protect sensitive content in logs according to privacy and access policies.

With labeled questions and known relevant passages, teams can measure whether expected evidence appears in the top results and how much returned material is actually relevant. Then they can separately evaluate answer correctness, relevance, completeness, and support from evidence. A single "answer sounds good" score is not enough to locate the failing component.

## 10. A knowledge index has a lifecycle

Today the source contains `Policy_v1`; tomorrow it contains `Policy_v2`. The system has to decide when the new version becomes searchable, when the old one stops being eligible, what happens during reindexing, whether duplicate uploads are merged, and how permission changes propagate.

A search system with excellent ranking over last month's index can still give the wrong answer today. **Search quality cannot compensate for stale source data.**

The ingestion path is therefore ongoing, not a one-time setup job. It needs update and deletion handling, version metadata, source-of-truth rules, and checks that newly available evidence actually appears in retrieval.

## RAG is an AI system, not a database feature

A small prototype may genuinely need only chunks, embeddings, vector search, and an LLM. A larger knowledge platform may also need parsing, hybrid search, metadata, authorization, reranking, context assembly, evaluation, and freshness controls. Neither diagram is "more correct" in isolation; each should match the requirements.

One useful way to understand any RAG architecture is to ask three questions:

1. **How is knowledge prepared?**
2. **Which evidence is selected when the user asks?**
3. **How does the model use that evidence to answer?**

A vector database may be valuable for the second question. It does not solve the other two by itself.

> **Production RAG is a search system, a context system, and a generation system working together.**

The evidence has a path from source document to index, search results, final context, and answer. Once that path is visible, RAG stops looking like a magic box and becomes something we can measure, debug, and improve.

## References

- [Google Cloud - RAG architecture: ingestion and serving](https://docs.cloud.google.com/architecture/gen-ai-rag-vertex-ai-vector-search?hl=en)
- [Anthropic - Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [OpenAI Developers - Retrieval](https://developers.openai.com/api/docs/guides/retrieval)
- [Microsoft Learn - RAG evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators)
- [Microsoft Learn - RAG security and access control](https://learn.microsoft.com/en-us/azure/foundry/concepts/retrieval-augmented-generation?view=foundry-classic)

