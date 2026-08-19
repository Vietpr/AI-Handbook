# Content Guide

The site has three content areas: **AI Handbook**, **Algorithms**, and **Blog**.

## AI Handbook (`section: "Handbook"`)

Use this for structured AI knowledge. A Handbook article should be complete enough that a beginner can build the right mental model without jumping across several posts.

Suggested structure:

1. What problem are we trying to solve?
2. Intuition / mental model.
3. How the mechanism works step by step.
4. Math, data flow or architecture when useful.
5. A small implementation or code path.
6. Common misunderstandings and failure cases.
7. Engineering trade-offs / production implications.
8. Related concepts and what to learn next.

Repository or paper analysis belongs **inside the relevant Handbook topic** when it helps explain the idea. There is no separate Deep Dives section.

Use `priority` and `order` so the article fits the learning path.

## Algorithms (`section: "Algorithm"`)

Use this for data structures, algorithms, problem-solving patterns and LeetCode solutions.

For a concept / pattern (`type: "Algorithm"`):

1. Problem shape.
2. Brute-force idea.
3. Key observation.
4. Algorithm / invariant.
5. Complexity.
6. Implementation.
7. Common mistakes.
8. Related problems.

For LeetCode (`type: "LeetCode"`): explain the reasoning before the accepted code. Avoid turning the article into a code dump.

## Blog (`section: "Blog"`)

Blog is intentionally lighter and more personal. Good examples:

- an interesting repository you found;
- a lesson from building an AI system;
- an experiment result;
- a technical opinion;
- a short note about a paper or tool;
- something worth revisiting later.

A Blog post can later become a Handbook article if the idea becomes stable, reusable knowledge.

## Languages

Use `language: "en"` or `language: "vi"` in frontmatter. The site UI supports EN / VI independently of article language.

If you later publish the same article in both languages, give both files the same `translationKey`. This field is reserved for linking translations in a future iteration.
