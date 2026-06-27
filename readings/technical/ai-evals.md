title: "Reading: AI Evals"

# AI Evals

Evaluating an AI system can be tricky.  A generative model (like a transformer or an image diffusion model) can generate many different correct outputs and evaluating whether an output is correct is more difficult than matching it to a known correct answer (as in classification problems) or subtracting the known answer from it (as in regression problems).

But the problem goes deeper than that.  A transformer can often solve problems correctly even where known correct answers are unavailable, and a transformer can be wrong in important ways even when it reaches a "correct" answer.

This is particularly true for chatbot applications, where a transformer is tasked with generating intelligent text to continue a very specific conversation on a very specific topic.

We are going to start with the idea of [Error Analysis](https://hamel.dev/blog/posts/evals-faq/#q-why-is-error-analysis-so-important-in-llm-evals-and-how-is-it-performed), the task of identifying where a transformer needs to improve using real-world data.

In addition to the link above, which gives a succinct summary of the steps in error analysis, this [blog post](https://hamel.dev/blog/posts/field-guide/) by the same author (Hamel Husain) gives more motivation for why error analysis is such a critical part of AI Evals process.

Finally, I first learned about Hamel Hussain (and his coauthor Shreya Shankar) when [they appeared on Lenny's podcast](https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill).  That podcast both discuses all of the steps from the first link above while giving practical motivating examples.

Consider the blog-post first.

**1. What is the most common error that Hamel claims AI teams make when developing an AI system?**

Make clear which is which:

**2. Give an example of a result of top-down analysis, and an example for a bottom-up analysis. Why do we prefer bottom-up analysis when developing tests for AI systems?**

Now, consider the last subsection (Guidelines for Using Synthetic Data) of Section 4 (Bootstrapping Your AI With Synthetic Data Is Effective).

**3. Should you generate synthetic user inputs or synthetic system outputs? Why?**

Now, consider Section 5, (Maintaining Trust in Evals Is Critical)

**4. Why does Hamel prefer binary decisions to 1-5 scales?**

**5. How does Hamel recommend grounding LLM-as-judge tests in human experience?**

**6. How does Hamel recommend that we build corporate long-term trust in AI Evals?**

Here, we revisit the last question from a previous reading:

**7. How is it possible for an LLM to reliably judge an unreliable output from the same LLM?**

