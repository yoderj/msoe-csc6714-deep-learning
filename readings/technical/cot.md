title: "Reading: Chain of Thought Prompt Engineering"

# Chain of Thought

Consider the following two prompts:

```
What is the largest city in pizza's country of origin?

Respond in this format:

<answer>...</answer>
<reason>...</reason>
```

and 

```
What is the largest city in pizza's country of origin?

Respond in this format:

<reason>...</reason>
<answer>...</answer>
```

Now consider how an LLM generates tokens, one token at a time.

**1. Which of the two prompts do you think is more likely lead to a well-thought out response?  What problems might the LLM face in following the other prompt?**

# Formatting the response

The examples above give an XML format for a prompt. In my experience, this sort of HTML-like tagging works well for LLM prompting.  For providing structured data as *input* to a prompt, the [Token-Oriented Object Notation (TOON)](https://github.com/toon-format/toon) could be useful. But I suspect that for output, HTML-like tags remains one of the best formats for a model to use.

# Examples of Chain of Thought

Consider [the first prompt of Appendix C.2 (p. 20) in this paper](https://arxiv.org/pdf/2305.18290) and [Figure 5 in Appendix A (p. 14) of this paper](https://arxiv.org/pdf/2306.05685). Both examples are quite similar, putting the reason before the answer. But they are subtly different in how they instruct the reasoning stage.

**2. Does the first or the second example do a better job leading the model into open-ended reasoning? How?**

Many prompting strategies guide the chain of thought explicitly.  For example, if you were to guide an LLM into reasoning through the evaluation of an integral, you might ask:

> ```
> 1. Identify what integration strateg(ies) might be effective for this integral. Consider substitution, integration by parts, and the recognition of patterns within the integral.
>
> 2. For each strategy identified above, attempt to use it to find an antiderivative for the integrand.
>
> 3. Take the derivative for your result from each strategy in the previous part.
>
> 4. Can the derivatives of the antiderivative be consired identical to the integrand? If so continue. If not, repeat steps 2 and 3, trying another strategy.
>
> 5. Subsitute the boundary conditions in to the antiderivative to evalute the definite integral.
```

For a task like this, a good LLM should already know to do these steps without being told. But for niche domain tasks not yet represented in the training data, a series of steps like this can make a huge difference in the output quality of an LLM.

# The Bitter Lesson
Rich Sutton, one of the founders of the theory of reinforcement learning, wrote an influential blog post in 2019 called, "[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)".  Read this blog post.

It has been argued that spending a lot of time writing really good guided chain-of-thought prompts like the one above can be considered an example of the bitter lesson -- newer models can perform well without all the guidance, so why do we spend time developing them?

**3. Give an example of a task where a modern NN does NOT require chain-of-thought prompting to come to a solution.

**4. Give an example of a task where a modern NN fails to reach a right answer, but chain of thought prompting helps it to reach the answer more reliably**

**5. How might your answers to the two previous questions change as larger models arrive on the scene?**

Extra: Do you have a business idea that leans on LLMs having some form of human-like intelligence?  Developing an eval for when an LLM performs that tasks successfully can help you to monitor for if -- and when -- the LLMs are ready for your business idea to become a reality.
