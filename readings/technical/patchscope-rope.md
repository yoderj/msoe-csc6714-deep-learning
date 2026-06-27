title: Patchscope and RoPE

# Overview

In this week's reading, we will look at techniques for seeing what a residual embedding means, and also briefly look at positional embeddings, and RoPE, the most common positional embedding used in today's leading transformer models.

# Attribute Lens

This [webpage](https://lre.baulab.info/) and [arXiv paper](https://arxiv.org/pdf/2308.09124) introduce the idea of an **attribute lens**, a technique for determining attributes of an input token from its embedding within the residual stream.

Consider the first diagram within the webpage and Figure 1 of the paper.

**1. What linear relationship does the attribute lens learn?**

Consider Figure 3 of the arxiv paper or the blue bar-graph within the webpage. 

**2. For what sorts of relationships is the linear attribute lens reliable? For which is it unreliable?**

**3. What stereotypes in GPT-J does the LRE reveal? **

# Patchscope

Most of our focus in this reading will be on the (pair with google explorable on patchscope)[https://pair.withgoogle.com/explorables/patchscopes/) which accompanies Google's [arXiv paper on this topic](https://arxiv.org/pdf/2401.06102).

You will implement patchscope in your first lab sequence, starting from a transformer implementation such as the Gemma one from the first week's reading.

**4. Describe how to patch in patchscope**

**5. Explain how patching can reveal the meaning of an embedding within the residual stream**

For the next question, consider entity resolution, attribute extraction, or multi-hop reasoning applications of patchscope (*not* next-token prediction). Look for a discussion of multi-hop reasining in one of the papers.  Note that attribute extraction is distinct from next-token prediction, so avoid anything that is simply predicting output tokens rather than taking a subject in tokens and describing a desired attribute of that subject.

**6. In which source and target layers is patch scoping most reliable for attribute extraction? why do you think this is? **

**7. What advantages does patchscoping have over probing techniques like the attribute lens? Does it have any disadvantages?**

# Positional Encoding

In the [Attention is all you need paper](https://arxiv.org/pdf/1706.03762), section 3.5 describes positional encoding.  But the sinusoidal positional encoding described in this part of the paper is not used in leading transformers today. Instead, most transformers use [RoPE embeddings](https://arxiv.org/pdf/2104.09864).

I do not expect you to take the time to deeply understand how each of these techniques work, although I'm happy to discuss any questions you have about both of these papers during class.

But what I *would* like everyone to be able to answer are the following questions.

You may consider either of the papers mentioned above for the following question. Both provide good answers without needing to understand how the techniques proposed solve the problem.

Now read this course text on [positional encoding and attention](cross-attention-without-positional-encoding.md)

**8. When you consider the internals of the attention operation, why is positional encoding necessary? **

