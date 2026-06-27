title: "Reading: Mixtral 7B"

# The feed-forward layer for output token memorization

In this reading, we discuss the role of the feed-forward (multilayer perceptron; MLP) layers of a transformer in memorizing the correct extension of a text. 

Suppose that the attention layers of a network have embedded several tokens of text into a single embedding for the current token.

**1. Describe how the MLP layer may be able to look up the correct next token in the sequence based on the embedding of the current token **

In a mixture-of-experts transformer, each feed-forward layer has many MLPs, and, on a per-token basis, the network selects which MLPs to use to process the given token.

**2. Explain why a mixture of experts architecture could be useful for increasing the amount of content a transformer can memorize without increasing the amount of compute required**

(Of course, mixture of experts *significantly* increase the amount of memory required for the feed-forward layers!) 

Consider the *Mixtral of Experts* \[[2401.04088](https://arxiv.org/pdf/2401.04088.pdf)\] paper, in section 2.1 on p. 2.  The equations given there can be interpreted as source-code from which you can infer the architecture of the mixture-of-experts approach used by Mixtral.

**3. How does the mixture of experts determine which MLP expert will process the token in the Mixtral model?

Now read this course text: [Reading supplement: What fraction of a transformers learnable parameters are in the mixture of experts?](../text/mixture.md), which explores how the vast majority of learnable parameters are now in the mixture-of-expert layers of modern transformers.

**4. What fraction of a transformer's learnable parameters are in the mixture of experts? How well does this statistc support the argument that a network mostly memorizes internet content vs. the argument that LLMs can reason?**

# (Optional) The OpenAI open-source-software (OSS) model and its predecessors

The line of transformers from the original transformer \[[1706.03762](https://arxiv.org/pdf/1706.03762.pdf)\] to OpenAI's gpt-oss is refreshingly short.

* *LLaMA: Open and Efficient Foundation Language Models* \[[2302.13971](http://export.arxiv.org/pdf/2302.13971)\]: The Llama model is Facebook's first public model.  It lists three changes from the original transformer architecture (Section 2.2):
  * Pre-normalization. Normalization before instead of after each layer
  * SwiGLU activation (instead of ReLU activation) (and a 2/3 smaller hidden layer in the feed-forward block)
  * Rotary Embeddings. Instead of adding positional encodings once at the input, positional encodings are added at each layer of the network.  (RoPE, \[[2104.09864.pdf](https://arxiv.org/pdf/2104.09864.pdf)\])
* *Llama 2: Open Foundation and Fine-Tuned Chat Models* \[[2307.09288](https://arxiv.org/pdf/2307.09288.pdf)\]: The Llama 2 model adds one notable architectural change:
  * Section 2.2, Appendix A.2.1 (p. 47): Grouped-Query Attention: Keys and Values for a token can be cached and reused when that same token appears in later processing. However, this starts to require significant memory. Thus, it is common to reduce the number of keys and values used, reusing the same key-value pairs for several heads.  Llama settles on a variant of key-value sharing known as GQA (Grouped-query attention, \[[2305.13245](https://arxiv.org/pdf/2305.13245.pdf)\]
* *Mistral 7B* \[[2310.06825](https://arxiv.org/pdf/2310.06825.pdf)\]: 
  * Mistral uses a clever rotating context window that uses the space above the diagonal to hold more context for the earlier tokens in a sequence.  Because each layer builds on the tokens before it, this allows each layer to reach back one more context length into the past, allowing for $k \times W$ tokens to be considered at the output for a network with a context length of $k$ and a context window in each layer of $W$ tokens.  (Context lengths are limited in part because of using RoPE for the embedding instead of absolute positional embeddings)
* *Mixtral of Experts*: \[[2401.04088](https://arxiv.org/pdf/2401.04088.pdf)\] 
  * Mixtral uses nearly an identical architecture to Mistral, except that the feed-forward layers use a mixture of experts. Thus, only a couple of a pool of feedforward networks are selected for each forward pass instead of just using one feed-forward network.
  * It is interesting that these experts are used for the feed-forward layer, not for the attention layer.  This suggests that the feed-forward layer has an important role in memorizing large corpora of text.
* *gpt-oss*: \[[2508.10925](https://arxiv.org/pdf/2508.10925)\] \[[openAI announcement](https://openai.com/index/introducing-gpt-oss/)\]
  * OpenAI recently released a high-quality Chain-of-thought (CoT) reinforcement learning (RL) fine-tuned model.
  * The architecture of this model uses all of the innovations described above: pre-normalization (presumably), SwiGLU activations, RoPE, Grouped Query Attention (GQA), and Mixture of Experts.  This model also uses sparse attention in every other layer to cut down somewhat on the O(n) compute required for ordinary (dense) attention.
  * Note that although the model weights and source code of this model are open-sourced, the training data and training procedure used is proprietary. The arXiv model card DOES discuss some details of training, but not nearly as many as would be in a typical academic paper.

These papers show that the small set of new technologies introduced in these papers have become the de-facto standard for transformer architectures. However, in this week's reading, we will focus only on the mixture of experts in the MLP (feed-forward) layer.

This blog, [From GPT-2 to gpt-oss: Analyzing the Architectural Advances](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the), provides a nice summary of the new features used in modern architectures by comparing gpt-oss to GPT-2 (which was very similar to the original transformer) and Qwen (another modern open architecture).

You can play with the Mixtral and gpt-oss models for free (currently) here: [Groq playground](https://console.groq.com/playground)
