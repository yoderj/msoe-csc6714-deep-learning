---
title: What fraction of a transformer's learnable parameters are in the mixture of experts?
---

The feed-forward layers of a transformer are the layer that memorize next-token extensions of a text.  So it's natural to ask: What fraction of a transformer's  learnable parameters are in these next-token-memorizing layers?

# In a basic transformer
Let's start with the classic transformer introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). In this architecture, each feed-forward layer has a hidden dimension that is 4x the embedding dimension, and the attention embeddings embed to the same size as the embedding dimension.

So the attention layer requires four attention embeddings (for Q, K, V, and the output), each of size $n_{model} \times n_{model}$, for a total of $4 \times n_{model} \times n_{model}$ learnable parameters per attention layer.

But the feed forward layer requires two embeddings of size $n_{model} \times (4 n_{model})$, one to go to the hidden dimension and another to come from it. (We are ignoring the extra parameters needed for the bias here, but these are tiny in comparison to the main weight matrix.) So that's a total of $8 \times n_{model} \times n_{model}$ learnable parameters per feed-forward (MLP) layer.

So in the first transformer, the feed-forward layer required twice as many parameters as the attention layer and thus about 2/3 of the learnable parameters in the transformer.

# In a mixture of experts transformer

Let's consider gpt-oss from OpenAI, released Summer 2025, as an example of a modern cutting-edge open transformer.  This transformer uses a mixture of experts for the feed-forward layer.

[Talking with Claude about gpt-oss](https://claude.ai/share/0327d3e5-9f2c-4ebc-9c84-202e937b6b0b) gives a simple sketch suggesting that the vast majority of the weights in gpt-oss must come from the mixture-of-expert layers.  Indeed, Claude estimates that in a network with 117B parameters, roughly 115B parameters remain for the mixture of experts after considering all the other portions of the network.

But Claude struggles to see how this would work -- after all, Claude estimates that even with an expansion factor of 2x (vs the 4x in the original transformer) to the hidden dimension of one of the MLP experts, a single SwiGLU expert would require about 50M parameters, while the 115B remaining parameters only leave 25B parameters per expert. Could it be that the expansion factor for an expert is only 1?

Indeed it is!  Looking at the [open-source PyTorch implementation of gpt-oss](https://github.com/openai/gpt-oss/blob/d7032f6f0408ddb1267b30495a96922ab8fa8ff5/gpt_oss/torch/model.py#L18), we can see that the hidden size ($d_{model}$) and the intermediate size ($d_{ff}$ for a single expert) are both 2880.  So the model really uses an "expansion" factor of only 1x for each expert!

(Claude correctly points out that a SwiGLU expert needs _three_ $d_{model} \times d_{ff}$ mappings -- including one that controls a gate on the hidden neurons.  Two of these mappings are stored in the same tensor within the implementation linked above. But the intermediate size really is the hidden size of the SwiGLUs in this implementation!)

Given this information, the total number of parameters for all the experts is $n_{layers} \times n_{experts} \times 3 \times d_{model} \times d_{ff}$ $= 36 \times 128 \times 3 \times 2880 \times 2880 = 114.B parameters$.

With the reported total parameter count for this network being 117B, the gpt-oss uses 98% of its learnable parameters in the mixture-of-experts layer!

# Does using a mixture of experts hurt model performance?

It's natural to wonder if a massively-parallel memorization mechanism like this hurts the performance of a transformer.  After all, one of the strengths of a transformer is in making connections between seemingly unrelated areas in its massive linear layers. With only four of 128 experts on for any given token, gpt-oss will likely have trouble making connections from one expert to another.

Indeed, as [pointed out in a hugging face blog post on mixture-of-experts](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe) shared with me by a student:

> [Switch Transformers](https://arxiv.org/pdf/2101.03961) observed that at a fixed pretrain perplexity, the sparse model does worse than the dense counterpart in downstream tasks, especially on reasoning-heavy tasks such as SuperGLUE. On the other hand, for knowledge-heavy tasks such as TriviaQA, the sparse model performs disproportionately well.

Similarly, the largest gemma version is released in both [a mixture-of-experts format](https://huggingface.co/google/gemma-4-E2B#mixture-of-experts-moe-model) and a dense format, where the dense format is expected to get better performance but require more computation.

It seems that using mixture of experts allows us to build transformers that are more effective at memorizing the entire contents of the internet and being efficient at inference time, but these models may be slightly less capable at reasoning in novel spaces outside of their training text.
