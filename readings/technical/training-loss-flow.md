title: "Reading: LLM training techniques"

There are three broad categories of techniques for training a transformer:

* Pretraining
* Fine-tuning
* Reinforcement learning

**Pretraining** is the simplest technique, but the most compute-expensive.  In this technique, a large corpus of text (roughly the size of the internet) is loaded one context window at a time into the transformer.  The input tokens are shifted one token to the left to produce the desired output tokens, and the transformer is trained to predict the right output token for EVERY input token in the text simultaneously.  This is repeated for every context window within the training set, usually only once, though with overlap between the blocks.

**Supervised fine-tuning (SFT)** uses exactly the same algorithm as pretraining, but instead of training on the entire internet, we continue to train the pretrained network, but using a smaller dataset that illustrates some specific task we want the network to perform.  For example, in instruction-fine-tuning, we provide the network with example conversations where the text formats the chat agent talking with a person, following the instructions the person gives and continuing the conversation.  While the algorithm is exactly the same, it is unclear how to generate training sets in these formats.  Many open-source models use ChatGPT's output to create example conversations, leaving open the question of how OpenAI created the fine-tuning datasets they use with ChatGPT in the first place.  OpenAI answers that question in section 3.2 of their [RLHF paper](https://arxiv.org/pdf/2203.02155). (Note that this section is NOT talking about RLHF. It is talking about a dataset for "supervised fine tuning." (See section 3.5)

**1. Describe the difference between pretraining and fine-tuning**

**2. Describe the similarities between pretraining and fine-tuning**

How does information flow into the network weights when using pretraining or fine-tuning, and how does that impact the strengths and weaknesses of these strategies? In this reading, we look into these questions.

We start with the Dive into Deep Learning (D2L) textbook, which provides important details about how deep learning networks are trained:

Read [Section 11.9.3.1 - (Pretraining Decoder-Only) GPT and GPT-2](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#gpt-and-gpt-2) of D2L.

**3. Describe how the output data is created for training the transformer block **

Consider what the [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper says about masking in Section 3.2.3.

**4. Where in the transformer is the causal mask applied? How is it implemented?**

**5. Why is a causal mask necessary in a decoder-only transformer?**

Consider Figure 1 in [Linearity of Relation Decoding in Transformer Language Models](https://arxiv.org/pdf/2308.09124).  Consider that, during training, tokens flow *forward* from input text (like "Miles Davis") through various internal embeddings (like `s`) to an output token (like `o`, "trumpet"), then backwards from the correct next token (like "trumpet") through the various internal embeddings (like (`s`)) to the input text (like Miles Davis).

Further consider the training data you saw in question 1, where each token receives a training sample, and the causal masking you saw in questions 2 and 3. Consider that information reaches a token both in the forward pass and in the backward pass that occurs during backpropagation.

**6. Describe which input tokens can influence the training of an internal embedding at the 7th token in the input sequence.**

Now consider that, during backpropagation, the objective is to update the weights of each module so that all the outputs of the transformer that depend on that model will move closer to their known outputs provided during training.

**7. Describe which output tokens can influence the training of an internal embedding at the 7th token in the input sequence.**

Recall pretraining and supervised fine-tuning use exactly the same training architecture and have the same information flows.

**8. Describe the big picture of how information flows through a transformer during pretraining and supervised fine-tuning**

Consider that instruction-fine-tuning uses a different dataset that pretraining. (See definitions in last week's reading)

**9. Give two examples of things instruction fine-tuning can add which may not have been learned in pretraining**

**10. How did OpenAI create their first instruction-fine-tuning dataset?**

# Reinforcement Learning

So far, we have seen how supervised fine-tuning and pretraining are essentially the same algorithm, but using different datasets.

**Reinforcement Learning** uses a very different algorithms. Rather than using a fixed corpus, the transformer generates several tokens of output, and this output is graded against some external criterion.  The model weights are then adjusted to increase the score (or "reward") provided by that external criterion.

There are two basic sub-techniques that use reinforecement learning:

**Reinforcement Learning from Human Feedback (RLHF)**: In this technique, for each prompt, we generate two model outputs, perhaps from the same model, or perhaps from competing models.  A human labels which prompt is better, which becomes the reward signal.  The model works to maximize the reward it received from the human.

Once you have an LLM that can respond in the correct instruction or chat format, you can use RLHF to furtherm improve the model to answer questions in a way that is more pleasing to humans.

**Reinforcement Learning with Verifiable Rewards (RLVR)**: In this technique, the model is working to solve an objective answer. Once it determines the final answer, an external scoring algorithm determines whether that answer is correct. This works especially well in fields like mathematics or writing software, where there is an exact answer, or the code produced by the model can be executed against a series of unit tests.   The score becomes the reward signal used to train the model.

**11. Compare and contrast supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF).

**12. Give an example of a task that you might use reinforcement learning with verifiable rewards to solve.**

For the next two questions, consider the four basic training techniques that we have seen so far:

* Pretraining
* Supervised fine tuning
* Reinforcement Learning with human Feedback
* Reinforcement Learning with verifiable rewards

**13. If you were building a chat agent, what three techniques would you likely use to train the transformer? List them in the order you would apply them. Explain.**

**14. If you were training a math agent, what three techniques would you likely use to train the transformer? List them in the order you would apply them.**


# (Optional) Constitutional AI

**(optional) Would you classify Anthropic's "[Old Constitution](https://www.anthropic.com/news/claudes-constitution)" or [New Constitution](https://www.anthropic.com/news/claude-new-constitution)as a pretraining technique, an SFT technique, an RLHF technique, an RLVR technique, or something else?**
title: "Reading: Loss flow in LLM training techniques"

