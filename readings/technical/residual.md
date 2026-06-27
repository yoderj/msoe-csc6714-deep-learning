title: "Reading: The Residual Stream"

# Reading: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

Focus on Section 1, especially the second column in p1., the first column in p.2 and Figure 2.

**1. *Repeat* the definition of residual layer as given in this paper.** 

**2. *Explain* how residual connections make the learning task of a layer easier.**

# Return to [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

Focus on Figure 1 and Section 3.1.

**3. *Describe* where residual connetions are found in the Transformer architecture.**

Now focus on Figure 1 and Section 3.3.  This section describes one of the sub-layers.

**4. *Discuss* the relative size of the hidden embedding within the feed-forward layer to the residual stream embedding size.**

Now focus on the right side of Figure 2 and Sections 3.2.1 and 3.2.2. Recall that the output dimensions of a matrix multiplication are driven by the outer dimensions of the matrix multiplication.  Consider the embedding of a single attention head, *before* the heads are concatenated and multiplied by $W^O$.

**5. *Discuss* the relative size of the embedding for a token in the output of a single head's Attention(...) function.**

# On to [A mathematical framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

Noting that this paper makes [a lot of simplifications](https://transformer-circuits.pub/2021/framework/index.html#model-simplifications) take a look at the [High-Level Architecture](https://transformer-circuits.pub/2021/framework/index.html#high-level-architecture) diagram (which hopefully looks familiar) and the view of residual streams in the section [Virtual Weights and the Residual Stream as a Communication Channel](https://transformer-circuits.pub/2021/framework/index.html#residual-comms).

**6. *Explain* why the authors prefer the language *residual stream* to the alternative name *embeddings*.**

**7. *Describe* how the feed-forward layers and attention layers can "talk" to one another through the residual stream.**

**8. *Describe* how information can flow from one token's residual stream to another's.**

**9. *Describe* how the embeddings sizes and residual stream communication from this reading fit together to help you understand the role of each part in a Transformer.**

# Optional Extras

## For a definition of the ReLU activation function
... see: [AlexNet: ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). Focus on Section 3.1
