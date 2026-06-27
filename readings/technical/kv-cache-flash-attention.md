title: "Reading: Efficient transformer inference: KV-caching and flash attention"

# [Flash attention](https://arxiv.org/pdf/2205.14135)

Flash attention computes the same attention formulas we have been studying all term, but does so in such a way that less GPU memory is required.

Consider Figure 1 of the paper.

**1. What matrix is not fully realized at one time in memory when using flash attention? **

**2. Does the outer loop of flash attention loop over Q or K?  What would break in flash attention if these outer loops were swapped?**

**3. What is HBM? Compare and contrast it to GPU SRAM and CPU DRAM.**

**4. What is the primary advantage of Flash Attention over a more straight-forward implementation of attention?**

# KV-caching

In an earlier reading, we had an optional viewing of a video by Welch Labs, which provided an alternative perspective on the basics of the attention mechanism.

In this reading, we have a required viewing of this video, picking up where that first viewing left off.

Watch the video starting at [6:50](https://youtu.be/0VLAoVGf_74?t=410). 

**5. When using key-value caching, what values can be reused from past tokens when generating a new token?**

**6. When using key-value caching, what values don't need to be saved or reused at all to compute a new token's attention output?**

**7. When using key-value caching, what needs to be recomputed for each token generated?**

The kiwico commercial break ends at [12:23](https://www.youtube.com/watch?v=0VLAoVGf_74&t=743s).

**8. Describe how MQA and GQA help mitigate one of the biggest disadvantages of KV-caching**

(note that while you might find the DeepSeek latent query attention (LQA) part of the video fascinating, it's not required for this reading, either!  After watching this optional part of the video slowly several times, I've come to the conclusion that latent query attention effectively uses a single slightly larger key and value embedding for all heads within a layer, but then allows for the values to be re-embedded differently for each head. That this performs as well as it does is really surprising to me.)

