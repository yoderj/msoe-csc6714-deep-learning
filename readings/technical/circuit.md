title: Transformer Circuit Tracing

# Transformer Circuit Tracing

Consider [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 

**1. What part of the transformer does the cross-layer transcoder (CLT) replace to explain the behavior of the network?**
**2. In the local replacement model, what parts of the model are borrowed from the underlying model's activation on a fixed prompt?**
**3. In the local replacement model, what parts of the model are derived from the CLT?**
**4. Why is the attribution graph considered linear?**
**5. Why is pruning performed on the attribution graph?**
**6. How are the values of error nodes determined?**

Now consider the companion paper, [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (which unfortunately takes one of the few words distinguishing real neural networks from the artificial and applies it to the artificial!) 

Pick one of the twelve examples illustrated as tiled graphics

**7. What do you learn about the internal operation of LLMs from the first example you picked?**

Now pick another...

**8. What do you learn about the internal operation of LLMs from the second example you picked?**


One of the limitations of circuit tracing is that it does not consider the Q and K values matching each other in the attention mechanism -- it only tracks the fixed attention matrix and the flow of residual stream information through the V matrices.  Later work does consider Q and K:

[Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html)

**9. What parts of the attention pattern does this work explain?**

**10. What parts of the attention pattern does this work ignore?**

Consider one of the examples explaining a head in the paper. 

**11. What do you learn about the internal working of LLMs from the example you selected?**

**12. How much do circuit-tracing graphs change your understanding of how LLMs function? Why?**
