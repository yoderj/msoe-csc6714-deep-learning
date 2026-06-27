title: "Reading: Few-shot learning and RAG"

In *transfer learning*, a simpler task is learned by considering a larger dataset not customized to that task. For example, using the million-image ImageNet dataset during training improved accuracy on image classification tasks that used smaller datasets like CIFAR-10.

Fine-tuning is an example of transfer learning: An LLM fine-tuned on the internet carries over the knowledge of syntax, word meanings, etc. seen in the general internet to the fine-tuned task.

In *few-shot learning*, we take transfer learning to the extreme: The training set for the simpler task is very small. For example, in one-shot learning, only one input, along with its correct output, is shared with the model.  Transfer learning is essential in this context. The model must use its prior knowledge from the larger dataset on which it was trained to understand the example and its answer and extend these to new examples within the problem space.

In addition to one-shot learning, one can also see two-shot, three-shot, or other small numbers. In these cases, more examples with known answers are given to the model.

And zero-shot learning takes this to the extreme: In this case, the model is given no examples of correct output and is simply asked to solve problems in a new domain other than the one it was trained on. For example, a model might be asked to classify text into one of three named categories without being given any examples of what text should be assigned to each category. Clearly the generalized knowledge from transfer learning or pretraining is critical to these sorts of tasks!

LLMs perform well at few-shot learning tasks *even when we don't fine-tune them further*!  All one needs to do to perform "few-shot" learning with an LLM is to place examples (with the correct outputs) into the instructions in the model's context. And, if you've already written a prompt for an LLM that *didn't* include any examples, then you have already performed a zero-shot transfer learning task, even if you didn't know it at the time.

Consider this prompt:

```
Determine whether the item described is a vegetable or a fruit. Answer in this format:

<result>...</result>

Put only "vegetable" or "fruit" within the result tags.
```

**1. Is this prompt a zero-shot prompt? Why or why not?**

**2. Update this prompt to illustrate two-shot learning.**

**3. Give an example of how few-shot learning can be used to tweak prompt performance.**

In the grad version of this course, we took few-shot learning to the opposite extreme: We placed all the items in a grocery store into an LLM's context window and asked it to select the best matches for a short grocery list.  This zero-shot learning, with many classes.

Students got mixed results in this context.  For many students, the products cited in the model's response appeared nowhere within the list.  For others, the match was exact character for character.  Suppose the prompt is adjusted to encourage exact quotes from a list.  In this case, the model is performing a fuzzy lookup task:  Given a grocery item near the end of the prompt, the model searches for a similar item throughout the much longer list.

Suppose KV-caching is used. Consider Big-O runtimes like $O(\log n)$, $O(n)$, or $O(n^2)$, where $n$ is the number of items in the grocery store. 

**4. How much compute is required to generate each token of the model's response when doing a lookup in this context?**

Now, consider an alternative where each item is embedded with an embedding algorithm and stored in an approximate-nearest-neighbors structure such as a [Hierarchical Navigable Small Worlds](https://www.pinecone.io/learn/series/faiss/hnsw/) vector index. Suppose searching this index is $O(\log n)$. (In practice, it may be higher, but still less than $O(n)$.). To generate a response, each grocery-list item is embedded, the vector index is searched to yield the top three responses, and these three responses are fed to an LLM prompted to pick the best fit.

Again answer in Big-O runtime, considering the total time to embed, lookup an answer, and the generate an LLM response. 

**5. How much compute is required for search and answer generation combined in this context? Explain.**

Consider more broadly any task that might use a long context window for an LLM.

**6. What are the advantages of using a long context window with an LLM vs. a vector store? Give advantages for each**

(Just in passing, systems that incorporate both a vector store and an LLM summarizer or reranker are often called "Retrieval Augmented Generation" (RAG) systems, regardless of whether they are used as the basis for a chatbot or not. And, when performed through prompting, few-shot learning is one example of "in-context" learning.)
