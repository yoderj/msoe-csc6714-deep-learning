---
title: How ChatGPT uses Matrix Multiplications
---

In AI, it is common to want to “embed” words into vectors, such that words with similar meanings embed closer to each other within the vector-space.

# Matrix multiplication for embedding lookup

One way to do this – and indeed, the way used by the Transformer architecture – is to first “one-hot encode” every word in your language:

```
apples –  [1, 0, 0, 0, …]
oranges – [0, 1, 0, 0, …]
bananas – [0, 0, 1, 0, …..]
I –       [0, 0, 0, 1, …..]
eat  –    [0, 0, 0, 0, 1 …..]
and –     [0, 0, 0, 0, 0, 1 …..]
etc.
```

Then, you can encode any sentence, by just using the one-hot codes for that sentence:

“I like apples”

S for sentence:

```
S =
 [0, 0, 0, 1, …..]      -- I
 [0, 0, 0, 0, 1 …..]    -- eat
 [1, 0, 0, 0, …]        -- apples
 [0, 0, 0, 0, 0, 1 …..] -- and
 [0, 0, 1, 0, …..]      -- bananas
```

Here, the one-hot encoding is SOMETHING like a permutation matrix, but not quite – since we can reuse words, and don’t have to use every word.

Multiply this matrix by another matrix on the right, which we call the embedding matrix:

D for dictionary:

```
D = 
[ 0.1,   0.4,     0] - apples
[ 0.1,   0.3,     0] - oranges
[ 0.2,   0.4,     0] - bananas
[   0,     0,  -0.5] - I
[   0,     0,  -0.6] - eat
[   0,     0,   0.5] - and
```

And you get the embedding (E) of your sentence:

```
E = 
[   0,     0,  -0.5] - I
[   0,     0,  -0.6] - eat
[ 0.1,   0.4,     0] - apples
[   0,     0,   0.5] - and
[ 0.2,   0.4,     0] - bananas
```

This alone is pretty cool – you can represent dictionary lookup with a matrix multiplication.  In practice, we’d probably do something more efficient that didn’t require floating-point multiplication.  But it’s still a cool way to think about the embedding process because we will continue to re-embed even the embeddings with further matrix multiplications later on.

For example, suppose some dimensions of your embedding carry the meaning “fruit” and other dimensions can allow you to distinguish fruit.  (My example earlier isn’t complex enough to illustrate this idea.)  Doing a matrix multiplication allows you to project to a subspace that ignores differences between fruit or to a subspace that EMPHASIZES the differences between fruit.

# Matrix multiplication for attribute extraction

Now, suppose your embeddings are in some subspace that works well for what you are trying to accomplish.  For example, suppose we are in a subspace that keeps the original fruit embeddings but ignores other words in the sentence:

W for weights

```
W =
[1,  0, 0]
[0,  1, 0]

E * W  = Q = 
[   0,     0] - I
[   0,     0] - eat
[ 0.1,   0.4] - apples
[   0,     0] - and
[ 0.2,   0.4] - bananas
```

# Matrix multiplication for word (or token) linking

Now, we can match one fruit in the matrix to another fruit in the matrix by multiplying Q by itself (transposed):

(We use the name Q just following the Attention is All you need paper, https://arxiv.org/pdf/1706.03762.pdf. )

```
Q * Q^T =
[0, 0, 0,    0, 0   ] – I 
[0, 0, 0,    0, 0   ] - eat
[0, 0, 0.17, 0, 0.18] - apples
[0, 0, 0,    0, 0   ] – and 
[0, 0, 0,18, 0, 0.2 ] - bananas
```

This allows us to build an “attention” matrix that connects parts of a sentence that have similar meaning. This operation is core to transformers like ChatGPT.

Notice how multiplying the matrix by its transpose allows us to compute the dot-product of every row of Q with every other row of Q.  This is sort of an “outer product” and an “inner product” at the same time – it allows us to build a matrix where every element describes how two words in the input are connected.

Transformers (like ChatGPT) use matrix multiplication in a third way.  And this third way is most closely related with the linear algebra concept of a sub-space.

After performing the Q * Q^T multiplication (technically, we perform a multiplication of Q * K^T, where Q and K use DIFFERENT projections of the embedding, this is important so that we can make concepts like “verb” match up with concepts like “noun”, but that’s an aside I don’t want to expand upon unless asked… you could also see equation 1 of https://arxiv.org/pdf/1706.03762.pdf), we normalize the rows so that they sum to 1 (again, technically, this is done with a softmax operation, but here, I’ll just do the standard technique of dividing each row by its own sum which is how softmax normalizes anyway.  It also raises each element to a power, which I don’t feel like doing here.)

```
A = 
[1, 0, 0,    0, 0   ] – I 
[0, 1, 0,    0, 0   ] - eat
[0, 0, 0.48, 0, 0.52] - apples
[0, 0, 0,    1, 0   ] – and 
[0, 0, 0,47, 0, 0.53] - bananas
```

(Please don’t ask where the 1’s on the diagonal come from – they wouldn’t actually happen here, but in practice words are likely to embed better to themselves than to other words, thus normalizing to a value of 1 within their own spot within their own row.)

# Matrix multiplication for embedding blending

Now, if we multiply the matrix A by our original embeddings 

```
A * E = 
[   0,     0] - I
[   0,     0] - eat
[ 0.15,   0.4] - apples
[   0,     0] - and
[ 0.15,   0.4] - bananas
```

We can see that the two words are embedded based not only on their original embeddings, but also on the embeddings of other words in the same sentence – this is important to the network, because a single word’s token can build up other important, related words into its own meaning.  (This example might work better if, for example, we allowed one of a few choices of verbs to match to one of a few choices of subjects.)

As a Transformer repeats this process in many layers, eventually we reach a point where a whole sentence’s meaning can be embedded into a single word’s position within the sentence.  This then allows sentences to be matched through the Q * Q^T operation over long distances of text. This is why you can paste an entire reading assignment into ChatGPT and then ask ChatGPT about it: It embeds your question, embeds the various sentences within the text, and then matches the embeddings through a dot product (matrix multiplication) so that it can then copy info out of that part of the paper into its response.

But let’s pause and think about the linear algebra implications here.

When we multiply A * E, what is that doing?  In practice, most rows in most A matrices are all zeros except for one, two, or perhaps three or four positions (e.g. See Figure 4 of https://arxiv.org/pdf/1706.03762.pdf).  Thus the new embedding for that row will be a linear combination of the embeddings of the three or four words coming into that row.

Thus, the attention operation we are describing cannot invent “new” meanings. Instead, the meanings are convex interpolations within the subspace defined by the embeddings for the words which that row combines.

Thinking in terms of linear algebra leads us to many insights about how ChatGPT-like transformers operate.  In particular, we have seen that they use the same operation – matrix multiplication – in three ways:

* **Attribute extraction**: to perform projections or mappings of each embedding into a new subspace, independent of the other embeddings
* **Word linking**: To compare embeddings and find similarities across long or short distances within text
* **Embedding blending**: To combine embeddings to create new embeddings which are not entirely new, but rather fall within a subspace define by existing embeddings

ChatGPT itself is able to [identify these three ways](https://chat.openai.com/share/9609ffed-b7f1-44f9-a1d9-b6771519b2da) with very little prompting.

If you have questions about the linear algebra, please consult with your instructor. If you have questions about the deep learning, I would be happy to answer them. – Dr. Yoder

