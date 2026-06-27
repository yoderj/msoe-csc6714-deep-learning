---
title: "Project: Grocery Linking (Part 2: Embeddings)"
---

Embeddings provide an efficient way to process text according to its semantic meanings.

In this lab, you will use embeddings to find a candidate list of products and then use your LLM prompt on this shorter list to select a good candidate.

Consider the example code for the [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) model.  This example code computes the cosine similarity of every query to every document, with the queries along the rows of the matrix and the documents along the columns.

Apply Qwen3-Embedding-0.6B (or another small embedding model of your choice) to all the items available in `grocery_store.csv`.  Find the five or ten most similar items in the list.  For each grocery list item, provide the Llama LLM with the top choices and ask it to select the best item from the list.

(Note: In a commercial application like this, we would likely store the embeddings in a vector database that provides sub-linear search time such as [hierarchical navigable small world (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/) lookup. (That is, it will find the closest embeddings without needing to consider all embeddings at search time.) **You don't need to use these advanced techniques in this lab.**)

As with the previous part, ensure the response for each item is included in the output of one of your cells near the end of your notebook.

Furthermore, in a text cell near the end of your notebook, comment on the relative advantages and disadvantages of the embedding technique compared to using the LLM directly.
