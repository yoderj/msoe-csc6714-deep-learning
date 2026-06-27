---
title: "Project: Training an LLM (Part 3: Exploring your model's attention patterns)"
---

This is the third week in a multi-week lab on training LLMs.

In this part, you will examine the attention patterns of your model while it successfully calculates an arithmetic expression and make hypotheses about the roles of the various attention heads in your model.

# Rosie setup

You will again likely need Rosie for this assignment, although you could also try a colab notebook again.

On Rosie, you may want to use the conda environment set up for CSC 4611:

```
conda activate /data/csc4611/conda-csc4611/
```

The [instructions for setting up VS Code on Rosie](http://example.com) may also be useful here.

# Assignment
The provided notebook is built against the nanoGPT setup.  It is configured to take a trained model in `'out-shakespeare-char'` and run it to extend some text.  You will need to modify the start text and the function `def result_complete(...)` to work with your format.  Once complete, this notebook displays the attention patterns used to generate the last token of output.

Explore some of the attention heads illustrated in the overview of all the attention heads.  Describe the behavior of the heads which you believe have some of the most important roles in processing text, such as the head for copying the innermost expression, or the head(s) that play a key role in identifying the innermost expression, or any heads that have a role in evaluating the simple arithmeti expressions.

Find and comment on at least one example of a copying head.  The attention pattern of a copying head is a diagonal line showing each token attending a fixed distance back to the token it is copied from.

Because the value matrices projects the residual stream to a lower dimension, a single copying head cannot copy the full space of the residual stream.  When this is necessary, sometimes several or all the heads within a layer will learn the same copying attention pattern, but presumably with their value embeddings capturing different subspaces of the residual stream so that, together, the entire residual stream can be copied from one location to another.  Can you find an example of heads working together like this for your transformer? Comment on your search and any findings.

# Report

Submit: Your modified attention.ipynb, including the heads you chose to focus on and what you think their roles are.
