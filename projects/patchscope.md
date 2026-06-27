---
title: "Project: Attribute extraction with patchscope"
---

In this project, you will extract attributes from embeddings within a transformer using the [patchscope](https://pair.withgoogle.com/explorables/patchscopes/) technique.  Focus your experiment on attribute extraction with patchscope, not on next-token prediction. In [the paper](https://arxiv.org/pdf/2401.06102), next-token prediction is discussed in Section 4.1 and attribute extraction is discussed in Section 4.2. Again, focus your experiment on the latter. In the pair-with-google article, next-token prediction is not discussed, making it easier to avoid that topic.

Perform your experiments in a Jupyter notebook (.ipynb) and report on your experiments in a brief report (.pdf). Like your reading reflections, your report should be anonymous to support peer review.

In your experiments, you should:

* Implement a patchscope algorithm that successfully decodes a single known attribute from an entity named in the source prompt. (For example, identify that the largest city in the UK is London)
* Identify the source and target layers that given the most reliable performance when determining the attributes of about a dozen embeddings (For example, identifying what source and target layer give the best accuracy at predicting the capitals of ten countries)
* Do a little extra exploration of a topic of interest to you.  For example, you might:
  * Explore a second object-attribute relationship to see if the same source and target layers are also best for it.
  * Explore if you can use patch-scope to probe a higher-level concept whose performance is best later in the network
  * Implement some kind of linear attribute lens and compare its performance with the patch-scope

In your report, include title information (but not your name), an introduction, figures and tables to support your conclusions from your experiment, a separate section titled appropriate to whatever extra explorations you performed, and a brief conclusion.

# Resources

You may wish to use a notebook hosted on colab or on Rosie.

Start from the [Gemma walk-through](https://colab.research.google.com/github/DouglasOrr/DouglasOrr.github.io/blob/examples/2024-04-transformers/gemma_walkthrough.ipynb) you used in the first lab.  This can be hosted on Rosie fairly easily, once you are [set up to use the CSC4611 conda environment](csc4611/rosie-vscode.md) and setting up your token to access the Gemma model on Rosie. You can also do your development within Colab, being aware that your colab environment will clear after a timeout.

If you haven't done so yet, [create an access token](https://huggingface.co/settings/tokens) within your HuggingFace account. To download the Gemma model to Rosie, you may need to set the environment variable `HF_TOKEN` within your notebook.

# Peer Review

As with the readings, the peer review will become available after the grading of the assignment is complete -- the Monday after the assignment is due -- and will be due a week after the assignment is due.  Remember - don't put your name on your .ipynb or .pdf so that the peer review can be anonymous.

# Excellent Credit

The "little extra" exploration will be worth 5% of the grade of the lab.  You are free to opt out of it, but your maximum grade on the lab will be at most 95% in that case.

# Submission

Submit both your .iynb notebook with experiments and your .pdf report
