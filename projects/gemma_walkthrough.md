---
title: "Project: Exploring the internals of a transformer with the Gemma Walkthrough"
---

# Resources
Start from the [Gemma walk-through](gemma_walkthrough.ipynb). 

[Create an access token](https://huggingface.co/settings/tokens) within your HuggingFace account and read and agree to the Gemma terms. To download the Gemma model to Rosie, you may need to set the environment variable `HF_TOKEN` within your notebook.

Create a HuggingFace account (if you don't have one), then go to the [gemma-2b hugging face model page](https://huggingface.co/google/gemma-2b) and agree to the terms to get access to Gemma (it's not a completely open-source model, you see).  Look for a large panel near the top of the main text with a large button on it to request access.

You also need to [create an access token](https://huggingface.co/settings/tokens) within your HuggingFace account. 

On Rosie, you may find this code useful to get the model to download:
```
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = # TODO: Paste your token here. (And then delete it after your first successful download of the model!)
```
Be sure to delete your token out of your notebook before submitting your lab!

In a colab notebook, click on the key on the left panel, and add a new secret named `HF_TOKEN` whose value is your hugging face token that you copied after creating it.

If the above does not work to make your token accessible within a colab notebook, you *may* also need use this code. (Although I doubt it.)
```
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

This can be hosted on Rosie fairly easily, once you are [set up to use the CSC4611 conda environment](csc4611/rosie-vscode.md) and setting up your token to access the Gemma model on Rosie. You can also do your development within Colab, being aware that your colab environment will clear after a timeout.

# Filling out the notebook

Try your own text with the Gemma model. Note that this model is NOT set up in a chat mode. You need to give it a text that makes sense to extend, not just a prompt for a model.

Then, fill in the details in sections C.1. through C.6 as well as section C.10.  Come to class prepared to ask questions and share what you discovered while working through this notebook!

# Peer Review

As with the readings, the peer review will become available after the grading of the assignment is complete -- the Monday after the assignment is due -- and will be due a week after the assignment is due.  Remember - don't put your name on your .ipynb or .pdf so that the peer review can be anonymous.

# Submission

Submit your .iynb notebook with your TODOs filled in.
